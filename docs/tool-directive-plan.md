# Plan: Generic tool-directive model + `tools/` restructure

> Status: design locked, implementation pending. Step 0 first (pure relocation, no behavior change).

## The spine (what everything reduces to)

> **A tool's behavior is a *directive on its result* — `control_flow`
> (`CONTINUE` / `INTERRUPT` / `PAUSE`, extensible) + `observability`
> (`VISIBLE` / `SILENT`). The tool *type / origin* decides whether the
> directive is fixed or settable. The loop just reads it — never
> `isinstance`, never feature-aware.**

| Tool | control_flow | settable? | observability |
|------|--------------|-----------|---------------|
| `BaseTool` (user/external) | `CONTINUE` default | yes, but **never `INTERRUPT`** | `VISIBLE` default, settable |
| `BaseElicitationTool` / `ElicitationResponse` | `PAUSE` | **fixed** (class constant, no ctor arg) | `VISIBLE` fixed |
| `SystemTool` (ours only, guarded) | `CONTINUE`/`INTERRUPT` | yes | per-tool |
| `MCPTool` | decided at runtime from response | n/a | from chosen result type |

### Two orthogonal axes (complete; origination handled by *where the call is made*, not a tool axis)
- **control-flow**: what the loop does after the tool runs.
- **observability**: whether running it emits a user-facing event.
- Result-routing (to model / to user / consumed by agent) is *derivable* from control-flow, not a third axis.

### Rerun rule (agent owns it)
> After `invoke()` returns: **rerun iff (loaded state changed via `INTERRUPT`) AND (not returning to host).**
- Activation and rerun are **separate**: `load_skill` mutates the active set immediately (agent state); rerun is a later decision.
- **Elicitation wins**: if a turn yields both an interrupt and an elicitation, return to host; the activated skill persists (agent state + LMC snapshot in `agent_service.py`), and the next `run()` rebuilds prompt+tools for free.

### Key invariant
> **Only our `SystemTool` can emit `INTERRUPT`** — type-enforced via `__init_subclass__` (subclassing allowed only inside `echo.`). External/MCP tools structurally cannot trigger a recompute.

---

## Target structure

```
src/echo/
  tools/                       ← FRAMEWORK ONLY
    core/
      schemas.py               ← shared: directive enums, shared result base,
                                 ElicitationDetails/Status/Component, MCPServerConfig, errors
      constants.py
    base_tool.py               ← BaseTool (defaults CONTINUE/VISIBLE)        → core
    base_elicitation_tool.py   ← BaseElicitationTool (PAUSE/VISIBLE fixed)    → base_tool, core
    system/
      system_tool.py           ← SystemTool base + __init_subclass__ guard   → base_tool, core
    mcp/
      mcp_tool.py              ← MCPTool (runtime: elicit→PAUSE else default) → base_tool, core
      connection_manager.py    ← MCPConnectionManager                        → mcp_tool, core
    __init__.py                ← exports FRAMEWORK only; never imports skills/ or databases/

  skills/                      ← SKILL DOMAIN
    skill.py, runtime.py
    meta_tools.py              ← LoadSkill/UnloadSkill (SystemTool)           → tools/system, tools/core

  databases/                   ← DATABASE DOMAIN (resource + its tool)
    postgres/
      config.py, client.py, binder.py
      pg_query_tool.py         ← PgQueryTool(BaseTool)                        → tools/base_tool, tools/core, sibling client
```

### Dependency direction — one-way, no cycles
```
tools/core      → (nothing internal)          ROOT
tools/base_tool → core
tools/{system, mcp, base_elicitation} → base_tool, core
skills/         → tools/system, tools/core
databases/postgres → tools/base_tool, tools/core
agents/         → skills, databases, tools, llm
```
**The rule that prevents a cycle:** `tools/__init__` exports only the framework and **never imports a domain** (`skills`, `databases`). Each domain exports its own tools.

---

## Step 0 — Restructure `tools/` (pure relocation, NO behavior change; land & verify alone)
- Move `schemas.py` → `tools/core/schemas.py`; add `tools/core/constants.py`.
- Rename `base_elicitation.py` → `base_elicitation_tool.py`.
- Move `mcp_tool.py` → `tools/mcp/mcp_tool.py`, `mcp_connection_manager.py` → `tools/mcp/connection_manager.py`.
- Add `tools/system/system_tool.py` (SystemTool + `__init_subclass__` guard — defined here, adopted in later steps).
- Move `tools/databases/pg_query_tool.py` → `databases/postgres/pg_query_tool.py`; export from `echo.databases`. Delete `tools/databases/`.
- Move `tools/skills/meta_tools.py` → `skills/meta_tools.py`. Delete `tools/skills/`.
- `tools/__init__` re-exports framework only (BaseTool, BaseElicitationTool, MCPTool, MCPConnectionManager, ElicitationResponse, ElicitationDetails, MCPServerConfig, directive enums once added).
- **Importer updates:** `agents/base.py:17` (`echo.tools.skills`→`echo.skills`); `matrix/med_assist` (`echo.tools.mcp_tool`, `echo.tools.mcp_connection_manager`, `echo.tools.schemas` → new paths or `echo.tools` re-exports). Keep `from echo.tools import MCPTool/MCPConnectionManager/ElicitationResponse/MCPServerConfig` working.
- **Docs:** `echo-sdk/overview` skill module-map + `CLAUDE.md` layout section.
- ✅ Verify: import smoke + existing tests green before any behavior change.

## Step 1 — Directives in `core/`
- `ControlFlow` (`CONTINUE`, `INTERRUPT`, `PAUSE`; room for `STOP`) + `Observability` (`VISIBLE`, `SILENT`) enums.
- Shared result base exposing `control_flow` + `observability`. `ToolResult` defaults `CONTINUE`/`VISIBLE`, settable. `ElicitationResponse` subclasses it with both fixed as class constants (ctor unchanged for matrix).

## Step 2 — Tool types adopt directives
- `BaseTool`: default `CONTINUE`/`VISIBLE`; can set observability + `PAUSE`; **no path to `INTERRUPT`**.
- `BaseElicitationTool`: always `ElicitationResponse` (PAUSE/VISIBLE fixed).
- `SystemTool`: `__init_subclass__` guard; may stamp `INTERRUPT`, choose observability.
- `MCPTool.run()`: response→directive mapping (elicit ⇒ `ElicitationResponse`, else default `ToolResult`).

## Step 3 — Provider inner loop reads directive (`llm/`)
- `anthropic.py` first: replace `isinstance(tool_res, ElicitationResponse)` with `result.control_flow`. Break on `PAUSE`/`INTERRUPT` **after** appending tool-results message (valid re-entry). Use `observability` to gate `TOOL_CALL_*` events (replaces `is_elicitation` skips). Surface outcome via `LLMResponse` (`pending_context_reload` for INTERRUPT; existing `elicitations` for PAUSE).
- Mirror into `openai.py`, `bedrock.py`, `gemini.py` — same generic read.

## Step 4 — Agent outer loop (`agents/base.py`)
- Wrap invoke/invoke_stream in `_run_agent`/`_run_agent_stream` with **rerun-iff** loop: on INTERRUPT & not returning to host → recompute `_build_system_prompt()` + `_build_active_tools()` → re-invoke; bound by `max_skill_reloads`.
- Elicitation precedence: present → return to host (no rerun).
- Streaming: swallow intermediate `DONE`s; emit `DONE` only on final non-reloading pass.
- Reserve slot for loop-originated policy (auto-`summary` on token threshold) at same recompute site — not built now.

## Step 5 — Skill meta-tools rebase
- `LoadSkillTool`/`UnloadSkillTool` → subclass `SystemTool`, stamp `INTERRUPT` + (likely) `SILENT`. Live in `echo/skills/meta_tools.py`.

---

## Deferred (explicitly not now)
- `DatabaseTool` base — until a 2nd backend (avoid premature abstraction).
- `STOP` control-flow value — additive later.
- Summary tool + loop-originated policy — slot reserved in Step 4.
- Origination as a tool axis — not needed.

## Invariants preserved
orjson-only · async-first · `ConversationContext` immutable (rerun uses returned context) · optional-deps guarded · factory pattern untouched · Pydantic v2 for new schemas/enums · only `SystemTool` can `INTERRUPT`.

## Notes
- Existing `breakpoint()` calls at `anthropic.py:170` and `base.py:186` are intentionally left as-is per owner instruction.
