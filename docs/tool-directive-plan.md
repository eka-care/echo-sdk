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
  tools/                       ← FRAMEWORK ONLY (foundation + categories)
    core/                      ← FOUNDATION (dependency root; imports nothing else from echo)
      base_tool.py             ← BaseTool (universal contract; defaults CONTINUE/VISIBLE)
      schemas.py               ← shared: directive/result base, Elicitation* payloads, ToolOutput
      __init__.py              ← exports BaseTool + shared schemas
    elicitation/               ← CATEGORY
      base_elicitation_tool.py ← BaseElicitationTool (PAUSE/VISIBLE fixed)    → core
    mcp/                       ← CATEGORY (pulls optional mcp/httpx)
      mcp_tool.py              ← MCPTool (runtime: elicit→PAUSE else default) → core, .schemas
      connection_manager.py    ← MCPConnectionManager                        → .mcp_tool, .schemas
      schemas.py               ← MCP-private: MCPServerConfig, errors, transport
    system/                    ← CATEGORY (echo-internal)
      system_tool.py           ← SystemTool + __init_subclass__ guard         → core
    __init__.py                ← re-exports ONLY BaseTool (convenience); no re-aggregation

  skills/                      ← SKILL DOMAIN
    skill.py, runtime.py
    meta_tools.py              ← LoadSkill/UnloadSkill (SystemTool)           → tools/system, tools/core

  databases/                   ← DATABASE DOMAIN (resource + its tool)
    postgres/
      config.py, client.py, binder.py, registry.py
      pg_query_tool.py         ← PgQueryTool(BaseTool)                        → tools/core, sibling client
```

### Dependency direction — one-way, no cycles
```
tools/core      → (nothing internal)          ROOT
tools/{elicitation, mcp, system} → core
skills/         → tools/system, tools/core
databases/postgres → tools/core
agents/         → skills, databases, tools, llm
```
**Import policy (tools/):** each subpackage owns its public API; `tools/__init__` does NOT re-aggregate — import from `echo.tools.core` / `.mcp` / `.elicitation` / `.system`. Only `BaseTool` is re-exported at top level. Keeps `import echo.tools` lean and free of optional `mcp`/`httpx`. Schemas follow **lowest-common-ancestor** (cross-category → `core`; category-private → its category). `tools/__init__` never imports `skills`/`databases` → acyclic.

---

## Step 0 — Restructure `tools/` (pure relocation, NO behavior change) — ✅ DONE
- `schemas.py` split: shared → `tools/core/schemas.py`, MCP-private → `tools/mcp/schemas.py`.
- `base_tool.py` → `tools/core/base_tool.py`; `base_elicitation.py` → `tools/elicitation/base_elicitation_tool.py`.
- `mcp_tool.py` → `tools/mcp/mcp_tool.py`, `mcp_connection_manager.py` → `tools/mcp/connection_manager.py`.
- Added `tools/system/system_tool.py` (SystemTool + `__init_subclass__` guard; adopted in later steps).
- `tools/databases/pg_query_tool.py` → `databases/postgres/pg_query_tool.py`; default-client registry extracted to `databases/postgres/registry.py` (breaks cycle). Public path: `echo.databases.postgres`.
- `tools/skills/meta_tools.py` → `skills/meta_tools.py` (exported from `echo.skills`).
- **Lean `__init__` policy applied to tools/:** subpackages own their exports; `tools/__init__` re-exports only `BaseTool`. Import from `echo.tools.core` / `.mcp` / `.elicitation` / `.system`.
- **Importers updated:** echo-sdk src/tests/examples + all `matrix/med_assist` files (BaseTool via `echo.tools`; MCP names via `echo.tools.mcp`; elicitation via `echo.tools.core`).
- **Docs:** `echo-sdk/overview` skill + `CLAUDE.md` layout/import-policy notes.
- ✅ Verified: import smoke + `import echo` no longer drags optional `mcp`/`httpx`; SystemTool guard rejects external subclasses; full suite 112 passed / 12 pre-existing failures (proven unrelated via stash test).

## Step 1 — Directives in `core/` — ✅ DONE
- Added `ControlFlow` (`CONTINUE`, `INTERRUPT`, `PAUSE`; room for `STOP`) + `Observability` (`VISIBLE`, `SILENT`) enums in `tools/core/schemas.py`, exported from `tools/core/__init__`.
- **Way 1 (directive on the result), no shared base / no envelope:**
  - `ToolResult` (`models/user_conversation.py`) gained settable `control_flow`/`observability` fields, `default` CONTINUE/VISIBLE, `Field(exclude=True)` so they never persist (model_dump/DB) — transient processing signals. New one-way edge `models → tools.core.schemas` (no cycle; tools never imports models).
  - `ElicitationResponse` exposes `control_flow`/`observability` as **read-only properties** returning PAUSE/VISIBLE — fixed, non-overridable (not fields). Loop reads both objects uniformly by duck-typing `.control_flow`/`.observability`.
- No loop / tool-type / `invoke_tool` behavior change. Verified: defaults, settability, persistence exclusion, elicitation fixedness; full suite 112 passed / 12 pre-existing.

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
