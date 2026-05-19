---
name: echo-sdk-agents
description: Building agents with BaseAgent / GenericAgent, including skill registration, activation modes (llm vs manual), and system-prompt composition. Use when editing src/echo/agents/ or building a new agent subclass.
---

# Agents

## What you're working with

- `BaseAgent` (abstract) in `src/echo/agents/base.py` — owns tool registry, skill registry, LLM, system-prompt composition.
- `GenericAgent` in `generic_agent.py` — the concrete 90% path; YAML-config-driven.
- `Skill` in `skill.py` — bundle of `(name, description, instructions, tool_names)`. Subclass only for `on_activate` / `on_deactivate` hooks.
- `AgentConfig` / `PersonaConfig` / `TaskConfig` in `config.py` + `load_agent_config()` for YAML.
- `AgentResult` in `schemas.py`.

## Rules

- **Subclass `BaseAgent` only when `GenericAgent` doesn't fit.** Most use cases just instantiate `GenericAgent(agent_config=..., llm_config=..., tools=..., skills=...)`.
- **Tool registry is name-keyed and dedup'd.** Duplicate names, empty names, and the reserved names `load_skill` / `unload_skill` are silently dropped with a warning.
- **`Skill.tool_names` must resolve** in the agent's tool registry at construction — unresolved skills are dropped with a warning.
- **`skill_activation`**:
  - `"manual"` — host calls `await agent.activate_skill(name, context)`.
  - `"llm"` (default) — `load_skill` / `unload_skill` meta-tools are auto-injected.
- **`base_tool_names`**: explicit empty list `[]` means "nothing default-visible — everything is skill-gated." `None` means "all non-meta tools visible by default."
- **Visibility is recomposed per turn** as a name-set union: `base ∪ ⋃(active skill.tool_names) ∪ meta`. See `_build_active_tools` in `base.py`.
- **Activation is idempotent.** Re-activating a live skill is a no-op; hooks don't re-fire.
- **`on_activate` / `on_deactivate`** receive `Optional[ConversationContext]` — `None` during setup, live context during a turn.

## System prompt layout

When skills are registered, the prompt becomes:
```
<base_user_prompt>...role/goal/task...</base_user_prompt>
<skill_mechanism>...</skill_mechanism>
<available_skills>...</available_skills>   (llm mode only)
<active_skills>...</active_skills>
<active_skill name="x">instructions</active_skill>
```
The mechanism template lives in `src/echo/prompts/templates/` (loaded by `load_template("skill_mechanism")`).

## Run methods

- `async def run(context, out_msg_id, **kwargs) -> AgentResult` — non-streaming.
- `async def run_stream(context, out_msg_id, **kwargs) -> AsyncGenerator[StreamEvent, None]` — streaming; terminal `DONE` event holds full response + updated context.

Both must catch exceptions and return/yield an error result (`AgentResult(error=...)` or `StreamEvent(type=ERROR)`) — never let the agentic loop bubble unhandled.

## Common mistakes

- **Mutating `self.tools` mid-conversation** → don't; tool list is computed per turn from the registry.
- **Putting auth tokens in skill instructions** → use `Skill.on_activate` to fetch tenant-scoped tokens at activation time.
- **Calling `activate_skill` before agent setup completes** without passing `context=None` → pass `None` explicitly during setup.
- **Forgetting that meta-tools count toward token budget** in `llm` mode — for very small contexts, prefer `manual` activation.

## See also

- Diagram: `.claude/diagrams/agent-skill-lifecycle.md`
- Sibling skills: `[[echo-sdk-llm]]`, `[[echo-sdk-tools]]`, `[[echo-sdk-models]]`, `[[echo-sdk-adding-a-skill-to-an-agent]]`
