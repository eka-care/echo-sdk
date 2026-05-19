---
name: echo-sdk-adding-a-skill-to-an-agent
description: End-to-end recipe for defining a Skill bundle and wiring it into a GenericAgent with the right activation mode and tool registry. Use when extending agent capability via swappable skills rather than baking tools in.
---

# Adding a Skill to an Agent

A `Skill` is a swappable bundle of `(instructions + tool_names + description)`. Use skills when the LLM should be able to load/unload capability mid-conversation, or when a host needs to scope which tools are visible per request.

## Quick recipe

```python
from echo.agents import GenericAgent, Skill, AgentConfig
from echo.tools import BaseTool

# 1. Define the tools (they live in the agent's registry by NAME).
class TriageQuery(BaseTool):
    name = "triage_query"
    description = "Look up triage history for a patient."
    @property
    def input_schema(self): return {"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]}
    async def run(self, patient_id: str, workspace_id: str):   # workspace_id from tool_context
        ...

# 2. Define the skill — references tools BY NAME, doesn't own them.
triage_skill = Skill(
    name="triage",
    description="Patient triage workflows.",
    instructions="When the user asks about triage, use triage_query first...",
    tool_names=["triage_query"],
)

# 3. Wire into the agent.
agent = GenericAgent(
    agent_config=AgentConfig(...),
    tools=[TriageQuery()],          # registry
    skills=[triage_skill],
    skill_activation="llm",         # or "manual"
    base_tool_names=[],             # nothing visible until a skill loads
)
```

## Rules

- **`Skill.tool_names` must resolve** in the agent's tool registry at construction. Unresolved skills are dropped with a warning — verify your test logs.
- **Skill does not own tool instances.** It references them by name. Two skills can share a tool; it appears once in the visible set.
- **Activation modes**:
  - `"llm"` — agent auto-injects `load_skill` / `unload_skill` meta-tools; the LLM picks.
  - `"manual"` — host calls `await agent.activate_skill("triage", context)`.
- **Lifecycle hooks**: subclass `Skill` and override `on_activate(context)` / `on_deactivate(context)` for setup/teardown (auth fetch, telemetry). Hooks are async. `context` is `None` during setup.
- **Idempotent activation** — re-activating a live skill is a no-op; hooks don't re-fire.
- **`base_tool_names=[]`** is the "everything skill-gated" pattern. `None` (the default) means "all non-meta tools visible by default."
- **Reserved tool names**: `load_skill`, `unload_skill` cannot be used by user tools when `skill_activation="llm"`.

## When NOT to use a skill

- Tool is always needed → just put it in `tools` and let `base_tool_names` default expose it.
- The "instructions" you'd put in a skill are core persona → put them in `AgentConfig.persona` instead.

## Common mistakes

- Forgetting that meta-tools (`load_skill`, `unload_skill`) consume tokens — for short-context models, prefer `manual`.
- Putting auth tokens in `Skill.instructions` (they end up in the prompt) → fetch in `on_activate`, pass via `tool_context`.
- Mutating skill state from a tool call → tools should be stateless w.r.t. the skill; skill state lives on the subclass instance.

## See also

- `[[echo-sdk-agents]]` for the registry mechanics.
- Diagram: `.claude/diagrams/agent-skill-lifecycle.md`
- `examples/skill_agent_usage.py`
