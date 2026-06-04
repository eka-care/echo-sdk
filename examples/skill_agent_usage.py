"""
Example: GenericAgent with Skills (LLM-driven and manual activation).

A Skill bundles `(prompt fragment + tool name references + description)` that
the agent attaches and detaches mid-conversation. Tools themselves live in
the agent's name-keyed registry; a skill just references the names it needs
visible while it is active. Two skills referencing the same tool name see
the same instance and the tool appears once in the LLM's tool list.

This example shows both activation modes side by side using stub tools, so
the wiring is visible without needing provider credentials.

Concepts demonstrated:
  - Defining a Skill that references tools by name (`tool_names`).
  - LLM-driven mode: the agent auto-injects `load_skill` / `unload_skill`
    meta-tools and an <available_skills> registry block. The model decides
    when to load.
  - Manual mode: the host calls `await agent.activate_skill(name)` directly.
    The model never sees a registry — it just sees whatever is currently
    active.
  - `base_tool_names` to mark a subset of `tools` as always visible; the
    rest become visible only when a skill names them.
  - Two skills sharing a tool: the tool appears exactly once in the per-turn
    tool list while both skills are active.

To wire this into a real LLM, swap the agent's `llm_config` for one that
matches your environment (see examples/mcp_agent_usage.py for a Bedrock
example) and feed messages via `await agent.run(context, out_msg_id)`.
"""

import asyncio
from typing import Any, Dict

from echo.agents import GenericAgent, Skill
from echo.prompts.schemas import AgentPrompt, PromptPersona, PromptTask
from echo.tools.core import BaseTool


# --- Stub tools (real skills would use real BaseTool subclasses) ---


class _StubTool(BaseTool):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    async def run(self, **kwargs: Any) -> Any:
        return f"[stub] {self.name}({kwargs!r})"


# --- Skill definitions ---

DOCTOR_BOOKING_INSTRUCTIONS = """\
You are now operating in the doctor booking flow.

1. Discover relevant doctors using the search_doctors tool.
2. Confirm the patient's preferred slot.
3. Confirm the booking and summarize.

Stay focused on doctor booking until the user explicitly asks for something
else. If the user pivots to a different need, call unload_skill so the agent
can route the new intent.
"""

LAB_BOOKING_INSTRUCTIONS = """\
You are now operating in the lab test booking flow.

1. Discover relevant lab packages using the search_lab_packages tool.
2. Confirm slot and address.
3. Confirm the booking and summarize.
"""


def build_tools() -> list:
    """Build every tool the agent can ever use. Both skills share `note_user`
    so we can demonstrate name-set dedup when both are simultaneously active.
    """
    return [
        _StubTool("search_doctors", "Search the directory of doctors."),
        _StubTool("search_lab_packages", "Search lab test packages."),
        _StubTool("note_user", "Persist a short note about the user."),
    ]


def build_doctor_skill() -> Skill:
    return Skill(
        name="doctor_booking",
        description=(
            "Use when the user wants to find or book a doctor appointment "
            "(by symptom, specialty, doctor name, or hospital)."
        ),
        instructions=DOCTOR_BOOKING_INSTRUCTIONS,
        tool_names=["search_doctors", "note_user"],
    )


def build_lab_skill() -> Skill:
    return Skill(
        name="lab_booking",
        description="Use when the user wants to book a diagnostic / lab test.",
        instructions=LAB_BOOKING_INSTRUCTIONS,
        tool_names=["search_lab_packages", "note_user"],
    )


# --- Agent setup (shared) ---

AGENT_CONFIG = AgentPrompt(
    persona=PromptPersona(
        role="hospital assistant",
        goal="Help patients accomplish their booking task.",
        backstory="You are an assistant for a multi-specialty hospital chain.",
    ),
    task=PromptTask(
        description=(
            "Greet the user, identify their intent, and use the right skill to "
            "complete their task."
        ),
        expected_output="A short, helpful response.",
    ),
)


# --- Demo: LLM-driven activation ---


async def demo_llm_mode() -> None:
    """Show what the LLM sees when skill_activation='llm' (the default).

    The agent auto-injects:
      - load_skill / unload_skill in the per-turn tool list
      - <available_skills> in the system prompt
    """
    print("=" * 70)
    print("LLM-DRIVEN MODE (default)")
    print("=" * 70)

    agent = GenericAgent(
        agent_prompt=AGENT_CONFIG,
        tools=build_tools(),
        skills=[build_doctor_skill(), build_lab_skill()],
        # Nothing is visible by default — every tool is skill-gated.
        base_tool_names=[],
        # skill_activation="llm" is the default; shown here for clarity.
        skill_activation="llm",
    )

    print("\n--- System prompt the LLM sees on turn 1 ---")
    print(agent._build_system_prompt())

    print("\n--- Tools available on turn 1 (no skill loaded yet) ---")
    for t in agent._build_active_tools():
        print(f"  - {t.name}: {t.description[:70]!r}")

    # The LLM would call load_skill itself. Here we simulate that call
    # by invoking the meta-tool directly.
    print("\n--- LLM calls load_skill(name='doctor_booking') ---")
    load_tool = next(t for t in agent._build_active_tools() if t.name == "load_skill")
    result = await load_tool.run(name="doctor_booking")
    print(f"  result: {result}")

    print("\n--- Tools available on turn 2 (doctor_booking active) ---")
    for t in agent._build_active_tools():
        print(f"  - {t.name}")

    # Load the second skill too — both reference `note_user` so name-set
    # dedup means it appears exactly once.
    print("\n--- LLM also calls load_skill(name='lab_booking') ---")
    result = await load_tool.run(name="lab_booking")
    print(f"  result: {result}")

    names = [t.name for t in agent._build_active_tools()]
    print("\n--- Tools available with BOTH skills active ---")
    print(f"  {names}")
    print(f"  note_user appears {names.count('note_user')} time(s) — dedup works")

    print("\n--- System prompt now contains both active skill blocks ---")
    prompt = agent._build_system_prompt()
    # Print just the skill-relevant section to keep output short.
    if "<active_skill" in prompt:
        start = prompt.index("<active_skill")
        print(prompt[start:])


# --- Demo: manual activation ---


async def demo_manual_mode() -> None:
    """Show how a host with its own router uses manual mode.

    No meta-tools, no registry block. The host decides activation entirely.
    """
    print("\n" + "=" * 70)
    print("MANUAL MODE (host-driven activation)")
    print("=" * 70)

    agent = GenericAgent(
        agent_prompt=AGENT_CONFIG,
        tools=build_tools(),
        skills=[build_doctor_skill(), build_lab_skill()],
        base_tool_names=[],
        skill_activation="manual",
    )

    print("\n--- System prompt with no skill active (no registry block) ---")
    print(agent._build_system_prompt())

    print("\n--- Tools available before activation ---")
    print([t.name for t in agent._build_active_tools()])

    # Host's upstream router decided the user wants lab booking.
    print("\n--- Host calls await agent.activate_skill('lab_booking') ---")
    await agent.activate_skill("lab_booking")

    print("\n--- Tools available after activation ---")
    print([t.name for t in agent._build_active_tools()])

    print("\n--- System prompt now ---")
    print(agent._build_system_prompt())


async def main() -> None:
    await demo_llm_mode()
    await demo_manual_mode()


if __name__ == "__main__":
    asyncio.run(main())
