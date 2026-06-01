"""Tests for the Skill primitive and the agent's skill registry / activation API.

Covers:
- Skill class shape and validation.
- Programmatic activate / deactivate (manual mode).
- Per-turn composition: prompt + tools include active-skill content.
- No-skills agents are equivalent to the pre-skills behavior.
- Tool registry uniqueness and skill tool_names resolution at construction.
- LLM-mode auto-injection: load_skill / unload_skill meta-tools and the
  <available_skills> registry block.
- LoadSkillTool / UnloadSkillTool execution (success + structured errors).
- Name-set dedup when active skills share a tool.
- The `base_tool_names` knob for skill-only tool visibility.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from echo.agents.generic_agent import GenericAgent
from echo.prompts.schemas import AgentPrompt, PromptPersona, PromptTask
from echo.skills import Skill
from echo.tools.base_tool import BaseTool


# --- Test fixtures ---


class _StubTool(BaseTool):
    """Minimal BaseTool subclass for tests. Records calls but does no work."""

    def __init__(self, name: str, description: str = "stub"):
        self.name = name
        self.description = description

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def run(self, **kwargs) -> Any:
        return None


def _make_agent_prompt() -> AgentPrompt:
    return AgentPrompt(
        persona=PromptPersona(
            role="test agent",
            goal="be useful",
            backstory="for tests",
        ),
        task=PromptTask(
            description="do the test task",
            expected_output="something",
        ),
    )


@pytest.fixture
def patched_get_llm():
    """Patch get_llm so BaseAgent construction doesn't require provider deps."""
    with patch("echo.agents.base.get_llm", return_value=MagicMock()) as p:
        yield p


def _new_agent(**kwargs) -> GenericAgent:
    """Construct a GenericAgent with a default config; pass kwargs to override."""
    kwargs.setdefault("agent_prompt", _make_agent_prompt())
    return GenericAgent(**kwargs)


# --- Skill class ---


def test_skill_construction_minimal():
    s = Skill(
        name="doctor_booking",
        description="Use to book a doctor",
        instructions="Walk the user through booking.",
    )
    assert s.name == "doctor_booking"
    assert s.tool_names == []


def test_skill_construction_with_tool_names():
    s = Skill(
        name="doctor_booking",
        description="d",
        instructions="i",
        tool_names=["search_doctors"],
    )
    assert s.tool_names == ["search_doctors"]


def test_skill_rejects_non_string_tool_names():
    with pytest.raises(ValueError, match="Skill.tool_names"):
        Skill(name="n", description="d", instructions="i", tool_names=[""])


@pytest.mark.parametrize(
    "field",
    ["name", "description", "instructions"],
)
def test_skill_required_fields_reject_empty(field):
    """Empty string for a required field raises ValueError naming the field."""
    kwargs = {"name": "n", "description": "d", "instructions": "i"}
    kwargs[field] = ""
    with pytest.raises(ValueError, match=f"Skill.{field}"):
        Skill(**kwargs)


# --- No-skills equivalence ---


def test_no_skills_agent_is_unchanged(patched_get_llm):
    """An agent with no skills= passed must behave like the pre-skills version."""
    base = _StubTool("base_tool")
    agent = _new_agent(tools=[base])

    # Skill machinery is initialized but inert.
    assert agent.skills == {}
    assert agent.active_skill_names() == []

    # _build_active_tools returns the same tools as input.
    assert agent._build_active_tools() == [base]

    # System prompt does not contain any skill-related content.
    prompt = agent._build_system_prompt()
    assert "<active_skill" not in prompt
    assert "<available_skills>" not in prompt


def test_no_skills_prompt_byte_equivalent_to_pre_skills(patched_get_llm):
    """Snapshot test: prompt with no skills is exactly what the old code produced."""
    agent = _new_agent()
    expected = (
        "You are a test agent\n\n"
        "Your goal is: be useful\n\n"
        "do the test task \n\n"
        "Expected Output: something"
    )
    assert agent._build_system_prompt() == expected


# --- Programmatic activation ---


async def test_activate_skill_adds_to_active_set(patched_get_llm):
    skill = Skill(name="doc", description="d", instructions="i")
    agent = _new_agent(skills=[skill])
    assert agent.active_skill_names() == []

    await agent.activate_skill("doc")
    assert agent.active_skill_names() == ["doc"]


async def test_deactivate_skill_removes_from_active_set(patched_get_llm):
    skill = Skill(name="doc", description="d", instructions="i")
    agent = _new_agent(skills=[skill])
    await agent.activate_skill("doc")

    await agent.deactivate_skill("doc")
    assert agent.active_skill_names() == []


async def test_activate_skill_is_idempotent(patched_get_llm):
    """Re-activating an active skill is a no-op; on_activate does not re-fire."""
    fired: List[str] = []

    class HookedSkill(Skill):
        async def on_activate(self, context):
            fired.append("activate")

    skill = HookedSkill(name="doc", description="d", instructions="i")
    agent = _new_agent(skills=[skill])

    await agent.activate_skill("doc")
    await agent.activate_skill("doc")
    assert fired == ["activate"]


async def test_deactivate_skill_is_idempotent(patched_get_llm):
    fired: List[str] = []

    class HookedSkill(Skill):
        async def on_deactivate(self, context):
            fired.append("deactivate")

    skill = HookedSkill(name="doc", description="d", instructions="i")
    agent = _new_agent(skills=[skill])
    await agent.activate_skill("doc")

    await agent.deactivate_skill("doc")
    await agent.deactivate_skill("doc")
    assert fired == ["deactivate"]


async def test_activate_unknown_skill_raises(patched_get_llm):
    agent = _new_agent(skills=[Skill(name="doc", description="d", instructions="i")])
    with pytest.raises(ValueError, match="Unknown skill"):
        await agent.activate_skill("not_registered")


async def test_hooks_receive_context(patched_get_llm):
    """on_activate / on_deactivate are called with the context the host passes in."""
    received: List[Any] = []

    class HookedSkill(Skill):
        async def on_activate(self, context):
            received.append(("activate", context))

        async def on_deactivate(self, context):
            received.append(("deactivate", context))

    skill = HookedSkill(name="doc", description="d", instructions="i")
    agent = _new_agent(skills=[skill])
    sentinel = object()

    await agent.activate_skill("doc", context=sentinel)
    await agent.deactivate_skill("doc", context=sentinel)

    assert received == [("activate", sentinel), ("deactivate", sentinel)]


# --- Per-turn composition (manual mode) ---


async def test_active_skill_appended_to_system_prompt(patched_get_llm):
    skill = Skill(
        name="doctor_booking",
        description="d",
        instructions="DOCTOR INSTRUCTIONS",
    )
    agent = _new_agent(skills=[skill])
    await agent.activate_skill("doctor_booking")

    prompt = agent._build_system_prompt()
    assert '<active_skill name="doctor_booking">' in prompt
    assert "DOCTOR INSTRUCTIONS" in prompt
    assert "</active_skill>" in prompt


async def test_inactive_skill_not_in_system_prompt(patched_get_llm):
    s1 = Skill(name="a", description="d", instructions="A_INSTRUCTIONS")
    s2 = Skill(name="b", description="d", instructions="B_INSTRUCTIONS")
    agent = _new_agent(skills=[s1, s2])
    await agent.activate_skill("a")

    prompt = agent._build_system_prompt()
    assert "A_INSTRUCTIONS" in prompt
    assert "B_INSTRUCTIONS" not in prompt


async def test_active_skill_tools_appended_to_tool_list(patched_get_llm):
    """Manual mode: tool list is base tools + the active skill's named tools."""
    base = _StubTool("base_tool")
    skill_tool = _StubTool("search_doctors")
    skill = Skill(
        name="doctor_booking",
        description="d",
        instructions="i",
        tool_names=["search_doctors"],
    )
    agent = _new_agent(
        tools=[base, skill_tool],
        skills=[skill],
        skill_activation="manual",
        base_tool_names=["base_tool"],
    )
    await agent.activate_skill("doctor_booking")

    tools = agent._build_active_tools()
    assert tools == [base, skill_tool]


async def test_inactive_skill_tools_not_in_tool_list(patched_get_llm):
    """Manual mode: a registered-but-inactive skill contributes no tools."""
    base = _StubTool("base_tool")
    skill_tool = _StubTool("search_doctors")
    skill = Skill(
        name="doc",
        description="d",
        instructions="i",
        tool_names=["search_doctors"],
    )
    agent = _new_agent(
        tools=[base, skill_tool],
        skills=[skill],
        skill_activation="manual",
        base_tool_names=["base_tool"],
    )

    tools = agent._build_active_tools()
    assert tools == [base]


async def test_manual_mode_does_not_inject_available_skills_block(patched_get_llm):
    """In manual mode, the <available_skills> registry block must NOT appear."""
    skill = Skill(name="doc", description="DOC_DESC", instructions="i")
    agent = _new_agent(skills=[skill], skill_activation="manual")
    await agent.activate_skill("doc")

    prompt = agent._build_system_prompt()
    assert "<available_skills>" not in prompt
    assert "DOC_DESC" not in prompt  # description only appears in registry


async def test_hot_swap_mid_conversation(patched_get_llm):
    """Deactivate A + activate B between turns; second composition reflects B only."""
    a_tool = _StubTool("a_tool")
    b_tool = _StubTool("b_tool")
    a = Skill(name="a", description="d", instructions="A_TEXT", tool_names=["a_tool"])
    b = Skill(name="b", description="d", instructions="B_TEXT", tool_names=["b_tool"])
    agent = _new_agent(
        tools=[a_tool, b_tool], skills=[a, b], base_tool_names=[]
    )

    await agent.activate_skill("a")
    prompt_1 = agent._build_system_prompt()
    tools_1 = agent._build_active_tools()
    assert "A_TEXT" in prompt_1
    assert "B_TEXT" not in prompt_1
    assert any(t.name == "a_tool" for t in tools_1)

    await agent.deactivate_skill("a")
    await agent.activate_skill("b")
    prompt_2 = agent._build_system_prompt()
    tools_2 = agent._build_active_tools()
    assert "B_TEXT" in prompt_2
    assert "A_TEXT" not in prompt_2
    assert any(t.name == "b_tool" for t in tools_2)
    assert not any(t.name == "a_tool" for t in tools_2)


# --- Tool registry and skill validation ---


def test_duplicate_skill_name_silently_dropped(patched_get_llm, caplog):
    s1 = Skill(name="doc", description="d", instructions="i")
    s2 = Skill(name="doc", description="d", instructions="i2")
    with caplog.at_level("WARNING"):
        agent = _new_agent(skills=[s1, s2])
    assert list(agent.skills) == ["doc"]
    assert agent.skills["doc"] is s1
    assert any("Duplicate skill dropped" in r.message for r in caplog.records)


def test_duplicate_tool_name_silently_dropped(patched_get_llm, caplog):
    """Second BaseTool with a colliding name is dropped; first wins."""
    a = _StubTool("shared")
    b = _StubTool("shared")
    with caplog.at_level("WARNING"):
        agent = _new_agent(tools=[a, b])
    assert agent.tools == [a]
    assert agent._tools_by_name["shared"] is a
    assert any("Duplicate tool dropped" in r.message for r in caplog.records)


def test_skill_referencing_unknown_tool_name_silently_dropped(patched_get_llm, caplog):
    """Skill whose tool_names don't all resolve is removed from the registry."""
    skill = Skill(
        name="ghost",
        description="d",
        instructions="i",
        tool_names=["does_not_exist"],
    )
    with caplog.at_level("WARNING"):
        agent = _new_agent(skills=[skill])
    assert "ghost" not in agent.skills
    assert any("missing tools" in r.message for r in caplog.records)


def test_base_tool_named_load_skill_is_dropped(patched_get_llm, caplog):
    """A base tool whose name shadows a reserved meta name is dropped silently."""
    skill = Skill(name="s", description="d", instructions="i")
    with caplog.at_level("WARNING"):
        agent = _new_agent(
            tools=[_StubTool("load_skill")],
            skills=[skill],
            skill_activation="llm",
        )
    # The reserved-name base tool was dropped; the real meta tool occupies that name.
    assert agent._tools_by_name["load_skill"].__class__.__name__ == "LoadSkillTool"
    assert any("Duplicate tool dropped" in r.message for r in caplog.records)


def test_invalid_base_tool_name_silently_ignored(patched_get_llm, caplog):
    """Entries in base_tool_names that don't exist (or are reserved) are dropped."""
    with caplog.at_level("WARNING"):
        agent = _new_agent(tools=[_StubTool("yes")], base_tool_names=["nope"])
    assert agent._base_tool_names == []
    assert any("invalid" in r.message for r in caplog.records)


# --- Name-set dedup ---


async def test_shared_tool_listed_once_when_two_skills_active(patched_get_llm):
    """A tool referenced by two active skills appears exactly once in the schema."""
    shared = _StubTool("shared")
    only_a = _StubTool("only_a")
    a = Skill(
        name="a", description="d", instructions="i", tool_names=["shared", "only_a"]
    )
    b = Skill(name="b", description="d", instructions="i", tool_names=["shared"])
    agent = _new_agent(
        tools=[shared, only_a],
        skills=[a, b],
        skill_activation="manual",
        base_tool_names=[],
    )
    await agent.activate_skill("a")
    await agent.activate_skill("b")

    names = [t.name for t in agent._build_active_tools()]
    assert names.count("shared") == 1
    assert names.count("only_a") == 1


async def test_shared_tool_stays_visible_when_one_of_two_owners_unloads(patched_get_llm):
    """Unloading one skill that names a tool does NOT remove it if another active skill still names it."""
    shared = _StubTool("shared")
    a = Skill(name="a", description="d", instructions="i", tool_names=["shared"])
    b = Skill(name="b", description="d", instructions="i", tool_names=["shared"])
    agent = _new_agent(
        tools=[shared],
        skills=[a, b],
        skill_activation="manual",
        base_tool_names=[],
    )
    await agent.activate_skill("a")
    await agent.activate_skill("b")

    await agent.deactivate_skill("a")
    names = [t.name for t in agent._build_active_tools()]
    assert "shared" in names


# --- base_tool_names knob ---


async def test_base_tool_names_partitions_default_vs_skill_only(patched_get_llm):
    """Tool not in base_tool_names is invisible until a skill names it."""
    default = _StubTool("default")
    skill_only = _StubTool("skill_only")
    s = Skill(
        name="s",
        description="d",
        instructions="i",
        tool_names=["skill_only"],
    )
    agent = _new_agent(
        tools=[default, skill_only],
        skills=[s],
        skill_activation="manual",
        base_tool_names=["default"],
    )

    # Before activation: only `default` is visible.
    before = [t.name for t in agent._build_active_tools()]
    assert "default" in before
    assert "skill_only" not in before

    # After activation: skill_only joins the visible set.
    await agent.activate_skill("s")
    after = [t.name for t in agent._build_active_tools()]
    assert "default" in after
    assert "skill_only" in after


# --- LLM-driven activation surface ---


def test_llm_mode_is_the_default_when_skills_are_passed(patched_get_llm):
    """Sanity check: skill_activation defaults to 'llm'."""
    skill = Skill(name="doc", description="d", instructions="i")
    agent = _new_agent(skills=[skill])
    assert agent.skill_activation == "llm"


def test_llm_mode_without_skills_is_a_no_op(patched_get_llm):
    """LLM mode with no skills registered must not break or inject anything."""
    base = _StubTool("base_tool")
    agent = _new_agent(tools=[base], skills=None, skill_activation="llm")
    # No meta-tools, no registry block, base tool list unchanged.
    assert agent._build_active_tools() == [base]
    assert "<available_skills>" not in agent._build_system_prompt()


def test_llm_mode_injects_meta_tools_into_tool_list(patched_get_llm):
    """LLM mode adds load_skill + unload_skill to every per-turn tool list."""
    skill_tool = _StubTool("search_doctors")
    skill = Skill(
        name="doc",
        description="d",
        instructions="i",
        tool_names=["search_doctors"],
    )
    agent = _new_agent(
        tools=[skill_tool], skills=[skill], skill_activation="llm"
    )

    # Even with no skills active, the meta-tools are present so the LLM
    # can call load_skill on the first turn.
    names = [t.name for t in agent._build_active_tools()]
    assert "load_skill" in names
    assert "unload_skill" in names


def test_llm_mode_injects_available_skills_block_in_prompt(patched_get_llm):
    """LLM mode appends an <available_skills> registry to the system prompt."""
    s1 = Skill(name="doctor", description="DOC_DESC", instructions="i")
    s2 = Skill(name="lab", description="LAB_DESC", instructions="i")
    agent = _new_agent(skills=[s1, s2], skill_activation="llm")

    prompt = agent._build_system_prompt()
    assert "<available_skills>" in prompt
    assert "doctor: DOC_DESC" in prompt
    assert "lab: LAB_DESC" in prompt
    assert "</available_skills>" in prompt


# --- LoadSkillTool / UnloadSkillTool execution ---


def _meta_tool(agent: GenericAgent, name: str) -> BaseTool:
    return agent._tools_by_name[name]


async def test_load_skill_tool_activates_named_skill(patched_get_llm):
    skill = Skill(name="doc", description="d", instructions="i")
    agent = _new_agent(skills=[skill], skill_activation="llm")
    load_tool = _meta_tool(agent, "load_skill")

    result = await load_tool.run(name="doc")
    assert result == {"status": "loaded", "skill": "doc"}
    assert agent.active_skill_names() == ["doc"]


async def test_unload_skill_tool_deactivates_named_skill(patched_get_llm):
    skill = Skill(name="doc", description="d", instructions="i")
    agent = _new_agent(skills=[skill], skill_activation="llm")
    await agent.activate_skill("doc")

    unload_tool = _meta_tool(agent, "unload_skill")
    result = await unload_tool.run(name="doc")
    assert result == {"status": "unloaded", "skill": "doc"}
    assert agent.active_skill_names() == []


async def test_load_skill_tool_returns_error_for_unknown_skill(patched_get_llm):
    """Unknown skill name must surface as a structured error, not a crash."""
    skill = Skill(name="doc", description="d", instructions="i")
    agent = _new_agent(skills=[skill], skill_activation="llm")
    load_tool = _meta_tool(agent, "load_skill")

    result = await load_tool.run(name="not_registered")
    assert result["status"] == "error"
    assert "Unknown skill" in result["message"]
    # The agent's active set must be unchanged.
    assert agent.active_skill_names() == []


async def test_load_skill_tool_returns_error_for_missing_name(patched_get_llm):
    skill = Skill(name="doc", description="d", instructions="i")
    agent = _new_agent(skills=[skill], skill_activation="llm")
    load_tool = _meta_tool(agent, "load_skill")

    result = await load_tool.run()
    assert result["status"] == "error"
    assert "Missing required argument" in result["message"]


async def test_unload_skill_tool_is_idempotent_for_inactive_skill(patched_get_llm):
    """Unloading a skill that was never active is a silent success, not an error."""
    skill = Skill(name="doc", description="d", instructions="i")
    agent = _new_agent(skills=[skill], skill_activation="llm")
    unload_tool = _meta_tool(agent, "unload_skill")

    result = await unload_tool.run(name="doc")
    assert result == {"status": "unloaded", "skill": "doc"}
