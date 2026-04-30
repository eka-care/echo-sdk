"""Tests for the Skill primitive and the agent's skill registry / activation API.

PR1 scope:
- Skill class shape and validation.
- Programmatic activate / deactivate (manual mode).
- Per-turn composition: prompt + tools include active-skill content.
- No-skills agents are byte-for-byte equivalent to the pre-skills behavior.
- Tool-name collision validator at construction.
- skill_activation='llm' raises NotImplementedError in PR1.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig
from echo.agents.generic_agent import GenericAgent
from echo.agents.skill import Skill
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


def _make_agent_config() -> AgentConfig:
    return AgentConfig(
        persona=PersonaConfig(
            role="test agent",
            goal="be useful",
            backstory="for tests",
        ),
        task=TaskConfig(
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
    kwargs.setdefault("agent_config", _make_agent_config())
    return GenericAgent(**kwargs)


# --- Skill class ---


def test_skill_construction_minimal():
    s = Skill(
        name="doctor_booking",
        description="Use to book a doctor",
        instructions="Walk the user through booking.",
    )
    assert s.name == "doctor_booking"
    assert s.tools == []


def test_skill_construction_with_tools():
    t = _StubTool("search_doctors")
    s = Skill(
        name="doctor_booking",
        description="d",
        instructions="i",
        tools=[t],
    )
    assert s.tools == [t]


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
    agent = _new_agent()

    # Skill machinery is initialized but inert.
    assert agent.skills == []
    assert agent.active_skill_names() == []

    # _build_active_tools returns the exact base tools list (same object).
    base_tools: List[BaseTool] = [_StubTool("base_tool")]
    agent.tools = base_tools
    assert agent._build_active_tools() is base_tools

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
    base = _StubTool("base_tool")
    skill_tool = _StubTool("search_doctors")
    skill = Skill(
        name="doctor_booking",
        description="d",
        instructions="i",
        tools=[skill_tool],
    )
    agent = _new_agent(tools=[base], skills=[skill])
    await agent.activate_skill("doctor_booking")

    tools = agent._build_active_tools()
    assert tools == [base, skill_tool]


async def test_inactive_skill_tools_not_in_tool_list(patched_get_llm):
    base = _StubTool("base_tool")
    skill_tool = _StubTool("search_doctors")
    skill = Skill(name="doc", description="d", instructions="i", tools=[skill_tool])
    agent = _new_agent(tools=[base], skills=[skill])
    # Skill registered but never activated.

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
    a = Skill(name="a", description="d", instructions="A_TEXT", tools=[_StubTool("a_tool")])
    b = Skill(name="b", description="d", instructions="B_TEXT", tools=[_StubTool("b_tool")])
    agent = _new_agent(skills=[a, b])

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


# --- Tool-name collision validator ---


def test_collision_two_skills_with_same_name(patched_get_llm):
    s1 = Skill(name="doc", description="d", instructions="i")
    s2 = Skill(name="doc", description="d", instructions="i")
    with pytest.raises(ValueError, match="Duplicate Skill.name"):
        _new_agent(skills=[s1, s2])


def test_collision_two_skills_with_same_tool_name(patched_get_llm):
    s1 = Skill(name="a", description="d", instructions="i", tools=[_StubTool("shared")])
    s2 = Skill(name="b", description="d", instructions="i", tools=[_StubTool("shared")])
    with pytest.raises(ValueError, match="Tool name collision.*shared"):
        _new_agent(skills=[s1, s2])


def test_collision_skill_tool_with_base_tool(patched_get_llm):
    base = _StubTool("shared")
    skill = Skill(name="a", description="d", instructions="i", tools=[_StubTool("shared")])
    with pytest.raises(ValueError, match="Tool name collision.*shared"):
        _new_agent(tools=[base], skills=[skill])


# --- LLM mode is not implemented yet (PR1) ---


def test_llm_mode_raises_not_implemented(patched_get_llm):
    skill = Skill(name="doc", description="d", instructions="i")
    with pytest.raises(NotImplementedError, match="skill_activation='llm'"):
        _new_agent(skills=[skill], skill_activation="llm")


def test_llm_mode_without_skills_does_not_raise(patched_get_llm):
    """LLM mode is meaningless without skills; it must not raise in that case."""
    # Skipping: skill_activation default is "manual"; setting "llm" with no
    # skills shouldn't raise because no LLM-mode behavior is exercised.
    _new_agent(skills=None, skill_activation="llm")
