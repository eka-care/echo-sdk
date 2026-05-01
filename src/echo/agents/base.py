"""
Base agent interface for Echo SDK.

Provides a framework-agnostic interface for agents with adapters.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Literal, Optional

from echo.agents.config import AgentConfig
from echo.agents.skill import Skill
from echo.llm import LLMConfig, get_llm
from echo.llm.schemas import StreamEvent, StreamEventType
from echo.tools.base_tool import BaseTool
from echo.tools.skills import LoadSkillTool, UnloadSkillTool

from .schemas import AgentResult

logger = logging.getLogger(__name__)

# Reserved tool names for the LLM-driven activation surface (PR2).
# Skill tools cannot use these names when skill_activation == "llm".
_RESERVED_META_TOOL_NAMES = ("load_skill", "unload_skill")

SkillActivation = Literal["llm", "manual"]


if TYPE_CHECKING:
    from echo.models.user_conversation import ConversationContext


class BaseAgent(ABC):
    """
    Abstract base class for Echo agents.

    Provides framework-agnostic agent definition with adapters for
    different execution frameworks (CrewAI, LangGraph, standalone, etc.).

    Each concrete agent must:
    - Set `name` class attribute (matches YAML config filename)
    - Implement `run()` method for standalone execution
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the agent."""
        pass

    def __init__(
        self,
        agent_config: Optional[AgentConfig] = None,
        llm_config: Optional[LLMConfig] = None,
        tools: Optional[List[BaseTool]] = None,
        skills: Optional[List[Skill]] = None,
        skill_activation: SkillActivation = "llm",
    ):
        """
        Initialize agent with config from YAML.

        Args:
            agent_config: Persona + task configuration.
            llm_config: Optional LLM configuration. Defaults to Bedrock Haiku.
            tools: Always-on base tools available on every turn.
            skills: Optional registry of skills the agent can activate.
                When None or empty, all skill machinery is bypassed and
                behavior is identical to a no-skills agent.
            skill_activation: How skills are activated.
                "llm"    (default) — agent auto-injects load_skill /
                          unload_skill meta-tools and an <available_skills>
                          registry block so the LLM picks. Use this for any
                          agent that should self-route between skills.
                "manual" — host calls agent.activate_skill(name) explicitly.
                          Use this when you have an upstream router (a
                          classifier, rules engine, or separate LLM) that
                          decides activation, or when you want deterministic
                          behavior for testing.
        """
        # Initialize tools as empty list (subclasses can override)
        self.tools = tools or []

        # Skill registry and runtime state. The registry is stored as a dict
        # keyed by Skill.name so lookups are O(1). Insertion order is
        # preserved (Python 3.7+), so prompt assembly remains deterministic.
        # The constructor still accepts a list because that's the natural
        # input shape; we detect duplicate names while converting.
        skills_input = list(skills or [])
        self.skills: Dict[str, Skill] = {}
        for skill in skills_input:
            if skill.name in self.skills:
                raise ValueError(
                    f"Duplicate Skill.name {skill.name!r}. "
                    "Each registered skill must have a unique name."
                )
            self.skills[skill.name] = skill
        self.skill_activation: SkillActivation = skill_activation
        # Ordered set: a dict's keys preserve insertion order with O(1)
        # membership tests. Iteration yields activation order, which keeps
        # prompt and tool-list assembly deterministic across runs.
        self._active_skill_names: Dict[str, None] = {}

        # Validate registry consistency before anything else can use it.
        self._validate_skill_registry()

        # When LLM-driven activation is enabled and skills are registered,
        # bind a pair of meta-tools to this agent. They translate LLM tool
        # calls into the same activate_skill / deactivate_skill API that
        # programmatic callers use.
        self._meta_tools: List[BaseTool] = []
        if self.skills and self.skill_activation == "llm":
            self._meta_tools = [LoadSkillTool(self), UnloadSkillTool(self)]

        # Load config from YAML (both agent and task)
        self.role = agent_config.persona.role
        self.goal = agent_config.persona.goal
        self.backstory = agent_config.persona.backstory
        self.task_description = agent_config.task.description
        self.expected_output = agent_config.task.expected_output

        # Set LLM config, defaults to Bedrock Haiku
        self.llm_config = llm_config or LLMConfig()
        self.llm = get_llm(self.llm_config)

    # --- Skill registry ---

    def _validate_skill_registry(self) -> None:
        """Check tool names don't collide across base tools and skills.

        Skill-name uniqueness is already enforced by the dict registry built
        in __init__ (duplicate input raises before this method is called).
        Runs at construction so problems surface fast instead of at first
        activation. Raises ValueError identifying the colliding name and its
        two owners.
        """
        if not self.skills:
            return

        # Tool names must be unique across base tools and all skills.
        tool_owner: Dict[str, str] = {}
        for tool in self.tools:
            if tool.name:
                tool_owner[tool.name] = "<base agent>"
        for skill in self.skills.values():
            for tool in skill.tools:
                if not tool.name:
                    continue
                if tool.name in tool_owner:
                    raise ValueError(
                        f"Tool name collision: {tool.name!r} is declared by "
                        f"{tool_owner[tool.name]} and skill {skill.name!r}. "
                        "Tool names must be unique across the agent's base "
                        "tools and all registered skills."
                    )
                tool_owner[tool.name] = f"skill {skill.name!r}"

        # Reserved meta-tool names cannot be used in LLM mode.
        if self.skill_activation == "llm":
            for reserved in _RESERVED_META_TOOL_NAMES:
                if reserved in tool_owner:
                    raise ValueError(
                        f"Tool name {reserved!r} is reserved for the "
                        f"skill_activation='llm' meta-tools but is already "
                        f"declared by {tool_owner[reserved]}."
                    )

    async def activate_skill(
        self,
        name: str,
        context: Optional["ConversationContext"] = None,
    ) -> None:
        """Add a registered skill to the active set.

        Idempotent: re-activating an already-active skill is a no-op (hooks
        do not re-fire). Raises ValueError if the name is not in the registry.

        Args:
            name: Skill.name as registered in the agent's `skills` registry.
            context: Optional live ConversationContext, forwarded to the
                skill's on_activate hook. Pass None when activating during
                agent setup, before any conversation has started.
        """
        skill = self.skills.get(name)
        if skill is None:
            raise ValueError(
                f"Unknown skill {name!r}. Registered skills: "
                f"{list(self.skills)!r}"
            )
        if name in self._active_skill_names:
            return
        self._active_skill_names[name] = None
        await skill.on_activate(context)

    async def deactivate_skill(
        self,
        name: str,
        context: Optional["ConversationContext"] = None,
    ) -> None:
        """Remove a skill from the active set.

        Idempotent: deactivating a skill that isn't active is a no-op.
        """
        if name not in self._active_skill_names:
            return
        skill = self.skills.get(name)
        self._active_skill_names.pop(name, None)
        if skill is not None:
            await skill.on_deactivate(context)

    def active_skill_names(self) -> List[str]:
        """Return the names of currently-active skills (in activation order)."""
        return list(self._active_skill_names)

    def _build_active_tools(self) -> List[BaseTool]:
        """Compose the per-turn tool list.

        Layout: base tools + active skill tools + meta-tools (LLM mode only).

        For agents with no skills registered, returns the base `self.tools`
        list as-is — same object — to guarantee byte-for-byte equivalence
        with the pre-skills behavior.
        """
        if not self.skills:
            return self.tools

        tools: List[BaseTool] = list(self.tools)
        for name in self._active_skill_names:
            skill = self.skills.get(name)
            if skill is not None:
                tools.extend(skill.tools)
        tools.extend(self._meta_tools)
        return tools

    @abstractmethod
    async def run(
        self,
        context: "ConversationContext",
        out_msg_id: str,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Execute the agent's task (standalone mode).

        Args:
            context: ConversationContext with:
                     - messages: conversation history
                     - system_context.tool_context: hidden params for tools

        Returns:
            AgentResult with json_output, raw response, parse_error, and metadata
        """
        pass

    @abstractmethod
    async def run_stream(
        self,
        context: "ConversationContext",
        out_msg_id: str,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream the agent's response.

        Yields StreamEvent objects as the LLM generates its response.
        The final DONE event contains the complete LLMResponse and updated context.

        Args:
            context: ConversationContext with:
                     - messages: conversation history
                     - system_context.tool_context: hidden params for tools

        Yields:
            StreamEvent objects (TEXT, TOOL_CALL_START, TOOL_CALL_END, DONE, ERROR)
        """
        pass

    def _build_system_prompt(
        self, skip_goal: bool = False, skip_expected_output: bool = False
    ) -> str:
        """Build system prompt from agent config (and active skills, if any)."""
        system_prompt = ""
        if self.role:
            system_prompt = f"You are a {self.role}\n\n"
        if not skip_goal and self.goal:
            system_prompt += f"Your goal is: {self.goal}\n\n"

        if not self.task_description:
            logger.error("Task description is required for agent: %s", self.name)
            raise Exception("Task description is required")
        system_prompt += f"{self.task_description} \n\n"

        if not skip_expected_output and self.expected_output:
            system_prompt += f"Expected Output: {self.expected_output}"

        # Append active skill blocks (and registry in LLM mode). When no
        # skills are registered this is a no-op and produces the same string
        # as before.
        if self.skills:
            system_prompt = self._append_skill_content(system_prompt)

        return system_prompt

    def _append_skill_content(self, base_prompt: str) -> str:
        """Append <active_skill> blocks (and <available_skills> in LLM mode).

        Active skills are appended in activation order (the order they were
        loaded). The <available_skills> registry, when present, lists every
        registered skill in registration order.
        """
        parts: List[str] = [base_prompt]
        for name in self._active_skill_names:
            skill = self.skills.get(name)
            if skill is not None:
                parts.append(
                    f"\n\n<active_skill name=\"{skill.name}\">\n"
                    f"{skill.instructions}\n"
                    f"</active_skill>"
                )
        if self.skill_activation == "llm":
            registry = "\n".join(
                f"- {s.name}: {s.description}" for s in self.skills.values()
            )
            parts.append(
                f"\n\n<available_skills>\n{registry}\n</available_skills>"
            )
        return "".join(parts)

    async def _run_agent(
        self, context: "ConversationContext", out_msg_id: str
    ) -> AgentResult:
        """Run the agent (non-streaming)."""
        try:
            # Build system prompt with task(mandatory) & expected output,role(optional)
            system_prompt = self._build_system_prompt(skip_goal=True)

            # Call LLM with tools - tool_context automatically injected.
            # Tool list is recomposed per turn from base tools + active skill
            # tools. With no skills registered this returns self.tools as-is.
            llm_response, updated_context = await self.llm.invoke(
                context=context,
                tools=self._build_active_tools(),
                system_prompt=system_prompt,
                out_msg_id=out_msg_id,
            )
            return AgentResult(
                llm_response=llm_response,
                context=updated_context,
                agent_name=self.name,
            )

        except Exception as e:
            context_info = str(context.system_context) if context else ""
            logger.error(
                f"Agent {self.name} failed during run: {e}, with context: {context_info}",
                exc_info=True,
            )
            return AgentResult(
                llm_response=None,
                context=context,
                agent_name=self.name,
                error=str(e),
            )

    async def _run_agent_stream(
        self, context: "ConversationContext", out_msg_id: str
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run the agent with streaming."""
        try:
            # Build system prompt with task(mandatory) & expected output,role(optional)
            system_prompt = self._build_system_prompt(skip_goal=True)

            async for event in self.llm.invoke_stream(
                context=context,
                tools=self._build_active_tools(),
                system_prompt=system_prompt,
                out_msg_id=out_msg_id,
            ):
                yield event
        except Exception as e:
            context_info = str(context.system_context) if context else ""
            logger.error(
                f"Agent {self.name} failed during stream: {e}, with context: {context_info}",
                exc_info=True,
            )
            yield StreamEvent(type=StreamEventType.ERROR, error=str(e))

    # --- Framework Adapters ---
    def to_crewai_agent(self, **kwargs) -> Any:
        """
        Convert to CrewAI Agent.

        Args:
            **kwargs: Additional CrewAI Agent arguments

        Returns:
            CrewAI Agent instance

        Raises:
            ImportError: If crewai is not installed
        """
        try:
            from crewai import Agent
        except ImportError:
            raise ImportError(
                "crewai is required for to_crewai_agent(). "
                "Install with: pip install crewai"
            )

        # Convert tools
        crewai_tools = [tool.to_crewai_tool() for tool in self.tools]

        # Default CrewAI agent settings
        defaults = {
            "verbose": False,
            "allow_delegation": False,
            "max_iter": 1,
            "cache": False,
            "memory": False,
        }
        defaults.update(kwargs)

        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            llm=self.llm_config.to_crewai_llm(),
            tools=crewai_tools,
            **defaults,
        )

    def to_langgraph_node(self) -> Any:
        """
        Convert to LangGraph node.

        Returns:
            LangGraph-compatible node

        Raises:
            NotImplementedError: LangGraph support coming soon
        """
        raise NotImplementedError("LangGraph adapter coming soon")

    def to_dict(self) -> Dict[str, Any]:
        """Get agent metadata as dict."""
        return {
            "name": self.name,
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "tools": [tool.to_dict() for tool in self.tools],
        }
