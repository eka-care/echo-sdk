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
        skill_activation: SkillActivation = "manual",
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
                "manual" — host calls agent.activate_skill(name) explicitly.
                "llm"    — agent auto-injects load_skill / unload_skill
                           meta-tools and an <available_skills> registry block
                           so the LLM picks. (Not implemented in PR1.)
        """
        # Initialize tools as empty list (subclasses can override)
        self.tools = tools or []

        # Skill registry and runtime state
        self.skills: List[Skill] = list(skills or [])
        self.skill_activation: SkillActivation = skill_activation
        self._active_skill_names: set = set()

        # Validate registry consistency before anything else can use it.
        self._validate_skill_registry()

        # PR1: LLM-driven activation is not yet implemented. Surface a clear
        # error rather than silently accepting it.
        if self.skills and self.skill_activation == "llm":
            raise NotImplementedError(
                "skill_activation='llm' is not implemented in this version. "
                "Use skill_activation='manual' and call agent.activate_skill() "
                "/ agent.deactivate_skill() programmatically."
            )

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
        """Check skill names are unique and tool names don't collide.

        Runs at construction so problems surface fast instead of at first
        activation. Raises ValueError with a specific message identifying the
        colliding name and its two owners.
        """
        if not self.skills:
            return

        # 1. Skill names must be unique.
        seen: set = set()
        for skill in self.skills:
            if skill.name in seen:
                raise ValueError(
                    f"Duplicate Skill.name {skill.name!r}. "
                    "Each registered skill must have a unique name."
                )
            seen.add(skill.name)

        # 2/3. Tool names must be unique across base tools and all skills.
        tool_owner: Dict[str, str] = {}
        for tool in self.tools:
            if tool.name:
                tool_owner[tool.name] = "<base agent>"
        for skill in self.skills:
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

        # 4. Reserved meta-tool names cannot be used in LLM mode.
        if self.skill_activation == "llm":
            for reserved in _RESERVED_META_TOOL_NAMES:
                if reserved in tool_owner:
                    raise ValueError(
                        f"Tool name {reserved!r} is reserved for the "
                        f"skill_activation='llm' meta-tools but is already "
                        f"declared by {tool_owner[reserved]}."
                    )

    def _get_skill_by_name(self, name: str) -> Optional[Skill]:
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    async def activate_skill(
        self,
        name: str,
        context: Optional["ConversationContext"] = None,
    ) -> None:
        """Add a registered skill to the active set.

        Idempotent: re-activating an already-active skill is a no-op (hooks
        do not re-fire). Raises ValueError if the name is not in the registry.

        Args:
            name: Skill.name as registered in the agent's `skills` list.
            context: Optional live ConversationContext, forwarded to the
                skill's on_activate hook. Pass None when activating during
                agent setup, before any conversation has started.
        """
        skill = self._get_skill_by_name(name)
        if skill is None:
            raise ValueError(
                f"Unknown skill {name!r}. Registered skills: "
                f"{[s.name for s in self.skills]!r}"
            )
        if name in self._active_skill_names:
            return
        self._active_skill_names.add(name)
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
        skill = self._get_skill_by_name(name)
        self._active_skill_names.discard(name)
        if skill is not None:
            await skill.on_deactivate(context)

    def active_skill_names(self) -> List[str]:
        """Return the names of currently-active skills (in registration order)."""
        return [s.name for s in self.skills if s.name in self._active_skill_names]

    def _build_active_tools(self) -> List[BaseTool]:
        """Compose the per-turn tool list: base tools + active skill tools.

        For agents with no skills registered, returns the base `self.tools`
        list as-is — same object — to guarantee byte-for-byte equivalence
        with the pre-skills behavior.

        (PR2 will append load_skill/unload_skill meta-tools when
        skill_activation == "llm".)
        """
        if not self.skills:
            return self.tools

        tools: List[BaseTool] = list(self.tools)
        for skill in self.skills:
            if skill.name in self._active_skill_names:
                tools.extend(skill.tools)
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

        Active skills are appended in registration order so the assembled
        prompt is deterministic regardless of activation order.
        """
        parts: List[str] = [base_prompt]
        for skill in self.skills:
            if skill.name in self._active_skill_names:
                parts.append(
                    f"\n\n<active_skill name=\"{skill.name}\">\n"
                    f"{skill.instructions}\n"
                    f"</active_skill>"
                )
        if self.skill_activation == "llm":
            registry = "\n".join(
                f"- {s.name}: {s.description}" for s in self.skills
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
