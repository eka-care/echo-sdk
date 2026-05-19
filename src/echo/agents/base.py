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
from echo.prompts.templates import load_template
from echo.tools.base_tool import BaseTool
from echo.tools.skills import LoadSkillTool, UnloadSkillTool

from .schemas import AgentResult

logger = logging.getLogger(__name__)

# Skill tools cannot use these names when skill_activation == "llm";
# they are reserved for the auto-injected load_skill / unload_skill meta-tools.
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
        base_tool_names: Optional[List[str]] = None,
    ):
        """
        Agent Init with config, tools, and skills.
        Args:
            agent_config: Persona + task configuration.
            llm_config: Optional LLM configuration. Defaults to Bedrock Haiku.
            tools: Registry of all tools the agent can use. Duplicate names are silently dropped.
            skills: Optional skill registry the agent can activate. A skill referenced tool
                   (`Skill.tool_names`) must appear in `tools`.
            skill_activation: How skills are activated.
                "llm"    (default) — agent auto-injects load_skill / unload_skill meta-tools.
                "manual" — host calls agent.activate_skill(name) explicitly.
            base_tool_names: Names of registered tools that should be visible to the LLM by default.
        """
        # 1. Build the canonical tool registry. self.tools stays as caller-passed list (no meta tools)
        #    so external readers like to_crewai_agent / to_dict aren't surprised.
        self.tools: List[BaseTool] = []
        self._tools_by_name: Dict[str, BaseTool] = {}
        for t in tools or []:
            # Silently drop duplicate and empty tool names
            if (
                not t.name
                or t.name in self._tools_by_name
                or t.name in _RESERVED_META_TOOL_NAMES
            ):
                logger.warning(
                    f"Duplicate tool dropped: name {t.name!r}. Tool names must be unique across the agent's registry."
                )
                continue
            self.tools.append(t)
            self._tools_by_name[t.name] = t

        # 2. Skill registry — dict keyed by Skill.name so lookups are O(1)
        #    and insertion order survives for deterministic prompt assembly.
        self.skills: Dict[str, Skill] = {}
        self.skill_activation: SkillActivation = skill_activation
        self._active_skill_names: Dict[str, None] = {}

        for skill in skills or []:
            # Silently drop duplicate or empty skill names
            if not skill.name or skill.name in self.skills:
                logger.warning(
                    f"Duplicate skill dropped: name {skill.name!r}. Each registered skill must have a unique name."
                )
                continue
            self.skills[skill.name] = skill

        self._meta_tool_names: List[str] = []
        if self.skills:
            # 4. Validate every Skill.tool_names entry resolves against the tool registry
            self._validate_skill_registry()

            # 5. Insert meta-tools into the registry
            if self.skill_activation == "llm":
                for meta in (LoadSkillTool(self), UnloadSkillTool(self)):
                    self._tools_by_name[meta.name] = meta
                    self._meta_tool_names.append(meta.name)

        # 6. Resolve the "always visible" tool set. Explicit empty list means
        #    "nothing default-visible — everything is skill-gated."
        if base_tool_names is not None:
            valid_base_tool_names = []
            for n in base_tool_names:
                if n not in self._tools_by_name or n in _RESERVED_META_TOOL_NAMES:
                    logger.warning(
                        f"Base tool name {n!r} is invalid and will be ignored."
                    )
                    continue
                valid_base_tool_names.append(n)

            self._base_tool_names: List[str] = valid_base_tool_names
        else:
            self._base_tool_names = [
                n for n in self._tools_by_name if n not in _RESERVED_META_TOOL_NAMES
            ]

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
        """Drop any skill whose tool_names don't all resolve in the registry."""
        to_drop: List[str] = []
        for skill in self.skills.values():
            missing = [tn for tn in skill.tool_names if tn not in self._tools_by_name]
            if missing:
                logger.warning(
                    f"Skill {skill.name!r} dropped due to missing tools: {missing!r}."
                )
                to_drop.append(skill.name)
        for name in to_drop:
            del self.skills[name]

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
                f"Unknown skill {name!r}. Registered skills: " f"{list(self.skills)!r}"
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
        """Compose the per-turn tool list as a name-set union.

        Visible set = base_tool_names ∪ ⋃(active skill.tool_names) ∪ meta
        (LLM mode only). A dict-as-ordered-set preserves insertion order
        (base → active skills in activation order → meta) and dedups by
        construction, so a tool referenced by two active skills appears
        exactly once.
        """
        visible: Dict[str, None] = {n: None for n in self._base_tool_names}
        for skill_name in self._active_skill_names:
            skill = self.skills.get(skill_name)
            if skill:
                for tn in skill.tool_names:
                    visible.setdefault(tn, None)
        for n in self._meta_tool_names:
            visible.setdefault(n, None)
        return [self._tools_by_name[n] for n in visible]

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
        system_prompt_parts, base_user_prompt = [], ""
        if self.role:
            base_user_prompt = f"You are a {self.role}\n\n"
        if not skip_goal and self.goal:
            base_user_prompt += f"Your goal is: {self.goal}\n\n"

        if not self.task_description:
            logger.error("Task description is required for agent: %s", self.name)
            raise Exception("Task description is required")
        base_user_prompt += f"{self.task_description}"
        if not skip_expected_output and self.expected_output:
            base_user_prompt += f"\n\nExpected Output: {self.expected_output}"

        system_prompt_parts.append(base_user_prompt)

        # Append active skill blocks (and registry in LLM mode). When no
        # skills are registered this is a no-op and produces the same string
        # as before.
        if self.skills:
            system_prompt_parts[0] = (
                "<base_user_prompt>\n"
                + system_prompt_parts[0]
                + "\n</base_user_prompt>"
            )
            skill_prompt_parts = self._get_skill_content()
            system_prompt_parts.extend(skill_prompt_parts)

        system_prompt = "\n\n".join(system_prompt_parts)
        return system_prompt

    def _get_skill_content(self) -> List[str]:
        """Get skill sections in attention-friendly order.

        Order: <skill_mechanism> → <available_skills> registry (llm mode)
        → <active_skills> name list → <active_skill> bodies. The mechanism
        block precedes the registry it references, and the active-name list
        precedes the active-body content it indexes.
        """
        parts = [load_template("skill_mechanism")]

        # [2] <available_skills> — registry list (llm mode only).
        if self.skill_activation == "llm":
            registry = "\n".join(
                f"- **{s.name}**: {s.description}" for s in self.skills.values()
            )
            parts.append(f"<available_skills>\n{registry}\n</available_skills>")

        # [3] <active_skills> — names of currently-active skills.
        if self._active_skill_names:
            active_list = "\n".join(f"- {n}" for n in self._active_skill_names)
            parts.append(f"<active_skills>\n{active_list}\n</active_skills>")

            # [4] <active_skill> bodies — full instructions, in activation order.
            for name in self._active_skill_names:
                skill = self.skills.get(name)
                if skill is not None:
                    parts.append(
                        f'<active_skill name="{skill.name}">\n'
                        f"{skill.instructions}\n"
                        f"</active_skill>"
                    )
        else:
            parts.append("<active_skills>\nNo active skills\n</active_skills>")

        return parts

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
