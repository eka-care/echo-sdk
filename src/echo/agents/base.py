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

    # --- AG-UI public API ---

    async def run_stream_with_ag_ui(
        self,
        context: "ConversationContext",
        run_input: Any,  # ag_ui.core.RunAgentInput
        state: Any,  # echo.ag_ui.AgUiState subclass
        out_msg_id: str,
        paused_run_store: Optional[Any] = None,  # echo.ag_ui.PausedRunStore
        pause_metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Any, None]:
        """Run the agent and yield AG-UI BaseEvents.

        Thin convenience wrapper around AgUiRunner that constructs the
        dispatcher from `run_input.tools` (FE-declared UI tools) and
        plumbs through the optional paused-run store.

        On a UI tool call the runner persists state, emits the tool
        events, and returns without RunFinished — the FE issues a
        /resume request to continue (see resume_run_with_ag_ui).

        Args:
            context: ConversationContext, mutated as the agent runs.
            run_input: ag_ui.core.RunAgentInput. Used for thread_id,
                run_id, and the FE-declared UI tool list. The
                run_input.state field is NOT auto-applied here — the
                host is expected to have already applied it to `state`.
            state: AgUiState subclass instance.
            out_msg_id: Echo-SDK message id for grouping LLM responses.
            paused_run_store: Required when UI tools are declared.
            pause_metadata: Optional dict attached to the persisted
                PausedRun (e.g. b_id, document_id) so the host can
                re-locate it on resume.

        Yields:
            ag_ui.core.BaseEvent instances.
        """
        from echo.ag_ui import AgUiRunner, AgUiToolDispatcher

        ui_tool_names = {t.name for t in run_input.tools}
        dispatcher = AgUiToolDispatcher(ui_tool_names=ui_tool_names)
        runner = AgUiRunner(
            agent=self,
            state=state,
            thread_id=run_input.thread_id,
            run_id=run_input.run_id,
            tool_dispatcher=dispatcher,
            paused_run_store=paused_run_store,
            pause_metadata=pause_metadata,
        )
        async for ev in runner.stream(context, out_msg_id):
            yield ev

    async def resume_run_with_ag_ui(
        self,
        paused_run_store: Any,  # echo.ag_ui.PausedRunStore
        thread_id: str,
        run_id: str,
        tool_call_id: str,
        tool_result: Any,
        state: Any,  # AgUiState
        context: "ConversationContext",
        out_msg_id: str,
        ui_tool_names: Optional[List[str]] = None,
        pause_metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Any, None]:
        """Resume a previously paused run with the FE-supplied tool result.

        Caller responsibilities:
          - Have validated the resume request (auth, etc.).
          - Have rehydrated `context` and `state` from the saved
            PausedRun's snapshots — the host owns deserialization since
            it knows its domain models.

        This method:
          1. Loads the PausedRun by (thread_id, run_id) and validates
             tool_call_id matches.
          2. Emits a TOOL_CALL_RESULT event so the FE sees the
             resolution.
          3. Appends a TOOL message carrying the tool_result to
             `context`.
          4. Invokes run_stream_with_ag_ui() to continue. RUN_STARTED
             and STATE_SNAPSHOT fire again — the FE replaces its state
             from the snapshot, which is consistent because the host
             already restored state from PausedRun.

        On clean completion the paused-run entry is deleted by the
        runner. On re-pause it's overwritten in place.
        """
        import orjson
        from ag_ui.core import (
            EventType,
            RunAgentInput,
            RunErrorEvent,
            Tool,
            ToolCallResultEvent,
        )

        from echo.ag_ui import make_pause_key
        from echo.models.user_conversation import (
            Message,
            MessageRole,
            ToolResult,
        )

        key = make_pause_key(thread_id, run_id)
        paused = await paused_run_store.load(key)
        if paused is None:
            yield RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=f"paused run not found or expired: {key}",
                code="paused_run_expired",
            )
            return
        if paused.tool_call_id != tool_call_id:
            yield RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=(
                    f"tool_call_id mismatch (paused on {paused.tool_call_id}, "
                    f"resume tried {tool_call_id})"
                ),
                code="tool_call_id_mismatch",
            )
            return

        # Tell the FE we received the result.
        if isinstance(tool_result, str):
            result_str = tool_result
        else:
            result_str = orjson.dumps(tool_result).decode()

        yield ToolCallResultEvent(
            type=EventType.TOOL_CALL_RESULT,
            message_id=out_msg_id,
            tool_call_id=tool_call_id,
            content=result_str,
            role="tool",
        )

        # Inject the tool result into the conversation context.
        context.add_message(
            Message(
                role=MessageRole.TOOL,
                content=[ToolResult(tool_id=tool_call_id, result=result_str)],
                msg_id=out_msg_id,
            )
        )

        # Build a fresh RunAgentInput re-declaring the UI tools so the
        # next agent turn can pause again on a different UI tool if
        # needed.
        fresh_tools = [
            Tool(name=n, description="", parameters=None)
            for n in (ui_tool_names or [])
        ]
        fresh_input = RunAgentInput(
            thread_id=thread_id,
            run_id=run_id,
            state={},
            messages=[],
            tools=fresh_tools,
            context=[],
            forwarded_props={},
        )

        async for ev in self.run_stream_with_ag_ui(
            context=context,
            run_input=fresh_input,
            state=state,
            out_msg_id=out_msg_id,
            paused_run_store=paused_run_store,
            pause_metadata=pause_metadata,
        ):
            yield ev

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
