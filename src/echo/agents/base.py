"""
Base agent interface for Echo SDK.

Provides a framework-agnostic interface for agents with adapters.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

from echo.agents.config import AgentConfig
from echo.llm import LLMConfig, get_llm
from echo.llm.schemas import StreamEvent, StreamEventType
from echo.tools.base_tool import BaseTool

from .schemas import AgentResult

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
    ):
        """
        Initialize agent with config from YAML.

        Args:
            llm_config: Optional LLM configuration. Defaults to Bedrock Haiku.
        """
        # Initialize tools as empty list (subclasses can override)
        self.tools = tools or []

        # Load config from YAML (both agent and task)
        self.role = agent_config.persona.role
        self.goal = agent_config.persona.goal
        self.backstory = agent_config.persona.backstory
        self.task_description = agent_config.task.description
        self.expected_output = agent_config.task.expected_output

        # Set LLM config, defaults to Bedrock Haiku
        self.llm_config = llm_config or LLMConfig()
        self.llm = get_llm(self.llm_config)

    @abstractmethod
    async def run(
        self,
        context: "ConversationContext",
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
        """Build system prompt from agent config."""
        if self.role:
            system_prompt = f"You are a {self.role}\n\n"
        if not skip_goal and self.goal:
            system_prompt += f"Your goal is: {self.goal}\n\n"

        if not self.task_description:
            raise Exception("Task description is required")
        system_prompt += f"Task Description: {self.task_description} \n\n"

        if not skip_expected_output and self.expected_output:
            system_prompt += f"Expected Output: {self.expected_output}"

        return system_prompt

    async def _run_agent(
        self, context: "ConversationContext"
    ) -> AgentResult:
        """Run the agent (non-streaming)."""
        try:
            # Build system prompt with task(mandatory) & expected output,role(optional)
            system_prompt = self._build_system_prompt(skip_goal=True)

            # Call LLM with tools - tool_context automatically injected
            llm_response, updated_context = await self.llm.invoke(
                context=context,
                tools=self.tools,
                system_prompt=system_prompt,
            )
            return AgentResult(
                llm_response=llm_response,
                context=updated_context,
                agent_name=self.name,
            )

        except Exception as e:
            return AgentResult(
                llm_response=None,
                context=context,
                agent_name=self.name,
                error=str(e),
            )

    async def _run_agent_stream(
        self, context: "ConversationContext"
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run the agent with streaming."""
        try:
            # Build system prompt with task(mandatory) & expected output,role(optional)
            system_prompt = self._build_system_prompt(skip_goal=True)

            async for event in self.llm.invoke_stream(
                context=context,
                tools=self.tools,
                system_prompt=system_prompt,
            ):
                yield event

        except Exception as e:
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
