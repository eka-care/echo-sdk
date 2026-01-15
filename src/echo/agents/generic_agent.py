"""
Generic Agent - Flexible agent for any task.

Can be configured with custom persona, task description, and tools.
Supports both regular and streaming invocation.
"""

from typing import AsyncGenerator

from echo.agents.base import BaseAgent
from echo.llm.schemas import StreamEvent
from echo.models.user_conversation import ConversationContext

from .schemas import AgentResult


class GenericAgent(BaseAgent):
    """
    Generic agent that can be used to perform any task.
    """

    def __init__(self, **kwargs) -> None:
        if not kwargs.get("agent_config"):
            raise Exception("Agent Config Mandatory for generic agent")
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return "generic_agent"

    async def run(
        self,
        context: ConversationContext,
    ) -> AgentResult:
        """
        Generic Agent to run for any use case
        Returns:
            AgentResult with JSON output containing:
        """
        # Add you own code here or call the helper method _run_agent from base class
        # No changes to the base run agent needed for generic agent
        return await self._run_agent(context)

    async def run_stream(
        self,
        context: ConversationContext,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream the agent's response. Overrides the base class method.
        Args:
            context: ConversationContext with messages and tool_context
        Yields:
            StreamEvent objects (TEXT, TOOL_CALL_START, TOOL_CALL_END, DONE, ERROR)
        """
        # Add you own code here or call the helper method _run_agent_stream from base class
        # No changes to the base run agent needed for generic agent
        async for event in self._run_agent_stream(context):
            yield event
