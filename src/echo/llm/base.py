"""
Base LLM interface for Echo SDK.

Provides a framework-agnostic interface for LLM calls.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from echo.models.user_conversation import ConversationContext, ToolCall, ToolResult
from echo.tools.base_tool import BaseTool
from echo.tools.schemas import ElicitationResponse, ElicitationDetails

from .config import LLMConfig
from .schemas import LLMResponse, StreamEvent


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.max_iterations = config.max_iterations

    @abstractmethod
    async def invoke(
        self,
        context: ConversationContext,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        out_msg_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[LLMResponse, ConversationContext]:
        """
        Unified LLM invocation.

        Handles both simple prompts and agentic tool-use loops.
        When tools are provided, runs an agentic loop where the LLM can call
        tools and receive results until it produces a final text response.

        Tool context injection: When executing tools, `context.tool_context`
        is automatically merged with tool inputs, allowing hidden parameters
        (user_id, workspace_id, etc.) to be passed to tools without the LLM
        needing to know about them.

        Args:
            context: Conversation context with messages and system_context.
                     The last message should be the user's query.
            tools: Optional list of BaseTool instances available for the LLM.
                   If provided, enables agentic loop with tool calling.
            system_prompt: Optional system prompt for LLM behavior.
            **kwargs: Additional provider-specific arguments.

        Returns:
            Tuple of (LLMResponse, updated_context):
            - LLMResponse: Structured response with text, pending_tool_result_processing, error, elicitations
            - updated_context: Context with tool calls/results appended
        """
        pass

    async def invoke_tool(
        self,
        tool_map: Dict[str, BaseTool],
        tool_call: ToolCall,
        tool_context: Dict[str, Any],
    ) -> ToolResult | ElicitationResponse:
        """
        Args:
            tool_call (ToolCall): _description_
            extras (Dict[str, Any]): _description_

        Returns:
            ToolResult: _description_
        """
        is_elicitation = False
        try:
            tool = tool_map.get(tool_call.tool_name)
            if not tool:
                return ToolResult(
                    tool_id=tool_call.tool_id,
                    result="Error: Tool not found, don't use this tool",
                )

            is_elicitation = tool.is_elicitation
            full_input = {**tool_call.tool_input, **{"tool_context": tool_context}}
            result = await tool.run(**full_input)
            if tool.is_elicitation or isinstance(result, ElicitationDetails):
                return ElicitationResponse(
                    tool_id=tool_call.tool_id,
                    tool_name=tool.name,
                    details=result,
                )
            else:
                return ToolResult(tool_id=tool_call.tool_id, result=result)
        except Exception as e:
            if is_elicitation:
                return ElicitationResponse(
                    tool_id=tool_call.tool_id, error=f"Error running tool: {e}"
                )
            else:
                return ToolResult(
                    tool_id=tool_call.tool_id, result=f"Error running tool: {e}"
                )

    async def invoke_stream(
        self,
        context: ConversationContext,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        out_msg_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Streaming LLM invocation using Bedrock converse_stream API.

        Yields StreamEvent objects as response is generated.
        Handles tool use by yielding TOOL_CALL_START/END events.

        Args:
            context: Conversation context with messages
            tools: Optional list of tools available for the LLM
            system_prompt: Optional system prompt
            **kwargs: Additional arguments (max_tokens, temperature)

        Yields:
            StreamEvent objects with type indicating what happened
        """
        pass
