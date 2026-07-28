"""
Base LLM interface for Echo SDK.

Provides a framework-agnostic interface for LLM calls.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from echo.models.user_conversation import ConversationContext, ToolCall, ToolResult
from echo.tools.core import BaseTool
from echo.tools.core.schemas import (
    ControlFlow,
    ElicitationDetails,
    ElicitationResponse,
    Observability,
    ToolOutput,
)
from echo.tools.system import SystemTool

from .config import LLMConfig
from .schemas import LLMResponse, StreamEvent

logger = logging.getLogger(__name__)


def prompt_cache_id(system_prompt: Optional[str]) -> Optional[str]:
    """Stable ID for a cacheable system prompt: the MD5 of its bytes.

    Two requests share a prompt cache iff their cacheable prefixes are
    byte-identical, so this hash *is* the cache identity — same agent (and
    prompt version) across sessions gives one ID, different agents give
    different ones. Providers that expose a cache-routing key (OpenAI's
    ``prompt_cache_key``) send it; everywhere else it is a log field for
    confirming which requests are meant to share an entry.

    Not a security primitive — MD5 is used for its speed and short digest.
    """
    if not system_prompt:
        return None
    return hashlib.md5(system_prompt.encode("utf-8")).hexdigest()


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
        system_suffix: Optional[str] = None,
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
            system_prompt: Optional system prompt for LLM behavior. Treated as
                     the cacheable prefix — keep it byte-stable across sessions.
            system_suffix: Optional volatile context (user, session, current
                     time). Rendered after the prompt-cache breakpoint, so it
                     may change per request without invalidating the cache.
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
            tool_result = await tool.run(**full_input)

            if is_elicitation or isinstance(tool_result, ElicitationDetails):
                meta = {}
                if (
                    hasattr(tool, "_manager")
                    and hasattr(tool._manager, "_config")
                    and tool._manager._config.url
                ):
                    meta["mcp_url"] = str(tool._manager._config.url)
                if hasattr(tool, "meta") and tool.meta:
                    meta.update(tool.meta)
                return ElicitationResponse(
                    tool_id=tool_call.tool_id,
                    tool_name=tool.name,
                    details=tool_result,
                    meta=meta if meta else None,
                )

            # Resolve the loop directives the tool declared. observability is
            # honored from any tool; INTERRUPT is honored ONLY from SystemTools
            # (the unfakeable marker) — any other tool declaring it is coerced
            # to CONTINUE so external/user/MCP tools can never force a recompute.
            observability = tool.observability
            if isinstance(tool, SystemTool):
                control_flow = tool.control_flow
            else:
                control_flow = ControlFlow.CONTINUE
                if tool.control_flow == ControlFlow.INTERRUPT:
                    logger.warning(
                        "Tool %r declared control_flow=INTERRUPT but is not a "
                        "SystemTool; coercing to CONTINUE.",
                        tool.name,
                    )

            if isinstance(tool_result, ToolOutput):
                return ToolResult(
                    tool_id=tool_call.tool_id,
                    result=tool_result.result,
                    meta=tool_result.meta,
                    control_flow=control_flow,
                    observability=observability,
                )

            return ToolResult(
                tool_id=tool_call.tool_id,
                result=tool_result,
                control_flow=control_flow,
                observability=observability,
            )

        except Exception as e:
            logger.error(
                "Error running tool '%s' :: context : %s :: error : %s",
                tool_call.tool_name,
                tool_context,
                e,
                exc_info=True,
            )
            return ToolResult(
                tool_id=tool_call.tool_id, result=f"Error running tool: {e}"
            )

    async def invoke_stream(
        self,
        context: ConversationContext,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        system_suffix: Optional[str] = None,
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
            system_prompt: Optional system prompt (cacheable prefix)
            system_suffix: Optional volatile context, sent after the
                     prompt-cache breakpoint
            **kwargs: Additional arguments (max_tokens, temperature)

        Yields:
            StreamEvent objects with type indicating what happened
        """
        pass
