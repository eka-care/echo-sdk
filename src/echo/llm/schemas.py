from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from echo.models.user_conversation import ConversationContext, LLMUsageMetrics
from echo.tools.core.schemas import ElicitationResponse


class VerboseResponseItem(BaseModel):
    type: Literal["text", "tool"] = "text"
    text: Optional[str] = None
    tool_name: Optional[str] = None


class LLMResponse(BaseModel):
    """Structured response from LLM invocation."""

    verbose: List[VerboseResponseItem] = []
    text: str = ""
    # Usage of the final LLM call (includes prompt-cache read/write tokens).
    usage: Optional[LLMUsageMetrics] = None
    details: Optional[Dict[str, Any]] = None
    pending_tool_result_processing: bool = False
    # Set when a tool changed the agent's loaded state (control_flow=INTERRUPT):
    # the provider loop stops and hands back so the agent can recompute the
    # system prompt + tool list and re-invoke. Consumed by the agent outer loop.
    pending_context_reload: bool = False
    error: Optional[str] = None
    elicitations: Optional[List[ElicitationResponse]] = None


class StreamEventType(str, Enum):
    """Event types for streaming LLM responses."""

    TEXT = "text"  # Partial text chunk
    TOOL_CALL_START = "tool_start"  # Tool invocation detected
    TOOL_CALL_ARGS = "tool_args"  # Tool args fragment (streaming)
    TOOL_CALL_END = "tool_end"  # Tool execution finished
    DONE = "eos"  # Stream complete
    ERROR = "error"  # Error occurred


class StreamEvent(BaseModel):
    """Event emitted during streaming LLM invocation."""

    type: StreamEventType
    text: Optional[str] = None  # For TYPE TEXT
    details: Optional[Dict[str, Any]] = None  # For TYPE TOOL_CALL_START/END

    llm_response: Optional[LLMResponse] = None  # For TYPE DONE LLM_RESPONSE
    context: Optional[ConversationContext] = None  # For TYPE DONE CONTEXT
    error: Optional[str] = None  # For TYPE ERROR
