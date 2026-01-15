import json
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator

from echo.models.providers import Provider

# --------- ENUMS ---------


class MessageRole(str, Enum):
    """Role of message sender."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageType(str, Enum):
    """Type of message content."""

    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


# --------- CONTENT TYPES ---------


class TextMessage(BaseModel):
    """Text content block."""

    type: MessageType = MessageType.TEXT
    text: str


class ToolCall(BaseModel):
    """Tool call content block."""

    type: MessageType = MessageType.TOOL_CALL
    tool_id: str
    tool_name: str
    tool_input: Dict[str, Any]


class ToolResult(BaseModel):
    """Tool result content block."""

    type: MessageType = MessageType.TOOL_RESULT
    tool_id: str
    result: Any


# Type alias for content items
ContentItem = Union[TextMessage, ToolCall, ToolResult]


class LLMUsageMetrics(BaseModel):
    """Usage metrics from LLM response."""

    in_t: int = 0
    op_t: int = 0
    latency_ms: int = 0


# --------- MESSAGE MODEL ---------


class Message(BaseModel):
    """
    A single message in a conversation.

    The content field is a list that can contain multiple items:
    - TextMessage: Text content
    - ToolCall: Tool invocation request
    - ToolResult: Result from a tool execution

    This allows for batch tool calls (multiple tools in one message)
    and mixed content (text + tool calls together).
    """

    role: MessageRole
    content: List[ContentItem]
    usage: LLMUsageMetrics = Field(default_factory=LLMUsageMetrics)
    timestamp: Optional[int] = Field(
        default=None, description="Unix timestamp of the message"
    )
    msg_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @model_validator(mode="after")
    def validate_role_content(self) -> "Message":
        """Validate that content types match the message role."""
        has_text = any(isinstance(c, TextMessage) for c in self.content)
        has_tool_call = any(isinstance(c, ToolCall) for c in self.content)
        has_tool_result = any(isinstance(c, ToolResult) for c in self.content)

        if self.role == MessageRole.USER:
            if has_tool_call or has_tool_result:
                raise ValueError("USER messages can only contain TextMessage")

        elif self.role == MessageRole.ASSISTANT:
            if has_tool_result:
                raise ValueError("ASSISTANT messages cannot contain ToolResult")

        elif self.role == MessageRole.TOOL:
            if has_text or has_tool_call:
                raise ValueError("TOOL messages can only contain ToolResult")

        return self

    def to_dict(self, format: Optional[Provider] = None) -> Dict[str, Any]:
        """Convert to dictionary for the specified provider format."""
        if format == Provider.ANTHROPIC:
            return self.to_anthropic_message()
        elif format == Provider.OPENAI:
            # OpenAI may return multiple messages, return first for compatibility
            messages = self.to_openai_messages()
            return messages[0] if messages else {}
        elif format == Provider.BEDROCK:
            return self.to_bedrock_message()

        return self.model_dump()

    def to_anthropic_message(self) -> Dict[str, Any]:
        """Convert to Anthropic message format."""
        blocks = []
        for item in self.content:
            if isinstance(item, TextMessage):
                blocks.append({"type": "text", "text": item.text})
            elif isinstance(item, ToolCall):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": item.tool_id,
                        "name": item.tool_name,
                        "input": item.tool_input,
                    }
                )
            elif isinstance(item, ToolResult):
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": item.tool_id,
                        "content": str(item.result),
                    }
                )
        # Anthropic expects tool results in 'user' role messages
        role = "user" if self.role == MessageRole.TOOL else self.role.value
        return {"role": role, "content": blocks}

    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """
        Convert to OpenAI message format.

        Returns a list because OpenAI requires separate messages for tool results.
        """
        messages = []

        # Separate content types
        text_parts = [c for c in self.content if isinstance(c, TextMessage)]
        tool_calls = [c for c in self.content if isinstance(c, ToolCall)]
        tool_results = [c for c in self.content if isinstance(c, ToolResult)]

        # Assistant message with text and/or tool_calls
        if self.role == MessageRole.ASSISTANT:
            msg: Dict[str, Any] = {"role": "assistant"}
            if text_parts:
                msg["content"] = " ".join(t.text for t in text_parts)
            else:
                msg["content"] = None
            if tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.tool_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.tool_input),
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(msg)

        # User text message
        elif self.role == MessageRole.USER and text_parts:
            messages.append(
                {
                    "role": "user",
                    "content": " ".join(t.text for t in text_parts),
                }
            )

        # Tool results become separate messages (OpenAI requires this)
        for tr in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tr.tool_id,
                    "content": str(tr.result),
                }
            )

        return messages

    def to_bedrock_message(self) -> Dict[str, Any]:
        """Convert to Bedrock Converse API format."""
        blocks = []
        for item in self.content:
            if isinstance(item, TextMessage):
                blocks.append({"text": item.text})
            elif isinstance(item, ToolCall):
                blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": item.tool_id,
                            "name": item.tool_name,
                            "input": item.tool_input,
                        }
                    }
                )
            elif isinstance(item, ToolResult):
                blocks.append(
                    {
                        "toolResult": {
                            "toolUseId": item.tool_id,
                            "content": [{"text": str(item.result)}],
                        }
                    }
                )
        # Bedrock expects tool results in 'user' role messages
        role = "user" if self.role == MessageRole.TOOL else self.role.value
        return {"role": role, "content": blocks}


# --------- CONVERSATION CONTEXT ---------


class ConversationContext(BaseModel):
    """
    Context for LLM calls including conversation history and system context.

    - messages: List of conversation messages (user, assistant, tool calls/results)
    - conversation_summary: Optional summary of earlier conversation
    - system_context: Hidden context not shown to LLM but used for execution
        - tool_context: Dict injected into all tool calls (user_id, workspace_id, etc.)
    """

    messages: List[Message] = Field(default_factory=list)
    conversation_summary: Optional[str] = Field(
        default=None, description="Brief summary of conversation before recent messages"
    )
    system_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hidden context for execution (not shown to LLM)",
    )

    @property
    def tool_context(self) -> Dict[str, Any]:
        """
        Hidden context injected into all tool calls.

        Use this for user_id, workspace_id, auth tokens, etc.
        that tools need but LLM shouldn't ask the user for.
        """
        return self.system_context.get("tool_context", {})

    # --------- HELPER METHODS ---------

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)

    # --------- PROVIDER-SPECIFIC CONVERTERS ---------

    def to_anthropic_messages(self) -> List[Dict[str, Any]]:
        """
        Convert to Anthropic message format.

        Handles text, tool_use, and tool_result blocks.
        """
        result = []
        for msg in self.messages:
            msg_dict = msg.to_anthropic_message()
            if msg_dict and msg_dict.get("content"):
                result.append(msg_dict)
        return result

    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """
        Convert to OpenAI message format.

        Handles text, tool_calls, and tool role messages.
        Note: Each Message may produce multiple OpenAI messages (for tool results).
        """
        result = []
        for msg in self.messages:
            openai_msgs = msg.to_openai_messages()
            result.extend(openai_msgs)
        return result

    def to_bedrock_messages(self) -> List[Dict[str, Any]]:
        """
        Convert to Bedrock Converse API message format.

        Handles text, toolUse, and toolResult blocks.
        """
        result = []
        for msg in self.messages:
            msg_dict = msg.to_bedrock_message()
            if msg_dict and msg_dict.get("content"):
                result.append(msg_dict)
        return result

    # --------- STRING REPRESENTATIONS ---------

    def get_conversations_context_str(self) -> str:
        """Get conversation as a formatted string (for prompts)."""
        if not self.messages:
            return ""

        conversation_str = "\nConversation Context:\n"
        for msg in self.messages:
            for item in msg.content:
                if isinstance(item, TextMessage):
                    role_label = "User" if msg.role == MessageRole.USER else "Bot"
                    conversation_str += f"- {role_label}: {item.text}\n"
        return conversation_str

    def get_conversation_summary_str(self) -> str:
        """Get conversation summary as formatted string."""
        if not self.conversation_summary:
            return ""

        msg_count = len(self.messages)
        if msg_count > 0:
            return (
                f"\nSummary before last {msg_count} turns:\n{self.conversation_summary}"
            )
        return f"\nSummary: {self.conversation_summary}"
