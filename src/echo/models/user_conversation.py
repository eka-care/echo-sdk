import base64
import json
import logging
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, HttpUrl, model_validator

from echo.models.providers import Provider

logger = logging.getLogger(__name__)

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
    IMAGE = "image"
    DOCUMENT = "document"


class ContentSourceType(str, Enum):
    """Source type for media content."""

    BASE64 = "base64"
    URL = "url"


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


class ImageContent(BaseModel):
    """Image content block for multimodal messages."""

    type: MessageType = MessageType.IMAGE
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
    source_type: ContentSourceType
    data: Optional[str] = None
    url: Optional[HttpUrl] = None

    @model_validator(mode="after")
    def validate_source(self):
        if self.source_type == ContentSourceType.URL and not self.url:
            raise ValueError("url is required when source_type is 'url'")
        if self.source_type == ContentSourceType.BASE64 and not self.data:
            raise ValueError("data is required when source_type is 'base64'")
        return self


class DocumentContent(BaseModel):
    """Document content block (e.g. PDF) for multimodal messages."""

    type: MessageType = MessageType.DOCUMENT
    media_type: Literal["application/pdf"] = "application/pdf"
    source_type: ContentSourceType
    data: Optional[str] = None
    url: Optional[HttpUrl] = None
    name: Optional[str] = None

    @model_validator(mode="after")
    def validate_source(self):
        if self.source_type == ContentSourceType.URL and not self.url:
            raise ValueError("url is required when source_type is 'url'")
        if self.source_type == ContentSourceType.BASE64 and not self.data:
            raise ValueError("data is required when source_type is 'base64'")
        return self


# Type alias for content items
ContentItem = Union[TextMessage, ToolCall, ToolResult, ImageContent, DocumentContent]


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
    usage: Optional[LLMUsageMetrics] = None
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
                raise ValueError(
                    "USER messages can only contain TextMessage, ImageContent, or DocumentContent"
                )

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
        elif format == Provider.GEMINI:
            return self.to_gemini_message()

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
            elif isinstance(item, ImageContent):
                if item.source_type == ContentSourceType.URL:
                    source = {"type": "url", "url": str(item.url)}
                else:
                    source = {
                        "type": "base64",
                        "media_type": item.media_type,
                        "data": item.data,
                    }
                blocks.append({"type": "image", "source": source})
            elif isinstance(item, DocumentContent):
                if item.source_type == ContentSourceType.URL:
                    source = {"type": "url", "url": str(item.url)}
                else:
                    source = {
                        "type": "base64",
                        "media_type": item.media_type,
                        "data": item.data,
                    }
                blocks.append({"type": "document", "source": source})
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

        # User message (text and/or multimodal)
        elif self.role == MessageRole.USER:
            media_parts = [
                c
                for c in self.content
                if isinstance(c, (ImageContent, DocumentContent))
            ]
            if media_parts:
                content_blocks = []
                for t in text_parts:
                    content_blocks.append({"type": "text", "text": t.text})
                for item in media_parts:
                    if isinstance(item, ImageContent):
                        if item.source_type == ContentSourceType.URL:
                            url_str = str(item.url)
                        else:
                            url_str = f"data:{item.media_type};base64,{item.data}"
                        content_blocks.append(
                            {"type": "image_url", "image_url": {"url": url_str}}
                        )
                    elif isinstance(item, DocumentContent):
                        if item.source_type == ContentSourceType.URL:
                            content_blocks.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": str(item.url)},
                                }
                            )
                        else:
                            logger.warning(
                                "OpenAI has limited support for base64 PDF documents"
                            )
                            content_blocks.append(
                                {
                                    "type": "text",
                                    "text": f"[Document: {item.name or 'document.pdf'}]",
                                }
                            )
                messages.append({"role": "user", "content": content_blocks})
            elif text_parts:
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
            elif isinstance(item, ImageContent):
                if item.source_type == ContentSourceType.URL:
                    logger.warning(
                        "Bedrock does not support image URLs directly; provide base64 data instead"
                    )
                    continue
                fmt = item.media_type.split("/")[-1]
                blocks.append(
                    {
                        "image": {
                            "format": fmt,
                            "source": {"bytes": base64.b64decode(item.data)},
                        }
                    }
                )
            elif isinstance(item, DocumentContent):
                if item.source_type == ContentSourceType.URL:
                    logger.warning(
                        "Bedrock does not support document URLs directly; provide base64 data instead"
                    )
                    continue
                fmt = item.media_type.split("/")[-1]
                doc: Dict[str, Any] = {
                    "format": fmt,
                    "source": {"bytes": base64.b64decode(item.data)},
                }
                if item.name:
                    doc["name"] = item.name
                blocks.append({"document": doc})
        # Bedrock expects tool results in 'user' role messages
        role = "user" if self.role == MessageRole.TOOL else self.role.value
        return {"role": role, "content": blocks}

    def to_gemini_message(self) -> Dict[str, Any]:
        """Convert to Google Gemini message format."""
        parts = []
        for item in self.content:
            if isinstance(item, TextMessage):
                parts.append({"text": item.text})
            elif isinstance(item, ToolCall):
                parts.append(
                    {
                        "function_call": {
                            "name": item.tool_name,
                            "args": item.tool_input,
                        }
                    }
                )
            elif isinstance(item, ToolResult):
                parts.append(
                    {
                        "function_response": {
                            "name": item.tool_id,
                            "response": {"result": str(item.result)},
                        }
                    }
                )
            elif isinstance(item, ImageContent):
                if item.source_type == ContentSourceType.URL:
                    parts.append(
                        {
                            "file_data": {
                                "mime_type": item.media_type,
                                "file_uri": str(item.url),
                            }
                        }
                    )
                else:
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": item.media_type,
                                "data": item.data,
                            }
                        }
                    )
            elif isinstance(item, DocumentContent):
                if item.source_type == ContentSourceType.URL:
                    parts.append(
                        {
                            "file_data": {
                                "mime_type": item.media_type,
                                "file_uri": str(item.url),
                            }
                        }
                    )
                else:
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": item.media_type,
                                "data": item.data,
                            }
                        }
                    )
        role = "model" if self.role == MessageRole.ASSISTANT else "user"
        return {"role": role, "parts": parts}


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

    def to_gemini_messages(self) -> List[Dict[str, Any]]:
        """Convert to Google Gemini message format."""
        result = []
        for msg in self.messages:
            msg_dict = msg.to_gemini_message()
            if msg_dict and msg_dict.get("parts"):
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
