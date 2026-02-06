"""
Anthropic LLM implementation.
"""

import uuid
from typing import Any, AsyncGenerator, List, Optional, Tuple

import orjson

from echo.models.user_conversation import (
    ConversationContext,
    LLMUsageMetrics,
    Message,
    MessageRole,
    TextMessage,
    ToolCall,
)
from echo.tools.base_tool import BaseTool
from echo.tools.schemas import ElicitationResponse

from .base import BaseLLM
from .config import LLMConfig
from .schemas import LLMResponse, StreamEvent, StreamEventType, VerboseResponseItem


class AnthropicLLM(BaseLLM):
    """Anthropic LLM provider (direct API, not Bedrock)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        self.thinking_budget_tokens = config.thinking.budget_tokens if config.thinking else None

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            import anthropic

            # Use config api_key if provided, otherwise falls back to ANTHROPIC_API_KEY env var
            self._client = anthropic.Anthropic(api_key=self.config.api_key)
        return self._client

    def _supports_extended_thinking(self) -> bool:
        """Check if model supports extended thinking."""
        # Claude 4/4.5 models support extended thinking
        return any(
            x in self.model
            for x in ["claude-sonnet-4", "claude-haiku-4", "claude-opus-4"]
        )

    def _parse_response(self, response, msg_id: str) -> Message:
        """Parse Anthropic response into a Message."""
        content_items = []
        for block in response.content:
            if block.type == "text":
                content_items.append(TextMessage(text=block.text))
            elif block.type == "tool_use":
                content_items.append(
                    ToolCall(
                        tool_id=block.id,
                        tool_name=block.name,
                        tool_input=block.input,
                    )
                )

        return Message(
            role=MessageRole.ASSISTANT,
            content=content_items,
            msg_id=msg_id,
            usage=LLMUsageMetrics(
                in_t=response.usage.input_tokens,
                op_t=response.usage.output_tokens,
                latency_ms=0,  # Anthropic doesn't provide this directly
            ),
        )

    async def invoke(
        self,
        context: ConversationContext,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        out_msg_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[LLMResponse, ConversationContext]:
        """
        Unified LLM invocation using Anthropic API.

        Handles both simple prompts and agentic tool-use loops.
        Injects context.tool_context into all tool calls.
        """
        final_response = LLMResponse()
        elicitations = []
        msg_id = out_msg_id or str(uuid.uuid4())

        # Build tool schemas if tools provided
        tool_schemas = None
        tool_map = {}
        if tools:
            tool_schemas = [tool.to_anthropic_schema() for tool in tools]
            tool_map = {tool.name: tool for tool in tools}

        # Build messages from context
        messages = context.to_anthropic_messages()

        # Build the base request kwargs
        request_kwargs = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": messages,
        }

        # Extended thinking for Claude 4/4.5
        if self.thinking_budget_tokens and self._supports_extended_thinking():
            request_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }

        if system_prompt:
            request_kwargs["system"] = system_prompt
        if tool_schemas:
            request_kwargs["tools"] = tool_schemas

        # call the LLM for the given number of iterations
        # No tools = single iteration
        iterations = self.max_iterations if tool_schemas else 1

        for _ in range(iterations):

            # Call Anthropic
            response = self.client.messages.create(**request_kwargs)

            # Parse response into Message
            assistant_msg = self._parse_response(response, msg_id)
            context.add_message(assistant_msg)
            messages.append(assistant_msg.to_anthropic_message())

            tool_results = []
            for content_item in assistant_msg.content:
                if isinstance(content_item, TextMessage):
                    final_response.verbose.append(
                        VerboseResponseItem(type="text", text=content_item.text)
                    )
                elif isinstance(content_item, ToolCall):
                    tool_res = await self.invoke_tool(
                        tool_map, content_item, context.tool_context
                    )
                    if isinstance(tool_res, ElicitationResponse):
                        elicitations.append(tool_res)
                    else:
                        final_response.verbose.append(
                            VerboseResponseItem(
                                type="tool", tool_name=content_item.tool_name
                            )
                        )
                        tool_results.append(tool_res)

            # Add all tool results as a single user message (Anthropic convention)
            if tool_results:
                results_msg = Message(
                    role=MessageRole.TOOL,
                    content=tool_results,
                    msg_id=msg_id,
                )
                context.add_message(results_msg)
                messages.append(results_msg.to_anthropic_message())
                final_response.pending_tool_result_processing = True
            else:
                final_response.pending_tool_result_processing = False

            # if we have elicitations, end loop and return to user
            if elicitations:
                break

            request_kwargs["messages"] = messages
            # if we have no tool results, only text, end loop and return to user
            if not tool_results:
                break

        # Max iterations reached / no tool use / elicitations - extract last response
        final_text = ""
        # in case toolResults are present, we need to use the last message before that
        last_message = (
            context.messages[-1]
            if context.messages[-1].role == MessageRole.ASSISTANT
            else context.messages[-2]
        )
        for item in last_message.content:
            if isinstance(item, TextMessage):
                final_text += item.text

        final_response.text = final_text.strip()
        final_response.elicitations = elicitations or None
        return final_response, context

    async def invoke_stream(
        self,
        context: ConversationContext,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        out_msg_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Streaming LLM invocation using Anthropic streaming API.

        Yields StreamEvent objects as response is generated.
        Handles tool use by yielding TOOL_CALL_START/END events.

        Args:
            context: Conversation context with messages
            tools: Optional list of tools available for the LLM
            system_prompt: Optional system prompt
            out_msg_id: Optional message ID for grouping messages
            **kwargs: Additional arguments (max_tokens, temperature)

        Yields:
            StreamEvent objects with type indicating what happened
        """
        msg_id = out_msg_id or str(uuid.uuid4())

        # Build tool schemas if tools provided
        tool_schemas = None
        tool_map = {}
        if tools:
            tool_schemas = [tool.to_anthropic_schema() for tool in tools]
            tool_map = {tool.name: tool for tool in tools}

        messages = context.to_anthropic_messages()

        # Build the base request kwargs
        request_kwargs = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "messages": messages,
        }

        # Extended thinking for Claude 4/4.5
        if self.thinking_budget_tokens and self._supports_extended_thinking():
            request_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }

        if system_prompt:
            request_kwargs["system"] = system_prompt
        if tool_schemas:
            request_kwargs["tools"] = tool_schemas

        iterations = self.max_iterations if tool_schemas else 1

        final_response = LLMResponse()
        elicitations = []

        for _ in range(iterations):
            try:
                # Use streaming API
                with self.client.messages.stream(**request_kwargs) as stream:
                    blocks = {}  # block index to content block
                    content_items = []
                    tool_results = []
                    usage_metrics = None

                    for event in stream:
                        if event.type == "content_block_start":
                            block_id = event.index
                            block = event.content_block
                            if block.type == "tool_use":
                                tool = tool_map.get(block.name)
                                is_elicitation = tool.is_elicitation if tool else False
                                blocks[block_id] = {
                                    "type": "tool",
                                    "tool_id": block.id,
                                    "tool_name": block.name,
                                    "input_json": "",
                                    "is_elicitation": is_elicitation,
                                }
                                if not is_elicitation:
                                    yield StreamEvent(
                                        type=StreamEventType.TOOL_CALL_START,
                                        details={
                                            "tool_id": block.id,
                                            "tool_name": block.name,
                                        },
                                    )
                            else:
                                blocks[block_id] = {
                                    "type": "text",
                                    "text": "",
                                }

                        elif event.type == "content_block_delta":
                            block_id = event.index
                            delta = event.delta

                            if delta.type == "text_delta":
                                blocks[block_id]["text"] += delta.text
                                yield StreamEvent(
                                    type=StreamEventType.TEXT, text=delta.text
                                )
                            elif delta.type == "input_json_delta":
                                blocks[block_id]["input_json"] += delta.partial_json

                        elif event.type == "content_block_stop":
                            block_id = event.index
                            if blocks[block_id]["type"] == "tool":
                                # Tool block complete - parse input and execute
                                input_json_str = blocks[block_id]["input_json"]
                                parsed_input = (
                                    orjson.loads(input_json_str)
                                    if input_json_str
                                    else {}
                                )
                                tool_call = ToolCall(
                                    tool_id=blocks[block_id]["tool_id"],
                                    tool_name=blocks[block_id]["tool_name"],
                                    tool_input=parsed_input,
                                )
                                content_items.append((block_id, tool_call))
                                tool_res = await self.invoke_tool(
                                    tool_map, tool_call, context.tool_context
                                )
                                # progress message event (skip for elicitation tools)
                                if not blocks[block_id].get("is_elicitation"):
                                    yield StreamEvent(
                                        type=StreamEventType.TOOL_CALL_END,
                                        details={
                                            "tool_name": blocks[block_id]["tool_name"],
                                            "tool_id": blocks[block_id]["tool_id"],
                                        },
                                    )

                                if isinstance(tool_res, ElicitationResponse):
                                    elicitations.append(tool_res)
                                else:
                                    tool_results.append(tool_res)
                            else:
                                content_items.append(
                                    (
                                        block_id,
                                        TextMessage(text=blocks[block_id]["text"]),
                                    )
                                )

                        elif event.type == "message_delta":
                            # Contains usage info
                            if hasattr(event, "usage") and event.usage:
                                usage_metrics = LLMUsageMetrics(
                                    in_t=getattr(event.usage, "input_tokens", 0),
                                    op_t=getattr(event.usage, "output_tokens", 0),
                                    latency_ms=0,
                                )

                # -- end of stream --

                # Build content items list and verbose list
                content_items = sorted(content_items, key=lambda x: x[0])
                content_items_list = []
                for _, item in content_items:
                    content_items_list.append(item)
                    if isinstance(item, TextMessage):
                        final_response.verbose.append(
                            VerboseResponseItem(type="text", text=item.text)
                        )
                    elif isinstance(item, ToolCall):
                        final_response.verbose.append(
                            VerboseResponseItem(type="tool", tool_name=item.tool_name)
                        )

                # Build the assistant message and add to context
                llm_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=content_items_list,
                    msg_id=msg_id,
                    usage=usage_metrics,
                )
                context.add_message(llm_message)
                messages.append(llm_message.to_anthropic_message())

                if tool_results:
                    results_msg = Message(
                        role=MessageRole.TOOL,
                        content=tool_results,
                        msg_id=msg_id,
                    )
                    context.add_message(results_msg)
                    messages.append(results_msg.to_anthropic_message())
                    final_response.pending_tool_result_processing = True
                else:
                    final_response.pending_tool_result_processing = False

                if elicitations:
                    break

                request_kwargs["messages"] = messages
                if not tool_results:
                    break

            except Exception as e:
                yield StreamEvent(type=StreamEventType.ERROR, error=str(e))
                return

        final_response.elicitations = elicitations or None
        yield StreamEvent(
            type=StreamEventType.DONE, llm_response=final_response, context=context
        )
