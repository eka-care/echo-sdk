"""
OpenAI LLM implementation.
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


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def _parse_response(self, response) -> Message:
        """Parse OpenAI response into a Message."""
        message = response.choices[0].message
        content_items = []

        if message.content:
            content_items.append(TextMessage(text=message.content))

        if message.tool_calls:
            for tc in message.tool_calls:
                content_items.append(
                    ToolCall(
                        tool_id=tc.id,
                        tool_name=tc.function.name,
                        tool_input=orjson.loads(tc.function.arguments),
                    )
                )

        return Message(
            role=MessageRole.ASSISTANT,
            content=content_items,
            msg_id=str(uuid.uuid4()),
            usage=LLMUsageMetrics(
                in_t=response.usage.prompt_tokens,
                op_t=response.usage.completion_tokens,
                latency_ms=0,
            ),
        )

    async def invoke(
        self,
        context: ConversationContext,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[LLMResponse, ConversationContext]:
        """
        Unified LLM invocation using OpenAI API.

        Handles both simple prompts and agentic tool-use loops.
        Injects context.tool_context into all tool calls.
        """
        final_response = LLMResponse()
        elicitations = []

        # Build tool schemas if tools provided
        openai_tools = None
        tool_map = {}
        if tools:
            openai_tools = [tool.to_openai_schema() for tool in tools]
            tool_map = {tool.name: tool for tool in tools}

        # Build messages from context once
        messages = context.to_openai_messages()

        # Add system message if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        # Build the base request kwargs once
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if openai_tools:
            request_kwargs["tools"] = openai_tools

        # No tools = single iteration
        iterations = self.max_iterations if openai_tools else 1

        for _ in range(iterations):

            # Call OpenAI
            response = self.client.chat.completions.create(**request_kwargs)

            # Parse response into Message
            assistant_msg = self._parse_response(response)
            context.add_message(assistant_msg)
            messages.append(assistant_msg.to_openai_message())

            tool_results = []
            for content_item in assistant_msg.content:
                if isinstance(content_item, TextMessage):
                    final_response.verbose.append(
                        VerboseResponseItem(type="text", text=content_item.text)
                    )
                elif isinstance(content_item, ToolCall):
                    tool_result = await self.invoke_tool(
                        tool_map, content_item, context.tool_context
                    )
                    if isinstance(tool_result, ElicitationResponse):
                        elicitations.append(tool_result)
                    else:
                        final_response.verbose.append(
                            VerboseResponseItem(
                                type="tool", tool_name=content_item.tool_name
                            )
                        )
                        # OpenAI requires each tool result as a separate message
                        result_msg = Message(
                            role=MessageRole.TOOL,
                            content=[tool_result],
                            msg_id=str(uuid.uuid4()),
                        )
                        context.add_message(result_msg)
                        messages.append(result_msg.to_openai_message())
                        tool_results.append(tool_result)
                        final_response.pending_tool_result_processing = True

            if not tool_results:
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
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Streaming LLM invocation using OpenAI streaming API.

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
        # Build tool schemas if tools provided
        openai_tools = None
        tool_map = {}
        if tools:
            openai_tools = [tool.to_openai_schema() for tool in tools]
            tool_map = {tool.name: tool for tool in tools}

        messages = context.to_openai_messages()

        # Add system message if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        # Build the base request kwargs
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if openai_tools:
            request_kwargs["tools"] = openai_tools

        iterations = self.max_iterations if openai_tools else 1

        final_response = LLMResponse()
        elicitations = []

        for _ in range(iterations):
            try:
                # Call streaming API
                stream = self.client.chat.completions.create(**request_kwargs)

                accumulated_text = ""
                tool_calls_map = {}  # index -> {id, name, arguments}
                usage_metrics = None

                for chunk in stream:
                    if not chunk.choices:
                        # Usage info comes in final chunk with empty choices
                        if chunk.usage:
                            usage_metrics = LLMUsageMetrics(
                                in_t=chunk.usage.prompt_tokens,
                                op_t=chunk.usage.completion_tokens,
                                latency_ms=0,
                            )
                        continue

                    delta = chunk.choices[0].delta

                    # Handle text content
                    if delta.content:
                        accumulated_text += delta.content
                        yield StreamEvent(type=StreamEventType.TEXT, text=delta.content)

                    # Handle tool calls
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index

                            if idx not in tool_calls_map:
                                # New tool call starting
                                tool_calls_map[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": (
                                        tc_delta.function.name
                                        if tc_delta.function
                                        else ""
                                    ),
                                    "arguments": "",
                                }
                                if tc_delta.id and tc_delta.function:
                                    yield StreamEvent(
                                        type=StreamEventType.TOOL_CALL_START,
                                        details={
                                            "tool_id": tc_delta.id,
                                            "tool_name": tc_delta.function.name,
                                        },
                                    )
                            else:
                                # Update existing tool call
                                if tc_delta.id:
                                    tool_calls_map[idx]["id"] = tc_delta.id
                                if tc_delta.function and tc_delta.function.name:
                                    tool_calls_map[idx]["name"] = tc_delta.function.name

                            # Accumulate arguments
                            if tc_delta.function and tc_delta.function.arguments:
                                tool_calls_map[idx][
                                    "arguments"
                                ] += tc_delta.function.arguments

                # -- end of stream --

                # Build content items
                content_items = []
                if accumulated_text:
                    content_items.append(TextMessage(text=accumulated_text))
                    final_response.verbose.append(
                        VerboseResponseItem(type="text", text=accumulated_text)
                    )

                # Process tool calls and execute them
                tool_results = []
                for idx in sorted(tool_calls_map.keys()):
                    tc_data = tool_calls_map[idx]
                    parsed_args = (
                        orjson.loads(tc_data["arguments"])
                        if tc_data["arguments"]
                        else {}
                    )
                    tool_call = ToolCall(
                        tool_id=tc_data["id"],
                        tool_name=tc_data["name"],
                        tool_input=parsed_args,
                    )
                    content_items.append(tool_call)

                    tool_res = await self.invoke_tool(
                        tool_map, tool_call, context.tool_context
                    )

                    yield StreamEvent(
                        type=StreamEventType.TOOL_CALL_END,
                        details={
                            "tool_name": tc_data["name"],
                            "tool_id": tc_data["id"],
                        },
                    )

                    if isinstance(tool_res, ElicitationResponse):
                        elicitations.append(tool_res)
                    else:
                        final_response.verbose.append(
                            VerboseResponseItem(type="tool", tool_name=tc_data["name"])
                        )
                        tool_results.append(tool_res)

                # Build assistant message and add to context
                if content_items:
                    llm_message = Message(
                        role=MessageRole.ASSISTANT,
                        content=content_items,
                        msg_id=str(uuid.uuid4()),
                        usage=usage_metrics,
                    )
                    context.add_message(llm_message)
                    messages.append(llm_message.to_openai_message())

                # OpenAI requires each tool result as a separate message
                if tool_results:
                    for tool_res in tool_results:
                        result_msg = Message(
                            role=MessageRole.TOOL,
                            content=[tool_res],
                            msg_id=str(uuid.uuid4()),
                        )
                        context.add_message(result_msg)
                        messages.append(result_msg.to_openai_message())
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
