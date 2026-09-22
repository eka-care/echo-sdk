"""
OpenAI LLM implementation.
"""

import logging
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
from echo.tools.core import BaseTool
from echo.tools.core.schemas import ControlFlow, Observability

from .base import BaseLLM, prompt_cache_id
from .config import LLMConfig
from .schemas import LLMResponse, StreamEvent, StreamEventType, VerboseResponseItem

logger = logging.getLogger(__name__)

# Reasoning effort levels per the OpenAI model pages.
_GPT_5_6_EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh", "max"})
_GPT_6_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        self.reasoning_effort = config.thinking.reasoning_effort if config.thinking else None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            # Use config api_key if provided, otherwise falls back to OPENAI_API_KEY env var
            if self.config.api_key:
                self._client = OpenAI(api_key=self.config.api_key)
            else:
                self._client = OpenAI()
        return self._client

    @staticmethod
    def _system_content(system_prompt: str, system_suffix: Optional[str] = None) -> str:
        """Join the cacheable prefix and the volatile context into one system
        message, stable half first.

        OpenAI caches automatically on the longest common prefix — there is no
        breakpoint to place — so ordering is the whole mechanism: anything that
        varies per user, per session, or per turn has to come last, or the
        agent's shared prefix stops matching.
        """
        if not system_suffix:
            return system_prompt
        return f"{system_prompt}\n\n{system_suffix}"

    def _uses_max_completion_tokens(self) -> bool:
        """Check if model uses max_completion_tokens instead of max_tokens.

        Newer OpenAI models (GPT-5.x, GPT-4.1, o-series) require max_completion_tokens
        while legacy models (gpt-4o, gpt-4o-mini) still use max_tokens.
        """
        legacy_models = ("gpt-4o",)  # gpt-4o and gpt-4o-mini use max_tokens
        return not self.model.startswith(legacy_models)

    def _is_reasoning_model(self) -> bool:
        """Check if model is a reasoning model that doesn't support temperature."""
        return self.model.startswith(("o1", "o3", "o4-mini", "gpt-5.6", "gpt-6"))

    def _supports_reasoning_effort(self) -> bool:
        """Check if model supports reasoning_effort parameter."""
        return self.model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4-mini"))

    def _supported_efforts(self) -> Optional[frozenset]:
        """Effort levels the model documents; None means unknown (pass through)."""
        if self.model.startswith("gpt-6"):
            return _GPT_6_EFFORTS
        if self.model.startswith("gpt-5.6"):
            return _GPT_5_6_EFFORTS
        return None

    def _resolve_reasoning_effort(self) -> Optional[str]:
        """The effort to send, or None to let the model use its default."""
        if not self.reasoning_effort or not self._supports_reasoning_effort():
            return None
        effort = self.reasoning_effort.value
        supported = self._supported_efforts()
        if supported is None or effort in supported:
            return effort
        # Only none/minimal fall through here (GPT-6 Astra has neither).
        # OpenAI's migration guidance is to start from "low".
        logger.warning(
            "reasoning_effort=%r is not supported by %s; using 'low'", effort, self.model
        )
        return "low"

    def _build_request_kwargs(
        self,
        messages: List[dict],
        openai_tools: Optional[List[dict]],
        system_prompt: Optional[str],
        kwargs: dict,
    ) -> dict:
        """Chat Completions request shared by invoke() and invoke_stream()."""
        request_kwargs = {"model": self.model, "messages": messages}

        # OpenAI caches automatically on the longest common prefix (GPT-5.6+:
        # 30m TTL, the default and only value). The key routes requests sharing
        # a prefix to the same cache — keyed on the cacheable half only, so
        # every session of an agent lands together and different agents stay
        # apart. On GPT-5.6+ it only separates cache accounting.
        cache_key = prompt_cache_id(system_prompt)
        if cache_key:
            request_kwargs["prompt_cache_key"] = cache_key

        max_tokens_value = kwargs.get("max_tokens", self.max_tokens)
        if self._uses_max_completion_tokens():
            request_kwargs["max_completion_tokens"] = max_tokens_value
        else:
            request_kwargs["max_tokens"] = max_tokens_value

        if not self._is_reasoning_model():
            request_kwargs["temperature"] = kwargs.get("temperature", self.temperature)

        effort = self._resolve_reasoning_effort()
        # GPT-5.6 on Chat Completions 400s on function tools with any effort
        # but "none" — including the implicit default (medium) — so it must be
        # sent explicitly. Reasoning with tools needs the Responses API.
        if openai_tools and self.model.startswith("gpt-5.6"):
            if effort not in (None, "none"):
                logger.warning(
                    "%s does not support reasoning_effort=%r with tools on Chat "
                    "Completions; using 'none'",
                    self.model,
                    effort,
                )
            effort = "none"
        if effort:
            request_kwargs["reasoning_effort"] = effort

        if openai_tools:
            request_kwargs["tools"] = openai_tools
        return request_kwargs

    def _uses_responses_api(self, openai_tools: Optional[List[dict]]) -> bool:
        """GPT-6 only accepts function tools on the Responses API."""
        return bool(openai_tools) and self.model.startswith("gpt-6")

    @staticmethod
    def _to_responses_input(messages: List[dict]) -> List[dict]:
        """Translate Chat Completions messages into Responses API input items."""
        items: List[dict] = []
        for msg in messages:
            if msg["role"] == "tool":
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg["tool_call_id"],
                        "output": msg["content"],
                    }
                )
                continue

            content = msg.get("content")
            if isinstance(content, list):
                content = [
                    {"type": "input_image", "image_url": part["image_url"]["url"]}
                    if part["type"] == "image_url"
                    else {"type": "input_text", "text": part["text"]}
                    for part in content
                ]
            if content:
                items.append({"role": msg["role"], "content": content})

            for tc in msg.get("tool_calls") or []:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                )
        return items

    def _build_responses_kwargs(
        self,
        input_items: List[dict],
        openai_tools: List[dict],
        system_prompt: Optional[str],
        system_suffix: Optional[str],
        kwargs: dict,
    ) -> dict:
        """Responses API request shared by the invoke() and invoke_stream() paths."""
        request_kwargs = {
            "model": self.model,
            "input": input_items,
            # Responses defaults function tools to strict mode, which rejects
            # schemas that aren't fully closed; Chat Completions never did.
            "tools": [
                {"type": "function", **tool["function"], "strict": False}
                for tool in openai_tools
            ],
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            # Stay stateless like the Chat Completions path: nothing is stored
            # server-side, so reasoning comes back encrypted and is replayed
            # within the tool loop (the model rejects tool turns without it).
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if system_prompt:
            request_kwargs["instructions"] = self._system_content(
                system_prompt, system_suffix
            )

        cache_key = prompt_cache_id(system_prompt)
        if cache_key:
            request_kwargs["prompt_cache_key"] = cache_key

        if not self._is_reasoning_model():
            request_kwargs["temperature"] = kwargs.get("temperature", self.temperature)

        effort = self._resolve_reasoning_effort()
        if effort:
            request_kwargs["reasoning"] = {"effort": effort}
        return request_kwargs

    @staticmethod
    def _responses_usage_metrics(usage) -> Optional[LLMUsageMetrics]:
        """Map Responses API usage to LLMUsageMetrics."""
        if usage is None:
            return None
        details = getattr(usage, "input_tokens_details", None)
        return LLMUsageMetrics(
            in_t=usage.input_tokens,
            op_t=usage.output_tokens,
            latency_ms=0,
            cache_read_t=getattr(details, "cached_tokens", 0) or 0,
        )

    def _parse_responses_output(self, response, msg_id: str) -> Message:
        """Parse a Responses API response into a Message.

        Reasoning items are wire-only: they are replayed within the tool loop
        but stay out of ConversationContext, like Anthropic thinking blocks.
        """
        text = ""
        tool_calls = []
        for item in response.output:
            if item.type == "message":
                text += "".join(
                    part.text for part in item.content if part.type == "output_text"
                )
            elif item.type == "function_call":
                tool_calls.append(
                    ToolCall(
                        tool_id=item.call_id,
                        tool_name=item.name,
                        tool_input=orjson.loads(item.arguments) if item.arguments else {},
                    )
                )

        content_items = [TextMessage(text=text)] if text else []
        content_items.extend(tool_calls)
        return Message(
            role=MessageRole.ASSISTANT,
            content=content_items,
            msg_id=msg_id,
            usage=self._responses_usage_metrics(response.usage),
        )

    @staticmethod
    def _usage_metrics(usage) -> Optional[LLMUsageMetrics]:
        """Map OpenAI usage (incl. prompt-cache reads/writes) to LLMUsageMetrics."""
        if usage is None:
            return None
        details = getattr(usage, "prompt_tokens_details", None)
        return LLMUsageMetrics(
            in_t=usage.prompt_tokens,
            op_t=usage.completion_tokens,
            latency_ms=0,
            cache_read_t=getattr(details, "cached_tokens", 0) or 0,
            cache_write_t=getattr(details, "cache_write_tokens", 0) or 0,
        )

    def _parse_response(self, response, msg_id: str) -> Message:
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
            msg_id=msg_id,
            usage=self._usage_metrics(response.usage),
        )

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
        Unified LLM invocation using OpenAI API.

        Handles both simple prompts and agentic tool-use loops.
        Injects context.tool_context into all tool calls.
        """
        final_response = LLMResponse()
        elicitations = []
        msg_id = out_msg_id or str(uuid.uuid4())

        # Build tool schemas if tools provided
        openai_tools = None
        tool_map = {}
        if tools:
            openai_tools = [tool.to_openai_schema() for tool in tools]
            tool_map = {tool.name: tool for tool in tools}

        if self._uses_responses_api(openai_tools):
            return await self._invoke_responses(
                context, openai_tools, tool_map, system_prompt, system_suffix, msg_id, kwargs
            )

        # Build messages from context once
        messages = context.to_openai_messages()

        # Add system message if provided
        if system_prompt:
            messages = [
                {
                    "role": "system",
                    "content": self._system_content(system_prompt, system_suffix),
                }
            ] + messages

        request_kwargs = self._build_request_kwargs(
            messages, openai_tools, system_prompt, kwargs
        )

        # No tools = single iteration
        iterations = self.max_iterations if openai_tools else 1

        for _ in range(iterations):

            try:
                # Call OpenAI
                response = self.client.chat.completions.create(**request_kwargs)
            except Exception as e:
                logger.error("OpenAI invoke error: %s", e, exc_info=True)
                raise

            # Parse response into Message
            assistant_msg = self._parse_response(response, msg_id)
            context.add_message(assistant_msg)
            messages.extend(assistant_msg.to_openai_messages())

            tool_results = []
            interrupt = False  # a tool changed loaded state → recompute & rerun
            for content_item in assistant_msg.content:
                if isinstance(content_item, TextMessage):
                    final_response.verbose.append(
                        VerboseResponseItem(type="text", text=content_item.text)
                    )
                elif isinstance(content_item, ToolCall):
                    tool_result = await self.invoke_tool(
                        tool_map, content_item, context.tool_context
                    )
                    # Dispatch on the result's declared directive — never on type.
                    if tool_result.control_flow == ControlFlow.PAUSE:
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
                            msg_id=msg_id,
                        )
                        context.add_message(result_msg)
                        messages.extend(result_msg.to_openai_messages())
                        tool_results.append(tool_result)
                        final_response.pending_tool_result_processing = True
                        if tool_result.control_flow == ControlFlow.INTERRUPT:
                            interrupt = True

            if not tool_results:
                final_response.pending_tool_result_processing = False

            # Elicitation wins: end loop and return to the user.
            if elicitations:
                break

            request_kwargs["messages"] = messages

            # A tool changed loaded state: stop so the agent can recompute the
            # prompt + tool list and re-invoke (results already in context).
            if interrupt:
                final_response.pending_context_reload = True
                break

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
        system_suffix: Optional[str] = None,
        out_msg_id: Optional[str] = None,
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
            out_msg_id: Optional message ID for grouping messages
            **kwargs: Additional arguments (max_tokens, temperature)

        Yields:
            StreamEvent objects with type indicating what happened
        """
        msg_id = out_msg_id or str(uuid.uuid4())

        # Build tool schemas if tools provided
        openai_tools = None
        tool_map = {}
        if tools:
            openai_tools = [tool.to_openai_schema() for tool in tools]
            tool_map = {tool.name: tool for tool in tools}

        if self._uses_responses_api(openai_tools):
            async for event in self._invoke_stream_responses(
                context, openai_tools, tool_map, system_prompt, system_suffix, msg_id, kwargs
            ):
                yield event
            return

        messages = context.to_openai_messages()

        # Add system message if provided
        if system_prompt:
            messages = [
                {
                    "role": "system",
                    "content": self._system_content(system_prompt, system_suffix),
                }
            ] + messages

        request_kwargs = self._build_request_kwargs(
            messages, openai_tools, system_prompt, kwargs
        )
        request_kwargs["stream"] = True
        request_kwargs["stream_options"] = {"include_usage": True}

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
                            usage_metrics = self._usage_metrics(chunk.usage)
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
                                tool_name = (
                                    tc_delta.function.name if tc_delta.function else ""
                                )
                                tool = tool_map.get(tool_name)
                                # Emit generic TOOL_CALL_* events only for VISIBLE
                                # tools (SILENT = elicitation/system tools).
                                visible = (
                                    tool.observability == Observability.VISIBLE
                                    if tool
                                    else True
                                )
                                tool_calls_map[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": tool_name,
                                    "arguments": "",
                                    "visible": visible,
                                }
                                if tc_delta.id and tc_delta.function and visible:
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
                                # forward the partial json fragment as a streaming TOOL_CALL_ARGS event so any partial data consumers like ag-ui etc
                                # can render args as they arrive. skip for SILENT tools, mirroring the TOOL_CALL_START / TOOL_CALL_END skip below.
                                if tool_calls_map[idx].get("visible"):
                                    yield StreamEvent(
                                        type=StreamEventType.TOOL_CALL_ARGS,
                                        details={
                                            "tool_id": tool_calls_map[idx]["id"],
                                            "tool_name": tool_calls_map[idx]["name"],
                                            "delta": tc_delta.function.arguments,
                                        },
                                    )

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
                interrupt = False  # a tool changed loaded state → recompute
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

                    # progress message event (skip for SILENT tools)
                    if tc_data.get("visible"):
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_END,
                            details={
                                "tool_name": tc_data["name"],
                                "tool_id": tc_data["id"],
                            },
                        )

                    # Dispatch on the result's declared directive.
                    if tool_res.control_flow == ControlFlow.PAUSE:
                        elicitations.append(tool_res)
                    else:
                        final_response.verbose.append(
                            VerboseResponseItem(type="tool", tool_name=tc_data["name"])
                        )
                        tool_results.append(tool_res)
                        if tool_res.control_flow == ControlFlow.INTERRUPT:
                            interrupt = True

                # Build assistant message and add to context
                if content_items:
                    llm_message = Message(
                        role=MessageRole.ASSISTANT,
                        content=content_items,
                        msg_id=msg_id,
                        usage=usage_metrics,
                    )
                    context.add_message(llm_message)
                    messages.extend(llm_message.to_openai_messages())

                # OpenAI requires each tool result as a separate message
                if tool_results:
                    for tool_res in tool_results:
                        result_msg = Message(
                            role=MessageRole.TOOL,
                            content=[tool_res],
                            msg_id=msg_id,
                        )
                        context.add_message(result_msg)
                        messages.extend(result_msg.to_openai_messages())
                    final_response.pending_tool_result_processing = True
                else:
                    final_response.pending_tool_result_processing = False

                if elicitations:
                    break

                request_kwargs["messages"] = messages

                if interrupt:
                    final_response.pending_context_reload = True
                    break

                if not tool_results:
                    break

            except Exception as e:
                logger.error("OpenAILLM streaming error: %s", e, exc_info=True)
                yield StreamEvent(type=StreamEventType.ERROR, error=str(e))
                return

        final_response.elicitations = elicitations or None
        yield StreamEvent(
            type=StreamEventType.DONE, llm_response=final_response, context=context
        )

    async def _invoke_responses(
        self,
        context: ConversationContext,
        openai_tools: List[dict],
        tool_map: dict,
        system_prompt: Optional[str],
        system_suffix: Optional[str],
        msg_id: str,
        kwargs: dict,
    ) -> Tuple[LLMResponse, ConversationContext]:
        """invoke() over the Responses API (tool calling on GPT-6)."""
        final_response = LLMResponse()
        elicitations = []

        input_items = self._to_responses_input(context.to_openai_messages())
        request_kwargs = self._build_responses_kwargs(
            input_items, openai_tools, system_prompt, system_suffix, kwargs
        )

        for _ in range(self.max_iterations):
            try:
                response = self.client.responses.create(**request_kwargs)
            except Exception as e:
                logger.error("OpenAI Responses invoke error: %s", e, exc_info=True)
                raise

            assistant_msg = self._parse_responses_output(response, msg_id)
            context.add_message(assistant_msg)
            # Replay the raw output, reasoning items included.
            input_items.extend(item.model_dump(exclude_none=True) for item in response.output)

            tool_results = []
            interrupt = False
            for content_item in assistant_msg.content:
                if isinstance(content_item, TextMessage):
                    final_response.verbose.append(
                        VerboseResponseItem(type="text", text=content_item.text)
                    )
                elif isinstance(content_item, ToolCall):
                    tool_result = await self.invoke_tool(
                        tool_map, content_item, context.tool_context
                    )
                    if tool_result.control_flow == ControlFlow.PAUSE:
                        elicitations.append(tool_result)
                    else:
                        final_response.verbose.append(
                            VerboseResponseItem(
                                type="tool", tool_name=content_item.tool_name
                            )
                        )
                        result_msg = Message(
                            role=MessageRole.TOOL,
                            content=[tool_result],
                            msg_id=msg_id,
                        )
                        context.add_message(result_msg)
                        input_items.extend(
                            self._to_responses_input(result_msg.to_openai_messages())
                        )
                        tool_results.append(tool_result)
                        final_response.pending_tool_result_processing = True
                        if tool_result.control_flow == ControlFlow.INTERRUPT:
                            interrupt = True

            if not tool_results:
                final_response.pending_tool_result_processing = False
            if elicitations:
                break
            if interrupt:
                final_response.pending_context_reload = True
                break
            if not tool_results:
                break

        final_text = ""
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

    async def _invoke_stream_responses(
        self,
        context: ConversationContext,
        openai_tools: List[dict],
        tool_map: dict,
        system_prompt: Optional[str],
        system_suffix: Optional[str],
        msg_id: str,
        kwargs: dict,
    ) -> AsyncGenerator[StreamEvent, None]:
        """invoke_stream() over the Responses API (tool calling on GPT-6)."""
        input_items = self._to_responses_input(context.to_openai_messages())
        request_kwargs = self._build_responses_kwargs(
            input_items, openai_tools, system_prompt, system_suffix, kwargs
        )
        request_kwargs["stream"] = True

        final_response = LLMResponse()
        elicitations = []

        for _ in range(self.max_iterations):
            try:
                stream = self.client.responses.create(**request_kwargs)

                calls = {}  # output_index -> {id, name, visible}
                completed = None

                for event in stream:
                    if event.type == "response.output_text.delta":
                        yield StreamEvent(type=StreamEventType.TEXT, text=event.delta)

                    elif (
                        event.type == "response.output_item.added"
                        and event.item.type == "function_call"
                    ):
                        tool = tool_map.get(event.item.name)
                        # Emit generic TOOL_CALL_* events only for VISIBLE
                        # tools (SILENT = elicitation/system tools).
                        visible = (
                            tool.observability == Observability.VISIBLE if tool else True
                        )
                        calls[event.output_index] = {
                            "id": event.item.call_id,
                            "name": event.item.name,
                            "visible": visible,
                        }
                        if visible:
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_START,
                                details={
                                    "tool_id": event.item.call_id,
                                    "tool_name": event.item.name,
                                },
                            )

                    elif event.type == "response.function_call_arguments.delta":
                        call = calls.get(event.output_index)
                        if call and call["visible"]:
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_ARGS,
                                details={
                                    "tool_id": call["id"],
                                    "tool_name": call["name"],
                                    "delta": event.delta,
                                },
                            )

                    elif event.type in ("response.completed", "response.incomplete"):
                        completed = event.response
                        if event.type == "response.incomplete":
                            logger.warning(
                                "%s response incomplete: %s",
                                self.model,
                                completed.incomplete_details,
                            )

                    elif event.type == "response.failed":
                        error = event.response.error
                        raise RuntimeError(error.message if error else "response failed")

                    elif event.type == "error":
                        raise RuntimeError(event.message)

                # -- end of stream --
                if completed is None:
                    raise RuntimeError("Responses stream ended without a final response")

                # The completed response is authoritative for text and tool args.
                assistant_msg = self._parse_responses_output(completed, msg_id)
                input_items.extend(
                    item.model_dump(exclude_none=True) for item in completed.output
                )
                visible_by_id = {c["id"]: c["visible"] for c in calls.values()}

                tool_results = []
                interrupt = False
                for content_item in assistant_msg.content:
                    if isinstance(content_item, TextMessage):
                        final_response.verbose.append(
                            VerboseResponseItem(type="text", text=content_item.text)
                        )
                        continue

                    tool_res = await self.invoke_tool(
                        tool_map, content_item, context.tool_context
                    )
                    if visible_by_id.get(content_item.tool_id, True):
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL_END,
                            details={
                                "tool_name": content_item.tool_name,
                                "tool_id": content_item.tool_id,
                            },
                        )

                    if tool_res.control_flow == ControlFlow.PAUSE:
                        elicitations.append(tool_res)
                    else:
                        final_response.verbose.append(
                            VerboseResponseItem(type="tool", tool_name=content_item.tool_name)
                        )
                        tool_results.append(tool_res)
                        if tool_res.control_flow == ControlFlow.INTERRUPT:
                            interrupt = True

                if assistant_msg.content:
                    context.add_message(assistant_msg)

                if tool_results:
                    for tool_res in tool_results:
                        result_msg = Message(
                            role=MessageRole.TOOL,
                            content=[tool_res],
                            msg_id=msg_id,
                        )
                        context.add_message(result_msg)
                        input_items.extend(
                            self._to_responses_input(result_msg.to_openai_messages())
                        )
                    final_response.pending_tool_result_processing = True
                else:
                    final_response.pending_tool_result_processing = False

                if elicitations:
                    break
                if interrupt:
                    final_response.pending_context_reload = True
                    break
                if not tool_results:
                    break

            except Exception as e:
                logger.error("OpenAILLM Responses streaming error: %s", e, exc_info=True)
                yield StreamEvent(type=StreamEventType.ERROR, error=str(e))
                return

        final_response.elicitations = elicitations or None
        yield StreamEvent(
            type=StreamEventType.DONE, llm_response=final_response, context=context
        )
