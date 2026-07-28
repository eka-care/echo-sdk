"""
Anthropic LLM implementation.
"""

import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

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

from .base import BaseLLM
from .config import LLMConfig
from .model_capabilities import claude_capabilities
from .schemas import LLMResponse, StreamEvent, StreamEventType, VerboseResponseItem

logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLM):
    """Anthropic LLM provider (direct API, not Bedrock)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None
        self.thinking_budget_tokens = (
            config.thinking.budget_tokens if config.thinking else None
        )
        self.thinking_effort = config.thinking.effort if config.thinking else None
        # What this model's request surface accepts — sampling params, which
        # thinking form, whether thinking is on when unset. See
        # model_capabilities for the per-generation rules.
        self.capabilities = claude_capabilities(config.model)

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            import anthropic

            # Use config api_key if provided, otherwise falls back to ANTHROPIC_API_KEY env var
            self._client = anthropic.Anthropic(api_key=self.config.api_key)
        return self._client

    @staticmethod
    def _cached_system(system_prompt: str) -> List[dict]:
        """
        Wrap the system prompt as a single cached text block.

        The breakpoint goes on the system prompt because it's the stable prefix
        (fixed per prompt/version/variables); the volatile user message stays
        uncached. If the prompt is below the model's minimum cacheable size the
        marker silently no-ops — harmless, so it's placed unconditionally.
        """
        return [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _thinking_request_kwargs(self) -> dict:
        """
        Build the thinking-related request fields for this model.

        The thinking parameter is not portable across Claude generations, so the
        configured intent is mapped onto whatever the target model accepts:

        - effort + a model with the effort knob -> adaptive thinking + effort
        - budget_tokens + a model that still takes it (Claude 4.x) -> unchanged
        - either, on an adaptive-only model (Sonnet 5, Opus 4.7+) -> adaptive
        - nothing configured -> on models that think by default and support the
          effort knob (5-series), send adaptive thinking at low effort

        `output_config` goes through `extra_body`: it is a wire-level field the
        pinned anthropic SDK does not expose as a named parameter yet, and
        `extra_body` keeps working once it does.
        """
        caps = self.capabilities
        budget, effort = self.thinking_budget_tokens, self.thinking_effort

        if not budget and not effort:
            if (
                caps.thinking_on_by_default
                and caps.supports_effort
                and caps.can_disable_thinking
            ):
                return {
                    "thinking": {"type": "adaptive"},
                    "extra_body": {"output_config": {"effort": "low"}},
                }
            return {}

        if effort and caps.supports_effort and caps.supports_adaptive_thinking:
            return {
                "thinking": {"type": "adaptive"},
                "extra_body": {"output_config": {"effort": effort.value}},
            }

        if budget and caps.accepts_budget_tokens:
            return {"thinking": {"type": "enabled", "budget_tokens": budget}}

        if caps.supports_adaptive_thinking:
            logger.info(
                "Model %s does not accept thinking.budget_tokens; using adaptive "
                "thinking instead. Set thinking.effort to control depth.",
                self.model,
            )
            return {"thinking": {"type": "adaptive"}}

        logger.warning(
            "Model %s does not support thinking; ignoring thinking config.",
            self.model,
        )
        return {}

    def _build_request_kwargs(
        self, messages: List[dict], overrides: Dict[str, Any]
    ) -> dict:
        """Assemble the per-model base request shared by invoke and streaming."""
        request_kwargs = {
            "model": self.model,
            "max_tokens": overrides.get("max_tokens", self.max_tokens),
            "messages": messages,
        }

        # Sonnet 5 / Opus 4.7+ removed the sampling parameters and 400 on them,
        # so temperature is dropped rather than passed through.
        if self.capabilities.accepts_sampling_params:
            request_kwargs["temperature"] = overrides.get(
                "temperature", self.temperature
            )

        request_kwargs.update(self._thinking_request_kwargs())
        return request_kwargs

    @staticmethod
    def _serialize_block(block) -> Optional[dict]:
        """
        Serialize one response content block back into request form.

        The assistant turn is echoed back from these blocks rather than rebuilt
        from the parsed `Message`: rebuilding drops `thinking` blocks, and
        Anthropic rejects a tool-use turn whose thinking blocks were stripped
        while thinking is on. Thinking stays out of `ConversationContext` — it
        is wire-only state for the current tool loop, not conversation history.
        """
        if block.type == "text":
            return {"type": "text", "text": block.text}
        if block.type == "tool_use":
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            }
        if block.type == "thinking":
            # `thinking` is an empty string when display is omitted (the default
            # on the 5-series); the block still has to go back unmodified.
            return {
                "type": "thinking",
                "thinking": block.thinking,
                "signature": block.signature,
            }
        if block.type == "redacted_thinking":
            return {"type": "redacted_thinking", "data": block.data}
        return None

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
                cache_write_t=getattr(
                    response.usage, "cache_creation_input_tokens", 0
                )
                or 0,
                cache_read_t=getattr(response.usage, "cache_read_input_tokens", 0)
                or 0,
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

        # Build the base request kwargs (model-specific: sampling params and the
        # thinking form differ by Claude generation)
        request_kwargs = self._build_request_kwargs(messages, kwargs)

        if system_prompt:
            request_kwargs["system"] = self._cached_system(system_prompt)
        if tool_schemas:
            request_kwargs["tools"] = tool_schemas

        # call the LLM for the given number of iterations
        # No tools = single iteration
        iterations = self.max_iterations if tool_schemas else 1

        for _ in range(iterations):

            try:
                # Call Anthropic
                response = self.client.messages.create(**request_kwargs)
            except Exception as e:
                logger.error("AnthropicLLM invoke error: %s", e, exc_info=True)
                raise

            # Parse response into Message
            assistant_msg = self._parse_response(response, msg_id)
            final_response.usage = assistant_msg.usage
            context.add_message(assistant_msg)
            # Echo the raw blocks (thinking included) rather than re-serializing
            # the parsed message — see _serialize_block.
            wire_blocks = [
                b
                for b in (self._serialize_block(block) for block in response.content)
                if b is not None
            ]
            messages.append({"role": "assistant", "content": wire_blocks})

            tool_results = []
            interrupt = False  # a tool changed loaded state → recompute & rerun
            for content_item in assistant_msg.content:
                if isinstance(content_item, TextMessage):
                    final_response.verbose.append(
                        VerboseResponseItem(type="text", text=content_item.text)
                    )
                elif isinstance(content_item, ToolCall):
                    tool_res = await self.invoke_tool(
                        tool_map, content_item, context.tool_context
                    )
                    # Dispatch on the result's declared directive — never on type.
                    if tool_res.control_flow == ControlFlow.PAUSE:
                        elicitations.append(tool_res)
                    else:
                        final_response.verbose.append(
                            VerboseResponseItem(
                                type="tool", tool_name=content_item.tool_name
                            )
                        )
                        tool_results.append(tool_res)
                        if tool_res.control_flow == ControlFlow.INTERRUPT:
                            interrupt = True

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

            # Elicitation wins: end loop and return to the user.
            if elicitations:
                break

            request_kwargs["messages"] = messages

            # A tool changed loaded state: stop so the agent can recompute the
            # prompt + tool list and re-invoke (tool results are already in
            # context, so re-entry is valid).
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

        # Build the base request kwargs (model-specific: sampling params and the
        # thinking form differ by Claude generation)
        request_kwargs = self._build_request_kwargs(messages, kwargs)

        if system_prompt:
            request_kwargs["system"] = self._cached_system(system_prompt)
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
                    # Block index to request-form block, rebuilt as the stream
                    # completes. Kept separate from content_items because
                    # thinking must go back on the wire but never enters the
                    # persisted context.
                    wire_blocks = {}
                    content_items = []
                    tool_results = []
                    interrupt = False  # a tool changed loaded state → recompute
                    usage_metrics = None

                    for event in stream:
                        if event.type == "message_start":
                            # Input + cache read/write tokens are reported here
                            # (message_delta only carries the output count).
                            u = getattr(event.message, "usage", None)
                            if u:
                                usage_metrics = LLMUsageMetrics(
                                    in_t=getattr(u, "input_tokens", 0) or 0,
                                    op_t=getattr(u, "output_tokens", 0) or 0,
                                    latency_ms=0,
                                    cache_write_t=getattr(
                                        u, "cache_creation_input_tokens", 0
                                    )
                                    or 0,
                                    cache_read_t=getattr(
                                        u, "cache_read_input_tokens", 0
                                    )
                                    or 0,
                                )
                        elif event.type == "content_block_start":
                            block_id = event.index
                            block = event.content_block
                            if block.type == "tool_use":
                                tool = tool_map.get(block.name)
                                # Emit generic TOOL_CALL_* events only for VISIBLE
                                # tools (decided up-front from the tool; elicitation
                                # and system tools are SILENT).
                                visible = (
                                    tool.observability == Observability.VISIBLE
                                    if tool
                                    else True
                                )
                                blocks[block_id] = {
                                    "type": "tool",
                                    "tool_id": block.id,
                                    "tool_name": block.name,
                                    "input_json": "",
                                    "visible": visible,
                                }
                                if visible:
                                    yield StreamEvent(
                                        type=StreamEventType.TOOL_CALL_START,
                                        details={
                                            "tool_id": block.id,
                                            "tool_name": block.name,
                                        },
                                    )
                            elif block.type in ("thinking", "redacted_thinking"):
                                # Accumulated for the wire echo only — no
                                # StreamEvent, since thinking is not shown to
                                # callers and carries no text when display is
                                # omitted (the 5-series default).
                                blocks[block_id] = {
                                    "type": block.type,
                                    "thinking": getattr(block, "thinking", "") or "",
                                    "signature": getattr(block, "signature", "") or "",
                                    "data": getattr(block, "data", "") or "",
                                }
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
                            elif delta.type == "thinking_delta":
                                blocks[block_id]["thinking"] += delta.thinking
                            elif delta.type == "signature_delta":
                                blocks[block_id]["signature"] += delta.signature
                            elif delta.type == "input_json_delta":
                                blocks[block_id]["input_json"] += delta.partial_json
                                # forward the partial json fragment as a streaming TOOL_CALL_ARGS event so any partial data consumers like ag-ui etc
                                # can render args as they arrive. skip for SILENT tools, mirroring the TOOL_CALL_START / TOOL_CALL_END skip below.
                                if blocks[block_id].get("visible"):
                                    yield StreamEvent(
                                        type=StreamEventType.TOOL_CALL_ARGS,
                                        details={
                                            "tool_id": blocks[block_id]["tool_id"],
                                            "tool_name": blocks[block_id]["tool_name"],
                                            "delta": delta.partial_json,
                                        },
                                    )

                        elif event.type == "content_block_stop":
                            block_id = event.index
                            block_type = blocks[block_id]["type"]
                            if block_type == "thinking":
                                wire_blocks[block_id] = {
                                    "type": "thinking",
                                    "thinking": blocks[block_id]["thinking"],
                                    "signature": blocks[block_id]["signature"],
                                }
                            elif block_type == "redacted_thinking":
                                wire_blocks[block_id] = {
                                    "type": "redacted_thinking",
                                    "data": blocks[block_id]["data"],
                                }
                            elif block_type == "tool":
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
                                wire_blocks[block_id] = {
                                    "type": "tool_use",
                                    "id": tool_call.tool_id,
                                    "name": tool_call.tool_name,
                                    "input": parsed_input,
                                }
                                tool_res = await self.invoke_tool(
                                    tool_map, tool_call, context.tool_context
                                )
                                # progress message event (skip for SILENT tools)
                                if blocks[block_id].get("visible"):
                                    yield StreamEvent(
                                        type=StreamEventType.TOOL_CALL_END,
                                        details={
                                            "tool_name": blocks[block_id]["tool_name"],
                                            "tool_id": blocks[block_id]["tool_id"],
                                        },
                                    )

                                # Dispatch on the result's declared directive.
                                if tool_res.control_flow == ControlFlow.PAUSE:
                                    elicitations.append(tool_res)
                                else:
                                    tool_results.append(tool_res)
                                    if tool_res.control_flow == ControlFlow.INTERRUPT:
                                        interrupt = True
                            else:
                                text = blocks[block_id]["text"]
                                content_items.append((block_id, TextMessage(text=text)))
                                if text:
                                    # An empty text block is rejected on replay,
                                    # so it never goes back on the wire.
                                    wire_blocks[block_id] = {
                                        "type": "text",
                                        "text": text,
                                    }

                        elif event.type == "message_delta":
                            # Running output-token count; input + cache tokens
                            # were already seeded from message_start above.
                            if hasattr(event, "usage") and event.usage:
                                if usage_metrics is None:
                                    usage_metrics = LLMUsageMetrics(latency_ms=0)
                                usage_metrics.op_t = (
                                    getattr(event.usage, "output_tokens", 0)
                                    or usage_metrics.op_t
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
                final_response.usage = usage_metrics
                context.add_message(llm_message)
                # Echo the streamed blocks (thinking included) rather than
                # re-serializing the parsed message — see _serialize_block.
                messages.append(
                    {
                        "role": "assistant",
                        "content": [wire_blocks[i] for i in sorted(wire_blocks)],
                    }
                )

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

                # Elicitation wins: end loop and return to the user.
                if elicitations:
                    break

                request_kwargs["messages"] = messages

                # A tool changed loaded state: stop so the agent can recompute
                # and re-invoke (results already in context → valid re-entry).
                if interrupt:
                    final_response.pending_context_reload = True
                    break

                if not tool_results:
                    break

            except Exception as e:
                logger.error("AnthropicLLM streaming error: %s", e, exc_info=True)
                yield StreamEvent(type=StreamEventType.ERROR, error=str(e))
                return

        final_response.elicitations = elicitations or None
        yield StreamEvent(
            type=StreamEventType.DONE, llm_response=final_response, context=context
        )
