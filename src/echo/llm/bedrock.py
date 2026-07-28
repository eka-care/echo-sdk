"""
AWS Bedrock LLM implementation using Converse API.
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

from .base import BaseLLM
from .config import LLMConfig
from .model_capabilities import claude_capabilities
from .schemas import LLMResponse, StreamEvent, StreamEventType, VerboseResponseItem

logger = logging.getLogger(__name__)


class BedrockLLM(BaseLLM):
    """AWS Bedrock LLM provider using Converse API."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.region = config.region or "ap-south-1"
        self._client = None
        # Bedrock Claude IDs (`anthropic.claude-sonnet-5`) carry the same
        # per-generation request rules as the first-party API. Non-Claude models
        # resolve to the permissive pre-5 surface, so this is a no-op for them.
        self.capabilities = claude_capabilities(config.model)

    def _inference_config(self, overrides: dict) -> dict:
        """Build `inferenceConfig` for this model."""
        config = {"maxTokens": overrides.get("max_tokens", self.max_tokens)}
        # Sonnet 5 / Opus 4.7+ removed the sampling parameters and 400 on them.
        if self.capabilities.accepts_sampling_params:
            config["temperature"] = overrides.get("temperature", self.temperature)
        return config

    def _system_blocks(
        self, system_prompt: str, system_suffix: Optional[str] = None
    ) -> List[dict]:
        """
        Build the Converse ``system`` blocks, with a cache point if the model
        supports one.

        Caching is a byte-exact prefix match, so the ``cachePoint`` goes right
        after the agent's stable persona + task and before ``system_suffix``,
        which holds whatever varies per user, per session, or per turn. That
        keeps one cache entry per agent prompt rather than one per session.

        Models that predate Bedrock prompt caching reject a ``cachePoint``
        block outright, so it is omitted for them and both halves are still
        sent — same prompt, no caching.
        """
        blocks = [{"text": system_prompt}]
        if self.capabilities.supports_prompt_caching:
            blocks.append({"cachePoint": {"type": "default"}})
        if system_suffix:
            blocks.append({"text": system_suffix})
        return blocks

    def _additional_request_fields(self) -> dict:
        """
        Model-specific fields Converse passes straight through to the model.

        The 5-series thinks by default when `thinking` is unset, where every
        older model did not. For those models, set adaptive thinking with low
        effort so the default is explicit and stable.
        """
        caps = self.capabilities
        if (
            caps.thinking_on_by_default
            and caps.supports_effort
            and caps.can_disable_thinking
        ):
            return {
                "additionalModelRequestFields": {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "low"},
                }
            }
        return {}

    @property
    def client(self):
        """Lazy initialization of Bedrock client."""
        if self._client is None:
            import boto3

            # Use config credentials if provided, otherwise falls back to boto3 default chain
            # (env vars AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, ~/.aws/credentials, IAM role, etc.)
            kwargs = {"region_name": self.region}
            if self.config.aws_access_key_id and self.config.aws_secret_access_key:
                kwargs["aws_access_key_id"] = self.config.aws_access_key_id
                kwargs["aws_secret_access_key"] = self.config.aws_secret_access_key
            self._client = boto3.client("bedrock-runtime", **kwargs)
        return self._client

    def _parse_response(self, response, msg_id: str) -> Message:
        """Parse Bedrock response into a Message."""
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])
        usage = response.get("usage", {})

        content_items = []
        for block in content:
            if "text" in block:
                content_items.append(TextMessage(text=block["text"]))
            elif "toolUse" in block:
                tu = block["toolUse"]
                content_items.append(
                    ToolCall(
                        tool_id=tu["toolUseId"],
                        tool_name=tu["name"],
                        tool_input=tu["input"],
                    )
                )

        return Message(
            role=MessageRole.ASSISTANT,
            content=content_items,
            msg_id=msg_id,
            usage=LLMUsageMetrics(
                in_t=usage.get("inputTokens", 0),
                op_t=usage.get("outputTokens", 0),
                latency_ms=response.get("metrics", {}).get("latencyMs", 0),
                cache_write_t=usage.get("cacheWriteInputTokens", 0) or 0,
                cache_read_t=usage.get("cacheReadInputTokens", 0) or 0,
            ),
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
        Unified LLM invocation using Bedrock Converse API.

        Handles both simple prompts and agentic tool-use loops.
        Injects context.tool_context into all tool calls.
        """
        final_response = LLMResponse()
        elicitations = []
        msg_id = out_msg_id or str(uuid.uuid4())

        # Build tool config if tools provided
        tool_config = None
        tool_map = {}
        if tools:
            tool_config = {"tools": [tool.to_bedrock_schema() for tool in tools]}
            tool_map = {tool.name: tool for tool in tools}

        # Build messages from context once
        messages = context.to_bedrock_messages()

        # Build the base request kwargs once
        request_kwargs = {
            "modelId": self.model,
            "messages": messages,
            "inferenceConfig": self._inference_config(kwargs),
            **self._additional_request_fields(),
        }

        if system_prompt:
            request_kwargs["system"] = self._system_blocks(system_prompt, system_suffix)

        if tool_config:
            request_kwargs["toolConfig"] = tool_config

        # No tools = single iteration
        iterations = self.max_iterations if tool_config else 1

        for _ in range(iterations):

            try:
                # Call Bedrock Converse API
                response = self.client.converse(**request_kwargs)
            except Exception as e:
                logger.error("BedrockLLM invoke error: %s", e, exc_info=True)
                raise

            # Parse response into Message and add to context and bedrock messages list
            assistant_msg = self._parse_response(response, msg_id)
            context.add_message(assistant_msg)
            messages.append(assistant_msg.to_bedrock_message())

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

            # Add all tool results as a TOOL message (adapter transforms to 'user' for API)
            if tool_results:
                results_msg = Message(
                    role=MessageRole.TOOL,
                    content=tool_results,
                    msg_id=msg_id,
                )
                context.add_message(results_msg)

                # Update messages with the correct LLM-based structure for tool results
                messages.append(results_msg.to_bedrock_message())
                final_response.pending_tool_result_processing = True
            else:
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

        # Max iterations reached/ no tool use / elicitations - extract last response
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
        # Build tool config if tools provided
        tool_config = None
        tool_map = {}
        if tools:
            tool_config = {"tools": [tool.to_bedrock_schema() for tool in tools]}
            tool_map = {tool.name: tool for tool in tools}

        messages = context.to_bedrock_messages()

        # Build the base request kwargs
        request_kwargs = {
            "modelId": self.model,
            "messages": messages,
            "inferenceConfig": self._inference_config(kwargs),
            **self._additional_request_fields(),
        }

        if system_prompt:
            request_kwargs["system"] = self._system_blocks(system_prompt, system_suffix)

        if tool_config:
            request_kwargs["toolConfig"] = tool_config

        iterations = self.max_iterations if tool_config else 1

        final_response = LLMResponse()
        elicitations = []

        # we'll be calling the stream API and returning important data
        for _ in range(iterations):
            try:
                # Call streaming API
                response = self.client.converse_stream(**request_kwargs)

                blocks = {}  # blockid to content block
                content_items = []
                tool_results = []
                interrupt = False  # a tool changed loaded state → recompute
                usage_metrics = None
                # stop_reason = None - not needed for time being

                # Process stream events
                for event in response["stream"]:
                    if "contentBlockStart" in event:
                        block_id = event["contentBlockStart"].get("contentBlockIndex")
                        start = event["contentBlockStart"].get("start") or {}
                        if start.get("toolUse"):
                            tool_name = start["toolUse"]["name"]
                            tool = tool_map.get(tool_name)
                            # Emit generic TOOL_CALL_* events only for VISIBLE
                            # tools (SILENT = elicitation/system tools).
                            visible = (
                                tool.observability == Observability.VISIBLE
                                if tool
                                else True
                            )
                            blocks[block_id] = {
                                "type": "tool",
                                "tool_id": start["toolUse"]["toolUseId"],
                                "tool_name": tool_name,
                                "input_json": "",
                                "visible": visible,
                            }
                            if visible:
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_START,
                                    details={
                                        "tool_id": blocks[block_id]["tool_id"],
                                        "tool_name": blocks[block_id]["tool_name"],
                                    },
                                )
                        else:
                            blocks[block_id] = {
                                "type": "text",
                                "text": "",
                            }

                    elif "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        block_id = event["contentBlockDelta"].get("contentBlockIndex")

                        # Create block on-the-fly if contentBlockStart was not sent
                        # (Bedrock skips contentBlockStart for simple text responses)
                        if block_id not in blocks:
                            blocks[block_id] = {
                                "type": "text",
                                "text": "",
                            }

                        # send back text chunks as they are generated and accumulate final text response
                        if delta.get("text"):
                            blocks[block_id]["text"] += delta["text"]
                            yield StreamEvent(
                                type=StreamEventType.TEXT, text=delta["text"]
                            )
                        elif "toolUse" in delta:
                            tool_input_fragment = delta["toolUse"].get("input", "")
                            blocks[block_id]["input_json"] += tool_input_fragment
                            # forward the partial json fragment as a streaming TOOL_CALL_ARGS event so any partial data consumers like ag-ui etc
                            # can render args as they arrive. skip for SILENT tools, mirroring the TOOL_CALL_START / TOOL_CALL_END skip below.
                            if blocks[block_id].get("visible"):
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_ARGS,
                                    details={
                                        "tool_id": blocks[block_id]["tool_id"],
                                        "tool_name": blocks[block_id]["tool_name"],
                                        "delta": tool_input_fragment,
                                    },
                                )

                    elif "contentBlockStop" in event:
                        block_id = event["contentBlockStop"].get("contentBlockIndex")
                        if blocks[block_id]["type"] == "tool":
                            # Tool block complete - parse input and execute
                            input_json_str = blocks[block_id]["input_json"]
                            blocks[block_id]["input_json"] = (
                                orjson.loads(input_json_str) if input_json_str else {}
                            )
                            tool_call = ToolCall(
                                tool_id=blocks[block_id]["tool_id"],
                                tool_name=blocks[block_id]["tool_name"],
                                tool_input=blocks[block_id]["input_json"],
                            )
                            # add this tool call block to final content items list
                            content_items.append((block_id, tool_call))
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
                            # add this text block to final content items list
                            content_items.append(
                                (block_id, TextMessage(text=blocks[block_id]["text"]))
                            )

                    # not needed for time being
                    # elif "messageStop" in event:
                    #     # this is the second last event in the stream
                    #     stop_reason = event["messageStop"].get("stopReason", "")

                    elif "metadata" in event:
                        # this is the last event in the stream

                        usage = event["metadata"].get("usage", {})
                        metrics = event["metadata"].get("metrics", {})
                        if usage:
                            usage_metrics = LLMUsageMetrics(
                                in_t=usage.get("inputTokens", 0),
                                op_t=usage.get("outputTokens", 0),
                                latency_ms=metrics.get("latencyMs", 0),
                                cache_write_t=usage.get("cacheWriteInputTokens", 0)
                                or 0,
                                cache_read_t=usage.get("cacheReadInputTokens", 0) or 0,
                            )
                        break

                # -- end of stream -------

                # build the content items list and verbose list from the content items
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

                # build the llmMessage from the content items list and add to context
                llm_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=content_items_list,
                    msg_id=out_msg_id,
                    usage=usage_metrics,
                )
                context.add_message(llm_message)
                messages.append(llm_message.to_bedrock_message())

                if tool_results:
                    llm_message = Message(
                        role=MessageRole.TOOL,
                        content=tool_results,
                        msg_id=out_msg_id,
                    )
                    context.add_message(llm_message)
                    messages.append(llm_message.to_bedrock_message())
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
                logger.error("BedrockLLM streaming error: %s", e, exc_info=True)
                yield StreamEvent(type=StreamEventType.ERROR, error=str(e))
                return

        # Not needed , this is important for full text response like invoke()
        # final_text = ""
        # last_message = (
        #     context.messages[-1]
        #     if context.messages[-1].role == MessageRole.ASSISTANT
        #     else context.messages[-2]
        # )
        # for item in last_message.content:
        #     if isinstance(item, TextMessage):
        #         final_text += item.text

        # final_response.text = final_text.strip()
        final_response.elicitations = elicitations or None
        # Max iterations reached
        yield StreamEvent(
            type=StreamEventType.DONE, llm_response=final_response, context=context
        )


# Sample response from Bedrock Converse API
"""
{'ResponseMetadata': {'RequestId': '4545f411-413b-461e-9055-fb4ef5208332', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Sat, 10 Jan 2026 11:11:25 GMT', 'content-type': 'application/json', 'content-length': '505', 'connection': 'keep-alive', 'x-amzn-requestid': '4545f411-413b-461e-9055-fb4ef5208332'}, 'RetryAttempts': 0}, 'output': {'message': {'role': 'assistant', 'content': [{'text': "Okay, let's get started with booking your appointment. First, I'll need to authenticate your phone number. Could you please provide your mobile number?"}, {'toolUse': {'toolUseId': 'tooluse_rzu8A0Y1T5yQP9RXT5Vwbw', 'name': 'phone_authentication_tool', 'input': {'requires_elicitation': True, 'state': 'otp'}}}]}}, 'stopReason': 'tool_use', 'usage': {'inputTokens': 9362, 'outputTokens': 109, 'totalTokens': 9471}, 'metrics': {'latencyMs': 1501}

"""
