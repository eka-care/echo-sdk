"""
Google Gemini LLM implementation.
"""

import os
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from echo.models.user_conversation import (
    ConversationContext,
    LLMUsageMetrics,
    Message,
    MessageRole,
    TextMessage,
    ToolCall,
    ToolResult,
)
from echo.tools.base_tool import BaseTool
from echo.tools.schemas import ElicitationResponse

from .base import BaseLLM
from .config import LLMConfig
from .schemas import LLMResponse, StreamEvent, StreamEventType, VerboseResponseItem


class GeminiLLM(BaseLLM):
    """Google Gemini LLM provider."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            from google import genai

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _to_gemini_contents(self, context: ConversationContext) -> List[Dict[str, Any]]:
        """Convert conversation context to Gemini message format."""
        contents = []
        for msg in context.messages:
            parts = []
            for item in msg.content:
                if isinstance(item, TextMessage):
                    parts.append({"text": item.text})
                elif isinstance(item, ToolCall):
                    parts.append({
                        "function_call": {
                            "name": item.tool_name,
                            "args": item.tool_input,
                        }
                    })
                elif isinstance(item, ToolResult):
                    parts.append({
                        "function_response": {
                            "name": item.tool_id,
                            "response": {"result": str(item.result)},
                        }
                    })
            if parts:
                role = "model" if msg.role == MessageRole.ASSISTANT else "user"
                contents.append({"role": role, "parts": parts})
        return contents

    def _build_gemini_tools(self, tools: List[BaseTool]) -> List[Dict[str, Any]]:
        """Build Gemini tool declarations."""
        function_declarations = []
        for tool in tools:
            schema = tool.to_gemini_schema()
            # Clean up schema to match Gemini expectations
            params = dict(schema.get("parameters", {}))
            # Gemini doesn't support 'required: []' (empty list) well
            if "required" in params and not params["required"]:
                del params["required"]
            function_declarations.append({
                "name": schema["name"],
                "description": schema["description"],
                "parameters": params,
            })
        return [{"function_declarations": function_declarations}]

    def _parse_response(self, response, msg_id: str) -> Message:
        """Parse Gemini response into a Message."""
        content_items = []
        candidate = response.candidates[0]

        for part in candidate.content.parts:
            if part.text:
                content_items.append(TextMessage(text=part.text))
            elif part.function_call:
                fc = part.function_call
                content_items.append(
                    ToolCall(
                        tool_id=str(uuid.uuid4()),
                        tool_name=fc.name,
                        tool_input=dict(fc.args) if fc.args else {},
                    )
                )

        usage = response.usage_metadata
        return Message(
            role=MessageRole.ASSISTANT,
            content=content_items,
            msg_id=msg_id,
            usage=LLMUsageMetrics(
                in_t=getattr(usage, "prompt_token_count", 0) or 0,
                op_t=getattr(usage, "candidates_token_count", 0) or 0,
                latency_ms=0,
            ),
        )

    def _tool_result_to_gemini_part(self, tool_result: ToolResult, tool_name: str) -> Dict[str, Any]:
        """Convert a ToolResult to a Gemini function_response part."""
        return {
            "function_response": {
                "name": tool_name,
                "response": {"result": str(tool_result.result)},
            }
        }

    async def invoke(
        self,
        context: ConversationContext,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        out_msg_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[LLMResponse, ConversationContext]:
        """Unified LLM invocation using Gemini API."""
        from google.genai import types

        final_response = LLMResponse()
        elicitations = []
        msg_id = out_msg_id or str(uuid.uuid4())

        tool_config = None
        tool_map = {}
        if tools:
            tool_config = self._build_gemini_tools(tools)
            tool_map = {tool.name: tool for tool in tools}

        contents = self._to_gemini_contents(context)

        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        if system_prompt:
            config.system_instruction = system_prompt
        if tool_config:
            config.tools = tool_config

        iterations = self.max_iterations if tool_config else 1

        for _ in range(iterations):
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            assistant_msg = self._parse_response(response, msg_id)
            context.add_message(assistant_msg)

            # Append assistant message to contents
            assistant_parts = []
            for item in assistant_msg.content:
                if isinstance(item, TextMessage):
                    assistant_parts.append({"text": item.text})
                elif isinstance(item, ToolCall):
                    assistant_parts.append({
                        "function_call": {
                            "name": item.tool_name,
                            "args": item.tool_input,
                        }
                    })
            contents.append({"role": "model", "parts": assistant_parts})

            tool_results = []
            tool_result_parts = []
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
                        tool_result_parts.append(
                            self._tool_result_to_gemini_part(tool_res, content_item.tool_name)
                        )

            if tool_results:
                results_msg = Message(
                    role=MessageRole.TOOL,
                    content=tool_results,
                    msg_id=msg_id,
                )
                context.add_message(results_msg)
                contents.append({"role": "user", "parts": tool_result_parts})
                final_response.pending_tool_result_processing = True
            else:
                final_response.pending_tool_result_processing = False

            if elicitations:
                break

            if not tool_results:
                break

        # Extract final text
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

    async def invoke_stream(
        self,
        context: ConversationContext,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        out_msg_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Streaming LLM invocation using Gemini streaming API."""
        from google.genai import types

        msg_id = out_msg_id or str(uuid.uuid4())

        tool_config = None
        tool_map = {}
        if tools:
            tool_config = self._build_gemini_tools(tools)
            tool_map = {tool.name: tool for tool in tools}

        contents = self._to_gemini_contents(context)

        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        if system_prompt:
            config.system_instruction = system_prompt
        if tool_config:
            config.tools = tool_config

        iterations = self.max_iterations if tool_config else 1

        final_response = LLMResponse()
        elicitations = []

        for _ in range(iterations):
            try:
                stream = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=config,
                )

                content_items = []
                tool_results = []
                tool_result_parts = []
                accumulated_text = ""
                accumulated_tool_calls = []
                usage_metrics = None

                for chunk in stream:
                    if not chunk.candidates:
                        # May contain usage metadata only
                        if chunk.usage_metadata:
                            usage_metrics = LLMUsageMetrics(
                                in_t=getattr(chunk.usage_metadata, "prompt_token_count", 0) or 0,
                                op_t=getattr(chunk.usage_metadata, "candidates_token_count", 0) or 0,
                                latency_ms=0,
                            )
                        continue

                    candidate = chunk.candidates[0]
                    if not candidate.content or not candidate.content.parts:
                        continue

                    for part in candidate.content.parts:
                        if part.text:
                            accumulated_text += part.text
                            yield StreamEvent(
                                type=StreamEventType.TEXT, text=part.text
                            )
                        elif part.function_call:
                            fc = part.function_call
                            tool_call = ToolCall(
                                tool_id=str(uuid.uuid4()),
                                tool_name=fc.name,
                                tool_input=dict(fc.args) if fc.args else {},
                            )
                            accumulated_tool_calls.append(tool_call)

                            tool = tool_map.get(fc.name)
                            is_elicitation = tool.is_elicitation if tool else False

                            if not is_elicitation:
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_START,
                                    details={
                                        "tool_id": tool_call.tool_id,
                                        "tool_name": tool_call.tool_name,
                                    },
                                )

                            tool_res = await self.invoke_tool(
                                tool_map, tool_call, context.tool_context
                            )

                            if not is_elicitation:
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_END,
                                    details={
                                        "tool_name": tool_call.tool_name,
                                        "tool_id": tool_call.tool_id,
                                    },
                                )

                            if isinstance(tool_res, ElicitationResponse):
                                elicitations.append(tool_res)
                            else:
                                tool_results.append(tool_res)
                                tool_result_parts.append(
                                    self._tool_result_to_gemini_part(tool_res, fc.name)
                                )

                    if chunk.usage_metadata:
                        usage_metrics = LLMUsageMetrics(
                            in_t=getattr(chunk.usage_metadata, "prompt_token_count", 0) or 0,
                            op_t=getattr(chunk.usage_metadata, "candidates_token_count", 0) or 0,
                            latency_ms=0,
                        )

                # -- end of stream --

                # Build content items list
                content_items_list = []
                if accumulated_text:
                    content_items_list.append(TextMessage(text=accumulated_text))
                    final_response.verbose.append(
                        VerboseResponseItem(type="text", text=accumulated_text)
                    )
                for tc in accumulated_tool_calls:
                    content_items_list.append(tc)
                    final_response.verbose.append(
                        VerboseResponseItem(type="tool", tool_name=tc.tool_name)
                    )

                # Build assistant message and add to context
                llm_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=content_items_list,
                    msg_id=msg_id,
                    usage=usage_metrics,
                )
                context.add_message(llm_message)

                # Append to contents for next iteration
                assistant_parts = []
                for item in content_items_list:
                    if isinstance(item, TextMessage):
                        assistant_parts.append({"text": item.text})
                    elif isinstance(item, ToolCall):
                        assistant_parts.append({
                            "function_call": {
                                "name": item.tool_name,
                                "args": item.tool_input,
                            }
                        })
                contents.append({"role": "model", "parts": assistant_parts})

                if tool_results:
                    results_msg = Message(
                        role=MessageRole.TOOL,
                        content=tool_results,
                        msg_id=msg_id,
                    )
                    context.add_message(results_msg)
                    contents.append({"role": "user", "parts": tool_result_parts})
                    final_response.pending_tool_result_processing = True
                else:
                    final_response.pending_tool_result_processing = False

                if elicitations:
                    break

                if not tool_results:
                    break

            except Exception as e:
                yield StreamEvent(type=StreamEventType.ERROR, error=str(e))
                return

        final_response.elicitations = elicitations or None
        yield StreamEvent(
            type=StreamEventType.DONE, llm_response=final_response, context=context
        )
