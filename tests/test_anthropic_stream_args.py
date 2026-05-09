"""
Unit tests for echo.llm.anthropic.AnthropicLLM.invoke_stream tool-args
streaming (PR-S5).

Verifies the provider emits StreamEventType.TOOL_CALL_ARGS as
input_json_delta fragments arrive, and skips emission for elicitation
tools (mirroring the existing TOOL_CALL_START / TOOL_CALL_END skip).

The Anthropic SDK is mocked at the client level — we don't make
real network calls.
"""

from types import SimpleNamespace
from typing import List

import pytest

from echo.llm.anthropic import AnthropicLLM
from echo.llm.config import LLMConfig
from echo.llm.schemas import StreamEvent, StreamEventType
from echo.models.user_conversation import (
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
)
from echo.tools.base_tool import BaseTool


# ----- helpers: mock Anthropic stream -----


class _FakeStream:
    """Stand-in for the context manager returned by client.messages.stream."""

    def __init__(self, events: List[SimpleNamespace]):
        self._events = events

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._events)


class _FakeMessages:
    def __init__(self, events: List[SimpleNamespace]):
        self._events = events

    def stream(self, **kwargs):
        return _FakeStream(self._events)


class _FakeAnthropicClient:
    def __init__(self, events: List[SimpleNamespace]):
        self.messages = _FakeMessages(events)


def _block_start_tool(index: int, tool_id: str, tool_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_start",
        index=index,
        content_block=SimpleNamespace(
            type="tool_use", id=tool_id, name=tool_name
        ),
    )


def _block_delta_input_json(index: int, partial_json: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(type="input_json_delta", partial_json=partial_json),
    )


def _block_delta_text(index: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        index=index,
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _block_stop(index: int) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_stop", index=index)


def _message_delta_with_usage(in_t=10, op_t=20) -> SimpleNamespace:
    return SimpleNamespace(
        type="message_delta",
        usage=SimpleNamespace(input_tokens=in_t, output_tokens=op_t),
    )


# ----- minimal BaseTool stubs -----


class _BackendTool(BaseTool):
    name = "emit_section"
    description = "test backend tool"

    @property
    def input_schema(self):
        return {"type": "object"}

    async def run(self, **kwargs):
        return {"ok": True}


class _ElicitationTool(BaseTool):
    name = "request_field_input"
    description = "test elicitation tool"

    @property
    def is_elicitation(self) -> bool:
        return True

    @property
    def input_schema(self):
        return {"type": "object"}

    async def run(self, **kwargs):
        from echo.tools.schemas import ElicitationDetails

        return ElicitationDetails(component=self.name, input=kwargs)


# ----- tests -----


def _llm(client) -> AnthropicLLM:
    # Real model name to satisfy LLMConfig's allowlist; the client is
    # mocked so the api_key is never used. max_iterations=1 prevents the
    # agentic loop from re-streaming the same scripted events.
    cfg = LLMConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        api_key="test-stub",
        max_iterations=1,
    )
    inst = AnthropicLLM(cfg)
    inst._client = client  # bypass lazy init
    return inst


@pytest.mark.asyncio
async def test_input_json_delta_emits_tool_call_args_event_for_backend_tool():
    events = [
        _block_start_tool(0, "tc1", "emit_section"),
        _block_delta_input_json(0, '{"key":"meds"'),
        _block_delta_input_json(0, ',"display_name":"Meds"}'),
        _block_stop(0),
        _message_delta_with_usage(),
    ]
    llm = _llm(_FakeAnthropicClient(events))

    ctx = ConversationContext()
    ctx.add_message(
        Message(role=MessageRole.USER, content=[TextMessage(text="go")])
    )

    yielded: List[StreamEvent] = []
    async for ev in llm.invoke_stream(
        context=ctx,
        tools=[_BackendTool()],
        system_prompt="sys",
        out_msg_id="msg1",
    ):
        yielded.append(ev)

    args_evs = [e for e in yielded if e.type == StreamEventType.TOOL_CALL_ARGS]
    assert len(args_evs) == 2
    assert args_evs[0].details["tool_id"] == "tc1"
    assert args_evs[0].details["tool_name"] == "emit_section"
    assert args_evs[0].details["delta"] == '{"key":"meds"'
    assert args_evs[1].details["delta"] == ',"display_name":"Meds"}'


@pytest.mark.asyncio
async def test_input_json_delta_skipped_for_elicitation_tool():
    events = [
        _block_start_tool(0, "tc1", "request_field_input"),
        _block_delta_input_json(0, '{"rowId":"m0"}'),
        _block_stop(0),
        _message_delta_with_usage(),
    ]
    llm = _llm(_FakeAnthropicClient(events))

    ctx = ConversationContext()
    ctx.add_message(
        Message(role=MessageRole.USER, content=[TextMessage(text="go")])
    )

    yielded: List[StreamEvent] = []
    async for ev in llm.invoke_stream(
        context=ctx,
        tools=[_ElicitationTool()],
        system_prompt="sys",
        out_msg_id="msg1",
    ):
        yielded.append(ev)

    # Elicitation tool: no TOOL_CALL_START / ARGS / END from the stream.
    assert not any(e.type == StreamEventType.TOOL_CALL_START for e in yielded)
    assert not any(e.type == StreamEventType.TOOL_CALL_ARGS for e in yielded)
    assert not any(e.type == StreamEventType.TOOL_CALL_END for e in yielded)


@pytest.mark.asyncio
async def test_text_delta_unaffected_by_args_streaming():
    """Sanity: text streaming still works alongside the new args path."""
    events = [
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="text", text=""),
        ),
        _block_delta_text(0, "hello "),
        _block_delta_text(0, "world"),
        _block_stop(0),
        _message_delta_with_usage(),
    ]
    llm = _llm(_FakeAnthropicClient(events))

    ctx = ConversationContext()
    ctx.add_message(
        Message(role=MessageRole.USER, content=[TextMessage(text="hi")])
    )

    yielded: List[StreamEvent] = []
    async for ev in llm.invoke_stream(
        context=ctx,
        tools=None,
        system_prompt="sys",
        out_msg_id="msg1",
    ):
        yielded.append(ev)

    text_evs = [e for e in yielded if e.type == StreamEventType.TEXT]
    assert [t.text for t in text_evs] == ["hello ", "world"]
    assert not any(e.type == StreamEventType.TOOL_CALL_ARGS for e in yielded)


@pytest.mark.asyncio
async def test_args_event_ordering_around_start_and_end():
    """ARGS events arrive after START and before END for the same tool_id."""
    events = [
        _block_start_tool(0, "tc1", "emit_section"),
        _block_delta_input_json(0, '{"a":1'),
        _block_delta_input_json(0, ',"b":2}'),
        _block_stop(0),
        _message_delta_with_usage(),
    ]
    llm = _llm(_FakeAnthropicClient(events))

    ctx = ConversationContext()
    ctx.add_message(
        Message(role=MessageRole.USER, content=[TextMessage(text="go")])
    )

    seq: List[StreamEventType] = []
    async for ev in llm.invoke_stream(
        context=ctx,
        tools=[_BackendTool()],
        system_prompt="sys",
        out_msg_id="msg1",
    ):
        if ev.type in (
            StreamEventType.TOOL_CALL_START,
            StreamEventType.TOOL_CALL_ARGS,
            StreamEventType.TOOL_CALL_END,
        ):
            seq.append(ev.type)

    assert seq == [
        StreamEventType.TOOL_CALL_START,
        StreamEventType.TOOL_CALL_ARGS,
        StreamEventType.TOOL_CALL_ARGS,
        StreamEventType.TOOL_CALL_END,
    ]
