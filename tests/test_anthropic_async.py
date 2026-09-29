"""The Anthropic provider must not hold the event loop while it waits.

invoke and invoke_stream are coroutines that run on the caller's loop. With the
sync client they blocked it for the whole request, so every other task on the
worker stalled: other agents' runs, websocket sends, and in a voice pipeline the
audio of every caller on that worker.

These tests drive the provider with a fake AsyncAnthropic whose stream pauses
between events the way a network read does, and check four things: events come
out in order, the loop stays free, concurrent runs do not cross, and abandoning
a stream closes it.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

from echo.llm.anthropic import AnthropicLLM
from echo.llm.config import LLMConfig
from echo.llm.schemas import StreamEventType
from echo.models.user_conversation import (
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
)
from echo.tools import BaseTool

MODEL = "claude-sonnet-4-20250514"


# --- a fake AsyncAnthropic ---------------------------------------------------


def _ev(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


def _usage(**kw):
    base = dict(
        input_tokens=10,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return SimpleNamespace(**{**base, **kw})


def text_script(chunks):
    return [
        _ev("message_start", message=SimpleNamespace(usage=_usage())),
        _ev("content_block_start", index=0, content_block=SimpleNamespace(type="text")),
        *[
            _ev(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text=c),
            )
            for c in chunks
        ],
        _ev("content_block_stop", index=0),
        _ev("message_delta", usage=_usage(output_tokens=len(chunks))),
        _ev("message_stop"),
    ]


def tool_script(tool_name, json_parts):
    return [
        _ev("message_start", message=SimpleNamespace(usage=_usage())),
        _ev(
            "content_block_start",
            index=0,
            content_block=SimpleNamespace(type="tool_use", id="toolu_1", name=tool_name),
        ),
        *[
            _ev(
                "content_block_delta",
                index=0,
                delta=SimpleNamespace(type="input_json_delta", partial_json=p),
            )
            for p in json_parts
        ],
        _ev("content_block_stop", index=0),
        _ev("message_delta", usage=_usage(output_tokens=3)),
        _ev("message_stop"),
    ]


class FakeStream:
    """What `client.messages.stream(...)` returns: an async context manager
    that is also an async iterator, pausing before each event like a socket."""

    def __init__(self, script, delay):
        self._script, self._delay = script, delay
        self.entered = self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc):
        self.exited = True
        return False

    async def __aiter__(self):
        for event in self._script:
            await asyncio.sleep(self._delay)
            yield event


class FakeClient:
    def __init__(self, respond, delay=0.0):
        self._respond, self._delay = respond, delay
        self.streams = []
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **request_kwargs):
        stream = FakeStream(self._respond(request_kwargs), self._delay)
        self.streams.append(stream)
        return stream


def _llm(respond, delay=0.0):
    llm = AnthropicLLM(LLMConfig(provider="anthropic", model=MODEL))
    llm._client = FakeClient(respond, delay)
    return llm


def _context(text="hello"):
    ctx = ConversationContext()
    ctx.add_message(Message(role=MessageRole.USER, content=[TextMessage(text=text)]))
    return ctx


async def _collect(llm, ctx, **kw):
    return [e async for e in llm.invoke_stream(context=ctx, system_prompt="p", **kw)]


def _text(events):
    return "".join(e.text for e in events if e.type == StreamEventType.TEXT)


class EchoTool(BaseTool):
    name = "echo_tool"
    description = "Echo the input back."
    input_schema = {"type": "object", "properties": {"q": {"type": "string"}}}

    def __init__(self):
        super().__init__()
        self.calls = []

    async def run(self, **kwargs):
        self.calls.append(kwargs.get("q"))
        return f"echo:{kwargs.get('q')}"


# --- same events, same order -------------------------------------------------


async def test_text_arrives_chunk_by_chunk_then_done():
    llm = _llm(lambda kw: text_script(["Take ", "rest ", "and fluids."]))

    events = await _collect(llm, _context())

    assert [e.type for e in events] == [StreamEventType.TEXT] * 3 + [StreamEventType.DONE]
    assert [e.text for e in events[:3]] == ["Take ", "rest ", "and fluids."]


async def test_tool_call_runs_then_the_answer_streams():
    tool = EchoTool()

    def respond(kw):
        asked_already = any(m["role"] == "assistant" for m in kw["messages"])
        if asked_already:
            return text_script(["Done."])
        return tool_script("echo_tool", ['{"q": ', '"fever"}'])

    llm = _llm(respond)
    events = await _collect(llm, _context(), tools=[tool])

    kinds = [e.type for e in events]
    assert kinds.index(StreamEventType.TOOL_CALL_START) < kinds.index(
        StreamEventType.TOOL_CALL_END
    )
    assert kinds.index(StreamEventType.TOOL_CALL_END) < kinds.index(StreamEventType.TEXT)
    assert kinds[-1] == StreamEventType.DONE
    assert tool.calls == ["fever"]
    assert _text(events) == "Done."


async def test_a_failing_stream_becomes_an_error_event():
    def respond(kw):
        raise RuntimeError("upstream 500")

    events = await _collect(_llm(respond), _context())

    assert events[0].type == StreamEventType.ERROR
    assert "upstream 500" in events[0].error


# --- the loop stays free -----------------------------------------------------


async def test_event_loop_keeps_running_during_a_stream():
    """A task due every 10ms must keep its schedule while the LLM is waited
    on. With the sync client it did not run once for the length of the call."""
    llm = _llm(lambda kw: text_script(["a", "b", "c", "d", "e"]), delay=0.1)

    gaps, running = [], True

    async def ticker():
        last = time.monotonic()
        while running:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    task = asyncio.create_task(ticker())
    started = time.monotonic()
    await _collect(llm, _context())
    took = time.monotonic() - started
    running = False
    await task

    assert took > 0.5, "the fake stream should have paused between events"
    assert max(gaps) < 0.08, f"loop stalled for {max(gaps) * 1000:.0f} ms"
    assert len(gaps) > 30


# --- concurrent runs do not cross --------------------------------------------


async def test_concurrent_streams_on_one_instance_stay_separate():
    """Ten runs interleaving on a single provider instance -- the hardest case,
    since any per-run state kept on the instance would be shared by all ten."""

    def respond(kw):
        said = kw["messages"][-1]["content"][-1]["text"]
        return text_script([f"reply", "-to-", said])

    llm = _llm(respond, delay=0.01)
    contexts = [_context(f"caller-{i}") for i in range(10)]

    results = await asyncio.gather(*[_collect(llm, c) for c in contexts])

    for i, (events, ctx) in enumerate(zip(results, contexts)):
        assert _text(events) == f"reply-to-caller-{i}"
        assert events[-1].type == StreamEventType.DONE
        # each run recorded its own reply in its own context, and only that
        replies = [m for m in ctx.messages if m.role == MessageRole.ASSISTANT]
        assert len(replies) == 1
        assert replies[0].content[0].text == f"reply-to-caller-{i}"


# --- abandoning a stream closes it -------------------------------------------


async def test_cancelling_mid_stream_closes_the_connection():
    """Barge-in in a voice call cancels the run. The HTTP stream must be closed
    on the way out, not left open until the model finishes."""
    llm = _llm(lambda kw: text_script(["a"] * 50), delay=0.05)
    seen = []

    async def consume():
        async for e in llm.invoke_stream(context=_context(), system_prompt="p"):
            seen.append(e)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.25)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    stream = llm._client.streams[0]
    assert stream.entered and stream.exited
    assert 0 < len(seen) < 50


async def test_closing_the_generator_closes_the_connection():
    llm = _llm(lambda kw: text_script(["a"] * 50), delay=0.01)

    gen = llm.invoke_stream(context=_context(), system_prompt="p")
    first = await gen.__anext__()
    await gen.aclose()

    assert first.type == StreamEventType.TEXT
    assert llm._client.streams[0].exited
