"""OpenAI and Gemini must not hold the event loop, and must release a stream
that is abandoned.

Run against three streaming paths: OpenAI chat completions, OpenAI responses,
and Gemini. Each fake stream pauses before every item the way a network read
does.

Releasing matters more here than it did for Anthropic, whose stream was already
inside a `with` block. These two were bare iterators; with the sync client a run
could not be interrupted mid-stream, so nothing ever abandoned one. Now a
cancelled run can, and an unclosed stream holds its connection until garbage
collection.
"""

import asyncio
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from echo.llm.config import LLMConfig
from echo.llm.gemini import GeminiLLM
from echo.llm.openai import OpenAILLM
from echo.llm.schemas import StreamEventType
from echo.models.user_conversation import (
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
)
from echo.tools import BaseTool

CHUNKS = ["Take ", "rest ", "and ", "drink ", "fluids."]


class FakeStream:
    """Async-iterable, pausing before each item. Closeable under both names:
    AsyncOpenAI's stream has close(), Gemini's async generator has aclose()."""

    def __init__(self, items, delay):
        self._items, self._delay = items, delay
        self.closed = False

    async def __aiter__(self):
        for item in self._items:
            await asyncio.sleep(self._delay)
            yield item

    async def close(self):
        self.closed = True

    aclose = close


class EchoTool(BaseTool):
    name = "echo_tool"
    description = "Echo the input back."
    input_schema = {"type": "object", "properties": {"q": {"type": "string"}}}

    async def run(self, **kwargs):
        return f"echo:{kwargs.get('q')}"


def _context():
    ctx = ConversationContext()
    ctx.add_message(Message(role=MessageRole.USER, content=[TextMessage(text="hello")]))
    return ctx


# --- one builder per streaming path ------------------------------------------
# Each returns (llm, invoke_kwargs, streams): `streams` collects every stream
# the fake client handed out.


def openai_chat(delay, chunks=CHUNKS):
    items = [
        NS(choices=[NS(delta=NS(content=c, tool_calls=None), finish_reason=None)], usage=None)
        for c in chunks
    ]
    llm = OpenAILLM(LLMConfig(provider="openai", model="gpt-6-luna"))
    llm._client, streams = MagicMock(), []

    async def create(**kw):
        streams.append(FakeStream(items, delay))
        return streams[-1]

    llm._client.chat.completions.create = AsyncMock(side_effect=create)
    return llm, {}, streams


def openai_responses(delay, chunks=CHUNKS):
    final = NS(
        output=[
            NS(
                type="message",
                role="assistant",
                content=[NS(type="output_text", text="".join(chunks))],
                model_dump=lambda exclude_none=False: {"type": "message"},
            )
        ],
        usage=NS(input_tokens=10, output_tokens=5, input_tokens_details=NS(cached_tokens=0)),
        incomplete_details=None,
    )
    items = [NS(type="response.output_text.delta", delta=c) for c in chunks]
    items.append(NS(type="response.completed", response=final))
    llm = OpenAILLM(LLMConfig(provider="openai", model="gpt-6-luna"))
    llm._client, streams = MagicMock(), []

    async def create(**kw):
        streams.append(FakeStream(items, delay))
        return streams[-1]

    llm._client.responses.create = AsyncMock(side_effect=create)
    # the responses path is the one taken when tools are present
    return llm, {"tools": [EchoTool()]}, streams


def gemini(delay, chunks=CHUNKS):
    items = [
        NS(
            candidates=[NS(content=NS(parts=[NS(text=c, function_call=None)]))],
            usage_metadata=None,
        )
        for c in chunks
    ]
    llm = GeminiLLM(LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key="x"))
    llm._client, streams = MagicMock(), []

    async def generate_content_stream(**kw):
        streams.append(FakeStream(items, delay))
        return streams[-1]

    llm._client.aio.models.generate_content_stream = AsyncMock(
        side_effect=generate_content_stream
    )
    return llm, {}, streams


PATHS = pytest.mark.parametrize(
    "build", [openai_chat, openai_responses, gemini], ids=lambda f: f.__name__
)


@PATHS
async def test_text_arrives_in_order_then_done(build):
    llm, kw, streams = build(delay=0)

    events = [e async for e in llm.invoke_stream(_context(), **kw)]

    assert [e.text for e in events if e.type == StreamEventType.TEXT] == CHUNKS
    assert events[-1].type == StreamEventType.DONE
    assert streams[0].closed, "a finished stream is released too"


@PATHS
async def test_event_loop_keeps_running_during_a_stream(build):
    llm, kw, _ = build(delay=0.1)
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
    events = [e async for e in llm.invoke_stream(_context(), **kw)]
    took = time.monotonic() - started
    running = False
    await task

    assert events[-1].type == StreamEventType.DONE
    assert took > 0.45, "the fake stream should have paused between items"
    assert max(gaps) < 0.08, f"loop stalled for {max(gaps) * 1000:.0f} ms"


@PATHS
async def test_cancelling_mid_stream_releases_it(build):
    llm, kw, streams = build(delay=0.05, chunks=["a"] * 50)
    seen = []

    async def consume():
        async for e in llm.invoke_stream(_context(), **kw):
            seen.append(e)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.25)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert 0 < len(seen) < 50
    assert streams[0].closed


@PATHS
async def test_closing_the_generator_releases_the_stream(build):
    llm, kw, streams = build(delay=0.01, chunks=["a"] * 50)

    gen = llm.invoke_stream(_context(), **kw)
    first = await gen.__anext__()
    await gen.aclose()

    assert first.type == StreamEventType.TEXT
    assert streams[0].closed
