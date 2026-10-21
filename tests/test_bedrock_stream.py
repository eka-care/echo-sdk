"""Bedrock stream reads must not block the event loop, and an abandoned stream
must be released."""

import asyncio
import time
from unittest.mock import MagicMock

from echo.llm.bedrock import BedrockLLM, _iter_in_thread
from echo.llm.config import LLMConfig
from echo.llm.schemas import StreamEventType
from echo.models.user_conversation import (
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
)


class FakeEventStream:
    """Blocking iterable like botocore's EventStream, with its close()."""

    def __init__(self, texts, delay):
        self._texts, self._delay = texts, delay
        self.closed = False

    def __iter__(self):
        for i, text in enumerate(self._texts):
            time.sleep(self._delay)
            yield {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": text}}}

    def close(self):
        self.closed = True


def _llm(delay):
    llm = BedrockLLM(LLMConfig(provider="bedrock"))
    llm._client, streams = MagicMock(), []

    def converse_stream(**kwargs):
        streams.append(FakeEventStream(["Take ", "rest."], delay))
        return {"stream": streams[-1]}

    llm._client.converse_stream.side_effect = converse_stream
    return llm, streams


def _context():
    ctx = ConversationContext()
    ctx.add_message(Message(role=MessageRole.USER, content=[TextMessage(text="hello")]))
    return ctx


def _slow_events():
    for i in range(3):
        time.sleep(0.2)  # a blocking network read on boto3's event stream
        yield i


async def test_iter_in_thread_yields_all_items():
    assert [e async for e in _iter_in_thread(_slow_events())] == [0, 1, 2]


async def test_iter_in_thread_keeps_loop_free():
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        _ = [e async for e in _iter_in_thread(_slow_events())]
    finally:
        task.cancel()

    # ~0.6s of blocking reads; a blocked loop would tick only between reads.
    assert ticks > 20


async def test_finished_stream_is_released():
    llm, streams = _llm(delay=0)
    events = [e async for e in llm.invoke_stream(_context(), out_msg_id="m1")]

    assert [e.text for e in events if e.type == StreamEventType.TEXT] == ["Take ", "rest."]
    assert events[-1].type == StreamEventType.DONE
    assert streams[0].closed


async def test_cancelling_mid_stream_releases_it():
    llm, streams = _llm(delay=0.2)

    async def consume():
        async for _ in llm.invoke_stream(_context(), out_msg_id="m1"):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert streams[0].closed
