"""Bedrock stream reads must not block the event loop."""

import asyncio
import time

from echo.llm.bedrock import _iter_in_thread


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
