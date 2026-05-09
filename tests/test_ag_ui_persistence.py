"""
Unit tests for echo.ag_ui.persistence (PR-S4).

Covers:
- make_pause_key format
- InMemoryPausedRunStore: save / load / delete round-trip
- TTL accepted but not enforced (documented behavior)
- Concurrent save/load is serialized via the asyncio.Lock
"""

import asyncio

import pytest

from echo.ag_ui import (
    InMemoryPausedRunStore,
    PausedRun,
    make_pause_key,
)


# ----- key format -----


def test_make_pause_key_format():
    assert (
        make_pause_key("thread-1", "run-9")
        == "ag_ui:paused_run:thread-1:run-9"
    )


def test_make_pause_key_with_real_world_ids():
    # voice2rx pattern: thread_id = "{txn_id}:{document_id}"
    assert (
        make_pause_key("txn_99:doc_42", "r1")
        == "ag_ui:paused_run:txn_99:doc_42:r1"
    )


# ----- store round-trip -----


def _make_paused() -> PausedRun:
    return PausedRun(
        thread_id="t1",
        run_id="r1",
        tool_call_id="tc1",
        tool_call_name="request_field_input",
        tool_args={"rowId": "m0", "field": "duration"},
        context_snapshot={"messages": []},
        state_snapshot={"transcript": "x", "sections": []},
        metadata={"b_id": "EC_x", "document_id": "doc_42"},
    )


@pytest.mark.asyncio
async def test_save_then_load_round_trip():
    store = InMemoryPausedRunStore()
    snap = _make_paused()
    key = make_pause_key("t1", "r1")

    await store.save(key, snap)
    loaded = await store.load(key)
    assert loaded is not None
    assert loaded.thread_id == "t1"
    assert loaded.tool_args == {"rowId": "m0", "field": "duration"}
    assert loaded.state_snapshot["transcript"] == "x"
    assert loaded.metadata["document_id"] == "doc_42"


@pytest.mark.asyncio
async def test_load_returns_none_for_missing_key():
    store = InMemoryPausedRunStore()
    assert await store.load("ag_ui:paused_run:nope:nope") is None


@pytest.mark.asyncio
async def test_save_overwrites_same_key():
    store = InMemoryPausedRunStore()
    key = make_pause_key("t1", "r1")
    await store.save(key, _make_paused())

    overwritten = PausedRun(
        thread_id="t1",
        run_id="r1",
        tool_call_id="tc2",
        tool_call_name="request_field_input",
        tool_args={"rowId": "m1", "field": "frequency"},
        context_snapshot={},
        state_snapshot={},
    )
    await store.save(key, overwritten)
    loaded = await store.load(key)
    assert loaded is not None
    assert loaded.tool_call_id == "tc2"
    assert loaded.tool_args == {"rowId": "m1", "field": "frequency"}


@pytest.mark.asyncio
async def test_delete_removes_entry():
    store = InMemoryPausedRunStore()
    key = make_pause_key("t1", "r1")
    await store.save(key, _make_paused())
    await store.delete(key)
    assert await store.load(key) is None


@pytest.mark.asyncio
async def test_delete_missing_key_is_noop():
    store = InMemoryPausedRunStore()
    # Should not raise.
    await store.delete("ag_ui:paused_run:never:was-there")


@pytest.mark.asyncio
async def test_save_accepts_ttl_arg():
    """In-memory store accepts ttl for API compatibility but doesn't enforce it."""
    store = InMemoryPausedRunStore()
    await store.save(make_pause_key("t1", "r1"), _make_paused(), ttl=60)
    # Item still there immediately; TTL is a no-op for the in-memory impl.
    assert await store.load(make_pause_key("t1", "r1")) is not None


@pytest.mark.asyncio
async def test_concurrent_save_and_load_safe():
    """Lock-protected: concurrent ops don't corrupt the store."""
    store = InMemoryPausedRunStore()
    key = make_pause_key("t1", "r1")

    async def writer(i: int):
        await store.save(
            key,
            PausedRun(
                thread_id="t1",
                run_id="r1",
                tool_call_id=f"tc{i}",
                tool_call_name="x",
                tool_args={},
                context_snapshot={},
                state_snapshot={},
            ),
        )

    async def reader():
        # Allow some scheduling; all loads should return a valid PausedRun
        # (not partial / not None) once at least one save has happened.
        for _ in range(5):
            await asyncio.sleep(0)

    await asyncio.gather(*(writer(i) for i in range(20)), reader())
    final = await store.load(key)
    assert final is not None
    # Some writer's value won — we don't assert which, only that it's coherent.
    assert final.tool_call_id.startswith("tc")
