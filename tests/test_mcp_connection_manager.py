"""Tests for MCPConnectionManager session caching and tool discovery."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from echo.tools.mcp_connection_manager import MCPConnection, MCPConnectionManager
from echo.tools.schemas import MCPExecutionError, MCPServerConfig, MCPTransport


@pytest.fixture(autouse=True)
async def reset_manager_state():
    """Reset class-level state between tests to avoid cross-test pollution."""
    async def _reset():
        MCPConnectionManager._sessions.clear()
        MCPConnectionManager._session_locks.clear()
        MCPConnectionManager._tools_cache.clear()
        MCPConnectionManager._tool_discovery_locks.clear()
        task = MCPConnectionManager._cleanup_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        MCPConnectionManager._cleanup_task = None

    await _reset()
    yield
    await _reset()


def make_fake_session(tool_names: List[str] = None, call_result: Any = None):
    """Build a fake mcp.ClientSession with stubbed initialize/list_tools/call_tool."""
    session = AsyncMock()
    session.initialize = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    tools = []
    for name in (tool_names or []):
        t = MagicMock()
        t.name = name
        t.description = f"fake tool {name}"
        t.inputSchema = {"type": "object", "properties": {}, "required": []}
        tools.append(t)
    list_tools_response = MagicMock()
    list_tools_response.tools = tools
    session.list_tools = AsyncMock(return_value=list_tools_response)

    session.call_tool = AsyncMock(return_value=call_result or MagicMock())
    return session


@asynccontextmanager
async def fake_transport_context(session):
    """Minimal stand-in for streamable_http_client / sse_client / stdio_client."""
    read_stream = MagicMock()
    write_stream = MagicMock()
    yield (read_stream, write_stream, lambda: "fake-session-id")


def build_config(url="https://mcp.test/mcp", transport=MCPTransport.STREAMABLE_HTTP, **kw) -> MCPServerConfig:
    """Factory for test configs."""
    return MCPServerConfig(transport=transport, url=url, **kw)


def test_fixtures_import_cleanly():
    """Smoke test that the fixtures module wires together."""
    cfg = build_config()
    assert cfg.url is not None


async def test_manager_has_independent_caches():
    """Manager should expose separate _sessions and _tools_cache class attrs."""
    assert hasattr(MCPConnectionManager, "_sessions")
    assert hasattr(MCPConnectionManager, "_tools_cache")
    assert hasattr(MCPConnectionManager, "_session_locks")
    # old state should be gone
    assert not hasattr(MCPConnectionManager, "_connections")


async def test_manager_construct_no_op():
    """Constructing a manager should not open any network connections."""
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)
    assert len(MCPConnectionManager._sessions) == 0
    assert len(MCPConnectionManager._tools_cache) == 0


async def test_get_tools_opens_throwaway_session_on_miss():
    """First get_tools call opens a session, lists tools, closes session, caches result."""
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)
    fake_session = make_fake_session(tool_names=["ping", "pong"])

    with patch.object(MCPConnectionManager, "_open_session",
                      new=AsyncMock(return_value=(fake_session, MagicMock(), None))):
        with patch.object(MCPConnectionManager, "_close_parts",
                          new=AsyncMock()):
            tools = await mgr.get_tools()

    assert {t.name for t in tools} == {"ping", "pong"}
    # cached now — cache key starts with transport value, includes url
    cache_keys = list(MCPConnectionManager._tools_cache.keys())
    assert len(cache_keys) == 1
    assert "mcp.test/mcp" in cache_keys[0]


async def test_get_tools_uses_cache_on_second_call():
    """Second get_tools call for the same manager hits the cache (no re-open)."""
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)
    fake_session = make_fake_session(tool_names=["ping"])
    open_mock = AsyncMock(return_value=(fake_session, MagicMock(), None))

    with patch.object(MCPConnectionManager, "_open_session", new=open_mock):
        with patch.object(MCPConnectionManager, "_close_parts", new=AsyncMock()):
            await mgr.get_tools()
            await mgr.get_tools()

    # _open_session called exactly once — second call hit the cache
    assert open_mock.call_count == 1


def test_tool_cache_partitions_by_key_headers():
    """Two configs with different tool_cache_key_headers values get different cache keys."""
    cfg_a = build_config(
        headers={"X-Workspace-Id": "a"},
        tool_cache_key_headers=["x-workspace-id"],
    )
    cfg_b = build_config(
        headers={"X-Workspace-Id": "b"},
        tool_cache_key_headers=["x-workspace-id"],
    )
    mgr_a = MCPConnectionManager(cfg_a)
    mgr_b = MCPConnectionManager(cfg_b)
    assert mgr_a._tool_cache_key != mgr_b._tool_cache_key


def test_tool_cache_ignores_unlisted_headers():
    """Headers NOT in tool_cache_key_headers must NOT affect the cache key."""
    cfg_a = build_config(
        headers={"X-Workspace-Id": "same", "Authorization": "Bearer alice"},
        tool_cache_key_headers=["x-workspace-id"],
    )
    cfg_b = build_config(
        headers={"X-Workspace-Id": "same", "Authorization": "Bearer bob"},
        tool_cache_key_headers=["x-workspace-id"],
    )
    mgr_a = MCPConnectionManager(cfg_a)
    mgr_b = MCPConnectionManager(cfg_b)
    assert mgr_a._tool_cache_key == mgr_b._tool_cache_key


async def test_execute_tool_fresh_path_opens_and_closes():
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)
    fake_session = make_fake_session()
    fake_session.call_tool = AsyncMock(return_value="fresh-result")
    open_mock = AsyncMock(return_value=(fake_session, MagicMock(), None))
    close_mock = AsyncMock()

    with patch.object(MCPConnectionManager, "_open_session", new=open_mock):
        with patch.object(MCPConnectionManager, "_close_parts", new=close_mock):
            result = await mgr.execute_tool("ping", {"arg": 1})

    assert result == "fresh-result"
    assert open_mock.call_count == 1
    assert close_mock.call_count == 1
    # no cached sessions
    assert len(MCPConnectionManager._sessions) == 0


async def test_execute_tool_fresh_path_parallel_uses_independent_sessions():
    """Parallel fresh calls each get their own session — no serialization."""
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)

    call_order = []

    async def slow_call_tool(name, arguments, meta=None):
        call_order.append(("enter", name))
        await asyncio.sleep(0.05)
        call_order.append(("exit", name))
        return f"result-{name}"

    def make_session_with_slow_call():
        s = make_fake_session()
        s.call_tool = AsyncMock(side_effect=slow_call_tool)
        return s

    open_calls = 0
    async def open_side_effect(self_mgr):
        nonlocal open_calls
        open_calls += 1
        return (make_session_with_slow_call(), MagicMock(), None)

    start = asyncio.get_event_loop().time()
    with patch.object(
        MCPConnectionManager, "_open_session",
        new=lambda self_: open_side_effect(self_),
    ):
        with patch.object(MCPConnectionManager, "_close_parts", new=AsyncMock()):
            results = await asyncio.gather(
                mgr.execute_tool("a", {}),
                mgr.execute_tool("b", {}),
                mgr.execute_tool("c", {}),
            )
    elapsed = asyncio.get_event_loop().time() - start

    assert set(results) == {"result-a", "result-b", "result-c"}
    assert open_calls == 3  # three independent sessions
    # Parallel: should be ~0.05s total (one sleep). Serial: ~0.15s (three sleeps).
    # Allow generous slack for CI scheduling jitter but still prove parallelism.
    assert elapsed < 0.12, (
        f"Expected parallel execution (<0.12s), got {elapsed:.3f}s — "
        "calls may be serialized."
    )


async def test_execute_tool_cached_reuses_session():
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)
    fake_session = make_fake_session()
    fake_session.call_tool = AsyncMock(return_value="cached-result")
    open_mock = AsyncMock(return_value=(fake_session, MagicMock(), None))

    with patch.object(MCPConnectionManager, "_open_session", new=open_mock):
        with patch.object(MCPConnectionManager, "_close_parts", new=AsyncMock()):
            r1 = await mgr.execute_tool("ping", {}, user_session_id="conv-1")
            r2 = await mgr.execute_tool("ping", {}, user_session_id="conv-1")

    assert r1 == "cached-result" and r2 == "cached-result"
    assert open_mock.call_count == 1  # session reused
    assert "conv-1" in MCPConnectionManager._sessions


async def test_execute_tool_cached_different_ids_get_different_sessions():
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)

    open_calls = 0
    async def make_session(self_):
        nonlocal open_calls
        open_calls += 1
        s = make_fake_session()
        s.call_tool = AsyncMock(return_value=f"session-{open_calls}")
        return (s, MagicMock(), None)

    with patch.object(MCPConnectionManager, "_open_session",
                      new=lambda self_: make_session(self_)):
        await mgr.execute_tool("ping", {}, user_session_id="conv-a")
        await mgr.execute_tool("ping", {}, user_session_id="conv-b")

    assert open_calls == 2
    assert set(MCPConnectionManager._sessions.keys()) == {"conv-a", "conv-b"}


async def test_concurrent_get_or_create_returns_single_session():
    """Two tasks racing on the same user_session_id must open ONE session."""
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)

    open_calls = 0
    async def slow_open(self_):
        nonlocal open_calls
        open_calls += 1
        await asyncio.sleep(0.05)
        s = make_fake_session()
        s.call_tool = AsyncMock(return_value="ok")
        return (s, MagicMock(), None)

    with patch.object(MCPConnectionManager, "_open_session",
                      new=lambda self_: slow_open(self_)):
        await asyncio.gather(
            mgr.execute_tool("ping", {}, user_session_id="conv-x"),
            mgr.execute_tool("ping", {}, user_session_id="conv-x"),
            mgr.execute_tool("ping", {}, user_session_id="conv-x"),
        )

    assert open_calls == 1  # lock prevented duplicate creates
    assert len(MCPConnectionManager._sessions) == 1


async def test_cached_session_failure_evicts():
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)
    fake_session = make_fake_session()
    fake_session.call_tool = AsyncMock(side_effect=RuntimeError("server blew up"))

    with patch.object(MCPConnectionManager, "_open_session",
                      new=AsyncMock(return_value=(fake_session, MagicMock(), None))):
        with patch.object(MCPConnectionManager, "_close_parts", new=AsyncMock()):
            with pytest.raises(MCPExecutionError):
                await mgr.execute_tool("boom", {}, user_session_id="conv-y")

    # Session should have been evicted so next call gets a fresh one
    assert "conv-y" not in MCPConnectionManager._sessions


async def test_forget_session_closes_and_removes():
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)
    fake_session = make_fake_session()
    close_mock = AsyncMock()

    with patch.object(MCPConnectionManager, "_open_session",
                      new=AsyncMock(return_value=(fake_session, MagicMock(), None))):
        with patch.object(MCPConnectionManager, "_close_parts", new=close_mock):
            await mgr.execute_tool("ping", {}, user_session_id="conv-z")
            assert "conv-z" in MCPConnectionManager._sessions

            await mgr.forget_session("conv-z")

    assert "conv-z" not in MCPConnectionManager._sessions
    assert "conv-z" not in MCPConnectionManager._session_locks
    # close_parts was called at least once at forget time
    assert close_mock.call_count >= 1


async def test_forget_session_unknown_id_is_noop():
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)
    await mgr.forget_session("never-existed")  # no raise


async def test_lru_eviction_when_cache_full(monkeypatch):
    """When sessions exceeds MAX_CACHED_SESSIONS, evict idle LRU."""
    monkeypatch.setattr(MCPConnectionManager, "MAX_CACHED_SESSIONS", 2)
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)

    async def open_mock(self_):
        s = make_fake_session()
        s.call_tool = AsyncMock(return_value="ok")
        return (s, MagicMock(), None)

    with patch.object(MCPConnectionManager, "_open_session",
                      new=lambda self_: open_mock(self_)):
        with patch.object(MCPConnectionManager, "_close_connection_static",
                          new=AsyncMock()):
            await mgr.execute_tool("a", {}, user_session_id="s1")
            # Force different last_used values deterministically
            MCPConnectionManager._sessions["s1"].last_used = 100.0
            await mgr.execute_tool("a", {}, user_session_id="s2")
            MCPConnectionManager._sessions["s2"].last_used = 200.0
            await mgr.execute_tool("a", {}, user_session_id="s3")
            # Yield once so the background close task (patched AsyncMock) can run
            await asyncio.sleep(0)

    # s1 is LRU — should have been evicted synchronously from the dict
    assert "s1" not in MCPConnectionManager._sessions
    assert "s2" in MCPConnectionManager._sessions
    assert "s3" in MCPConnectionManager._sessions
