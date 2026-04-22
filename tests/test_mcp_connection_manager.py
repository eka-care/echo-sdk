"""Tests for MCPConnectionManager (bare-metal edition — no caching)."""

import asyncio
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from echo.tools.mcp_connection_manager import MCPConnectionManager
from echo.tools.schemas import MCPExecutionError, MCPServerConfig, MCPTransport


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


def build_config(
    url="https://mcp.test/mcp",
    transport=MCPTransport.STREAMABLE_HTTP,
    **kw,
) -> MCPServerConfig:
    return MCPServerConfig(transport=transport, url=url, **kw)


def test_fixtures_import_cleanly():
    cfg = build_config()
    assert cfg.url is not None


async def test_manager_construct_no_op():
    """Constructing a manager should not open any network connections."""
    cfg = build_config()
    MCPConnectionManager(cfg)  # no state to populate


async def test_get_tools_opens_fresh_session_every_call():
    """Every get_tools call opens a fresh session (no caching)."""
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)
    fake_session = make_fake_session(tool_names=["ping", "pong"])
    open_mock = AsyncMock(return_value=(fake_session, MagicMock(), None))

    with patch.object(MCPConnectionManager, "_open_session", new=open_mock):
        with patch.object(MCPConnectionManager, "_close_parts", new=AsyncMock()):
            tools1 = await mgr.get_tools()
            tools2 = await mgr.get_tools()

    assert {t.name for t in tools1} == {"ping", "pong"}
    assert {t.name for t in tools2} == {"ping", "pong"}
    # bare-metal: every call opens a new session
    assert open_mock.call_count == 2


async def test_execute_tool_opens_and_closes():
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)
    fake_session = make_fake_session()
    fake_session.call_tool = AsyncMock(return_value="ok")
    open_mock = AsyncMock(return_value=(fake_session, MagicMock(), None))
    close_mock = AsyncMock()

    with patch.object(MCPConnectionManager, "_open_session", new=open_mock):
        with patch.object(MCPConnectionManager, "_close_parts", new=close_mock):
            result = await mgr.execute_tool("ping", {"arg": 1})

    assert result == "ok"
    assert open_mock.call_count == 1
    assert close_mock.call_count == 1


async def test_execute_tool_parallel_uses_independent_sessions():
    """Parallel execute_tool calls each get their own session — no serialization."""
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)

    async def slow_call_tool(name, arguments, meta=None):
        await asyncio.sleep(0.05)
        return f"result-{name}"

    def make_session_with_slow_call():
        s = make_fake_session()
        s.call_tool = AsyncMock(side_effect=slow_call_tool)
        return s

    open_calls = 0

    async def open_side_effect(self_):
        nonlocal open_calls
        open_calls += 1
        return (make_session_with_slow_call(), MagicMock(), None)

    start = asyncio.get_running_loop().time()
    with patch.object(
        MCPConnectionManager,
        "_open_session",
        new=lambda self_: open_side_effect(self_),
    ):
        with patch.object(MCPConnectionManager, "_close_parts", new=AsyncMock()):
            results = await asyncio.gather(
                mgr.execute_tool("a", {}),
                mgr.execute_tool("b", {}),
                mgr.execute_tool("c", {}),
            )
    elapsed = asyncio.get_running_loop().time() - start

    assert set(results) == {"result-a", "result-b", "result-c"}
    assert open_calls == 3
    # Parallel: ~0.05s. Serial: ~0.15s. Allow CI slack but still prove parallelism.
    assert elapsed < 0.12, (
        f"Expected parallel execution (<0.12s), got {elapsed:.3f}s"
    )


async def test_execute_tool_wraps_failures_in_execution_error():
    cfg = build_config()
    mgr = MCPConnectionManager(cfg)
    fake_session = make_fake_session()
    fake_session.call_tool = AsyncMock(side_effect=RuntimeError("server blew up"))

    with patch.object(
        MCPConnectionManager,
        "_open_session",
        new=AsyncMock(return_value=(fake_session, MagicMock(), None)),
    ):
        with patch.object(MCPConnectionManager, "_close_parts", new=AsyncMock()):
            with pytest.raises(MCPExecutionError):
                await mgr.execute_tool("boom", {})
