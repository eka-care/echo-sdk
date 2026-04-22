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
    MCPConnectionManager._sessions.clear()
    MCPConnectionManager._session_locks.clear()
    MCPConnectionManager._tools_cache.clear()
    if MCPConnectionManager._cleanup_task and not MCPConnectionManager._cleanup_task.done():
        MCPConnectionManager._cleanup_task.cancel()
        try:
            await MCPConnectionManager._cleanup_task
        except asyncio.CancelledError:
            pass
    MCPConnectionManager._cleanup_task = None
    yield
    MCPConnectionManager._sessions.clear()
    MCPConnectionManager._session_locks.clear()
    MCPConnectionManager._tools_cache.clear()


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
