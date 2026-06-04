"""
MCP Connection Manager for Echo SDK — bare-metal edition.

Every public call opens a fresh MCP ClientSession, does its work, and
closes the session. No caching, no pooling, no class-level state.

This is parallel-safe by construction: each call gets its own session
(its own Mcp-Session-Id on the server), so concurrent tool calls fan out
without serializing on a shared session. It costs ~1 initialize
round-trip per call (~10-50ms typical) but eliminates all classes of
cross-caller state bleed.
"""

import logging
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from mcp import ClientSession

from .mcp_tool import MCPTool
from ..core.schemas import (
    MCPConfigError,
    MCPConnectionError,
    MCPExecutionError,
    MCPServerConfig,
    MCPTransport,
)

logger = logging.getLogger(__name__)


class MCPConnectionManager:
    """
    Manages MCP tool discovery and execution.

    Every call to get_tools() or execute_tool() opens a fresh session,
    performs the action, and closes the session. No caching of any kind.
    """

    def __init__(self, config: MCPServerConfig):
        config.validate()
        self._config = config

    async def get_tools(
        self,
        filter_fn: Optional[Callable] = None,
        tool_names: Optional[List[str]] = None,
    ) -> List[MCPTool]:
        """Open a session, list tools, close. Returns filtered MCPTool list."""
        session, transport_ctx, http_client = await self._open_session()
        try:
            tools_response = await session.list_tools()
        finally:
            await self._close_parts(session, transport_ctx, http_client)

        tools = [
            MCPTool(
                manager=self,
                tool_name=t.name,
                tool_description=t.description or "",
                input_schema=getattr(t, "inputSchema", None),
            )
            for t in tools_response.tools
        ]
        return self._apply_filters(tools, filter_fn, tool_names)

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Open a session, call the tool, close."""
        session, transport_ctx, http_client = await self._open_session()
        try:
            result = await session.call_tool(tool_name, arguments=arguments, meta=meta)
            logger.debug("Executed %s", tool_name)
            return result
        except Exception as e:
            raise MCPExecutionError(f"Failed to execute '{tool_name}': {e}") from e
        finally:
            await self._close_parts(session, transport_ctx, http_client)

    # ---- Internals ----

    async def _open_session(self) -> Tuple[ClientSession, Any, Any]:
        """Open a fresh ClientSession. Returns (session, transport_ctx, http_client).
        Caller must close via _close_parts. Cleans up its own partial failures."""
        transport_ctx: Any = None
        http_client: Any = None
        cleaned_up = False
        try:
            if self._config.transport == MCPTransport.SSE:
                transport_ctx = self._connect_sse()
            elif self._config.transport == MCPTransport.STDIO:
                transport_ctx = self._connect_stdio()
            elif self._config.transport == MCPTransport.STREAMABLE_HTTP:
                transport_ctx, http_client = self._connect_streamable_http()
            else:
                raise MCPConfigError(
                    f"Unknown transport type: {self._config.transport}"
                )

            transport_result = await transport_ctx.__aenter__()
            if self._config.transport == MCPTransport.STREAMABLE_HTTP:
                read_stream, write_stream, _ = transport_result
            else:
                read_stream, write_stream = transport_result

            session = ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self._config.sse_read_timeout),
            )
            try:
                await session.__aenter__()
                await session.initialize()
            except Exception:
                # _close_parts closed session+transport+http_client already;
                # tell the outer handler not to double-close.
                await self._close_parts(session, transport_ctx, http_client)
                cleaned_up = True
                raise
            return session, transport_ctx, http_client
        except MCPConfigError:
            raise
        except Exception as e:
            if not cleaned_up:
                if transport_ctx is not None:
                    try:
                        await transport_ctx.__aexit__(None, None, None)
                    except Exception as close_err:
                        logger.debug("Transport close after failed open: %s", close_err)
                if http_client is not None:
                    try:
                        await http_client.aclose()
                    except Exception as close_err:
                        logger.debug("httpx close after failed open: %s", close_err)
            raise MCPConnectionError(f"Failed to open MCP session: {e}") from e

    @staticmethod
    async def _close_parts(
        session: Optional[ClientSession], transport_ctx: Any, http_client: Any
    ) -> None:
        """Close session, transport, and http_client in order, ignoring benign errors."""
        if session is not None:
            try:
                await session.__aexit__(None, None, None)
            except RuntimeError as e:
                if "cancel scope" not in str(e):
                    logger.debug("Session close error: %s", e)
            except Exception as e:
                logger.debug("Session close error: %s", e)
        if transport_ctx is not None:
            try:
                await transport_ctx.__aexit__(None, None, None)
            except RuntimeError as e:
                if "cancel scope" not in str(e):
                    logger.debug("Transport close error: %s", e)
            except Exception as e:
                logger.debug("Transport close error: %s", e)
        if http_client is not None:
            try:
                await http_client.aclose()
            except Exception as e:
                logger.debug("httpx close error: %s", e)

    def _connect_sse(self):
        from mcp.client.sse import sse_client

        return sse_client(
            url=str(self._config.url),
            headers=self._config.headers,
            timeout=self._config.timeout,
            sse_read_timeout=self._config.sse_read_timeout,
        )

    def _connect_stdio(self):
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args or [],
            env=self._config.env,
        )
        return stdio_client(server_params)

    def _connect_streamable_http(self):
        from mcp.client.streamable_http import streamable_http_client

        http_client = httpx.AsyncClient(
            headers=self._config.headers or {},
            timeout=httpx.Timeout(
                self._config.timeout, read=self._config.sse_read_timeout
            ),
        )
        ctx = streamable_http_client(url=str(self._config.url), http_client=http_client)
        return ctx, http_client

    def _apply_filters(
        self,
        tools: List[MCPTool],
        filter_fn: Optional[Callable],
        tool_names: Optional[List[str]],
    ) -> List[MCPTool]:
        result = tools
        if tool_names:
            wanted = set(tool_names)
            result = [t for t in result if t.name in wanted]
        if filter_fn:
            result = [t for t in result if filter_fn(t)]
        return result
