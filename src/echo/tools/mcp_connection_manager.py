"""
MCP Connection Manager for Echo SDK.

Handles connection pooling, automatic reconnection, and tool discovery
for MCP servers with resilience features.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, List, Optional

from mcp import ClientSession

from .mcp_tool import MCPTool
from .schemas import (
    MCPConfigError,
    MCPConnectionError,
    MCPExecutionError,
    MCPServerConfig,
    MCPTransport,
)

logger = logging.getLogger(__name__)


@dataclass
class MCPConnection:
    """Encapsulates MCP connection state."""

    session: ClientSession
    transport_context: Any
    config: MCPServerConfig
    connected_at: float
    last_used: float


class MCPConnectionManager:
    """
    Manages MCP connections with pooling, health checks, and automatic reconnection.

    Provides a simple API for discovering and executing MCP tools with built-in
    resilience features including exponential backoff retry and connection pooling.
    """

    # Class-level shared state
    _connections: ClassVar[Dict[str, MCPConnection]] = {}
    _locks: ClassVar[Dict[str, asyncio.Lock]] = {}
    _cleanup_task: ClassVar[Optional[asyncio.Task]] = None
    _tools_cache: ClassVar[Dict[str, List[MCPTool]]] = {}

    def __init__(self, config: MCPServerConfig):
        """
        Initialize with config and start cleanup if needed.

        Args:
            config: MCPServerConfig with connection settings
        """
        config.validate()
        self._config = config
        self._server_id = self._generate_server_id(config)

        # Ensure lock exists for this server
        if self._server_id not in self._locks:
            self._locks[self._server_id] = asyncio.Lock()

        # Start cleanup task immediately if not running
        try:
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        except RuntimeError:
            # No event loop yet, will start on first async call
            pass

    async def get_tools(
        self,
        filter_fn: Optional[Callable] = None,
        tool_names: Optional[List[str]] = None,
    ) -> List[MCPTool]:
        """
        Get tools from server, connecting if needed. Uses caching to avoid repeated discovery.

        Args:
            filter_fn: Optional function to filter tools
            tool_names: Optional list of tool names to include

        Returns:
            List of MCPTool instances

        Raises:
            MCPConnectionError: If connection fails
        """
        connection = await self._get_or_create_connection()

        async with self._locks[self._server_id]:
            # Check cache first
            if self._server_id not in self._tools_cache:
                # Discover and cache all tools
                tools_response = await connection.session.list_tools()
                all_tools = []
                for tool in tools_response.tools:
                    mcp_tool = MCPTool(
                        manager=self,
                        server_id=self._server_id,
                        tool_name=tool.name,
                        tool_description=tool.description or "",
                        input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else None,
                    )
                    all_tools.append(mcp_tool)
                self._tools_cache[self._server_id] = all_tools
                logger.info(f"Discovered and cached {len(all_tools)} tools from {self._server_id}")

            cached_tools = self._tools_cache[self._server_id]

        # Apply filters outside lock (filters are read-only)
        filtered_tools = cached_tools

        # 1. Apply config-level include filter (whitelist)
        if self._config.tool_include:
            include_set = set(self._config.tool_include)
            filtered_tools = [t for t in filtered_tools if t.name in include_set]

        # 2. Apply config-level exclude filter (blacklist)
        if self._config.tool_exclude:
            exclude_set = set(self._config.tool_exclude)
            filtered_tools = [t for t in filtered_tools if t.name not in exclude_set]

        # 3. Apply method-level filters
        if tool_names:
            tool_names_set = set(tool_names)
            filtered_tools = [t for t in filtered_tools if t.name in tool_names_set]

        if filter_fn:
            filtered_tools = [t for t in filtered_tools if filter_fn(t)]

        return filtered_tools

    async def refresh_tools_cache(self) -> None:
        """Force refresh of tools cache for this server."""
        async with self._locks[self._server_id]:
            self._tools_cache.pop(self._server_id, None)

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute tool with simple one-retry logic.

        If tool call fails, closes connection, reconnects, and tries once more.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            MCPExecutionError: If tool execution fails after reconnection attempt
        """
        conn = await self._get_or_create_connection()

        try:
            # Try once with current connection
            result = await conn.session.call_tool(tool_name, arguments=arguments)
            conn.last_used = time.time()
            logger.debug(f"Successfully executed tool: {tool_name}")
            return result

        except Exception as e:
            # Connection failed - reconnect and try once more
            logger.warning(f"Tool {tool_name} failed: {e}, reconnecting...")

            # Close bad connection and remove from pool
            async with self._locks[self._server_id]:
                if self._server_id in self._connections:
                    try:
                        await self._close_connection(conn)
                    except Exception as e:
                        logger.error(f"Error closing connection: {e}")
                    del self._connections[self._server_id]
                    self._tools_cache.pop(self._server_id, None)  # Clear tools cache too

            # Get fresh connection and retry once
            try:
                conn = await self._get_or_create_connection()
                result = await conn.session.call_tool(tool_name, arguments=arguments)
                conn.last_used = time.time()
                logger.info(f"Successfully executed {tool_name} after reconnection")
                return result

            except Exception as retry_error:
                raise MCPExecutionError(
                    f"Failed to execute '{tool_name}' after reconnection attempt"
                ) from retry_error

    async def _get_or_create_connection(self) -> MCPConnection:
        """Get connection from pool or create new one. No health checks."""
        async with self._locks[self._server_id]:
            conn = self._connections.get(self._server_id)

            # If exists in pool, return it (trust it until it fails)
            if conn:
                logger.debug(f"Reusing existing connection: {self._server_id}")
                return conn

            # Create new connection
            conn = await self._create_connection()
            self._connections[self._server_id] = conn
            logger.info(f"Connected to: {self._server_id}")
            return conn

    async def _create_connection(self) -> MCPConnection:
        """Create new MCP connection."""
        try:
            # Get transport context based on type
            transport_type = self._config.transport
            if transport_type == MCPTransport.SSE:
                transport_context = self._connect_sse()
            elif transport_type == MCPTransport.STDIO:
                transport_context = self._connect_stdio()
            elif transport_type == MCPTransport.STREAMABLE_HTTP:
                transport_context = self._connect_streamable_http()
            else:
                raise MCPConfigError(f"Unknown transport type: {transport_type}")

            # Enter transport context
            transport_result = await transport_context.__aenter__()
            if transport_type == MCPTransport.STREAMABLE_HTTP:
                read_stream, write_stream, _get_session_id = transport_result
            else:
                read_stream, write_stream = transport_result

            # Create and initialize session
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()

            now = time.time()
            return MCPConnection(
                session=session,
                transport_context=transport_context,
                config=self._config,
                connected_at=now,
                last_used=now,
            )

        except Exception as e:
            raise MCPConnectionError(f"Failed to create connection: {e}") from e

    async def _close_connection(self, conn: MCPConnection):
        """Close connection and clean up resources."""
        try:
            if conn.session:
                await conn.session.__aexit__(None, None, None)
        except RuntimeError as e:
            # Expected for streamable_http when cleaning up from different task
            if "cancel scope" in str(e):
                logger.debug(f"Ignoring expected cleanup error: {e}")
            else:
                logger.error(f"Error closing session: {e}")
        except Exception as e:
            logger.error(f"Error closing session: {e}")

        try:
            if conn.transport_context:
                await conn.transport_context.__aexit__(None, None, None)
        except RuntimeError as e:
            # Expected for streamable_http when cleaning up from different task
            if "cancel scope" in str(e):
                logger.debug(f"Ignoring expected cleanup error: {e}")
            else:
                logger.error(f"Error closing transport: {e}")
        except Exception as e:
            logger.error(f"Error closing transport: {e}")

        logger.debug(f"Closed connection: {self._server_id}")

    def _connect_sse(self):
        """Establish SSE connection and return context manager."""
        from mcp.client.sse import sse_client

        return sse_client(
            url=str(self._config.url),
            headers=self._config.headers,
            timeout=self._config.timeout,
            sse_read_timeout=self._config.sse_read_timeout,
        )

    def _connect_stdio(self):
        """Establish stdio connection and return context manager."""
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_params = StdioServerParameters(
            command=self._config.command,
            args=self._config.args or [],
            env=self._config.env,
        )
        return stdio_client(server_params)

    def _connect_streamable_http(self):
        """Establish Streamable HTTP connection and return context manager."""
        import httpx
        from mcp.client.streamable_http import streamable_http_client

        http_client = httpx.AsyncClient(
            headers=self._config.headers or {},
            timeout=httpx.Timeout(
                self._config.timeout, read=self._config.sse_read_timeout
            ),
        )
        return streamable_http_client(url=str(self._config.url), http_client=http_client)

    def _generate_server_id(self, config: MCPServerConfig) -> str:
        """Generate unique ID for connection pooling."""
        if config.transport in [MCPTransport.SSE, MCPTransport.STREAMABLE_HTTP]:
            return f"{config.transport.value}:{config.url}"
        else:  # STDIO
            return f"{config.transport.value}:{config.command}:{':'.join(config.args or [])}"

    async def _cleanup_loop(self):
        """Background task to remove expired connections (TTL only)."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Find expired connections (inline)
                now = time.time()
                expired = [
                    server_id
                    for server_id, conn in self._connections.items()
                    if (now - conn.connected_at) > self._config.connection_ttl
                ]

                # Remove expired connections
                for server_id in expired:
                    async with self._locks[server_id]:
                        if server_id in self._connections:
                            await self._close_connection(self._connections[server_id])
                            del self._connections[server_id]
                            self._tools_cache.pop(server_id, None)  # Clear tools cache too
                            logger.info(f"Cleaned up expired connection: {server_id}")

            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    @classmethod
    async def cleanup_all(cls):
        """Cleanup all connections (call at shutdown)."""
        if cls._cleanup_task:
            cls._cleanup_task.cancel()
            try:
                await cls._cleanup_task
            except asyncio.CancelledError:
                pass

        for server_id, conn in list(cls._connections.items()):
            try:
                if conn.session:
                    await conn.session.__aexit__(None, None, None)
            except RuntimeError as e:
                # Expected for streamable_http - cleanup from different task
                if "cancel scope" not in str(e):
                    logger.debug(f"Session cleanup error for {server_id}: {e}")
            except Exception as e:
                logger.debug(f"Session cleanup error for {server_id}: {e}")

            try:
                if conn.transport_context:
                    await conn.transport_context.__aexit__(None, None, None)
            except RuntimeError as e:
                # Expected for streamable_http - cleanup from different task
                if "cancel scope" not in str(e):
                    logger.debug(f"Transport cleanup error for {server_id}: {e}")
            except Exception as e:
                logger.debug(f"Transport cleanup error for {server_id}: {e}")

        cls._connections.clear()
        cls._tools_cache.clear()  # Clear tools cache too
        logger.info("Cleaned up all connections")
