"""
MCP Tool Provider for Echo SDK.

Handles connection to MCP servers (SSE, Streamable HTTP, or stdio) and provides
tool discovery with filtering support.
"""

from typing import Any, Callable, Dict, List, Optional

from .mcp_tool import MCPTool
from .schemas import MCPServerConfig, MCPTransport


class MCPToolProvider:
    """
    Provider for discovering and creating MCPTool instances from MCP servers.

    Handles connection lifecycle and tool filtering for both SSE and stdio transports.

    Example:
        # SSE with custom headers
        config = MCPServerConfig(
            transport=MCPTransport.SSE,
            url="http://mcp-server/sse",
            headers={"X-Tenant-ID": "123"}
        )
        provider = MCPToolProvider(config)

        async with provider.connect() as tools:
            # tools is List[MCPTool]
            agent = MyAgent(tools=tools)
            result = agent.run("query")

        # With filtering
        async with provider.connect(filter_fn=lambda t: t.name.startswith("search_")) as tools:
            # Only tools matching the filter
            ...
    """

    def __init__(self, config: MCPServerConfig):
        """
        Initialize the provider with server configuration.

        Args:
            config: MCPServerConfig with connection details
        """
        config.validate()
        self._config = config
        self._session = None
        self._context_stack = None

    def _connect_sse(self):
        """Establish SSE connection and return context manager."""
        from mcp.client.sse import sse_client

        return sse_client(
            url=self._config.url,
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

        async def log_request(request):
            print(f">>> {request.method} {request.url}")
            print(f">>> Headers: {dict(request.headers)}")
            if request.content:
                print(f">>> Body: {request.content.decode()}")

        async def log_response(response):
            print(f"<<< {response.status_code} {response.url}")

        # Create custom httpx client with headers, timeout, and logging hooks
        http_client = httpx.AsyncClient(
            headers=self._config.headers or {},
            timeout=httpx.Timeout(
                self._config.timeout, read=self._config.sse_read_timeout
            ),
            # event_hooks={"request": [log_request], "response": [log_response]},
        )

        return streamable_http_client(
            url=self._config.url,
            http_client=http_client,
        )

    def connect(
        self,
        filter_fn: Optional[Callable[[Any], bool]] = None,
        tool_names: Optional[List[str]] = None,
    ) -> "_MCPConnectionContext":
        """
        Create a connection context that yields discovered tools.

        Args:
            filter_fn: Optional function to filter tools. Receives MCP tool object,
                      returns True to include the tool.
            tool_names: Optional list of tool names to include. If provided,
                       only tools with matching names are returned.

        Returns:
            Async context manager that yields List[MCPTool]

        Example:
            async with provider.connect(tool_names=["search", "fetch"]) as tools:
                agent = MyAgent(tools=tools)
        """
        return _MCPConnectionContext(
            provider=self,
            filter_fn=filter_fn,
            tool_names=tool_names,
        )

    async def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        List available tools without creating MCPTool wrappers.

        Useful for inspecting what tools are available before connecting.

        Returns:
            List of tool metadata dicts with name, description, inputSchema
        """
        async with self.connect() as tools:
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]


class _MCPConnectionContext:
    """
    Async context manager for MCP connections.

    Handles connection lifecycle and tool discovery/filtering.
    """

    def __init__(
        self,
        provider: MCPToolProvider,
        filter_fn: Optional[Callable[[Any], bool]] = None,
        tool_names: Optional[List[str]] = None,
    ):
        self._provider = provider
        self._filter_fn = filter_fn
        self._tool_names = set(tool_names) if tool_names else None
        self._transport_context = None
        self._session_context = None
        self._session = None

    async def __aenter__(self) -> List[MCPTool]:
        """Connect to MCP server and return discovered tools."""
        from mcp import ClientSession

        # Connect based on transport type
        transport_type = self._provider._config.transport
        if transport_type == MCPTransport.SSE:
            self._transport_context = self._provider._connect_sse()
        elif transport_type == MCPTransport.STDIO:
            self._transport_context = self._provider._connect_stdio()
        elif transport_type == MCPTransport.STREAMABLE_HTTP:
            self._transport_context = self._provider._connect_streamable_http()
        else:
            raise ValueError(f"Unknown transport type: {transport_type}")

        # Enter transport context
        # Streamable HTTP returns (read, write, get_session_id), SSE/stdio return (read, write)
        transport_result = await self._transport_context.__aenter__()
        if transport_type == MCPTransport.STREAMABLE_HTTP:
            read_stream, write_stream, _get_session_id = transport_result
        else:
            read_stream, write_stream = transport_result

        # Create and initialize session
        self._session = ClientSession(read_stream, write_stream)
        await self._session.__aenter__()
        await self._session.initialize()

        # Discover tools
        tools_response = await self._session.list_tools()
        tools = []

        for tool in tools_response.tools:
            # Apply name filter
            if self._tool_names and tool.name not in self._tool_names:
                continue

            # Apply custom filter
            if self._filter_fn and not self._filter_fn(tool):
                continue

            # Create MCPTool wrapper
            mcp_tool = MCPTool(
                session=self._session,
                tool_name=tool.name,
                tool_description=tool.description or "",
                input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else None,
            )
            tools.append(mcp_tool)

        return tools

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up connection resources."""
        # Close session
        if self._session:
            await self._session.__aexit__(exc_type, exc_val, exc_tb)

        # Close transport
        if self._transport_context:
            await self._transport_context.__aexit__(exc_type, exc_val, exc_tb)

        return False
