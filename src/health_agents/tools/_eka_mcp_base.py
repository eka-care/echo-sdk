from __future__ import annotations

from typing import Callable, List, Optional

from echo.tools.mcp_connection_manager import MCPConnectionManager
from echo.tools.mcp_tool import MCPTool
from echo.tools.schemas import MCPServerConfig, MCPTransport


class EkaMcpClient:
    def __init__(
        self,
        bearer_token: str,
        *,
        url: str,
        timeout: int = 30,
        sse_read_timeout: Optional[int] = None,
    ) -> None:
        if not bearer_token:
            raise ValueError("bearer_token is required to reach the Eka MCP server.")
        self.url = url
        config_kwargs = dict(
            transport=MCPTransport.STREAMABLE_HTTP,
            url=url,
            headers={"Authorization": f"Bearer {bearer_token}"},
            timeout=timeout,
        )
        if sse_read_timeout is not None:
            config_kwargs["sse_read_timeout"] = sse_read_timeout
        self._config = MCPServerConfig(**config_kwargs)
        self.manager = MCPConnectionManager(self._config)

    async def get_tools(
        self,
        *,
        tool_names: Optional[List[str]] = None,
        filter_fn: Optional[Callable[[MCPTool], bool]] = None,
    ) -> List[MCPTool]:
        return await self.manager.get_tools(filter_fn=filter_fn, tool_names=tool_names)
