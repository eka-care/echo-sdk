from __future__ import annotations

from typing import Callable, List, Optional

from echo.tools.mcp_tool import MCPTool

from .._eka_mcp_base import EkaMcpClient

#: Default streamable-HTTP endpoint for the Eka EMR MCP server.
EKA_EMR_MCP_URL = "https://mcp.eka.care/mcp"


class EkaEmrMcpClient(EkaMcpClient):
    def __init__(
        self,
        bearer_token: str,
        *,
        url: str = EKA_EMR_MCP_URL,
        timeout: int = 30,
        sse_read_timeout: Optional[int] = None,
    ) -> None:
        super().__init__(
            bearer_token,
            url=url,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
        )


async def get_eka_emr_tools(
    bearer_token: str,
    *,
    url: str = EKA_EMR_MCP_URL,
    tool_names: Optional[List[str]] = None,
    filter_fn: Optional[Callable[[MCPTool], bool]] = None,
    timeout: int = 30,
) -> List[MCPTool]:
    client = EkaEmrMcpClient(bearer_token, url=url, timeout=timeout)
    return await client.get_tools(tool_names=tool_names, filter_fn=filter_fn)
