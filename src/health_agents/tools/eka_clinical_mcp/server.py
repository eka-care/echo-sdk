from __future__ import annotations

from typing import Callable, List, Optional

from echo.tools.mcp_tool import MCPTool

from .._eka_mcp_base import EkaMcpClient

#: Default streamable-HTTP endpoint for the Eka clinical tools MCP server.
EKA_CLINICAL_MCP_URL = "https://medai-tools.eka.care/mcp"


class EkaClinicalMcpClient(EkaMcpClient):
    def __init__(
        self,
        bearer_token: str,
        *,
        url: str = EKA_CLINICAL_MCP_URL,
        timeout: int = 30,
        sse_read_timeout: Optional[int] = None,
    ) -> None:
        super().__init__(
            bearer_token,
            url=url,
            timeout=timeout,
            sse_read_timeout=sse_read_timeout,
        )


async def get_eka_clinical_tools(
    bearer_token: str,
    *,
    url: str = EKA_CLINICAL_MCP_URL,
    tool_names: Optional[List[str]] = None,
    filter_fn: Optional[Callable[[MCPTool], bool]] = None,
    timeout: int = 30,
) -> List[MCPTool]:
    client = EkaClinicalMcpClient(bearer_token, url=url, timeout=timeout)
    return await client.get_tools(tool_names=tool_names, filter_fn=filter_fn)
