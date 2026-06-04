"""MCP tool framework: wrap arbitrary MCP servers as `BaseTool`s.

MCP is generic tool *infrastructure* (it wraps any server), so it lives in
`tools/` rather than a business domain. Owns its category-private schemas
(transport, server config, error hierarchy); cross-category types live in
`tools/core`. Depends only on `core`.

Importing this package pulls in the optional `mcp`/`httpx` deps — which is
why the top-level `echo.tools` does NOT re-export it. Import from
`echo.tools.mcp` directly when you need MCP.
"""

from .connection_manager import MCPConnectionManager
from .mcp_tool import MCPTool
from .schemas import (
    MCPConfigError,
    MCPConnectionError,
    MCPError,
    MCPExecutionError,
    MCPServerConfig,
    MCPTransport,
)

__all__ = [
    "MCPConnectionManager",
    "MCPTool",
    "MCPConfigError",
    "MCPConnectionError",
    "MCPError",
    "MCPExecutionError",
    "MCPServerConfig",
    "MCPTransport",
]
