"""MCP tool framework: wrap arbitrary MCP servers as `BaseTool`s.

MCP is generic tool *infrastructure* (it wraps any server), so it lives in
`tools/` rather than a business domain. Depends only on `base_tool` and
`core`.
"""

from .connection_manager import MCPConnectionManager
from .mcp_tool import MCPTool

__all__ = ["MCPConnectionManager", "MCPTool"]
