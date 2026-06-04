"""MCP-specific schemas: transport, server config, and error hierarchy.

These are MCP-private (consumed by the mcp tool/connection manager and by
hosts configuring MCP servers). Cross-category schemas (directive/result
base, elicitation types) live in `tools/core` instead.
"""

import os
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_serializer


def _default_mcp_tool_timeout() -> int:
    """Read ECHO_MCP_TOOL_TIMEOUT from env, fall back to 10s."""
    try:
        return int(os.getenv("ECHO_MCP_TOOL_TIMEOUT", "10"))
    except ValueError:
        return 10


# MCP Exception Classes
class MCPError(Exception):
    """Base exception for MCP operations."""

    pass


class MCPConfigError(MCPError):
    """Configuration errors."""

    pass


class MCPConnectionError(MCPError):
    """Connection failures (e.g., transport or initialize failed)."""

    pass


class MCPExecutionError(MCPError):
    """Tool execution failures."""

    pass


class MCPTransport(str, Enum):
    """MCP transport type."""

    SSE = "sse"
    STDIO = "stdio"
    STREAMABLE_HTTP = (
        "streamable_http"  # For servers that use HTTP POST with optional SSE responses
    )


class MCPServerConfig(BaseModel):
    """
    Configuration for connecting to an MCP server. Find examples below.
    """

    transport: MCPTransport = MCPTransport.SSE

    # SSE options
    url: Optional[HttpUrl] = None
    headers: Optional[Dict[str, str]] = None
    timeout: int = 5
    sse_read_timeout: int = Field(default_factory=_default_mcp_tool_timeout)

    # stdio options
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None

    # Tool filtering (optional)
    tool_include: Optional[List[str]] = None  # Whitelist: only these tools available
    tool_exclude: Optional[List[str]] = None  # Blacklist: exclude these tools

    def validate(self) -> None:
        """Validate configuration based on transport type."""
        if self.transport == MCPTransport.SSE:
            if not self.url:
                raise ValueError("SSE transport requires 'url'")
        elif self.transport == MCPTransport.STREAMABLE_HTTP:
            if not self.url:
                raise ValueError("Streamable HTTP transport requires 'url'")
        elif self.transport == MCPTransport.STDIO:
            if not self.command:
                raise ValueError("stdio transport requires 'command'")

    @field_serializer("url")
    def serialize_url(self, url: HttpUrl) -> str:
        return str(url)
