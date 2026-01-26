from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, HttpUrl, field_serializer


# MCP Exception Classes
class MCPError(Exception):
    """Base exception for MCP operations."""

    pass


class MCPConfigError(MCPError):
    """Configuration errors (no retry)."""

    pass


class MCPConnectionError(MCPError):
    """Connection failures (retryable)."""

    pass


class MCPExecutionError(MCPError):
    """Tool execution failures after all retries."""

    pass


class ElicitationComponent(str, Enum):
    """Types of elicitation UI components."""

    pass


class ElicitationDetails(BaseModel):
    """Structured response from elicitation tools."""

    component: str  # Accept any string enum value for flexibility with subclasses
    input: Dict[str, Any]
    _meta: Optional[Dict[str, Any]] = None


class ElicitationResponse(BaseModel):
    """Structured response from elicitation tools."""

    tool_id: str
    tool_name: str
    details: ElicitationDetails
    error: Optional[str] = None


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
    sse_read_timeout: int = 300

    # stdio options
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None

    # Connection management
    connection_ttl: float = 600.0  # TTL for cleanup (10 minutes)

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
