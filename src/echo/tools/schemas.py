from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


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
    Configuration for connecting to an MCP server.

    For SSE transport:
        config = MCPServerConfig(
            transport=MCPTransport.SSE,
            url="http://localhost:8000/sse",
            headers={"Authorization": "Bearer token"}
        )

    For Streamable HTTP transport (HTTP POST with JSON-RPC):
        config = MCPServerConfig(
            transport=MCPTransport.STREAMABLE_HTTP,
            url="http://localhost:8000/mcp/",
            headers={"Authorization": "Bearer token"}
        )

    For stdio transport:
        config = MCPServerConfig(
            transport=MCPTransport.STDIO,
            command="python",
            args=["my_mcp_server.py"],
            env={"API_KEY": "secret"}
        )
    """

    transport: MCPTransport = MCPTransport.SSE

    # SSE options
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    timeout: float = 5.0
    sse_read_timeout: float = 300.0

    # stdio options
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None

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
