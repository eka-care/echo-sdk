from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, HttpUrl, field_serializer


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


@dataclass
class ToolOutput:
    """Structured return type for tools that want to attach metadata."""

    result: str
    meta: Optional[Dict[str, Any]] = field(default=None)


class ElicitationComponent(str, Enum):
    """Types of elicitation UI components."""

    pass


class ElicitationStatus(str, Enum):
    """Status for elicitation tools to facilitate callback"""

    IN_PROGRESS = "progress"
    DONE = "success"
    ERROR = "failure"


class ElicitationDetails(BaseModel):
    """Structured response from elicitation tools."""

    model_config = {"populate_by_name": True}

    component: str  # Accept any string enum value for flexibility with subclasses
    input: Dict[str, Any]
    meta: Optional[Dict[str, Any]] = Field(default=None, alias="_meta")
    status: Optional[ElicitationStatus] = None
    hidden_message: Optional[str] = None
    disp_toast_msg: Optional[str] = None
    mcp_meta_fields: Optional[list[str]] = None


class ElicitationResponse(BaseModel):
    """Structured response from elicitation tools."""

    tool_type: str = "elicitation"
    tool_id: str
    tool_name: str
    details: ElicitationDetails
    meta: Optional[Dict[str, Any]] = None
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
    connection_ttl: int = 600  # TTL for cleanup (10 minutes)

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
