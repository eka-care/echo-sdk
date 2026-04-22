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


class ElicitationResponse(BaseModel):
    """Structured response from elicitation tools."""

    tool_type: str = "elicitation"
    tool_id: str
    tool_name: str
    details: ElicitationDetails
    meta: Optional[Dict[str, Any]] = None


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

    # Session cache management (only applies when caller passes user_session_id
    # to execute_tool; fresh-session-per-call path is unaffected).
    session_idle_ttl: int = 600
    """Evict cached session after this many seconds since last use."""

    session_absolute_ttl: int = 3600
    """Hard cap on cached session lifetime from first connect. Guards against
    sessions outliving the caller's auth-token lifetime."""

    # Tool-schema cache keying.
    # Header names (case-insensitive) whose values participate in the tool
    # cache key alongside (transport, url). Default None/empty → tools from
    # the same URL are shared across all callers. Set this when the same URL
    # returns different tool catalogues for different header values — e.g.
    # tool_cache_key_headers=["x-workspace-id"] when each workspace has its
    # own tool set.
    tool_cache_key_headers: Optional[List[str]] = None

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
