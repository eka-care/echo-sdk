"""Tool framework for Echo SDK.

This package exposes the tool *framework* — base types, MCP infrastructure,
and shared schemas. Concrete *domain* tools live with their domain and import
this framework (one-way dependency):

- skill tools  → ``echo.skills``        (``LoadSkillTool``, ``UnloadSkillTool``)
- postgres tool → ``echo.databases.postgres`` (``PgQueryTool``)

`tools/__init__` deliberately imports nothing from `skills` or `databases`,
keeping the dependency graph acyclic.
"""

from .base_elicitation_tool import BaseElicitationTool
from .base_tool import BaseTool
from .core.schemas import (
    ElicitationDetails,
    ElicitationResponse,
    ElicitationStatus,
    MCPConfigError,
    MCPConnectionError,
    MCPError,
    MCPExecutionError,
    MCPServerConfig,
    MCPTransport,
)
from .mcp import MCPConnectionManager, MCPTool

__all__ = [
    "BaseElicitationTool",
    "BaseTool",
    "MCPConnectionManager",
    "MCPServerConfig",
    "MCPTransport",
    "MCPTool",
    "MCPError",
    "MCPConfigError",
    "MCPConnectionError",
    "MCPExecutionError",
    "ElicitationDetails",
    "ElicitationStatus",
    "ElicitationResponse",
]
