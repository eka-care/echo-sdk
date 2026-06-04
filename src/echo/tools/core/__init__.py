"""Core tool framework: shared schemas and constants.

This package is the dependency root of `tools/` — it imports nothing else
from `echo`. Everything in `tools/` (and domain tools that extend the
framework) points inward to here. Keep cross-module schemas (directive
enums, elicitation details, server config, errors) in this package so no
two sibling tool modules need to import each other.
"""

from .schemas import (
    ElicitationComponent,
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

__all__ = [
    "ElicitationComponent",
    "ElicitationDetails",
    "ElicitationResponse",
    "ElicitationStatus",
    "MCPConfigError",
    "MCPConnectionError",
    "MCPError",
    "MCPExecutionError",
    "MCPServerConfig",
    "MCPTransport",
]
