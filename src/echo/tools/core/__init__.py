"""Core tool framework: the foundation every tool category builds on.

Holds the universal contract (`BaseTool`) and cross-category schemas
(tool output, elicitation payloads). This package is the dependency root of
`tools/` — it imports nothing else from `echo`. Tool categories
(`elicitation`, `mcp`, `system`) and domain tools all point inward to here.
"""

from .base_tool import BaseTool
from .schemas import (
    ControlFlow,
    ElicitationComponent,
    ElicitationDetails,
    ElicitationResponse,
    ElicitationStatus,
    Observability,
    ToolOutput,
)

__all__ = [
    "BaseTool",
    "ControlFlow",
    "ElicitationComponent",
    "ElicitationDetails",
    "ElicitationResponse",
    "ElicitationStatus",
    "Observability",
    "ToolOutput",
]
