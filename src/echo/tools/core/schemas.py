"""Core tool schemas: shared types every tool category may depend on.

Holds the cross-category types (tool output, elicitation payloads). MCP-
private schemas live in `tools/mcp/schemas.py`. Elicitation types live here
(not in `tools/elicitation/`) because they are consumed by more than one
category — the elicitation tools, the MCP wrapper (which can surface MCP
elicitations), and the LLM providers — so they sit at the common ancestor.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


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
