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


class ControlFlow(str, Enum):
    """What the agentic loop should do after a tool runs.

    Read off a tool result by the provider loop (never via isinstance). The
    set is intentionally extensible — e.g. a terminal ``STOP`` can be added
    later without touching the loop, which dispatches on the value.

    - ``CONTINUE``  — feed the result back and keep looping (default).
    - ``INTERRUPT`` — the tool changed the agent's loaded state (active
      skills, history, ...); break so the agent can recompute the prompt +
      tool list and re-invoke. Only echo-internal ``SystemTool``s emit this.
    - ``PAUSE``     — stop and return to the user (elicitation).
    """

    CONTINUE = "continue"
    INTERRUPT = "interrupt"
    PAUSE = "pause"


class Observability(str, Enum):
    """Whether executing a tool emits a user-facing event.

    - ``VISIBLE`` — emit TOOL_CALL_* events (default; normal tools).
    - ``SILENT``  — suppress events (e.g. internal/system tools the user
      need not see).
    """

    VISIBLE = "visible"
    SILENT = "silent"


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

    # Directives are FIXED for elicitation and cannot be overridden — exposed
    # as read-only properties (not fields) so there is no way to construct an
    # elicitation that doesn't pause/stay visible. The loop reads these the
    # same way it reads ToolResult's settable fields.
    @property
    def control_flow(self) -> "ControlFlow":
        return ControlFlow.PAUSE

    @property
    def observability(self) -> "Observability":
        return Observability.VISIBLE
