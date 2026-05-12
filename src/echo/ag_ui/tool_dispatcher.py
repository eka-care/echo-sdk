"""Tool dispatcher for AG-UI runs: classifies tool calls as backend or UI and emits AG-UI events."""

from dataclasses import dataclass, field
from typing import List, Optional

from ag_ui.core import (
    BaseEvent,
    EventType,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

from echo.llm.schemas import StreamEvent, StreamEventType


@dataclass
class PauseSignal:
    """Surfaces a UI tool call that the runner should pause on."""

    tool_call_id: str
    tool_call_name: str
    tool_args: dict


@dataclass
class _ToolCallState:
    """Internal per-call bookkeeping."""

    tool_call_id: str
    tool_call_name: str
    is_ui_tool: bool
    args_buffer: str = ""
    args_parsed: dict = field(default_factory=dict)
    ended: bool = False


class AgUiToolDispatcher:
    """Tracks tool calls during a run and emits AG-UI tool events. Constructed once per run."""

    def __init__(self, ui_tool_names: Optional[set[str]] = None) -> None:
        self._ui_tool_names: set[str] = set(ui_tool_names or [])
        self._calls: dict[str, _ToolCallState] = {}

    def register_ui_tools(self, tool_names: List[str]) -> None:
        """Mark these tool names as FE-declared UI tools (idempotent)."""
        self._ui_tool_names.update(tool_names)

    def is_ui_tool(self, tool_name: str) -> bool:
        return tool_name in self._ui_tool_names

    def call_classification(self, tool_call_id: str) -> Optional[str]:
        """Return 'ui' or 'backend' for a tracked tool_call_id, else None."""
        state = self._calls.get(tool_call_id)
        if state is None:
            return None
        return "ui" if state.is_ui_tool else "backend"

    def translate(self, sev: StreamEvent) -> List[BaseEvent]:
        """Translate a tool-related StreamEvent to AG-UI events; returns [] for non-tool events."""
        if sev.type == StreamEventType.TOOL_CALL_START:
            details = sev.details or {}
            tool_id = str(details.get("tool_id", ""))
            tool_name = str(details.get("tool_name", ""))
            # track the call regardless of how it ends.
            self._calls[tool_id] = _ToolCallState(
                tool_call_id=tool_id,
                tool_call_name=tool_name,
                is_ui_tool=self.is_ui_tool(tool_name),
            )
            return [
                ToolCallStartEvent(
                    type=EventType.TOOL_CALL_START,
                    tool_call_id=tool_id,
                    tool_call_name=tool_name,
                )
            ]

        if sev.type == StreamEventType.TOOL_CALL_ARGS:
            details = sev.details or {}
            tool_id = str(details.get("tool_id", ""))
            delta = str(details.get("delta", ""))
            args_ev = self.append_args_delta(tool_id, delta)
            return [args_ev] if args_ev is not None else []

        if sev.type == StreamEventType.TOOL_CALL_END:
            details = sev.details or {}
            tool_id = str(details.get("tool_id", ""))
            state = self._calls.get(tool_id)
            if state:
                state.ended = True
            return [
                ToolCallEndEvent(
                    type=EventType.TOOL_CALL_END,
                    tool_call_id=tool_id,
                )
            ]

        return []

    def append_args_delta(
        self, tool_call_id: str, delta: str
    ) -> Optional[ToolCallArgsEvent]:
        """Append a streaming args fragment and emit ToolCallArgsEvent; returns None if id unknown."""
        state = self._calls.get(tool_call_id)
        if state is None:
            return None
        state.args_buffer += delta
        return ToolCallArgsEvent(
            type=EventType.TOOL_CALL_ARGS,
            tool_call_id=tool_call_id,
            delta=delta,
        )

    def record_tool_args(self, tool_call_id: str, args: dict) -> None:
        """Record the full parsed args for a tool call once they are fully known."""
        state = self._calls.get(tool_call_id)
        if state is not None:
            state.args_parsed = dict(args)

    def register_completed_call(
        self,
        tool_call_id: str,
        tool_call_name: str,
        is_ui_tool: bool,
        args: dict,
    ) -> None:
        """Register a tool call after-the-fact with full args known (used for elicitation flow)."""
        self._calls[tool_call_id] = _ToolCallState(
            tool_call_id=tool_call_id,
            tool_call_name=tool_call_name,
            is_ui_tool=is_ui_tool,
            args_parsed=dict(args),
            ended=True,
        )

    def pending_ui_calls(self) -> List[_ToolCallState]:
        """All recorded UI tool calls (for diagnostics / multi-call cases)."""
        return [s for s in self._calls.values() if s.is_ui_tool]

    def consume_pause_signal(self) -> Optional[PauseSignal]:
        """Return a PauseSignal for the first recorded UI tool call, or None."""
        for state in self._calls.values():
            if state.is_ui_tool:
                return PauseSignal(
                    tool_call_id=state.tool_call_id,
                    tool_call_name=state.tool_call_name,
                    tool_args=dict(state.args_parsed),
                )
        return None

    def emit_backend_result_ack(
        self, tool_call_id: str, message_id: str, content: str = "(server-executed)"
    ) -> Optional[ToolCallResultEvent]:
        """Emit a redacted TOOL_CALL_RESULT for a backend tool call; returns None for non-backend calls."""
        if self.call_classification(tool_call_id) != "backend":
            return None
        return ToolCallResultEvent(
            type=EventType.TOOL_CALL_RESULT,
            message_id=message_id,
            tool_call_id=tool_call_id,
            content=content,
            role="tool",
        )
