"""
Tool dispatcher for AG-UI runs.

Classifies tool calls during a run as either:

  * backend tools — defined in the agent's tool list and executed
    server-side by echo-sdk's normal LLM loop. Their results stay on
    the server (or are surfaced to the FE only as a redacted ack).

  * UI tools — declared by the frontend in RunAgentInput.tools and
    executed in the browser. The agent emits TOOL_CALL_* events and
    pauses; the FE renders, the user responds, and a /resume call
    feeds the result back into the agent.

PR-S3 scope: classification + StreamEvent → AG-UI event translation +
PauseSignal surfacing. PR-S4 wires the pause signal into the public
BaseAgent.run_stream_with_ag_ui() entry point and the PausedRunStore.
PR-S5 enables streaming TOOL_CALL_ARGS deltas from the Anthropic provider.
"""

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
    """Surfaces a UI tool call that the runner should pause on.

    Consumed by the run-stream entry point (PR-S4): on receipt, the host
    persists (ConversationContext, AgUiState, run_id, tool_call_id) to
    a PausedRunStore and exits the SSE generator. A subsequent /resume
    request rehydrates and continues.
    """

    tool_call_id: str
    tool_call_name: str
    tool_args: dict


@dataclass
class _ToolCallState:
    """Internal per-call bookkeeping."""

    tool_call_id: str
    tool_call_name: str
    is_ui_tool: bool
    args_buffer: str = ""  # accumulated input_json fragments (PR-S5)
    args_parsed: dict = field(default_factory=dict)
    ended: bool = False


class AgUiToolDispatcher:
    """Tracks tool calls during a run and emits AG-UI tool events.

    Construct once per run. Passed to AgUiRunner. UI tool names come
    from RunAgentInput.tools at run start; the host calls
    `register_ui_tools([t.name for t in run_input.tools])` before the
    runner begins streaming.

    The dispatcher is duck-typed at the StreamEvent boundary — it does
    not need echo-sdk's BaseTool to function. That makes it directly
    unit-testable.
    """

    def __init__(self, ui_tool_names: Optional[set[str]] = None) -> None:
        self._ui_tool_names: set[str] = set(ui_tool_names or [])
        self._calls: dict[str, _ToolCallState] = {}

    # --- registration / classification ---

    def register_ui_tools(self, tool_names: List[str]) -> None:
        """Mark these tool names as FE-declared UI tools.

        Idempotent. Safe to call mid-run, though normal usage is
        once at runner construction time.
        """
        self._ui_tool_names.update(tool_names)

    def is_ui_tool(self, tool_name: str) -> bool:
        return tool_name in self._ui_tool_names

    def call_classification(self, tool_call_id: str) -> Optional[str]:
        """Return 'ui' or 'backend' for a tracked tool_call_id, else None."""
        state = self._calls.get(tool_call_id)
        if state is None:
            return None
        return "ui" if state.is_ui_tool else "backend"

    # --- StreamEvent translation ---

    def translate(self, sev: StreamEvent) -> List[BaseEvent]:
        """Translate a tool-related StreamEvent to AG-UI events.

        Returns [] for non-tool events. The runner is responsible for
        dispatching only TOOL_CALL_* events here; this method tolerates
        anything but emits nothing for non-tool kinds.
        """
        if sev.type == StreamEventType.TOOL_CALL_START:
            details = sev.details or {}
            tool_id = str(details.get("tool_id", ""))
            tool_name = str(details.get("tool_name", ""))
            # Track the call regardless of how it ends.
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

    # --- args streaming (PR-S5 will call append_args_delta) ---

    def append_args_delta(
        self, tool_call_id: str, delta: str
    ) -> Optional[ToolCallArgsEvent]:
        """Append a streaming args fragment and emit ToolCallArgsEvent.

        Returns None if the tool_call_id is unknown (e.g. delta arrived
        before TOOL_CALL_START — should not happen with a well-behaved
        provider, but we degrade gracefully rather than raise).
        """
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
        """Record the full parsed args for a tool call.

        Call when args are fully known (echo-sdk's Anthropic provider
        knows them by content_block_stop). PR-S5 calls this; PR-S3 also
        accepts callers that already have the parsed args.
        """
        state = self._calls.get(tool_call_id)
        if state is not None:
            state.args_parsed = dict(args)

    # --- pause signal ---

    def pending_ui_calls(self) -> List[_ToolCallState]:
        """All recorded UI tool calls (for diagnostics / multi-call cases)."""
        return [s for s in self._calls.values() if s.is_ui_tool]

    def consume_pause_signal(self) -> Optional[PauseSignal]:
        """Return a PauseSignal for the first recorded UI tool call, or None.

        PR-S3 surfaces only the first one. Real-world Anthropic flows can
        emit multiple tool_use blocks per turn; PR-S4 may extend this to
        return all pending UI calls if FE round-tripping for several at
        once becomes a need.
        """
        for state in self._calls.values():
            if state.is_ui_tool:
                return PauseSignal(
                    tool_call_id=state.tool_call_id,
                    tool_call_name=state.tool_call_name,
                    tool_args=dict(state.args_parsed),
                )
        return None

    # --- backend result emission ---

    def emit_backend_result_ack(
        self, tool_call_id: str, message_id: str, content: str = "(server-executed)"
    ) -> Optional[ToolCallResultEvent]:
        """Emit a redacted TOOL_CALL_RESULT for a backend tool call.

        Backend tool results may carry secrets / PII / EMR data; we
        default to a short string ack so the FE sees the call completed
        without exposing the payload. Hosts can pass `content=` to
        substitute a custom ack string.

        Returns None when the classification isn't 'backend' (UI tool
        results come from the FE via /resume — the runner shouldn't
        emit them).
        """
        if self.call_classification(tool_call_id) != "backend":
            return None
        return ToolCallResultEvent(
            type=EventType.TOOL_CALL_RESULT,
            message_id=message_id,
            tool_call_id=tool_call_id,
            content=content,
            role="tool",
        )
