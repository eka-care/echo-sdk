"""
AG-UI integration for Echo SDK.

Bridges echo-sdk's agent runtime (StreamEvent-based) to the Agent-User
Interaction Protocol (AG-UI) — a typed, event-based protocol for streaming
structured agent state and tool interactions to a user-facing application
over SSE or WebSocket transports.

Public surface (filled in across PR-S1..PR-S4):
    AgUiState               — Pydantic base for any state that streams over AG-UI
    AgUiRunInput            — request body for an AG-UI run
    AgUiToolDef             — tool declaration shape (matches AG-UI spec)
    AgUiRunner              — drives an agent.run_stream() and emits AG-UI events
    PausedRunStore          — protocol for persisting paused-for-UI-tool runs
    InMemoryPausedRunStore  — default in-process implementation

See echo.agents.base.BaseAgent.run_stream_with_ag_ui() for the entry point.
"""

from .persistence import (
    InMemoryPausedRunStore,
    PausedRun,
    PausedRunStore,
    make_pause_key,
)
from .runner import AgUiRunner
from .schemas import AgUiResumeInput
from .state import AgUiState
from .tool_dispatcher import AgUiToolDispatcher, PauseSignal

__all__ = [
    "AgUiState",
    "AgUiRunner",
    "AgUiToolDispatcher",
    "PauseSignal",
    "PausedRun",
    "PausedRunStore",
    "InMemoryPausedRunStore",
    "make_pause_key",
    "AgUiResumeInput",
]
