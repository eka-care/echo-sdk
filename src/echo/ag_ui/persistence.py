"""
Paused-run persistence for AG-UI runs.

When an agent calls a UI tool (one declared by the FE in
RunAgentInput.tools), the runner pauses: it emits the TOOL_CALL_*
events, persists the in-flight state, and exits the SSE generator
without emitting RUN_FINISHED. The FE renders the prompt, the user
responds, and a /resume request continues the run.

This module defines:

  PausedRun           — serialized snapshot of a paused run
  PausedRunStore      — Protocol for the storage backend
  InMemoryPausedRunStore — default in-process store, for tests / dev

The voice2rx-be repo provides a Redis-backed implementation in PR-V10.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class PausedRun:
    """Snapshot of an agent run paused awaiting a UI-tool response.

    Fields:
        thread_id, run_id        — identify the paused run
        tool_call_id             — the UI tool call awaiting a response
        tool_call_name           — the UI tool's name
        tool_args                — the args the agent passed to the UI tool
        context_snapshot         — ConversationContext.model_dump(mode="json")
        state_snapshot           — AgUiState.snapshot()
        metadata                 — host-provided dict (b_id, document_id, etc.)
    """

    thread_id: str
    run_id: str
    tool_call_id: str
    tool_call_name: str
    tool_args: dict
    context_snapshot: dict
    state_snapshot: dict
    metadata: dict = field(default_factory=dict)


def make_pause_key(thread_id: str, run_id: str) -> str:
    """Canonical key format for paused-run storage.

    Backends should use this so multiple deployments hash consistently.
    """
    return f"ag_ui:paused_run:{thread_id}:{run_id}"


class PausedRunStore(Protocol):
    """Storage for paused runs.

    Implementations may apply TTL (default 30 min). The runtime persists
    on pause, loads on resume, and deletes on clean completion.
    """

    async def save(self, key: str, snapshot: PausedRun, ttl: int = 1800) -> None: ...

    async def load(self, key: str) -> Optional[PausedRun]: ...

    async def delete(self, key: str) -> None: ...


class InMemoryPausedRunStore:
    """In-process PausedRunStore. For tests and single-worker dev.

    NOT suitable for production: state is lost on process restart and
    not shared across workers. Use a Redis-backed implementation in
    multi-worker deployments. TTL is accepted for API compatibility
    but not enforced (tests don't run long enough to matter).
    """

    def __init__(self) -> None:
        self._runs: dict[str, PausedRun] = {}
        self._lock = asyncio.Lock()

    async def save(self, key: str, snapshot: PausedRun, ttl: int = 1800) -> None:
        async with self._lock:
            self._runs[key] = snapshot

    async def load(self, key: str) -> Optional[PausedRun]:
        async with self._lock:
            return self._runs.get(key)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._runs.pop(key, None)
