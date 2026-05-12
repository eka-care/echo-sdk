"""Paused-run persistence for AG-UI runs awaiting a UI-tool response."""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class PausedRun:
    """Snapshot of an agent run paused awaiting a UI-tool response."""

    thread_id: str
    run_id: str
    tool_call_id: str
    tool_call_name: str
    tool_args: dict
    context_snapshot: dict
    state_snapshot: dict
    metadata: dict = field(default_factory=dict)


def make_pause_key(thread_id: str, run_id: str) -> str:
    """Canonical key format for paused-run storage."""
    return f"ag_ui:paused_run:{thread_id}:{run_id}"


class PausedRunStore(Protocol):
    """Storage protocol for paused runs (save/load/delete with optional TTL)."""

    async def save(self, key: str, snapshot: PausedRun, ttl: int = 1800) -> None: ...

    async def load(self, key: str) -> Optional[PausedRun]: ...

    async def delete(self, key: str) -> None: ...


class InMemoryPausedRunStore:
    """In-process PausedRunStore for tests and single-worker dev. Not for production."""

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
