"""
Base state container for AG-UI integrations.

Subclasses are regular Pydantic models holding domain state (e.g.
ScribeState in voice2rx). The host (typically AgUiRunner) calls
begin_tracking() once after the initial STATE_SNAPSHOT is emitted, and
drain_pending_ops() between LLM events to compute STATE_DELTA frames.
"""

from typing import Any

from pydantic import BaseModel, PrivateAttr

from .delta import (
    JsonPatchOp,
    append_op,
    apply_ops,
    diff_to_ops,
    make_remove,
    make_replace,
)


class AgUiState(BaseModel):
    """Pydantic base for state that streams over AG-UI as STATE_SNAPSHOT
    + STATE_DELTA frames.

    Two ways to mutate state:

    1. Direct Pydantic mutation, then drain_pending_ops() returns the
       JSON Patch ops since begin_tracking() (or the previous drain).
       Recommended — natural, fully typed.

           state.sections.append(new_section)
           state.sections[0].status.state = "ready"
           ops = state.drain_pending_ops()

    2. Explicit ops via set_path() / append_at() / remove_at(). The
       change is applied to state and tracked by the same diff. Useful
       when you already have JSON Pointer paths in hand.

           state.append_at("/sections", new_section.model_dump(mode="json"))
           state.set_path("/sections/0/status/state", "ready")

    Both styles can be mixed in one run; drain returns the combined diff.
    """

    # Baseline snapshot used to compute deltas. None until begin_tracking()
    # is called.
    _baseline: dict | None = PrivateAttr(default=None)

    def begin_tracking(self) -> None:
        """Snapshot current state as the diff baseline.

        Subsequent drain_pending_ops() calls return ops since this baseline.
        Idempotent — calling again re-baselines to the current state.
        """
        self._baseline = self.snapshot()

    def stop_tracking(self) -> None:
        """Disable tracking. drain_pending_ops() will return [] until
        begin_tracking() is called again."""
        self._baseline = None

    def drain_pending_ops(self) -> list[JsonPatchOp]:
        """Compute JSON Patch ops from baseline to current state.

        Re-baselines so the next drain returns only ops since this drain.
        Returns [] when tracking is off or no changes have occurred.
        """
        if self._baseline is None:
            return []
        current = self.snapshot()
        ops = diff_to_ops(self._baseline, current)
        self._baseline = current
        return ops

    def snapshot(self) -> dict:
        """Current state as a JSON-serializable dict (model_dump mode='json')."""
        return self.model_dump(mode="json")

    # --- explicit op helpers ---

    def set_path(self, path: str, value: Any) -> None:
        """Apply a 'replace' op at `path`. Re-validates via Pydantic."""
        self._apply_and_reload([make_replace(path, value)])

    def append_at(self, list_path: str, value: Any) -> None:
        """Append `value` to the list at `list_path`."""
        self._apply_and_reload([append_op(list_path, value)])

    def remove_at(self, path: str) -> None:
        """Remove the value at `path`."""
        self._apply_and_reload([make_remove(path)])

    def _apply_and_reload(self, ops: list[JsonPatchOp]) -> None:
        new_snap = apply_ops(self.snapshot(), ops)
        validated = type(self).model_validate(new_snap)
        for field_name in type(self).model_fields:
            setattr(self, field_name, getattr(validated, field_name))
