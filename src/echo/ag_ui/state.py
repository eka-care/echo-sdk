"""Base state container for AG-UI integrations, streamed as STATE_SNAPSHOT + STATE_DELTA frames."""

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
    """Pydantic base for state streamed over AG-UI; supports direct mutation or explicit ops."""

    _baseline: dict | None = PrivateAttr(default=None)

    def begin_tracking(self) -> None:
        """Snapshot current state as the diff baseline."""
        self._baseline = self.snapshot()

    def stop_tracking(self) -> None:
        """Disable tracking; drain_pending_ops() returns [] until begin_tracking() is called again."""
        self._baseline = None

    def drain_pending_ops(self) -> list[JsonPatchOp]:
        """Compute JSON Patch ops from baseline to current state, then re-baseline."""
        if self._baseline is None:
            return []
        current = self.snapshot()
        ops = diff_to_ops(self._baseline, current)
        self._baseline = current
        return ops

    def snapshot(self) -> dict:
        """Current state as a JSON-serializable dict."""
        return self.model_dump(mode="json")

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
