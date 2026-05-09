"""
JSON Patch (RFC 6902) helpers for AG-UI state deltas.

Thin layer over the `jsonpatch` library so the rest of echo.ag_ui (and
downstream consumers) don't need to know the implementation. All ops
returned/accepted here conform to RFC 6902 — the same shape AG-UI's
`STATE_DELTA` event carries.
"""

from typing import Any

import jsonpatch

# A single JSON Patch operation (RFC 6902).
# Shape: {"op": "add"|"remove"|"replace"|"move"|"copy"|"test", "path": str, ...}
JsonPatchOp = dict


def diff_to_ops(old: dict, new: dict) -> list[JsonPatchOp]:
    """Compute JSON Patch ops to transform `old` into `new`.

    Returns an empty list if the two states are equal.
    """
    return list(jsonpatch.make_patch(old, new).patch)


def apply_ops(state: dict, ops: list[JsonPatchOp]) -> dict:
    """Apply a list of JSON Patch ops to `state`.

    Returns a new dict; does not mutate the input.
    """
    return jsonpatch.apply_patch(state, ops, in_place=False)


def make_replace(path: str, value: Any) -> JsonPatchOp:
    return {"op": "replace", "path": path, "value": value}


def make_add(path: str, value: Any) -> JsonPatchOp:
    return {"op": "add", "path": path, "value": value}


def make_remove(path: str) -> JsonPatchOp:
    return {"op": "remove", "path": path}


def append_op(list_path: str, value: Any) -> JsonPatchOp:
    """Op that appends `value` to the list at `list_path` (RFC 6902 `path/-`)."""
    return {"op": "add", "path": f"{list_path.rstrip('/')}/-", "value": value}
