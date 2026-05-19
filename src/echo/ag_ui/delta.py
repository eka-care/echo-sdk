"""JSON Patch (RFC 6902) helpers for AG-UI state deltas."""

from typing import Any

import jsonpatch

JsonPatchOp = dict


def diff_to_ops(old: dict, new: dict) -> list[JsonPatchOp]:
    """Compute JSON Patch ops to transform `old` into `new`."""
    return list(jsonpatch.make_patch(old, new).patch)


def apply_ops(state: dict, ops: list[JsonPatchOp]) -> dict:
    """Apply JSON Patch ops to `state`, returning a new dict."""
    return jsonpatch.apply_patch(state, ops, in_place=False)


def make_replace(path: str, value: Any) -> JsonPatchOp:
    return {"op": "replace", "path": path, "value": value}


def make_add(path: str, value: Any) -> JsonPatchOp:
    return {"op": "add", "path": path, "value": value}


def make_remove(path: str) -> JsonPatchOp:
    return {"op": "remove", "path": path}


def append_op(list_path: str, value: Any) -> JsonPatchOp:
    """Op that appends `value` to the list at `list_path`."""
    return {"op": "add", "path": f"{list_path.rstrip('/')}/-", "value": value}
