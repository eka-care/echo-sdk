"""
Scaffolding smoke tests for the echo.ag_ui namespace (PR-S0).

These tests verify the module exists, is importable, and that the
upstream dependencies (ag-ui-protocol, jsonpatch) are installed and
import correctly. Real surface tests land in PR-S1 onward.
"""


def test_ag_ui_namespace_importable():
    """The echo.ag_ui package can be imported."""
    import echo.ag_ui  # noqa: F401


def test_ag_ui_protocol_dep_installed():
    """ag-ui-protocol Python SDK is installed."""
    from ag_ui.core import EventType, RunAgentInput  # noqa: F401
    from ag_ui.encoder import EventEncoder  # noqa: F401


def test_jsonpatch_dep_installed():
    """jsonpatch (RFC 6902) is installed."""
    import jsonpatch  # noqa: F401

    # Exercise it briefly so a broken install fails loudly.
    patch = jsonpatch.make_patch({"a": 1}, {"a": 2})
    assert list(patch.patch) == [{"op": "replace", "path": "/a", "value": 2}]
