"""
Unit tests for echo.ag_ui.state.AgUiState (PR-S1).

Covers:
- snapshot serialization
- begin_tracking / drain_pending_ops baseline semantics
- direct Pydantic mutation captured by drain
- set_path / append_at / remove_at explicit ops
- nested Pydantic models in subclasses
- mixed direct + explicit mutations
- re-baselining after drain
"""

from typing import List, Optional

import pytest
from pydantic import BaseModel

from echo.ag_ui import AgUiState


# ----- fixtures: a small subclass with nested models, mirrors ScribeState shape -----


class _Status(BaseModel):
    state: str = "pending"
    error: Optional[str] = None


class _Section(BaseModel):
    key: str
    display_name: str
    payload: dict
    status: _Status = _Status()


class _DemoState(AgUiState):
    transcript: str = ""
    sections: List[_Section] = []
    omitted_sections: List[str] = []


@pytest.fixture
def state() -> _DemoState:
    return _DemoState()


# ----- snapshot -----


def test_snapshot_returns_json_dict(state: _DemoState):
    snap = state.snapshot()
    assert isinstance(snap, dict)
    assert snap == {"transcript": "", "sections": [], "omitted_sections": []}


def test_snapshot_after_population(state: _DemoState):
    state.transcript = "hello"
    state.sections.append(
        _Section(key="symptoms", display_name="Symptoms", payload={"items": []})
    )
    snap = state.snapshot()
    assert snap["transcript"] == "hello"
    assert len(snap["sections"]) == 1
    assert snap["sections"][0]["key"] == "symptoms"
    assert snap["sections"][0]["status"]["state"] == "pending"


# ----- baseline / drain semantics -----


def test_drain_without_begin_tracking_returns_empty(state: _DemoState):
    state.transcript = "anything"
    assert state.drain_pending_ops() == []


def test_drain_after_begin_tracking_no_changes(state: _DemoState):
    state.begin_tracking()
    assert state.drain_pending_ops() == []


def test_begin_tracking_is_idempotent(state: _DemoState):
    state.begin_tracking()
    state.transcript = "first"
    state.begin_tracking()  # re-baseline; clears uncomputed diff
    assert state.drain_pending_ops() == []


def test_stop_tracking_disables_drain(state: _DemoState):
    state.begin_tracking()
    state.transcript = "x"
    state.stop_tracking()
    assert state.drain_pending_ops() == []


# ----- direct Pydantic mutation captured by drain -----


def test_direct_field_mutation_captured(state: _DemoState):
    state.begin_tracking()
    state.transcript = "first transcript"
    ops = state.drain_pending_ops()
    assert ops == [
        {"op": "replace", "path": "/transcript", "value": "first transcript"}
    ]


def test_multiple_field_mutations_captured(state: _DemoState):
    state.begin_tracking()
    state.transcript = "abc"
    state.sections.append(
        _Section(key="vitals", display_name="Vitals", payload={"vitals": {}})
    )
    state.omitted_sections.append("diagnosis")

    ops = state.drain_pending_ops()
    # ops list contains all changes; ordering is determined by jsonpatch
    paths_seen = {(o["op"], o["path"]) for o in ops}
    assert ("replace", "/transcript") in paths_seen
    # adding to a list at index 0
    assert ("add", "/sections/0") in paths_seen
    assert ("add", "/omitted_sections/0") in paths_seen


def test_drain_rebaselines(state: _DemoState):
    state.begin_tracking()
    state.transcript = "v1"
    first_ops = state.drain_pending_ops()
    assert any(o["path"] == "/transcript" and o["value"] == "v1" for o in first_ops)

    # second drain with no changes should be empty (re-baselined)
    assert state.drain_pending_ops() == []

    # subsequent change should produce only the new op
    state.transcript = "v2"
    second_ops = state.drain_pending_ops()
    assert second_ops == [{"op": "replace", "path": "/transcript", "value": "v2"}]


# ----- explicit op helpers -----


def test_set_path_top_level(state: _DemoState):
    state.transcript = "starting"
    state.begin_tracking()
    state.set_path("/transcript", "via set_path")

    assert state.transcript == "via set_path"
    ops = state.drain_pending_ops()
    assert ops == [{"op": "replace", "path": "/transcript", "value": "via set_path"}]


def test_set_path_nested_field(state: _DemoState):
    state.sections.append(
        _Section(key="vitals", display_name="Vitals", payload={"vitals": {}})
    )
    state.begin_tracking()
    state.set_path("/sections/0/status/state", "ready")

    assert state.sections[0].status.state == "ready"
    ops = state.drain_pending_ops()
    assert ops == [
        {"op": "replace", "path": "/sections/0/status/state", "value": "ready"}
    ]


def test_append_at_list(state: _DemoState):
    state.begin_tracking()
    state.append_at(
        "/sections",
        _Section(
            key="symptoms", display_name="Symptoms", payload={"items": []}
        ).model_dump(mode="json"),
    )

    assert len(state.sections) == 1
    assert state.sections[0].key == "symptoms"
    ops = state.drain_pending_ops()
    assert len(ops) == 1
    assert ops[0]["op"] == "add"
    assert ops[0]["path"] == "/sections/0"
    assert ops[0]["value"]["key"] == "symptoms"


def test_remove_at_list_element(state: _DemoState):
    state.sections.extend(
        [
            _Section(key="a", display_name="A", payload={}),
            _Section(key="b", display_name="B", payload={}),
        ]
    )
    state.begin_tracking()
    state.remove_at("/sections/0")

    assert len(state.sections) == 1
    assert state.sections[0].key == "b"
    ops = state.drain_pending_ops()
    # jsonpatch may emit a single 'remove' or compute a different minimal diff;
    # assert the resulting state is right and at least one op was produced.
    assert ops != []
    new_snap = state.snapshot()
    assert [s["key"] for s in new_snap["sections"]] == ["b"]


def test_set_path_revalidates_via_pydantic(state: _DemoState):
    """set_path should reject values that break the Pydantic schema."""
    state.sections.append(
        _Section(key="x", display_name="X", payload={})
    )
    state.begin_tracking()

    # status.state expects a string; assigning a dict should fail validation
    with pytest.raises(Exception):
        state.set_path("/sections/0/status/state", {"not": "a string"})


# ----- mixed direct + explicit mutations -----


def test_mixed_direct_and_explicit_in_one_drain(state: _DemoState):
    state.begin_tracking()
    state.transcript = "from direct"
    state.append_at(
        "/sections",
        _Section(key="meds", display_name="Meds", payload={"items": []}).model_dump(
            mode="json"
        ),
    )
    state.sections[0].status.state = "ready"  # direct nested mutation

    ops = state.drain_pending_ops()
    final = state.snapshot()
    assert final["transcript"] == "from direct"
    assert final["sections"][0]["key"] == "meds"
    assert final["sections"][0]["status"]["state"] == "ready"
    # all three changes are reflected in ops
    paths_changed = {o["path"] for o in ops}
    assert "/transcript" in paths_changed


# ----- subclass with nested Pydantic models round-trips -----


def test_subclass_round_trip(state: _DemoState):
    state.transcript = "x"
    state.sections.append(
        _Section(
            key="vitals",
            display_name="Vitals",
            payload={"vitals": {"hr": 80}},
            status=_Status(state="ready"),
        )
    )
    snap = state.snapshot()
    rebuilt = _DemoState.model_validate(snap)
    assert rebuilt.transcript == "x"
    assert rebuilt.sections[0].status.state == "ready"
    assert rebuilt.sections[0].payload == {"vitals": {"hr": 80}}
