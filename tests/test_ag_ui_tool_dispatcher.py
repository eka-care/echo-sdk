"""
Unit tests for echo.ag_ui.tool_dispatcher.AgUiToolDispatcher (PR-S3).

Covers:
- UI-tool registration and classification
- StreamEvent → AG-UI event translation
- Streaming args via append_args_delta
- record_tool_args + consume_pause_signal interaction
- emit_backend_result_ack policy (backend yes, UI no, unknown no)
- Integration: AgUiRunner uses the dispatcher to translate tool events
"""

from typing import AsyncGenerator, List

import pytest
from ag_ui.core import (
    EventType,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

from echo.ag_ui import AgUiRunner, AgUiState, AgUiToolDispatcher, PauseSignal
from echo.llm.schemas import StreamEvent, StreamEventType


# ----- helpers shared with test_ag_ui_runner_text style -----


class _DemoState(AgUiState):
    transcript: str = ""


class FakeAgent:
    def __init__(self, events: List[StreamEvent]):
        self._events = events

    async def run_stream(
        self, context, out_msg_id: str, **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        for ev in self._events:
            yield ev


async def _collect(runner: AgUiRunner):
    return [ev async for ev in runner.stream(None, "msg1")]


# ----- registration & classification -----


def test_default_dispatcher_has_no_ui_tools():
    d = AgUiToolDispatcher()
    assert d.is_ui_tool("anything") is False


def test_register_ui_tools_marks_them_ui():
    d = AgUiToolDispatcher()
    d.register_ui_tools(["request_field_input", "confirm_action"])
    assert d.is_ui_tool("request_field_input") is True
    assert d.is_ui_tool("confirm_action") is True
    assert d.is_ui_tool("emit_section") is False


def test_register_ui_tools_is_idempotent():
    d = AgUiToolDispatcher()
    d.register_ui_tools(["a"])
    d.register_ui_tools(["a", "b"])
    assert d.is_ui_tool("a") is True
    assert d.is_ui_tool("b") is True


def test_constructor_accepts_set():
    d = AgUiToolDispatcher(ui_tool_names={"x", "y"})
    assert d.is_ui_tool("x") is True
    assert d.is_ui_tool("y") is True


def test_classification_is_none_for_untracked_call_id():
    d = AgUiToolDispatcher(ui_tool_names={"x"})
    assert d.call_classification("never-seen") is None


# ----- StreamEvent translation -----


def test_translate_tool_call_start_backend():
    d = AgUiToolDispatcher(ui_tool_names={"request_field_input"})
    sev = StreamEvent(
        type=StreamEventType.TOOL_CALL_START,
        details={"tool_id": "tc1", "tool_name": "emit_section"},
    )
    out = d.translate(sev)
    assert len(out) == 1
    assert isinstance(out[0], ToolCallStartEvent)
    assert out[0].tool_call_id == "tc1"
    assert out[0].tool_call_name == "emit_section"
    assert out[0].type == EventType.TOOL_CALL_START
    assert d.call_classification("tc1") == "backend"


def test_translate_tool_call_start_ui():
    d = AgUiToolDispatcher(ui_tool_names={"request_field_input"})
    sev = StreamEvent(
        type=StreamEventType.TOOL_CALL_START,
        details={"tool_id": "tc1", "tool_name": "request_field_input"},
    )
    d.translate(sev)
    assert d.call_classification("tc1") == "ui"


def test_translate_tool_call_end():
    d = AgUiToolDispatcher()
    d.translate(
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            details={"tool_id": "tc1", "tool_name": "x"},
        )
    )
    out = d.translate(
        StreamEvent(
            type=StreamEventType.TOOL_CALL_END,
            details={"tool_id": "tc1", "tool_name": "x"},
        )
    )
    assert len(out) == 1
    assert isinstance(out[0], ToolCallEndEvent)
    assert out[0].tool_call_id == "tc1"
    assert out[0].type == EventType.TOOL_CALL_END


def test_translate_non_tool_event_returns_empty():
    d = AgUiToolDispatcher()
    out = d.translate(StreamEvent(type=StreamEventType.TEXT, text="hi"))
    assert out == []
    out = d.translate(StreamEvent(type=StreamEventType.DONE))
    assert out == []
    out = d.translate(StreamEvent(type=StreamEventType.ERROR, error="x"))
    assert out == []


def test_translate_tool_call_start_with_missing_details_degrades_gracefully():
    d = AgUiToolDispatcher()
    out = d.translate(StreamEvent(type=StreamEventType.TOOL_CALL_START, details=None))
    assert len(out) == 1
    assert isinstance(out[0], ToolCallStartEvent)
    assert out[0].tool_call_id == ""
    assert out[0].tool_call_name == ""


# ----- args streaming -----


def test_append_args_delta_emits_tool_call_args():
    d = AgUiToolDispatcher()
    d.translate(
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            details={"tool_id": "tc1", "tool_name": "x"},
        )
    )
    ev = d.append_args_delta("tc1", '{"name":"Met')
    assert ev is not None
    assert isinstance(ev, ToolCallArgsEvent)
    assert ev.tool_call_id == "tc1"
    assert ev.delta == '{"name":"Met'
    assert ev.type == EventType.TOOL_CALL_ARGS


def test_append_args_delta_unknown_id_returns_none():
    d = AgUiToolDispatcher()
    assert d.append_args_delta("never-seen", '{"x":1}') is None


def test_append_args_delta_accumulates_buffer():
    d = AgUiToolDispatcher()
    d.translate(
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            details={"tool_id": "tc1", "tool_name": "x"},
        )
    )
    d.append_args_delta("tc1", "abc")
    d.append_args_delta("tc1", "def")
    # buffer is internal; verify by inspecting the underlying state via
    # consume_pause_signal once we set the parsed args.
    # Here we just verify we got two distinct ToolCallArgsEvents.
    e1 = d.append_args_delta("tc1", "ghi")
    assert e1.delta == "ghi"


# ----- record_tool_args + pause signal -----


def test_consume_pause_signal_returns_none_when_no_ui_calls():
    d = AgUiToolDispatcher(ui_tool_names={"request_field_input"})
    d.translate(
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            details={"tool_id": "tc1", "tool_name": "emit_section"},
        )
    )
    assert d.consume_pause_signal() is None


def test_consume_pause_signal_returns_first_ui_call():
    d = AgUiToolDispatcher(ui_tool_names={"request_field_input"})
    d.translate(
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            details={"tool_id": "tc1", "tool_name": "request_field_input"},
        )
    )
    d.record_tool_args("tc1", {"rowId": "m0", "field": "duration"})
    sig = d.consume_pause_signal()
    assert isinstance(sig, PauseSignal)
    assert sig.tool_call_id == "tc1"
    assert sig.tool_call_name == "request_field_input"
    assert sig.tool_args == {"rowId": "m0", "field": "duration"}


def test_consume_pause_signal_picks_ui_over_backend():
    d = AgUiToolDispatcher(ui_tool_names={"request_field_input"})
    # backend call first
    d.translate(
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            details={"tool_id": "tcA", "tool_name": "emit_section"},
        )
    )
    # UI call second
    d.translate(
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            details={"tool_id": "tcB", "tool_name": "request_field_input"},
        )
    )
    sig = d.consume_pause_signal()
    assert sig is not None
    assert sig.tool_call_id == "tcB"


def test_record_tool_args_for_unknown_id_is_silent_noop():
    d = AgUiToolDispatcher(ui_tool_names={"x"})
    d.record_tool_args("nope", {"a": 1})
    # No crash, no pause signal.
    assert d.consume_pause_signal() is None


def test_pending_ui_calls_lists_only_ui():
    d = AgUiToolDispatcher(ui_tool_names={"ui1"})
    d.translate(
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            details={"tool_id": "a", "tool_name": "backend1"},
        )
    )
    d.translate(
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            details={"tool_id": "b", "tool_name": "ui1"},
        )
    )
    d.translate(
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            details={"tool_id": "c", "tool_name": "backend2"},
        )
    )
    pending = d.pending_ui_calls()
    assert len(pending) == 1
    assert pending[0].tool_call_id == "b"


# ----- backend result ack -----


def test_emit_backend_result_ack_for_backend_call():
    d = AgUiToolDispatcher(ui_tool_names={"ui1"})
    d.translate(
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            details={"tool_id": "tc1", "tool_name": "emit_section"},
        )
    )
    ev = d.emit_backend_result_ack("tc1", message_id="msg42")
    assert ev is not None
    assert isinstance(ev, ToolCallResultEvent)
    assert ev.tool_call_id == "tc1"
    assert ev.message_id == "msg42"
    assert ev.role == "tool"
    assert ev.content == "(server-executed)"


def test_emit_backend_result_ack_skips_ui_call():
    d = AgUiToolDispatcher(ui_tool_names={"ui1"})
    d.translate(
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            details={"tool_id": "tc1", "tool_name": "ui1"},
        )
    )
    assert d.emit_backend_result_ack("tc1", "msg") is None


def test_emit_backend_result_ack_skips_unknown_id():
    d = AgUiToolDispatcher()
    assert d.emit_backend_result_ack("never-seen", "msg") is None


def test_emit_backend_result_ack_custom_content():
    d = AgUiToolDispatcher()
    d.translate(
        StreamEvent(
            type=StreamEventType.TOOL_CALL_START,
            details={"tool_id": "tc1", "tool_name": "lookup_rxnorm"},
        )
    )
    ev = d.emit_backend_result_ack("tc1", "m", content="ok")
    assert ev is not None
    assert ev.content == "ok"


# ----- runner integration -----


@pytest.mark.asyncio
async def test_runner_with_dispatcher_emits_tool_call_events():
    state = _DemoState()
    dispatcher = AgUiToolDispatcher(ui_tool_names={"request_field_input"})
    agent = FakeAgent(
        events=[
            StreamEvent(
                type=StreamEventType.TOOL_CALL_START,
                details={"tool_id": "tc1", "tool_name": "emit_section"},
            ),
            StreamEvent(
                type=StreamEventType.TOOL_CALL_END,
                details={"tool_id": "tc1", "tool_name": "emit_section"},
            ),
            StreamEvent(type=StreamEventType.DONE),
        ]
    )
    runner = AgUiRunner(agent, state, "t1", "r1", tool_dispatcher=dispatcher)

    events = await _collect(runner)
    types = [type(e).__name__ for e in events]
    assert types == [
        "RunStartedEvent",
        "StateSnapshotEvent",
        "ToolCallStartEvent",
        "ToolCallEndEvent",
        "RunFinishedEvent",
    ]
    # Classification was tracked.
    assert dispatcher.call_classification("tc1") == "backend"


@pytest.mark.asyncio
async def test_runner_default_dispatcher_classifies_all_as_backend():
    """Without UI tools registered, every tool call is backend."""
    state = _DemoState()
    agent = FakeAgent(
        events=[
            StreamEvent(
                type=StreamEventType.TOOL_CALL_START,
                details={"tool_id": "tc1", "tool_name": "anything"},
            ),
            StreamEvent(
                type=StreamEventType.TOOL_CALL_END,
                details={"tool_id": "tc1", "tool_name": "anything"},
            ),
            StreamEvent(type=StreamEventType.DONE),
        ]
    )
    runner = AgUiRunner(agent, state, "t1", "r1")  # no dispatcher passed
    events = await _collect(runner)
    type_names = [type(e).__name__ for e in events]
    assert "ToolCallStartEvent" in type_names
    assert "ToolCallEndEvent" in type_names
    assert runner.tool_dispatcher.call_classification("tc1") == "backend"


@pytest.mark.asyncio
async def test_runner_with_dispatcher_records_ui_call_for_pause():
    state = _DemoState()
    dispatcher = AgUiToolDispatcher(ui_tool_names={"request_field_input"})
    agent = FakeAgent(
        events=[
            StreamEvent(
                type=StreamEventType.TOOL_CALL_START,
                details={"tool_id": "tc9", "tool_name": "request_field_input"},
            ),
            StreamEvent(
                type=StreamEventType.TOOL_CALL_END,
                details={"tool_id": "tc9", "tool_name": "request_field_input"},
            ),
            StreamEvent(type=StreamEventType.DONE),
        ]
    )
    runner = AgUiRunner(agent, state, "t1", "r1", tool_dispatcher=dispatcher)
    await _collect(runner)

    dispatcher.record_tool_args("tc9", {"rowId": "m0", "field": "duration"})
    sig = dispatcher.consume_pause_signal()
    assert sig is not None
    assert sig.tool_call_id == "tc9"
    assert sig.tool_call_name == "request_field_input"
    assert sig.tool_args == {"rowId": "m0", "field": "duration"}


@pytest.mark.asyncio
async def test_runner_text_path_unaffected_by_dispatcher():
    """Adding a dispatcher must not change behavior for non-tool flows."""
    state = _DemoState()
    dispatcher = AgUiToolDispatcher(ui_tool_names={"request_field_input"})
    agent = FakeAgent(
        events=[
            StreamEvent(type=StreamEventType.TEXT, text="hello"),
            StreamEvent(type=StreamEventType.DONE),
        ]
    )
    runner = AgUiRunner(agent, state, "t1", "r1", tool_dispatcher=dispatcher)
    events = await _collect(runner)
    types = [type(e).__name__ for e in events]
    assert types == [
        "RunStartedEvent",
        "StateSnapshotEvent",
        "TextMessageChunkEvent",
        "RunFinishedEvent",
    ]
