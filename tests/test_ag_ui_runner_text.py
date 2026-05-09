"""
Unit tests for echo.ag_ui.runner.AgUiRunner — text-only path (PR-S2).

The runner is duck-typed against any object with an async run_stream()
method, so tests use a tiny FakeAgent rather than dragging in the full
BaseAgent + LLMConfig + ConversationContext machinery.

Tool dispatch is covered in PR-S3.
"""

from typing import AsyncGenerator, List

import pytest
from ag_ui.core import (
    EventType,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    TextMessageChunkEvent,
)

from echo.ag_ui import AgUiRunner, AgUiState
from echo.llm.schemas import StreamEvent, StreamEventType


# ----- helpers -----


class _DemoState(AgUiState):
    transcript: str = ""
    counter: int = 0


class FakeAgent:
    """Minimal stand-in for BaseAgent. Replays a scripted StreamEvent sequence.

    Optionally calls a `before_yield` hook on the host's state right before
    yielding each event — this lets tests simulate "host mutates state
    during agent execution" interleavings.
    """

    def __init__(
        self,
        events: List[StreamEvent],
        state: AgUiState | None = None,
        mutations_per_event: List[List[str]] | None = None,
    ):
        # mutations_per_event[i] is a list of new transcript values to set
        # on `state` BEFORE yielding events[i].
        self._events = events
        self._state = state
        self._mutations = mutations_per_event or [[] for _ in events]
        if len(self._mutations) != len(events):
            raise ValueError("mutations_per_event must match events length")
        self.run_stream_called_with: tuple | None = None

    async def run_stream(
        self, context, out_msg_id: str, **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        self.run_stream_called_with = (context, out_msg_id)
        for i, ev in enumerate(self._events):
            for new_transcript in self._mutations[i]:
                if self._state is not None:
                    self._state.transcript = new_transcript
            yield ev


async def _collect(runner: AgUiRunner, context=None, out_msg_id="msg_1"):
    return [ev async for ev in runner.stream(context, out_msg_id)]


# ----- empty / minimal lifecycle -----


@pytest.mark.asyncio
async def test_empty_run_emits_lifecycle_only():
    state = _DemoState()
    agent = FakeAgent(events=[StreamEvent(type=StreamEventType.DONE)])
    runner = AgUiRunner(agent, state, thread_id="t1", run_id="r1")

    events = await _collect(runner)

    types = [type(e).__name__ for e in events]
    assert types == ["RunStartedEvent", "StateSnapshotEvent", "RunFinishedEvent"]

    assert events[0].thread_id == "t1"
    assert events[0].run_id == "r1"
    assert events[0].type == EventType.RUN_STARTED
    assert events[1].snapshot == {"transcript": "", "counter": 0}
    assert events[2].thread_id == "t1"
    assert events[2].run_id == "r1"


@pytest.mark.asyncio
async def test_initial_state_snapshot_reflects_prepopulated_state():
    state = _DemoState(transcript="seeded", counter=5)
    agent = FakeAgent(events=[StreamEvent(type=StreamEventType.DONE)])
    runner = AgUiRunner(agent, state, "t1", "r1")

    events = await _collect(runner)
    snap = next(e for e in events if isinstance(e, StateSnapshotEvent))
    assert snap.snapshot == {"transcript": "seeded", "counter": 5}


# ----- text streaming -----


@pytest.mark.asyncio
async def test_single_text_chunk_translated():
    state = _DemoState()
    agent = FakeAgent(
        events=[
            StreamEvent(type=StreamEventType.TEXT, text="hello"),
            StreamEvent(type=StreamEventType.DONE),
        ]
    )
    runner = AgUiRunner(agent, state, "t1", "r1")

    events = await _collect(runner)
    text_events = [e for e in events if isinstance(e, TextMessageChunkEvent)]
    assert len(text_events) == 1
    assert text_events[0].delta == "hello"
    assert text_events[0].role == "assistant"
    assert text_events[0].message_id  # any non-empty id
    assert text_events[0].type == EventType.TEXT_MESSAGE_CHUNK


@pytest.mark.asyncio
async def test_multi_chunk_share_message_id():
    state = _DemoState()
    agent = FakeAgent(
        events=[
            StreamEvent(type=StreamEventType.TEXT, text="hel"),
            StreamEvent(type=StreamEventType.TEXT, text="lo "),
            StreamEvent(type=StreamEventType.TEXT, text="world"),
            StreamEvent(type=StreamEventType.DONE),
        ]
    )
    runner = AgUiRunner(agent, state, "t1", "r1")

    events = await _collect(runner)
    text_events = [e for e in events if isinstance(e, TextMessageChunkEvent)]
    assert [e.delta for e in text_events] == ["hel", "lo ", "world"]
    # All chunks share the same message_id.
    ids = {e.message_id for e in text_events}
    assert len(ids) == 1


@pytest.mark.asyncio
async def test_empty_text_chunk_yields_empty_delta():
    state = _DemoState()
    agent = FakeAgent(
        events=[
            StreamEvent(type=StreamEventType.TEXT, text=None),
            StreamEvent(type=StreamEventType.DONE),
        ]
    )
    runner = AgUiRunner(agent, state, "t1", "r1")
    events = await _collect(runner)
    text_events = [e for e in events if isinstance(e, TextMessageChunkEvent)]
    assert len(text_events) == 1
    assert text_events[0].delta == ""


# ----- state deltas interleave with text -----


@pytest.mark.asyncio
async def test_state_mutation_during_run_emits_state_delta():
    state = _DemoState()
    agent = FakeAgent(
        events=[
            StreamEvent(type=StreamEventType.TEXT, text="a"),
            StreamEvent(type=StreamEventType.DONE),
        ],
        state=state,
        mutations_per_event=[["mid-run"], []],
    )
    runner = AgUiRunner(agent, state, "t1", "r1")

    events = await _collect(runner)
    delta_events = [e for e in events if isinstance(e, StateDeltaEvent)]
    # The mutation happens before the TEXT yield, so it's drained after the
    # TEXT event is processed.
    assert len(delta_events) == 1
    assert any(
        op["op"] == "replace" and op["path"] == "/transcript" and op["value"] == "mid-run"
        for op in delta_events[0].delta
    )


@pytest.mark.asyncio
async def test_no_mutations_no_state_delta():
    state = _DemoState()
    agent = FakeAgent(
        events=[
            StreamEvent(type=StreamEventType.TEXT, text="just text"),
            StreamEvent(type=StreamEventType.DONE),
        ]
    )
    runner = AgUiRunner(agent, state, "t1", "r1")

    events = await _collect(runner)
    assert not any(isinstance(e, StateDeltaEvent) for e in events)


@pytest.mark.asyncio
async def test_multiple_mutations_interleave_with_text():
    state = _DemoState()
    agent = FakeAgent(
        events=[
            StreamEvent(type=StreamEventType.TEXT, text="a"),
            StreamEvent(type=StreamEventType.TEXT, text="b"),
            StreamEvent(type=StreamEventType.DONE),
        ],
        state=state,
        mutations_per_event=[["v1"], ["v2"], []],
    )
    runner = AgUiRunner(agent, state, "t1", "r1")

    events = await _collect(runner)
    # Expected ordering: RUN_STARTED, STATE_SNAPSHOT, TEXT(a), DELTA(v1),
    # TEXT(b), DELTA(v2), RUN_FINISHED.
    types = [type(e).__name__ for e in events]
    assert types == [
        "RunStartedEvent",
        "StateSnapshotEvent",
        "TextMessageChunkEvent",
        "StateDeltaEvent",
        "TextMessageChunkEvent",
        "StateDeltaEvent",
        "RunFinishedEvent",
    ]


# ----- error paths -----


@pytest.mark.asyncio
async def test_error_stream_event_emits_run_error_no_run_finished():
    state = _DemoState()
    agent = FakeAgent(
        events=[
            StreamEvent(type=StreamEventType.TEXT, text="partial"),
            StreamEvent(type=StreamEventType.ERROR, error="boom"),
        ]
    )
    runner = AgUiRunner(agent, state, "t1", "r1")

    events = await _collect(runner)
    type_names = [type(e).__name__ for e in events]
    assert "RunErrorEvent" in type_names
    assert "RunFinishedEvent" not in type_names

    err = next(e for e in events if isinstance(e, RunErrorEvent))
    assert err.message == "boom"
    assert err.code == "agent_stream_error"


@pytest.mark.asyncio
async def test_exception_in_agent_stream_emits_run_error():
    state = _DemoState()

    class ExplodingAgent:
        async def run_stream(self, context, out_msg_id, **kwargs):
            yield StreamEvent(type=StreamEventType.TEXT, text="ok")
            raise RuntimeError("kaboom")

    runner = AgUiRunner(ExplodingAgent(), state, "t1", "r1")
    events = await _collect(runner)
    type_names = [type(e).__name__ for e in events]
    assert "RunErrorEvent" in type_names
    assert "RunFinishedEvent" not in type_names
    err = next(e for e in events if isinstance(e, RunErrorEvent))
    assert err.code == "runner_exception"
    assert "kaboom" in err.message


# ----- TOOL_CALL events ignored in PR-S2 -----


@pytest.mark.asyncio
async def test_tool_call_events_ignored_in_pr_s2():
    state = _DemoState()
    agent = FakeAgent(
        events=[
            StreamEvent(
                type=StreamEventType.TOOL_CALL_START,
                details={"tool_id": "tc1", "tool_name": "x"},
            ),
            StreamEvent(
                type=StreamEventType.TOOL_CALL_END,
                details={"tool_id": "tc1", "tool_name": "x"},
            ),
            StreamEvent(type=StreamEventType.TEXT, text="post-tool"),
            StreamEvent(type=StreamEventType.DONE),
        ]
    )
    runner = AgUiRunner(agent, state, "t1", "r1")

    events = await _collect(runner)
    # TOOL_CALL_* events from the LLM stream produce no AG-UI events here.
    type_names = [type(e).__name__ for e in events]
    assert "ToolCallStartEvent" not in type_names
    assert "ToolCallEndEvent" not in type_names
    # But the trailing TEXT and lifecycle still flow.
    assert "TextMessageChunkEvent" in type_names
    assert "RunFinishedEvent" in type_names


# ----- forwarded args -----


@pytest.mark.asyncio
async def test_run_stream_called_with_passed_context_and_msg_id():
    state = _DemoState()
    agent = FakeAgent(events=[StreamEvent(type=StreamEventType.DONE)])
    runner = AgUiRunner(agent, state, "t1", "r1")
    sentinel_ctx = object()

    await _collect(runner, context=sentinel_ctx, out_msg_id="my_msg")

    assert agent.run_stream_called_with == (sentinel_ctx, "my_msg")
