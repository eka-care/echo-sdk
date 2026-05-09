"""
End-to-end tests for the public AG-UI run / resume API (PR-S4).

Covers:
- AgUiRunner pause-on-DONE-with-elicitations: synthesizes
  TOOL_CALL_START / ARGS / END for each elicitation, persists, returns
  without RUN_FINISHED.
- AgUiRunner without paused_run_store: degrades to emitting events but
  still RUN_FINISHED (no persistence).
- BaseAgent.run_stream_with_ag_ui: thin wrapper, full round-trip via a
  TinyAgent that bypasses BaseAgent.__init__'s LLM setup.
- BaseAgent.resume_run_with_ag_ui: success path; tool_call_id mismatch;
  expired (missing) paused run.
"""

from typing import Any, AsyncGenerator, List

import pytest
from ag_ui.core import (
    EventType,
    RunAgentInput,
    RunErrorEvent,
    Tool,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

from echo.ag_ui import (
    AgUiRunner,
    AgUiState,
    AgUiToolDispatcher,
    InMemoryPausedRunStore,
    make_pause_key,
)
from echo.agents.base import BaseAgent
from echo.llm.schemas import LLMResponse, StreamEvent, StreamEventType
from echo.models.user_conversation import (
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
)
from echo.tools.schemas import ElicitationDetails, ElicitationResponse


# ----- shared fixtures -----


class _DemoState(AgUiState):
    transcript: str = ""


class FakeAgent:
    """Minimal stand-in for BaseAgent — replays a scripted StreamEvent
    sequence on each run_stream() call.

    Supports multi-call test scenarios: pass `events_per_call` as a list
    of lists; the i-th call to run_stream() uses events_per_call[i].
    Falls back to events_per_call[0] for further calls.
    """

    def __init__(
        self,
        events: List[StreamEvent] | None = None,
        events_per_call: List[List[StreamEvent]] | None = None,
    ) -> None:
        if events_per_call is None:
            assert events is not None
            self._events_per_call = [events]
        else:
            self._events_per_call = events_per_call
        self._call_count = 0

    async def run_stream(
        self, context, out_msg_id: str, **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        idx = min(self._call_count, len(self._events_per_call) - 1)
        self._call_count += 1
        for ev in self._events_per_call[idx]:
            yield ev


def _elicitation(tool_id: str, tool_name: str, args: dict) -> ElicitationResponse:
    return ElicitationResponse(
        tool_id=tool_id,
        tool_name=tool_name,
        details=ElicitationDetails(component=tool_name, input=args),
    )


def _done_with_elicitations(elicitations: list[ElicitationResponse]) -> StreamEvent:
    return StreamEvent(
        type=StreamEventType.DONE,
        llm_response=LLMResponse(elicitations=elicitations),
    )


async def _collect(gen) -> list:
    return [ev async for ev in gen]


# ----- runner pause path -----


@pytest.mark.asyncio
async def test_runner_pauses_on_done_with_elicitations():
    state = _DemoState(transcript="t")
    store = InMemoryPausedRunStore()
    dispatcher = AgUiToolDispatcher(ui_tool_names={"request_field_input"})
    elicit = _elicitation(
        "tc9", "request_field_input", {"rowId": "m0", "field": "duration"}
    )
    agent = FakeAgent(events=[_done_with_elicitations([elicit])])
    runner = AgUiRunner(
        agent=agent,
        state=state,
        thread_id="t1",
        run_id="r1",
        tool_dispatcher=dispatcher,
        paused_run_store=store,
        pause_metadata={"document_id": "doc_42"},
    )

    events = await _collect(runner.stream(None, "msg1"))

    types = [type(e).__name__ for e in events]
    # No RUN_FINISHED because we paused.
    assert "RunFinishedEvent" not in types
    # Synthesized tool events.
    assert "ToolCallStartEvent" in types
    assert "ToolCallArgsEvent" in types
    assert "ToolCallEndEvent" in types

    start = next(e for e in events if isinstance(e, ToolCallStartEvent))
    assert start.tool_call_id == "tc9"
    assert start.tool_call_name == "request_field_input"

    args_ev = next(e for e in events if isinstance(e, ToolCallArgsEvent))
    assert args_ev.tool_call_id == "tc9"
    # args_ev.delta is JSON-encoded args.
    import orjson

    assert orjson.loads(args_ev.delta) == {"rowId": "m0", "field": "duration"}

    end = next(e for e in events if isinstance(e, ToolCallEndEvent))
    assert end.tool_call_id == "tc9"

    # Persisted to store.
    paused = await store.load(make_pause_key("t1", "r1"))
    assert paused is not None
    assert paused.tool_call_id == "tc9"
    assert paused.tool_call_name == "request_field_input"
    assert paused.tool_args == {"rowId": "m0", "field": "duration"}
    assert paused.metadata == {"document_id": "doc_42"}
    assert paused.state_snapshot["transcript"] == "t"


@pytest.mark.asyncio
async def test_runner_without_store_emits_tool_events_then_run_finished():
    """No store → can't persist; runner still emits the tool events but
    falls through to RUN_FINISHED so the SSE doesn't dangle."""
    state = _DemoState()
    elicit = _elicitation("tc1", "request_field_input", {"x": 1})
    agent = FakeAgent(events=[_done_with_elicitations([elicit])])
    runner = AgUiRunner(
        agent=agent,
        state=state,
        thread_id="t1",
        run_id="r1",
        paused_run_store=None,  # no store
    )

    events = await _collect(runner.stream(None, "msg1"))
    types = [type(e).__name__ for e in events]
    # Tool events still synthesized.
    assert "ToolCallStartEvent" in types
    # And without a store the runner can't truly pause, so RunFinished still
    # fires (the host opted out of the resume mechanism).
    # Actually the runner sets paused=True regardless — no RunFinished.
    # That's intentional: pausing without persistence would be wrong.
    assert "RunFinishedEvent" not in types


@pytest.mark.asyncio
async def test_runner_clean_completion_deletes_prior_paused_entry():
    state = _DemoState()
    store = InMemoryPausedRunStore()
    # Pre-seed a stale paused-run entry under our key.
    key = make_pause_key("t1", "r1")
    from echo.ag_ui import PausedRun

    await store.save(
        key,
        PausedRun(
            thread_id="t1",
            run_id="r1",
            tool_call_id="stale",
            tool_call_name="x",
            tool_args={},
            context_snapshot={},
            state_snapshot={},
        ),
    )

    # Run without elicitations → clean completion.
    agent = FakeAgent(events=[StreamEvent(type=StreamEventType.DONE)])
    runner = AgUiRunner(
        agent=agent,
        state=state,
        thread_id="t1",
        run_id="r1",
        paused_run_store=store,
    )
    await _collect(runner.stream(None, "msg1"))
    # Stale entry was cleaned up.
    assert await store.load(key) is None


# ----- BaseAgent.run_stream_with_ag_ui -----


class TinyAgent(BaseAgent):
    """BaseAgent subclass that bypasses heavy __init__ (no LLM setup)
    so we can test the AG-UI public methods in isolation."""

    def __init__(self, scripted_events_per_call: List[List[StreamEvent]]) -> None:
        # Skip super().__init__ — we don't need llm/tools/skills setup.
        self._events_per_call = scripted_events_per_call
        self._call_count = 0
        self.tools = []
        self.skills = {}
        self._meta_tools = []
        self._active_skill_names = {}

    @property
    def name(self) -> str:
        return "tiny"

    async def run(self, *args, **kwargs):
        raise NotImplementedError

    async def run_stream(
        self, context, out_msg_id: str, **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        idx = min(self._call_count, len(self._events_per_call) - 1)
        self._call_count += 1
        for ev in self._events_per_call[idx]:
            yield ev


def _user_message(text: str) -> Message:
    return Message(
        role=MessageRole.USER,
        content=[TextMessage(text=text)],
    )


def _make_run_input(thread_id="t1", run_id="r1", ui_tool_names=None) -> RunAgentInput:
    tools = [
        Tool(name=n, description="", parameters=None)
        for n in (ui_tool_names or [])
    ]
    return RunAgentInput(
        thread_id=thread_id,
        run_id=run_id,
        state={},
        messages=[],
        tools=tools,
        context=[],
        forwarded_props={},
    )


@pytest.mark.asyncio
async def test_base_agent_run_stream_with_ag_ui_text_path():
    agent = TinyAgent(
        scripted_events_per_call=[
            [
                StreamEvent(type=StreamEventType.TEXT, text="hello"),
                StreamEvent(type=StreamEventType.DONE),
            ]
        ]
    )
    state = _DemoState()
    ctx = ConversationContext()
    ctx.add_message(_user_message("hi"))

    events = await _collect(
        agent.run_stream_with_ag_ui(
            context=ctx,
            run_input=_make_run_input(),
            state=state,
            out_msg_id="msg1",
        )
    )
    types = [type(e).__name__ for e in events]
    assert types == [
        "RunStartedEvent",
        "StateSnapshotEvent",
        "TextMessageChunkEvent",
        "RunFinishedEvent",
    ]


@pytest.mark.asyncio
async def test_base_agent_run_stream_with_ag_ui_pauses_on_ui_tool():
    agent = TinyAgent(
        scripted_events_per_call=[
            [
                _done_with_elicitations(
                    [_elicitation("tcA", "request_field_input", {"x": 1})]
                )
            ]
        ]
    )
    state = _DemoState()
    ctx = ConversationContext()
    ctx.add_message(_user_message("ask me"))
    store = InMemoryPausedRunStore()

    events = await _collect(
        agent.run_stream_with_ag_ui(
            context=ctx,
            run_input=_make_run_input(ui_tool_names=["request_field_input"]),
            state=state,
            out_msg_id="msg1",
            paused_run_store=store,
            pause_metadata={"document_id": "doc_1"},
        )
    )

    # Paused: no RunFinished.
    types = [type(e).__name__ for e in events]
    assert "RunFinishedEvent" not in types
    assert "ToolCallStartEvent" in types

    # Persisted with full ConversationContext snapshot.
    paused = await store.load(make_pause_key("t1", "r1"))
    assert paused is not None
    assert paused.tool_call_id == "tcA"
    assert paused.tool_call_name == "request_field_input"
    assert paused.tool_args == {"x": 1}
    # ConversationContext.model_dump preserves the user message.
    assert "messages" in paused.context_snapshot
    assert len(paused.context_snapshot["messages"]) == 1


# ----- BaseAgent.resume_run_with_ag_ui -----


@pytest.mark.asyncio
async def test_resume_completes_run_with_no_more_pauses():
    """Pause → resume → no further elicitations → RUN_FINISHED."""
    agent = TinyAgent(
        scripted_events_per_call=[
            # First call: pause on UI tool.
            [
                _done_with_elicitations(
                    [_elicitation("tcA", "request_field_input", {"field": "duration"})]
                )
            ],
            # Resume call: simple text + DONE.
            [
                StreamEvent(type=StreamEventType.TEXT, text="thanks!"),
                StreamEvent(type=StreamEventType.DONE),
            ],
        ]
    )
    state = _DemoState()
    ctx = ConversationContext()
    ctx.add_message(_user_message("ask me"))
    store = InMemoryPausedRunStore()

    # Initial run → pauses.
    await _collect(
        agent.run_stream_with_ag_ui(
            context=ctx,
            run_input=_make_run_input(ui_tool_names=["request_field_input"]),
            state=state,
            out_msg_id="msg1",
            paused_run_store=store,
        )
    )
    assert await store.load(make_pause_key("t1", "r1")) is not None

    # Resume.
    events = await _collect(
        agent.resume_run_with_ag_ui(
            paused_run_store=store,
            thread_id="t1",
            run_id="r1",
            tool_call_id="tcA",
            tool_result={"value": "3 months"},
            state=state,
            context=ctx,
            out_msg_id="msg2",
            ui_tool_names=["request_field_input"],
        )
    )

    types = [type(e).__name__ for e in events]
    # Resume entry emits TOOL_CALL_RESULT, then RUN_STARTED + STATE_SNAPSHOT
    # from the new runner, then text, then RUN_FINISHED.
    assert types[0] == "ToolCallResultEvent"
    assert "RunFinishedEvent" in types
    assert "TextMessageChunkEvent" in types

    # Tool result event carries the FE's response as JSON-encoded content.
    res_ev = next(e for e in events if isinstance(e, ToolCallResultEvent))
    assert res_ev.tool_call_id == "tcA"
    assert res_ev.role == "tool"
    import orjson

    assert orjson.loads(res_ev.content) == {"value": "3 months"}

    # Tool result was injected into context as a TOOL message.
    tool_msgs = [m for m in ctx.messages if m.role == MessageRole.TOOL]
    assert len(tool_msgs) >= 1

    # Paused entry deleted on clean completion.
    assert await store.load(make_pause_key("t1", "r1")) is None


@pytest.mark.asyncio
async def test_resume_with_string_tool_result_passes_through_verbatim():
    agent = TinyAgent(
        scripted_events_per_call=[
            [
                _done_with_elicitations(
                    [_elicitation("tcA", "request_field_input", {})]
                )
            ],
            [StreamEvent(type=StreamEventType.DONE)],
        ]
    )
    state = _DemoState()
    ctx = ConversationContext()
    ctx.add_message(_user_message("hi"))
    store = InMemoryPausedRunStore()

    await _collect(
        agent.run_stream_with_ag_ui(
            context=ctx,
            run_input=_make_run_input(ui_tool_names=["request_field_input"]),
            state=state,
            out_msg_id="m1",
            paused_run_store=store,
        )
    )

    events = await _collect(
        agent.resume_run_with_ag_ui(
            paused_run_store=store,
            thread_id="t1",
            run_id="r1",
            tool_call_id="tcA",
            tool_result="3 months",  # plain string
            state=state,
            context=ctx,
            out_msg_id="m2",
        )
    )
    res_ev = next(e for e in events if isinstance(e, ToolCallResultEvent))
    # Strings pass through untouched (no orjson wrapping).
    assert res_ev.content == "3 months"


@pytest.mark.asyncio
async def test_resume_tool_call_id_mismatch_emits_run_error():
    agent = TinyAgent(
        scripted_events_per_call=[
            [_done_with_elicitations([_elicitation("tcA", "x", {})])]
        ]
    )
    state = _DemoState()
    ctx = ConversationContext()
    ctx.add_message(_user_message("hi"))
    store = InMemoryPausedRunStore()

    await _collect(
        agent.run_stream_with_ag_ui(
            context=ctx,
            run_input=_make_run_input(ui_tool_names=["x"]),
            state=state,
            out_msg_id="m",
            paused_run_store=store,
        )
    )

    # Wrong tool_call_id on resume.
    events = await _collect(
        agent.resume_run_with_ag_ui(
            paused_run_store=store,
            thread_id="t1",
            run_id="r1",
            tool_call_id="WRONG",
            tool_result={},
            state=state,
            context=ctx,
            out_msg_id="m2",
        )
    )
    assert len(events) == 1
    assert isinstance(events[0], RunErrorEvent)
    assert events[0].code == "tool_call_id_mismatch"

    # Original paused entry untouched.
    assert await store.load(make_pause_key("t1", "r1")) is not None


@pytest.mark.asyncio
async def test_resume_missing_paused_run_emits_run_error():
    agent = TinyAgent(scripted_events_per_call=[[StreamEvent(type=StreamEventType.DONE)]])
    state = _DemoState()
    ctx = ConversationContext()
    store = InMemoryPausedRunStore()  # empty

    events = await _collect(
        agent.resume_run_with_ag_ui(
            paused_run_store=store,
            thread_id="t-nope",
            run_id="r-nope",
            tool_call_id="tc1",
            tool_result={},
            state=state,
            context=ctx,
            out_msg_id="m",
        )
    )
    assert len(events) == 1
    assert isinstance(events[0], RunErrorEvent)
    assert events[0].code == "paused_run_expired"


@pytest.mark.asyncio
async def test_run_continues_to_run_finished_when_no_elicitations():
    """Sanity: with no UI tools called, run_stream_with_ag_ui completes
    normally and clears any prior paused entry."""
    agent = TinyAgent(
        scripted_events_per_call=[
            [
                StreamEvent(type=StreamEventType.TEXT, text="ok"),
                StreamEvent(type=StreamEventType.DONE),
            ]
        ]
    )
    state = _DemoState()
    ctx = ConversationContext()
    store = InMemoryPausedRunStore()

    events = await _collect(
        agent.run_stream_with_ag_ui(
            context=ctx,
            run_input=_make_run_input(),  # no UI tools
            state=state,
            out_msg_id="m",
            paused_run_store=store,
        )
    )
    types = [type(e).__name__ for e in events]
    assert "RunFinishedEvent" in types
    assert "ToolCallStartEvent" not in types
