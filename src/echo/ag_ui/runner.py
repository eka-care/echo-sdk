"""
AgUiRunner: translates an echo agent.run_stream() into AG-UI events.

The runner emits:

    RUN_STARTED
    STATE_SNAPSHOT(snapshot)
    [ TEXT_MESSAGE_CHUNK | TOOL_CALL_START | TOOL_CALL_ARGS | TOOL_CALL_END
      | STATE_DELTA(ops) ]*
    RUN_FINISHED                    (or RUN_ERROR on failure)

Tool dispatch is delegated to AgUiToolDispatcher: it classifies
each call as backend (server-executed by echo-sdk) or UI (FE-declared,
emitted as TOOL_CALL_* and resolved via /resume). Pause/resume wiring
into a PausedRunStore enables streaming
TOOL_CALL_ARGS deltas from the Anthropic/Open-ai or any LLM provider.

Caller responsibilities:

    1. Pre-populate the AgUiState before constructing the runner — the
       initial STATE_SNAPSHOT is taken from state.snapshot().
    2. Provide a fresh thread_id / run_id pair (typically from the FE's
       RunAgentInput).
    3. (Optional) Pass an AgUiToolDispatcher pre-loaded with the
       FE-declared UI tool names. If omitted, all tool calls are treated
       as backend (no pause).
    4. Consume the async generator until exhaustion. The runner guarantees
       exactly one terminal event (RUN_FINISHED or RUN_ERROR) under
       normal control flow.
"""

import logging
import uuid
from typing import TYPE_CHECKING, AsyncGenerator, Optional

import orjson
from ag_ui.core import (
    BaseEvent,
    EventType,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    TextMessageChunkEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

from echo.llm.schemas import StreamEvent, StreamEventType

from .persistence import PausedRun, PausedRunStore, make_pause_key
from .state import AgUiState
from .tool_dispatcher import AgUiToolDispatcher

if TYPE_CHECKING:
    from echo.agents.base import BaseAgent
    from echo.models.user_conversation import ConversationContext
    from echo.tools.schemas import ElicitationResponse

logger = logging.getLogger(__name__)


class AgUiRunner:
    """Drives an echo agent's run_stream() and yields AG-UI BaseEvents.

    See module docstring for the event sequence.
    """

    def __init__(
        self,
        agent: "BaseAgent",
        state: AgUiState,
        thread_id: str,
        run_id: str,
        tool_dispatcher: Optional[AgUiToolDispatcher] = None,
        paused_run_store: Optional[PausedRunStore] = None,
        pause_metadata: Optional[dict] = None,
    ) -> None:
        self.agent = agent
        self.state = state
        self.thread_id = thread_id
        self.run_id = run_id
        self.tool_dispatcher = tool_dispatcher or AgUiToolDispatcher()
        self.paused_run_store = paused_run_store
        self.pause_metadata = pause_metadata or {}
        # stable message_id so all TEXT chunks group into one assistant turn.
        self._assistant_message_id = str(uuid.uuid4())

    async def stream(
        self,
        context: "ConversationContext",
        out_msg_id: str,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Run the agent and yield AG-UI events.

        Terminates with exactly one of:
          - RUN_FINISHED  on clean completion;
          - RUN_ERROR     on stream/runner error;
          - (nothing)     when paused on a UI tool call — the FE will
                          resume via /resume which starts a fresh SSE.
        """
        try:
            yield RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=self.thread_id,
                run_id=self.run_id,
            )
            yield StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=self.state.snapshot(),
            )
            self.state.begin_tracking()

            errored = False
            paused = False

            async for sev in self.agent.run_stream(context, out_msg_id):
                for ev in self._translate(sev):
                    yield ev

                if sev.type == StreamEventType.ERROR:
                    errored = True
                    break

                # flush any state ops accumulated during this LLM event.
                ops = self.state.drain_pending_ops()
                if ops:
                    yield StateDeltaEvent(type=EventType.STATE_DELTA, delta=ops)

                if sev.type == StreamEventType.DONE:
                    elicitations = (
                        sev.llm_response.elicitations
                        if sev.llm_response and sev.llm_response.elicitations
                        else None
                    )
                    if elicitations:
                        # Synthesize AG-UI events for each elicitation
                        # (echo-sdk's Anthropic provider doesn't emit
                        # TOOL_CALL_START/END for is_elicitation tools,
                        # so we surface them here).
                        for elicit in elicitations:
                            for ag_ev in self._synthesize_ui_tool_events(elicit):
                                yield ag_ev

                        # final state flush before pause.
                        tail = self.state.drain_pending_ops()
                        if tail:
                            yield StateDeltaEvent(
                                type=EventType.STATE_DELTA, delta=tail
                            )

                        if self.paused_run_store is not None:
                            await self._persist_pause(context, elicitations[0])

                        paused = True
                    break

            # fail flush — in case the last event mutated state.
            if not paused:
                tail_ops = self.state.drain_pending_ops()
                if tail_ops:
                    yield StateDeltaEvent(
                        type=EventType.STATE_DELTA, delta=tail_ops
                    )

            if not errored and not paused:
                yield RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=self.thread_id,
                    run_id=self.run_id,
                )
                # clean completion: clear any prior paused-run entry
                # for this (thread_id, run_id) pair.
                if self.paused_run_store is not None:
                    await self.paused_run_store.delete(
                        make_pause_key(self.thread_id, self.run_id)
                    )

        except Exception as e:
            logger.error("AgUiRunner.stream raised: %s", e, exc_info=True)
            yield RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=str(e),
                code="runner_exception",
            )

    # --- pause helpers ---
    def _synthesize_ui_tool_events(
        self, elicitation: "ElicitationResponse"
    ) -> list[BaseEvent]:
        """Emit AG-UI ToolCallStart / Args / End events for an elicitation.

        Also registers the call in the dispatcher so consume_pause_signal()
        and call_classification() return the right answer afterward.
        """
        args = dict(elicitation.details.input or {})
        args_json = orjson.dumps(args).decode()

        self.tool_dispatcher.register_completed_call(
            tool_call_id=elicitation.tool_id,
            tool_call_name=elicitation.tool_name,
            is_ui_tool=True,
            args=args,
        )

        return [
            ToolCallStartEvent(
                type=EventType.TOOL_CALL_START,
                tool_call_id=elicitation.tool_id,
                tool_call_name=elicitation.tool_name,
            ),
            ToolCallArgsEvent(
                type=EventType.TOOL_CALL_ARGS,
                tool_call_id=elicitation.tool_id,
                delta=args_json,
            ),
            ToolCallEndEvent(
                type=EventType.TOOL_CALL_END,
                tool_call_id=elicitation.tool_id,
            ),
        ]

    async def _persist_pause(
        self,
        context: "ConversationContext",
        elicitation: "ElicitationResponse",
    ) -> None:
        """Persist the paused run to the configured store."""
        ctx_snap: dict
        try:
            ctx_snap = context.model_dump(mode="json")
        except Exception:
            # defensive: callers may pass non-Pydantic stand-ins (tests).
            ctx_snap = {}

        snapshot = PausedRun(
            thread_id=self.thread_id,
            run_id=self.run_id,
            tool_call_id=elicitation.tool_id,
            tool_call_name=elicitation.tool_name,
            tool_args=dict(elicitation.details.input or {}),
            context_snapshot=ctx_snap,
            state_snapshot=self.state.snapshot(),
            metadata=dict(self.pause_metadata),
        )
        key = make_pause_key(self.thread_id, self.run_id)
        await self.paused_run_store.save(key, snapshot)

    # ag-ui event translation helpers --- these are for AG-UI events that don't have a 1:1 mapping
    def _translate(self, sev: StreamEvent) -> list[BaseEvent]:
        """Map one StreamEvent to zero-or-more AG-UI events.

        Delegates TOOL_CALL_START / TOOL_CALL_END to the tool dispatcher.
        DONE is consumed by the outer loop and emits no event.
        """
        if sev.type == StreamEventType.TEXT:
            return [
                TextMessageChunkEvent(
                    type=EventType.TEXT_MESSAGE_CHUNK,
                    message_id=self._assistant_message_id,
                    role="assistant",
                    delta=sev.text or "",
                )
            ]
        if sev.type == StreamEventType.ERROR:
            return [
                RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message=sev.error or "agent error",
                    code="agent_stream_error",
                )
            ]
        if sev.type in (
            StreamEventType.TOOL_CALL_START,
            StreamEventType.TOOL_CALL_ARGS,
            StreamEventType.TOOL_CALL_END,
        ):
            return self.tool_dispatcher.translate(sev)
        return []
