"""
AgUiRunner: translates an echo agent.run_stream() into AG-UI events.

PR-S2 scope: text-only path. The runner emits:

    RUN_STARTED
    STATE_SNAPSHOT(snapshot)
    [ TEXT_MESSAGE_CHUNK ... STATE_DELTA(ops) ... ]*
    RUN_FINISHED                    (or RUN_ERROR on failure)

Tool dispatch (TOOL_CALL_*) and pause/resume (paused-run store) land in
PR-S3 and PR-S4. In PR-S2 the runner silently ignores TOOL_CALL_START /
TOOL_CALL_END events from the LLM stream — they will be wired up in PR-S3.

Caller responsibilities:

    1. Pre-populate the AgUiState before constructing the runner — the
       initial STATE_SNAPSHOT is taken from state.snapshot().
    2. Provide a fresh thread_id / run_id pair (typically from the FE's
       RunAgentInput).
    3. Consume the async generator until exhaustion. The runner guarantees
       exactly one terminal event (RUN_FINISHED or RUN_ERROR) under
       normal control flow.
"""

import logging
import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from ag_ui.core import (
    BaseEvent,
    EventType,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    TextMessageChunkEvent,
)

from echo.llm.schemas import StreamEvent, StreamEventType

from .state import AgUiState

if TYPE_CHECKING:
    from echo.agents.base import BaseAgent
    from echo.models.user_conversation import ConversationContext

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
    ) -> None:
        self.agent = agent
        self.state = state
        self.thread_id = thread_id
        self.run_id = run_id
        # Stable message_id so all TEXT chunks group into one assistant turn.
        self._assistant_message_id = str(uuid.uuid4())

    async def stream(
        self,
        context: "ConversationContext",
        out_msg_id: str,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Run the agent and yield AG-UI events.

        Always terminates with exactly one of RUN_FINISHED or RUN_ERROR
        under normal control flow.
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
            async for sev in self.agent.run_stream(context, out_msg_id):
                for ev in self._translate(sev):
                    yield ev

                if sev.type == StreamEventType.ERROR:
                    errored = True
                    break

                # Flush any state ops accumulated during this LLM event.
                ops = self.state.drain_pending_ops()
                if ops:
                    yield StateDeltaEvent(type=EventType.STATE_DELTA, delta=ops)

                if sev.type == StreamEventType.DONE:
                    break

            # Tail flush — in case the last event mutated state.
            tail_ops = self.state.drain_pending_ops()
            if tail_ops:
                yield StateDeltaEvent(type=EventType.STATE_DELTA, delta=tail_ops)

            if not errored:
                yield RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=self.thread_id,
                    run_id=self.run_id,
                )

        except Exception as e:
            logger.error("AgUiRunner.stream raised: %s", e, exc_info=True)
            yield RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=str(e),
                code="runner_exception",
            )

    # --- StreamEvent → AG-UI event translation ---

    def _translate(self, sev: StreamEvent) -> list[BaseEvent]:
        """Map one StreamEvent to zero-or-more AG-UI events.

        PR-S2 handles TEXT and ERROR. TOOL_CALL_START / TOOL_CALL_END are
        silently ignored here — PR-S3 introduces the tool dispatcher.
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
        return []
