"""AG-UI capable agent: extends BaseAgent with AG-UI streaming + resume.

Mirrors `GenericAgent` (constructed with the standard BaseAgent kwargs) and
adds two AG-UI-specific entry points on top of the inherited `run` /
`run_stream`:

- `ag_ui_stream`: drive the agent and translate `StreamEvent`s into AG-UI
  `BaseEvent`s (RUN_STARTED / state frames / tool-call frames / text chunks
  / RUN_FINISHED). Pauses without a RUN_FINISHED when the agent elicits a
  UI tool call.
- `ag_ui_resume_stream`: re-enter a paused run with a FE-supplied tool
  result and continue streaming.
"""

import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional

import orjson
from ag_ui.core import (
    BaseEvent,
    EventType,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    TextMessageChunkEvent,
    Tool,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)

from echo.ag_ui.persistence import PausedRun, PausedRunStore, make_pause_key
from echo.ag_ui.state import AgUiState
from echo.ag_ui.tool_dispatcher import AgUiToolDispatcher
from echo.agents.base import BaseAgent
from echo.agents.schemas import AgentResult
from echo.llm.schemas import StreamEvent, StreamEventType
from echo.models.user_conversation import (
    ConversationContext,
    Message,
    MessageRole,
    ToolResult,
)

if TYPE_CHECKING:
    from echo.tools.core.schemas import ElicitationResponse

logger = logging.getLogger(__name__)


class AgUiAgent(BaseAgent):
    """AG-UI capable agent with all GenericAgent capabilities."""

    def __init__(self, **kwargs) -> None:
        if not kwargs.get("agent_prompt"):
            logger.error("agent_prompt is mandatory for ag_ui agent")
            raise Exception("agent_prompt is mandatory for ag_ui agent")
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Return the name of the agent."""
        return "ag_ui_agent"

    async def run(
        self,
        context: ConversationContext,
        out_msg_id: str,
    ) -> AgentResult:
        """Run the agent (non-streaming). Same semantics as GenericAgent.run."""
        return await self._run_agent(context, out_msg_id)

    async def run_stream(
        self,
        context: ConversationContext,
        out_msg_id: str,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream the agent's response as StreamEvents (BaseAgent contract)."""
        async for event in self._run_agent_stream(context, out_msg_id):
            yield event

    async def ag_ui_stream(
        self,
        context: ConversationContext,
        run_input: RunAgentInput,
        state: AgUiState,
        out_msg_id: str,
        paused_run_store: Optional[PausedRunStore] = None,
        pause_metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Run and yield AG-UI BaseEvents; pauses without RunFinished on a UI tool call."""
        ui_tool_names = {t.name for t in run_input.tools}
        tool_dispatcher = AgUiToolDispatcher(ui_tool_names=ui_tool_names)
        ui_tools = list(run_input.tools)
        thread_id = run_input.thread_id
        run_id = run_input.run_id
        # stable message_id so all TEXT chunks group into one assistant turn.
        assistant_message_id = str(uuid.uuid4())
        pause_metadata = pause_metadata or {}

        try:
            yield RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=thread_id,
                run_id=run_id,
            )
            yield StateSnapshotEvent(
                type=EventType.STATE_SNAPSHOT,
                snapshot=state.snapshot(),
            )
            state.begin_tracking()

            errored = False
            paused = False

            async for sev in self._run_agent_stream(context, out_msg_id):
                for ev in self._translate(
                    sev, assistant_message_id, tool_dispatcher
                ):
                    yield ev

                if sev.type == StreamEventType.ERROR:
                    errored = True
                    break

                # flush any state ops accumulated during this LLM event.
                ops = state.drain_pending_ops()
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
                            for ag_ev in self._synthesize_ui_tool_events(
                                elicit, tool_dispatcher
                            ):
                                yield ag_ev

                        # final state flush before pause.
                        tail = state.drain_pending_ops()
                        if tail:
                            yield StateDeltaEvent(
                                type=EventType.STATE_DELTA, delta=tail
                            )

                        if paused_run_store is not None:
                            await self._persist_pause(
                                context=context,
                                elicitation=elicitations[0],
                                thread_id=thread_id,
                                run_id=run_id,
                                state=state,
                                ui_tools=ui_tools,
                                pause_metadata=pause_metadata,
                                paused_run_store=paused_run_store,
                            )

                        paused = True
                    break

            # tail flush — in case the last event mutated state.
            if not paused:
                tail_ops = state.drain_pending_ops()
                if tail_ops:
                    yield StateDeltaEvent(
                        type=EventType.STATE_DELTA, delta=tail_ops
                    )

            if not errored and not paused:
                yield RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=thread_id,
                    run_id=run_id,
                )
                # clean completion: clear any prior paused-run entry
                # for this (thread_id, run_id) pair.
                if paused_run_store is not None:
                    await paused_run_store.delete(
                        make_pause_key(thread_id, run_id)
                    )

        except Exception as e:
            logger.error("AgUiAgent.ag_ui_stream raised: %s", e, exc_info=True)
            yield RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=str(e),
                code="runner_exception",
            )

    async def ag_ui_resume_stream(
        self,
        paused_run_store: PausedRunStore,
        thread_id: str,
        run_id: str,
        tool_call_id: str,
        tool_result: Any,
        state: AgUiState,
        context: ConversationContext,
        out_msg_id: str,
        pause_metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Resume a previously paused run with the FE-supplied tool result."""
        key = make_pause_key(thread_id, run_id)
        paused = await paused_run_store.load(key)
        if paused is None:
            yield RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=f"paused run not found or expired: {key}",
                code="paused_run_expired",
            )
            return
        if paused.tool_call_id != tool_call_id:
            yield RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=(
                    f"tool_call_id mismatch (paused on {paused.tool_call_id}, "
                    f"resume tried {tool_call_id})"
                ),
                code="tool_call_id_mismatch",
            )
            return

        if isinstance(tool_result, str):
            result_str = tool_result
        else:
            result_str = orjson.dumps(tool_result).decode()

        yield ToolCallResultEvent(
            type=EventType.TOOL_CALL_RESULT,
            message_id=out_msg_id,
            tool_call_id=tool_call_id,
            content=result_str,
            role="tool",
        )

        context.add_message(
            Message(
                role=MessageRole.TOOL,
                content=[ToolResult(tool_id=tool_call_id, result=result_str)],
                msg_id=out_msg_id,
            )
        )

        # Rebuild the FE-side tool surface from the paused snapshot so the
        # next turn can pause again on any of the originally declared tools.
        # `state`, `messages`, `context`, `forwarded_props` are intentionally
        # blank: state is carried by the `state` arg, history by `context`.
        fresh_tools = [Tool.model_validate(d) for d in paused.ui_tools]
        fresh_input = RunAgentInput(
            thread_id=thread_id,
            run_id=run_id,
            state={},
            messages=[],
            tools=fresh_tools,
            context=[],
            forwarded_props={},
        )

        async for ev in self.ag_ui_stream(
            context=context,
            run_input=fresh_input,
            state=state,
            out_msg_id=out_msg_id,
            paused_run_store=paused_run_store,
            pause_metadata=pause_metadata,
        ):
            yield ev

    # --- AG-UI translation helpers ---
    def _translate(
        self,
        sev: StreamEvent,
        assistant_message_id: str,
        tool_dispatcher: AgUiToolDispatcher,
    ) -> List[BaseEvent]:
        """Map one StreamEvent to zero-or-more AG-UI events."""
        if sev.type == StreamEventType.TEXT:
            return [
                TextMessageChunkEvent(
                    type=EventType.TEXT_MESSAGE_CHUNK,
                    message_id=assistant_message_id,
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
            return tool_dispatcher.translate(sev)
        return []

    def _synthesize_ui_tool_events(
        self,
        elicitation: "ElicitationResponse",
        tool_dispatcher: AgUiToolDispatcher,
    ) -> List[BaseEvent]:
        """Emit AG-UI ToolCallStart / Args / End events for an elicitation."""
        args = dict(elicitation.details.input or {})
        args_json = orjson.dumps(args).decode()

        tool_dispatcher.register_completed_call(
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
        context: ConversationContext,
        elicitation: "ElicitationResponse",
        thread_id: str,
        run_id: str,
        state: AgUiState,
        ui_tools: List[Tool],
        pause_metadata: Dict[str, Any],
        paused_run_store: PausedRunStore,
    ) -> None:
        """Persist the paused run to the configured store."""
        snapshot = PausedRun(
            thread_id=thread_id,
            run_id=run_id,
            tool_call_id=elicitation.tool_id,
            tool_call_name=elicitation.tool_name,
            tool_args=dict(elicitation.details.input or {}),
            context_snapshot=context.model_dump(mode="json"),
            state_snapshot=state.snapshot(),
            ui_tools=[t.model_dump(mode="json") for t in ui_tools],
            metadata=dict(pause_metadata),
        )
        key = make_pause_key(thread_id, run_id)
        await paused_run_store.save(key, snapshot)
