"""AG-UI capable agent: composition wrapper around a BaseAgent.

External clients construct `AgUiAgent(some_base_agent)` and call
`run_stream` / `resume_stream` to drive an AG-UI front end. The wrapper is
deliberately decoupled from the agent class hierarchy so concrete agents
(GenericAgent and any subclass of BaseAgent) stay free of AG-UI deps.
"""

from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

import orjson
from ag_ui.core import (
    BaseEvent,
    EventType,
    RunAgentInput,
    RunErrorEvent,
    Tool,
    ToolCallResultEvent,
)

from echo.agents.base import BaseAgent
from echo.models.user_conversation import Message, MessageRole, ToolResult

from .persistence import PausedRunStore, make_pause_key
from .runner import AgUiRunner
from .state import AgUiState
from .tool_dispatcher import AgUiToolDispatcher

if TYPE_CHECKING:
    from echo.models.user_conversation import ConversationContext


class AgUiAgent:
    """AG-UI facade over a BaseAgent.

    Holds a reference to the underlying agent and translates its streaming
    output into AG-UI events. Pause/resume around UI-tool calls is handled
    here, not on the agent.
    """

    def __init__(self, agent: BaseAgent) -> None:
        self.agent = agent

    async def run_stream(
        self,
        context: "ConversationContext",
        run_input: RunAgentInput,
        state: AgUiState,
        out_msg_id: str,
        paused_run_store: Optional[PausedRunStore] = None,
        pause_metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Run the wrapped agent and yield AG-UI BaseEvents; pauses without RunFinished on a UI tool call."""
        ui_tool_names = {t.name for t in run_input.tools}
        dispatcher = AgUiToolDispatcher(ui_tool_names=ui_tool_names)
        runner = AgUiRunner(
            agent=self.agent,
            state=state,
            thread_id=run_input.thread_id,
            run_id=run_input.run_id,
            tool_dispatcher=dispatcher,
            paused_run_store=paused_run_store,
            pause_metadata=pause_metadata,
            ui_tools=list(run_input.tools),
        )
        async for ev in runner.stream(context, out_msg_id):
            yield ev

    async def resume_stream(
        self,
        paused_run_store: PausedRunStore,
        thread_id: str,
        run_id: str,
        tool_call_id: str,
        tool_result: Any,
        state: AgUiState,
        context: "ConversationContext",
        out_msg_id: str,
        pause_metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Resume a previously paused run with the FE-supplied tool result and continue streaming."""
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

        async for ev in self.run_stream(
            context=context,
            run_input=fresh_input,
            state=state,
            out_msg_id=out_msg_id,
            paused_run_store=paused_run_store,
            pause_metadata=pause_metadata,
        ):
            yield ev
