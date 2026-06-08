from __future__ import annotations

import uuid
from typing import Any, AsyncGenerator, List, Optional

from ag_ui.core import BaseEvent, RunAgentInput

# Import via the `echo.ag_ui` package (not `echo.agents.ag_ui_agent`) to avoid a
# circular import between that module and `echo.ag_ui.__init__`.
from echo.ag_ui import AgUiAgent
from echo.llm import LLMConfig
from echo.tools.base_tool import BaseTool

from ..tools.ag_ui import DocumentState, build_section_tools
from .base import HealthAgentMixin, build_clinical_context, new_message_id


class TranscriptToClinicalNotesAgUiAgent(HealthAgentMixin, AgUiAgent):
    SYSTEM_PROMPT = "transcript_to_clinical_notes_ag_ui"
    AGENT_NAME = "transcript_to_clinical_notes_ag_ui"

    def __init__(
        self,
        *,
        user_prompt: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        tools: Optional[List[BaseTool]] = None,
        date: Optional[str] = None,
        **variables: Any,
    ) -> None:
        agent_config = self.build_config(
            user_prompt=user_prompt, date=date, **variables
        )
        super().__init__(
            agent_config=agent_config,
            llm_config=llm_config,
            tools=tools if tools is not None else build_section_tools(),
        )

    async def stream(
        self,
        transcript: str,
        *,
        thread_id: Optional[str] = None,
        run_id: Optional[str] = None,
        out_msg_id: Optional[str] = None,
        document_state: Optional[DocumentState] = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        state = document_state or DocumentState()
        context = build_clinical_context(
            transcript, tool_context={"document_state": state}
        )
        run_input = RunAgentInput(
            thread_id=thread_id or str(uuid.uuid4()),
            run_id=run_id or str(uuid.uuid4()),
            state={},
            messages=[],
            tools=[],
            context=[],
            forwarded_props={},
        )
        async for event in self.ag_ui_stream(
            context=context,
            run_input=run_input,
            state=state,
            out_msg_id=out_msg_id or new_message_id(),
        ):
            yield event
