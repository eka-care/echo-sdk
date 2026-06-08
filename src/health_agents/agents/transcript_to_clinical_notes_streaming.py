from __future__ import annotations

from typing import Any, AsyncGenerator, Iterable, Optional

from echo.agents.generic_agent import GenericAgent
from echo.llm import LLMConfig
from echo.llm.schemas import StreamEvent

from .base import HealthAgentMixin, build_clinical_context, new_message_id


class TranscriptToClinicalNotesStreamingAgent(HealthAgentMixin, GenericAgent):
    SYSTEM_PROMPT = "transcript_to_clinical_notes_streaming"
    AGENT_NAME = "transcript_to_clinical_notes_streaming"

    def __init__(
        self,
        *,
        user_prompt: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        date: Optional[str] = None,
        **variables: Any,
    ) -> None:
        agent_config = self.build_config(
            user_prompt=user_prompt, date=date, **variables
        )
        super().__init__(agent_config=agent_config, llm_config=llm_config)

    async def generate_stream(
        self,
        transcript: str,
        *,
        out_msg_id: Optional[str] = None,
        extra_context: Optional[Iterable[str]] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        context = build_clinical_context(transcript, extra_context=extra_context)
        async for event in self.run_stream(context, out_msg_id or new_message_id()):
            yield event
