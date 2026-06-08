from __future__ import annotations

from typing import Any, Iterable, Optional

from echo.agents.generic_agent import GenericAgent
from echo.agents.schemas import AgentResult
from echo.llm import LLMConfig

from .base import HealthAgentMixin, build_clinical_context, new_message_id


class TranscriptToClinicalNotesAgent(HealthAgentMixin, GenericAgent):
    SYSTEM_PROMPT = "transcript_to_clinical_notes"
    AGENT_NAME = "transcript_to_clinical_notes"

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

    async def generate(
        self,
        transcript: str,
        *,
        out_msg_id: Optional[str] = None,
        extra_context: Optional[Iterable[str]] = None,
    ) -> AgentResult:
        context = build_clinical_context(transcript, extra_context=extra_context)
        return await self.run(context, out_msg_id or new_message_id())
