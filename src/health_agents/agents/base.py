from __future__ import annotations

import uuid
from datetime import date as _date
from typing import Any, ClassVar, Dict, Iterable, Optional

from echo.agents.config import AgentConfig
from health_agents.prompts import HealthPrompts
from echo.models.user_conversation import (
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
)


def _today() -> str:
    return _date.today().isoformat()


def new_message_id() -> str:
    return str(uuid.uuid4())


def build_clinical_context(
    transcript: str,
    *,
    extra_context: Optional[Iterable[str]] = None,
    tool_context: Optional[Dict[str, Any]] = None,
) -> ConversationContext:
    context = ConversationContext()
    if tool_context:
        context.system_context["tool_context"] = dict(tool_context)
    context.add_message(
        Message(role=MessageRole.USER, content=[TextMessage(text=transcript)])
    )
    for item in extra_context or []:
        context.add_message(
            Message(role=MessageRole.USER, content=[TextMessage(text=item)])
        )
    return context


class HealthAgentMixin:
    SYSTEM_PROMPT: ClassVar[str]
    AGENT_NAME: ClassVar[str]

    @property
    def name(self) -> str:
        return self.AGENT_NAME

    @classmethod
    def build_config(
        cls,
        *,
        user_prompt: Optional[str] = None,
        prompts: Optional[HealthPrompts] = None,
        date: Optional[str] = None,
        **variables: Any,
    ) -> AgentConfig:
        store = prompts or HealthPrompts()
        variables.setdefault("date", date or _today())
        return store.build_agent_config(cls.SYSTEM_PROMPT, user_prompt, **variables)
