from __future__ import annotations

from typing import Any, AsyncGenerator, List, Optional

from echo.agents.schemas import AgentResult
from echo.llm import LLMConfig
from echo.llm.schemas import StreamEvent
from echo.models.user_conversation import ConversationContext
from echo.tools.base_tool import BaseTool

from .base import HealthAgentMixin

_NOT_IMPLEMENTED = "DocAssist is scaffolded only; implementation is deferred."


class DocAssistAgent(HealthAgentMixin):
    SYSTEM_PROMPT = "docassist"
    AGENT_NAME = "docassist"

    def __init__(
        self,
        *,
        bearer_token: str,
        llm_config: Optional[LLMConfig] = None,
        user_prompt: Optional[str] = None,
        **variables: Any,
    ) -> None:
        self._bearer_token = bearer_token
        self._llm_config = llm_config
        self._user_prompt = user_prompt
        self._variables = variables
        self._tools: List[BaseTool] = []
        self._agent: Any = None 

    async def setup_tools(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    async def run(
        self,
        context: ConversationContext,
        out_msg_id: Optional[str] = None,
    ) -> AgentResult:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    async def stream(
        self,
        context: ConversationContext,
        out_msg_id: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        raise NotImplementedError(_NOT_IMPLEMENTED)
        yield  # pragma: no cover — marks this as an async generator
