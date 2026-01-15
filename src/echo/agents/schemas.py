"""
Result models for Echo SDK.

Standardized output models for agents and tools.
"""

from typing import Optional

from pydantic import BaseModel, Field

from echo.llm.schemas import LLMResponse
from echo.models.user_conversation import ConversationContext


class AgentResult(BaseModel):
    """
    Standardized result from agent execution.

    All agents return this model regardless of execution framework.
    """

    llm_response: Optional[LLMResponse] = Field(
        description="LLM response from the agent"
    )
    context: ConversationContext = Field(
        description="Conversation context from the agent"
    )
    agent_name: str = Field(description="Name of the agent")
    error: Optional[str] = Field(
        default=None, description="Error message if execution failed"
    )

    @classmethod
    def from_error(cls, error: str) -> "AgentResult":
        """Create a failed result from an error message."""
        return cls(
            error=error,
        )
