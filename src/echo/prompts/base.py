"""Base classes for Echo SDK prompt management."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel

from .schemas import AgentPrompt


class FetchedPrompt(BaseModel):
    """Prompt with ready-to-use AgentPrompt."""

    name: str
    version: Optional[str] = None
    agent_prompt: AgentPrompt  # Ready to use, already compiled

    model_config = {"arbitrary_types_allowed": True}


class PromptFetchError(Exception):
    """Raised when prompt fetch fails."""

    pass


class BasePromptProvider(ABC):
    """Abstract base class for prompt providers."""

    @abstractmethod
    async def get_prompt(
        self,
        name: str,
        version: Optional[str] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> FetchedPrompt:
        """
        Fetch prompt from provider and return ready-to-use AgentPrompt.

        Args:
            name: Prompt name/identifier
            version: Optional version number as string
            **variables: Variables to compile the prompt with

        Returns:
            FetchedPrompt with agent_prompt ready to use

        Raises:
            PromptFetchError: If fetch fails
        """
        pass
