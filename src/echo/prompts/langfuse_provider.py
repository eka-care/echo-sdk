"""Langfuse prompt provider for Echo SDK."""

import asyncio
import os
from typing import Any, Dict, Optional

from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig

from .base import BasePromptProvider, FetchedPrompt, PromptFetchError


class LangfusePromptProvider(BasePromptProvider):
    """Langfuse prompt provider with lazy client initialization."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Langfuse client using env vars."""
        if self._client is None:
            try:
                from langfuse import Langfuse
            except ImportError:
                raise ImportError(
                    "langfuse required. Install: pip install 'echo[langfuse]'"
                )

            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")

            if not public_key or not secret_key:
                raise ValueError(
                    "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY env vars"
                )

            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
            )
        return self._client

    async def get_prompt(
        self,
        name: str,
        version: Optional[int] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
    ) -> FetchedPrompt:
        """
        Fetch prompt from Langfuse and return ready-to-use AgentConfig.

        Args:
            name: Prompt name in Langfuse
            version: Optional version number
            **variables: Variables to compile the prompt with

        Returns:
            FetchedPrompt with agent_config ready to use

        Raises:
            PromptFetchError: If fetch fails
        """
        try:
            kwargs: dict[str, Any] = {}
            if version is not None:
                kwargs["version"] = version

            # Langfuse SDK is sync, run in executor
            loop = asyncio.get_event_loop()
            langfuse_prompt = await loop.run_in_executor(
                None, lambda: self.client.get_prompt(name, **kwargs)
            )

            # Compile with variables NOW
            task_description = langfuse_prompt.compile(**prompt_variables)

            # Extract config fields (provider-specific logic stays HERE)
            config = getattr(langfuse_prompt, "config", {}) or {}

            agent_config = AgentConfig(
                persona=PersonaConfig(
                    role=config.get("role"),
                    goal=config.get("goal"),
                    backstory=config.get("backstory"),
                ),
                task=TaskConfig(
                    description=task_description,
                    expected_output=config.get("expected_output"),
                ),
            )

            return FetchedPrompt(
                name=name,
                version=getattr(langfuse_prompt, "version", None),
                agent_config=agent_config,
            )

        except Exception as e:
            raise PromptFetchError(f"Failed to fetch '{name}': {e}")
