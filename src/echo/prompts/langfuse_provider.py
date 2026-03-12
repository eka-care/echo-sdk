"""Langfuse prompt provider for Echo SDK."""

import asyncio
import os
from typing import Any, Dict, Optional

from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig

from .base import BasePromptProvider, FetchedPrompt, PromptFetchError
from .observability import (
    PromptFetchMetadata,
    PromptObservationContext,
    PromptObservability,
    get_prompt_observability,
)


class LangfusePromptProvider(BasePromptProvider):
    """Langfuse prompt provider with lazy client initialization."""

    def __init__(self, observability: Optional[PromptObservability] = None):
        super().__init__(observability=observability or get_prompt_observability())
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
        version: Optional[str] = None,
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
        prompt_variables = dict(prompt_variables or {})
        metadata = PromptFetchMetadata(
            prompt_name=name,
            provider_name="langfuse",
            version=version,
            prompt_variables=prompt_variables,
        )

        context: Optional[PromptObservationContext] = None
        try:
            client = self.client
            context = self.observability.on_fetch_start(
                metadata, langfuse_client=client
            )

            kwargs: dict[str, Any] = {}
            if version is not None:
                kwargs["version"] = int(version)

            # Langfuse SDK is sync, run in executor
            loop = asyncio.get_event_loop()
            langfuse_prompt = await loop.run_in_executor(
                None, lambda: client.get_prompt(name, **kwargs)
            )

            # Compile with variables NOW — signature declares prompt_variables
            # as Optional, but Langfuse's compile(**x) rejects None. Normalize
            # at the provider boundary, right before the ** unpack.
            task_description = langfuse_prompt.compile(**(prompt_variables or {}))

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

            fetched_prompt = FetchedPrompt(
                name=name,
                version=str(getattr(langfuse_prompt, "version", "")) or None,
                agent_config=agent_config,
            )

            self.observability.on_fetch_success(
                context, metadata, {"version": fetched_prompt.version}
            )

            return fetched_prompt

        except Exception as exc:
            if context is None:
                context = self.observability.on_fetch_start(
                    metadata, langfuse_client=None
                )
            self.observability.on_fetch_failure(context, metadata, exc)
            raise PromptFetchError(f"Failed to fetch '{name}': {exc}") from exc
