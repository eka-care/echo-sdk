"""OpenAI-compatible LLM provider (plan decision #15).

Points the OpenAI client at ANY OpenAI-compatible endpoint — vLLM or Ollama
serving Qwen/MedGemma locally, LiteLLM proxies, or hosted services. Strips the
OpenAI-only request parameters (max_completion_tokens heuristics,
reasoning_effort) that open-model servers reject.

Config: LLMConfig(provider="openai_compatible", base_url=..., model=...,
api_key=...) with env fallbacks ECHO_LLM_BASE_URL / ECHO_LLM_API_KEY.
"""

from __future__ import annotations

import os

from .config import LLMConfig
from .openai import OpenAILLM


class OpenAICompatibleLLM(OpenAILLM):
    """OpenAI wire format against a configurable base_url."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = (
            getattr(config, "base_url", None)
            or os.getenv("ECHO_LLM_BASE_URL")
            or "http://localhost:11434/v1"  # Ollama default
        )

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = (
                self.config.api_key
                or os.getenv("ECHO_LLM_API_KEY")
                or "not-needed"  # local vLLM/Ollama don't check keys
            )
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    # Open-model servers speak plain max_tokens and reject OpenAI-only params.
    def _uses_max_completion_tokens(self) -> bool:
        return False

    def _supports_reasoning_effort(self) -> bool:
        return False

    def _is_reasoning_model(self) -> bool:
        return False
