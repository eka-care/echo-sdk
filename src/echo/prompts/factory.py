"""Factory for prompt provider instantiation."""

import os
from typing import Optional

from .base import BasePromptProvider

_provider_instance: Optional[BasePromptProvider] = None


def get_prompt_provider(reset: bool = False) -> BasePromptProvider:
    """
    Get prompt provider singleton.

    Provider is determined by ECHO_PROMPT_PROVIDER env var (default: langfuse).

    Args:
        reset: Force create new instance (useful for testing)

    Returns:
        BasePromptProvider singleton instance

    Raises:
        ValueError: If unsupported provider specified
    """
    global _provider_instance

    if _provider_instance is None or reset:
        provider = os.getenv("ECHO_PROMPT_PROVIDER", "langfuse").lower()

        if provider == "langfuse":
            from .langfuse_provider import LangfusePromptProvider

            _provider_instance = LangfusePromptProvider()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    return _provider_instance


def reset_prompt_provider() -> None:
    """Reset the singleton (for testing)."""
    global _provider_instance
    _provider_instance = None
