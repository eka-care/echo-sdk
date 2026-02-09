"""Factory for dataset and evaluation provider instantiation."""

import os
from typing import Optional

from .base import BaseEvalProvider

_provider_instance: Optional[BaseEvalProvider] = None


def get_eval_provider(reset: bool = False) -> BaseEvalProvider:
    """
    Get Dataset and Evaluation provider singleton.

    Provider is determined by ECHO_EVAL_PROVIDER env var (default: langfuse).

    Args:
        reset: Force create new instance (useful for testing)

    Returns:
        BaseEvalProvider singleton instance

    Raises:
        ValueError: If unsupported provider specified
    """
    global _provider_instance

    if _provider_instance is None or reset:
        provider = os.getenv("ECHO_EVAL_PROVIDER", "langfuse").lower()

        if provider == "langfuse":
            from .langfuse_provider import LangfuseEvalProvider

            _provider_instance = LangfuseEvalProvider()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    return _provider_instance


def reset_eval_provider() -> None:
    """Reset the singleton (for testing)."""
    global _provider_instance
    _provider_instance = None
