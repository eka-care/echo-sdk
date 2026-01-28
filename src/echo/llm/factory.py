"""
LLM factory for Echo SDK.

Creates LLM instances based on configuration.
"""

from typing import Optional

from .base import BaseLLM
from .config import LLMConfig


def generate_llm_config(
    provider: str,
    model: str,
    temperature: Optional[float] = 0.2,
    region: Optional[str] = None,
    max_tokens: Optional[int] = 4096,
    max_iterations: Optional[int] = 5,
) -> LLMConfig:
    """Generate LLM config from environment variables."""
    return LLMConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        region=region,
        max_tokens=max_tokens,
        max_iterations=max_iterations,
    )


def get_llm(config: LLMConfig) -> BaseLLM:
    """
    Get an LLM instance based on configuration.

    Args:
        config: LLM configuration. Defaults to Bedrock Haiku.

    Returns:
        BaseLLM instance

    Raises:
        ValueError: If provider is not supported
        ImportError: If provider dependencies are not installed
    """
    provider = config.provider.lower()

    if provider == "bedrock":
        try:
            from .bedrock import BedrockLLM

            return BedrockLLM(config)
        except ImportError:
            raise ImportError(
                "boto3 is required for Bedrock. Install with: pip install boto3"
            )

    elif provider == "openai":
        try:
            from .openai import OpenAILLM

            return OpenAILLM(config)
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI. Install with: pip install openai"
            )

    elif provider == "anthropic":
        try:
            from .anthropic import AnthropicLLM

            return AnthropicLLM(config)
        except ImportError:
            raise ImportError(
                "anthropic is required for Anthropic. Install with: pip install anthropic"
            )

    elif provider == "gemini":
        try:
            from .gemini import GeminiLLM

            return GeminiLLM(config)
        except ImportError:
            raise ImportError(
                "google-generativeai is required for Gemini. "
                "Install with: pip install google-generativeai"
            )

    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: bedrock, openai, anthropic, gemini"
        )
