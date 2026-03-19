"""
Audio transcriber factory.
"""

from .base import BaseTranscriber
from .config import TranscriptionConfig


def get_transcriber(config: TranscriptionConfig) -> BaseTranscriber:
    """Get a transcriber instance based on configuration.

    Args:
        config: Transcription configuration.

    Returns:
        BaseTranscriber instance.

    Raises:
        ValueError: If provider is not supported.
        ImportError: If provider dependencies are not installed.
    """
    provider = config.provider.lower()

    if provider == "openai":
        try:
            from .openai import OpenAITranscriber

            return OpenAITranscriber(config)
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI transcription. "
                "Install with: pip install openai"
            )

    elif provider == "gemini":
        try:
            from .gemini import GeminiTranscriber

            return GeminiTranscriber(config)
        except ImportError:
            raise ImportError(
                "google-genai is required for Gemini transcription. "
                "Install with: pip install google-genai"
            )

    else:
        raise ValueError(
            f"Unsupported transcription provider: {provider}. "
            f"Supported providers: openai, gemini"
        )
