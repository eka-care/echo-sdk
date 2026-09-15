"""Factory for the audio transcription module."""

import logging
from typing import Optional

from .base import BaseStreamingTranscriber, BaseTranscriber
from .config import STREAMING_PROVIDERS, TranscriberConfig

logger = logging.getLogger(__name__)


def generate_transcriber_config(
    provider: str = "gemini",
    model: str = "models/gemini-2.5-flash",
    api_key: Optional[str] = None,
    language: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
) -> TranscriberConfig:
    return TranscriberConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        language=language,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def get_transcriber(config: TranscriberConfig) -> BaseTranscriber:
    """Get a transcriber instance for the configured provider.

    Raises:
        ValueError: If provider is not supported.
        ImportError: If provider dependencies are not installed.
    """
    provider = config.provider.lower()

    if provider == "gemini":
        try:
            from .gemini import GeminiTranscriber

            return GeminiTranscriber(config)
        except ImportError as e:
            raise ImportError(
                "google-genai is required for Gemini transcription. "
                "Install with: pip install 'echo-sdk[gemini]'"
            ) from e

    if provider == "ekacare":
        try:
            from .ekacare import EkaCareTranscriber

            return EkaCareTranscriber(config)
        except ImportError as e:
            raise ImportError(
                "httpx is required for Eka Care transcription. "
                "Install with: pip install 'echo-sdk[ekacare]'"
            ) from e

    if provider == "sarvam":
        try:
            from .sarvam import SarvamTranscriber

            return SarvamTranscriber(config)
        except ImportError as e:
            raise ImportError(
                "sarvamai is required for Sarvam transcription. "
                "Install with: pip install 'echo-sdk[sarvam]'"
            ) from e

    if provider == "elevenlabs":
        # Only the realtime (Scribe v2) path is wired up today.
        raise ValueError(
            "Provider 'elevenlabs' is streaming-only in echo; "
            "use get_streaming_transcriber(config) instead."
        )

    raise ValueError(
        f"Unsupported transcription provider: {provider!r}. "
        f"Supported: gemini, ekacare, sarvam"
    )


def get_streaming_transcriber(config: TranscriberConfig) -> BaseStreamingTranscriber:
    """Get a real-time (websocket) transcriber for the configured provider.

    Raises:
        ValueError: If the provider does not support streaming.
        ImportError: If provider dependencies are not installed.
    """
    provider = config.provider.lower()

    if provider == "sarvam":
        try:
            from .sarvam import SarvamStreamingTranscriber

            return SarvamStreamingTranscriber(config)
        except ImportError as e:
            raise ImportError(
                "sarvamai is required for Sarvam streaming transcription. "
                "Install with: pip install 'echo-sdk[sarvam]'"
            ) from e

    if provider == "elevenlabs":
        try:
            from .elevenlabs import ElevenLabsStreamingTranscriber

            return ElevenLabsStreamingTranscriber(config)
        except ImportError as e:
            raise ImportError(
                "elevenlabs is required for ElevenLabs streaming transcription. "
                "Install with: pip install 'echo-sdk[elevenlabs]'"
            ) from e

    raise ValueError(
        f"Streaming transcription not supported for provider {provider!r}. "
        f"Supported: {', '.join(STREAMING_PROVIDERS)}"
    )
