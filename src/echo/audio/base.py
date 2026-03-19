"""
Base transcriber abstract class.
"""

from abc import ABC, abstractmethod

from .config import TranscriptionConfig
from .schemas import TranscriptionResult


class BaseTranscriber(ABC):
    """Abstract base class for audio transcription providers."""

    def __init__(self, config: TranscriptionConfig):
        self.config = config

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data.
            mime_type: MIME type of the audio (e.g. "audio/wav", "audio/mp3").
            **kwargs: Additional provider-specific arguments.

        Returns:
            TranscriptionResult with text and optional segments.
        """
