"""Base transcriber interface for the audio transcription module."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from .config import TranscriberConfig
from .schemas import AudioInput, TranscriptionResponse

logger = logging.getLogger(__name__)


class BaseTranscriber(ABC):
    """Abstract base class for audio transcription providers."""

    def __init__(self, config: TranscriberConfig):
        self.config = config
        self.model = config.model

    @abstractmethod
    async def transcribe(
        self,
        audio: AudioInput,
        prompt: Optional[str] = None,
        mime_type: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptionResponse:
        """Transcribe audio. If `prompt` is None, returns a verbatim transcription."""
        pass
