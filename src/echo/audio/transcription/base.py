"""Base transcriber interfaces for the audio transcription module.

Two families live here:

* ``BaseTranscriber`` — one-shot (batch) transcription of a complete audio
  payload.
* ``BaseStreamingTranscriber`` / ``StreamingTranscriptionSession`` — real-time
  transcription over a provider websocket. Audio goes in as raw PCM chunks and
  ``StreamingTranscriptEvent``s come out as the provider produces them.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncContextManager, AsyncIterator, Optional

from .config import TranscriberConfig
from .schemas import AudioInput, StreamingTranscriptEvent, TranscriptionResponse

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


class StreamingTranscriptionSession(ABC):
    """One open provider stream, obtained from ``BaseStreamingTranscriber.stream()``.

    Contract:

    * ``send_audio`` takes raw ``pcm_s16le`` mono bytes at ``config.sample_rate``.
      The provider class does any base64 / envelope wrapping. Callers own chunk
      sizing; 100-250 ms per call (3 200-8 000 bytes at 16 kHz) works for both
      providers.
    * ``events()`` yields until a ``DONE`` event or an ``ERROR`` with
      ``recoverable=False``; after that the session is finished.
    * ``flush()`` asks the provider to commit whatever audio it is holding.
    * ``close()`` is idempotent and safe to call from ``finally``.
    """

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None: ...

    @abstractmethod
    async def flush(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def events(self) -> AsyncIterator[StreamingTranscriptEvent]: ...

    def __aiter__(self) -> AsyncIterator[StreamingTranscriptEvent]:
        return self.events()


class BaseStreamingTranscriber(ABC):
    """Abstract base class for real-time (websocket) transcription providers."""

    def __init__(self, config: TranscriberConfig):
        self.config = config
        self.model = config.model

    @abstractmethod
    def stream(self) -> AsyncContextManager[StreamingTranscriptionSession]:
        """Open a provider stream.

        Usage::

            async with transcriber.stream() as session:
                await session.send_audio(pcm_chunk)
                async for event in session:
                    ...
        """
        ...
