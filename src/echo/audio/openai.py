"""
OpenAI audio transcription implementation.
"""

from io import BytesIO
from typing import Optional

from .base import BaseTranscriber
from .config import TranscriptionConfig
from .schemas import TranscriptionResult, TranscriptionSegment

_MIME_TO_EXT = {
    "audio/flac": "audio.flac",
    "audio/mp3": "audio.mp3",
    "audio/mpeg": "audio.mp3",
    "audio/mp4": "audio.mp4",
    "audio/m4a": "audio.m4a",
    "audio/ogg": "audio.ogg",
    "audio/wav": "audio.wav",
    "audio/webm": "audio.webm",
}


class OpenAITranscriber(BaseTranscriber):
    """OpenAI audio transcription provider (Whisper / GPT-4o-transcribe)."""

    def __init__(self, config: TranscriptionConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            if self.config.api_key:
                self._client = OpenAI(api_key=self.config.api_key)
            else:
                self._client = OpenAI()
        return self._client

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio using OpenAI's transcription API."""
        filename = _MIME_TO_EXT.get(mime_type, "audio.wav")
        response_format = self.config.response_format or "verbose_json"

        request_kwargs: dict = {
            "file": (filename, BytesIO(audio_bytes), mime_type),
            "model": self.config.model,
            "response_format": response_format,
        }

        if self.config.language:
            request_kwargs["language"] = self.config.language

        request_kwargs.update(kwargs)

        response = self.client.audio.transcriptions.create(**request_kwargs)

        return self._parse_response(response, response_format)

    def _parse_response(
        self, response, response_format: str
    ) -> TranscriptionResult:
        """Parse OpenAI transcription response."""
        # verbose_json returns an object with segments, language, duration
        if response_format == "verbose_json":
            segments: Optional[list] = None
            raw_segments = getattr(response, "segments", None)
            if raw_segments:
                segments = [
                    TranscriptionSegment(
                        text=seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", ""),
                        start=seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", None),
                        end=seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", None),
                    )
                    for seg in raw_segments
                ]

            return TranscriptionResult(
                text=getattr(response, "text", ""),
                segments=segments,
                language=getattr(response, "language", None),
                duration=getattr(response, "duration", None),
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        # Plain text or other formats
        text = response.text if hasattr(response, "text") else str(response)
        return TranscriptionResult(text=text)
