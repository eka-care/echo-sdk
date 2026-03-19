"""
Google Gemini audio transcription implementation.
"""

import os
from typing import Optional

import orjson

from .base import BaseTranscriber
from .config import TranscriptionConfig
from .schemas import TranscriptionResult, TranscriptionSegment

_DEFAULT_PROMPT = (
    "Transcribe the following audio accurately. "
    "Return a JSON object with the following fields:\n"
    '- "text": the full transcription text\n'
    '- "segments": an array of objects with "text", "start" (seconds), "end" (seconds)\n'
    '- "language": the detected language code (e.g. "en")\n'
    "Return ONLY the JSON object, no markdown or extra text."
)


class GeminiTranscriber(BaseTranscriber):
    """Google Gemini audio transcription provider."""

    def __init__(self, config: TranscriptionConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            from google import genai

            api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key required for Gemini: provide api_key in config "
                    "or set GOOGLE_API_KEY env var"
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio using Gemini's multimodal API."""
        from google.genai import types

        prompt = self.config.prompt or _DEFAULT_PROMPT

        response = await self.client.aio.models.generate_content(
            model=self.config.model,
            contents=[
                prompt,
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            ],
        )

        response_text = response.text or ""
        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> TranscriptionResult:
        """Parse Gemini response, attempting JSON first, falling back to plain text."""
        # Strip markdown code fences if present
        text = response_text.strip()
        if text.startswith("```"):
            # Remove opening fence (e.g. ```json)
            first_newline = text.index("\n") if "\n" in text else len(text)
            text = text[first_newline + 1 :]
            # Remove closing fence
            if text.endswith("```"):
                text = text[: -3].strip()

        try:
            data = orjson.loads(text)
        except (orjson.JSONDecodeError, ValueError):
            return TranscriptionResult(text=response_text.strip())

        segments: Optional[list] = None
        raw_segments = data.get("segments")
        if raw_segments and isinstance(raw_segments, list):
            segments = [
                TranscriptionSegment(
                    text=seg.get("text", ""),
                    start=seg.get("start"),
                    end=seg.get("end"),
                    speaker=seg.get("speaker"),
                )
                for seg in raw_segments
                if isinstance(seg, dict)
            ]

        return TranscriptionResult(
            text=data.get("text", response_text.strip()),
            segments=segments,
            language=data.get("language"),
            duration=data.get("duration"),
            raw_response=data,
        )
