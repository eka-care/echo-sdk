"""Gemini audio transcription provider."""

import asyncio
import logging
import os
import urllib.request
from typing import Any, Optional

from .base import BaseTranscriber
from .config import TranscriberConfig
from .schemas import AudioInput, TokenUsage, TranscriptionResponse

logger = logging.getLogger(__name__)

DEFAULT_TRANSCRIPTION_PROMPT = (
    "Transcribe the audio verbatim. Preserve the spoken language and script. "
    "Do not annotate non-speech sounds (music, noise, laughter, breaths). "
    "Output only the transcription, with no preamble, commentary, or formatting."
)


class GeminiTranscriber(BaseTranscriber):
    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai

            api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "Gemini API key not provided. "
                    "Set GOOGLE_API_KEY env var or pass api_key in TranscriberConfig."
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    async def transcribe(
        self,
        audio: AudioInput,
        prompt: Optional[str] = None,
        mime_type: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptionResponse:
        try:
            from google.genai import types

            audio_part = self._build_audio_part(audio, mime_type)
            user_prompt = prompt or DEFAULT_TRANSCRIPTION_PROMPT

            contents = [
                types.Content(
                    role="user",
                    parts=[audio_part, types.Part.from_text(text=user_prompt)],
                )
            ]
            cfg = types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
            )

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=contents,
                config=cfg,
            )
            return self._parse_response(response)

        except Exception as e:
            logger.error("GeminiTranscriber transcribe error: %s", e, exc_info=True)
            return TranscriptionResponse(text="", error=str(e))

    def _build_audio_part(self, audio: AudioInput, mime_type: Optional[str]):
        from google.genai import types

        if isinstance(audio, bytes):
            if not mime_type:
                raise ValueError("mime_type is required when passing raw audio bytes")
            return types.Part.from_bytes(data=audio, mime_type=mime_type)

        if isinstance(audio, str):
            if audio.startswith(("http://", "https://")):
                req = urllib.request.Request(audio, headers={"User-Agent": "echo-sdk"})
                with urllib.request.urlopen(
                    req, timeout=self.config.request_timeout_s
                ) as resp:
                    data = resp.read()
                    detected_mime = mime_type or (
                        resp.headers.get("Content-Type", "").split(";")[0].strip()
                    )
                if not detected_mime:
                    raise ValueError(f"Could not determine mime_type for URL {audio!r}")
                return types.Part.from_bytes(data=data, mime_type=detected_mime)

            if not mime_type:
                file_obj = self.client.files.get(name=audio)
                mime_type = file_obj.mime_type
            return types.Part.from_uri(file_uri=audio, mime_type=mime_type)

        raise TypeError(f"Unsupported audio input type: {type(audio).__name__}")

    def _parse_response(self, response) -> TranscriptionResponse:
        text = ""
        if response.candidates and response.candidates[0].content.parts:
            text = "".join(
                p.text
                for p in response.candidates[0].content.parts
                if getattr(p, "text", None)
            )

        usage = None
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            usage = TokenUsage(
                input_tokens=getattr(um, "prompt_token_count", None),
                output_tokens=getattr(um, "candidates_token_count", None),
                total_tokens=getattr(um, "total_token_count", None),
            )

        return TranscriptionResponse(
            text=text.strip(),
            usage=usage,
            details={"model": self.model},
        )
