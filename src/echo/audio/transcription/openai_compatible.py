"""OpenAI-compatible audio transcription provider.

Speaks the OpenAI audio wire format (``POST {base_url}/audio/transcriptions``,
multipart ``file`` + ``model``) against a configurable base_url, so any
self-hosted STT server that exposes the OpenAI API works: faster-whisper
server, speaches, vLLM audio, LocalAI, or an actual OpenAI-compatible proxy.

Config: TranscriberConfig(provider="openai_compatible", base_url=..., model=...,
api_key=...) with env fallbacks ECHO_TRANSCRIBER_BASE_URL /
ECHO_TRANSCRIBER_API_KEY. base_url follows the OpenAI convention and should
include the version prefix (e.g. ``http://stt.local:8000/v1``).
"""

import io
import logging
import os
from typing import Any, Optional, Tuple

import orjson

from .base import BaseTranscriber
from .config import TranscriberConfig
from .schemas import AudioInput, TranscriptionResponse

logger = logging.getLogger(__name__)

TRANSCRIPTIONS_PATH = "/audio/transcriptions"

_MIME_EXT = {
    "audio/m4a": "m4a",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/aac": "aac",
}


class OpenAICompatibleTranscriber(BaseTranscriber):
    """OpenAI audio wire format against a configurable base_url."""

    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
        self.base_url = (
            config.base_url or os.getenv("ECHO_TRANSCRIBER_BASE_URL") or ""
        ).rstrip("/")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self.config.request_timeout_s)
        return self._client

    async def transcribe(
        self,
        audio: AudioInput,
        prompt: Optional[str] = None,
        mime_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptionResponse:
        try:
            if not self.base_url:
                raise ValueError(
                    "openai_compatible transcriber needs a base_url — pass "
                    "TranscriberConfig(base_url=...) or set "
                    "ECHO_TRANSCRIBER_BASE_URL (e.g. http://stt.local:8000/v1)."
                )
            audio_bytes, resolved_mime = self._resolve_audio(audio, mime_type)
            lang = language or self.config.language

            data = {"model": self.model}
            if lang:
                # OpenAI wire expects ISO-639-1 — pass "hi-IN" style through as
                # its language part; servers that accept full tags still work.
                data["language"] = lang.split("-")[0]
            if prompt:
                data["prompt"] = prompt

            ext = _MIME_EXT.get((resolved_mime or "").lower(), "bin")
            files = {
                "file": (
                    f"audio.{ext}",
                    io.BytesIO(audio_bytes),
                    resolved_mime or "application/octet-stream",
                )
            }
            headers = {}
            api_key = self.config.api_key or os.getenv("ECHO_TRANSCRIBER_API_KEY")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            url = f"{self.base_url}{TRANSCRIPTIONS_PATH}"
            response = await self.client.post(
                url, data=data, files=files, headers=headers
            )
            return self._parse_response(response, lang)

        except Exception as e:
            logger.error(
                "OpenAICompatibleTranscriber transcribe error: %s", e, exc_info=True
            )
            return TranscriptionResponse(text="", error=str(e))

    def _resolve_audio(
        self, audio: AudioInput, mime_type: Optional[str]
    ) -> Tuple[bytes, Optional[str]]:
        if isinstance(audio, bytes):
            return audio, mime_type
        raise TypeError(
            "openai_compatible transcriber accepts raw audio bytes only; "
            f"got {type(audio).__name__}."
        )

    def _parse_response(
        self, response, language: Optional[str]
    ) -> TranscriptionResponse:
        if response.status_code != 200:
            body_preview = (response.text or "")[:200]
            return TranscriptionResponse(
                text="",
                error=f"openai_compatible STT HTTP {response.status_code}: {body_preview}",
            )
        data = orjson.loads(response.content)
        return TranscriptionResponse(
            text=(data.get("text") or "").strip(),
            language_detected=data.get("language") or language,
            duration_s=data.get("duration"),
        )
