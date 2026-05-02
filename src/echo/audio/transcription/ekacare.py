"""Eka Care audio transcription provider (HTTP ASR API)."""

import io
import logging
import os
import urllib.request
from typing import Any, Optional, Tuple

import orjson

from .base import BaseTranscriber
from .config import EKACARE_LANGUAGES, TranscriberConfig
from .schemas import AudioInput, TranscriptionResponse

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "en-IN"
DEFAULT_BASE_URL = "https://api.eka.care"
ASR_PATH = "/voice/get-asr"

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
}


class EkaCareTranscriber(BaseTranscriber):
    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
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
        if prompt is not None:
            preview = prompt[:80] + ("..." if len(prompt) > 80 else "")
            logger.warning(
                "EkaCareTranscriber received prompt=%r but the Eka ASR API "
                "does not accept prompts; ignoring.",
                preview,
            )

        try:
            lang = language or self.config.language or DEFAULT_LANGUAGE
            if lang not in EKACARE_LANGUAGES:
                raise ValueError(
                    f"Language {lang!r} not supported. Supported: {EKACARE_LANGUAGES}"
                )

            audio_bytes, resolved_mime = self._resolve_audio(audio, mime_type)

            token = self.config.api_key or os.getenv("EKA_API_TOKEN")
            if not token:
                raise ValueError(
                    "Eka Care auth token not provided. "
                    "Set EKA_API_TOKEN env var or pass api_key in TranscriberConfig."
                )

            url = f"{DEFAULT_BASE_URL}{ASR_PATH}"
            params = {"language": lang}

            ext = _MIME_EXT.get(resolved_mime, "bin") if resolved_mime else "bin"
            files = {
                "file": (
                    f"audio.{ext}",
                    io.BytesIO(audio_bytes),
                    resolved_mime or "application/octet-stream",
                )
            }
            headers = {"Authorization": f"Bearer {token}"}

            response = await self.client.post(
                url, params=params, headers=headers, files=files
            )
            return self._parse_response(response, lang)

        except Exception as e:
            logger.error("EkaCareTranscriber transcribe error: %s", e, exc_info=True)
            return TranscriptionResponse(text="", error=str(e))

    def _resolve_audio(
        self, audio: AudioInput, mime_type: Optional[str]
    ) -> Tuple[bytes, Optional[str]]:
        if isinstance(audio, bytes):
            return audio, mime_type

        if isinstance(audio, str):
            if audio.startswith(("http://", "https://")):
                req = urllib.request.Request(audio, headers={"User-Agent": "echo-sdk"})
                with urllib.request.urlopen(
                    req, timeout=self.config.request_timeout_s
                ) as resp:
                    data = resp.read()
                    detected_mime = mime_type or (
                        resp.headers.get("Content-Type", "").split(";")[0].strip()
                        or None
                    )
                return data, detected_mime

            raise ValueError(
                f"Files API URI {audio!r} is not supported by Eka Care; "
                "pass raw bytes or a public URL."
            )

        raise TypeError(f"Unsupported audio input type: {type(audio).__name__}")

    def _parse_response(self, response, language: str) -> TranscriptionResponse:
        if response.status_code != 200:
            body_preview = (response.text or "")[:200]
            return TranscriptionResponse(
                text="",
                error=f"Eka ASR HTTP {response.status_code}: {body_preview}",
            )

        data = orjson.loads(response.content)
        return TranscriptionResponse(
            text=(data.get("text") or "").strip(),
            language_detected=language,
            duration_s=data.get("audio_duration"),
            details={"txn_id": data.get("txn_id")},
        )
