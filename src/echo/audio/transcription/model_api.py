"""Self-hosted MODEL API transcription provider.

Speaks the bare multipart transcription wire used by self-deployed model
services:

    POST {endpoint}?language=<code>
    multipart/form-data: file=<audio>

e.g. ``curl 'http://<model-host>/transcribe?language=hi' --form 'file=@1.mp3'``

The endpoint is the FULL transcribe URL — TranscriberConfig(base_url=...) or
env ``MODEL_API_TRANSCRIBE_URL``. Auth is optional (``api_key`` /
``MODEL_API_AUTH_TOKEN`` sent as a Bearer when set). The response is parsed
leniently: JSON with the transcript under ``text`` / ``transcript`` /
``transcription`` (optionally inside a ``data`` envelope), or a plain-text
body.
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


class ModelApiTranscriber(BaseTranscriber):
    """Multipart file upload against a configurable transcribe endpoint."""

    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
        self.endpoint = (
            config.base_url or os.getenv("MODEL_API_TRANSCRIBE_URL") or ""
        ).strip()
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
            logger.warning(
                "ModelApiTranscriber received a prompt but the model API does "
                "not accept prompts; ignoring."
            )
        try:
            if not self.endpoint:
                raise ValueError(
                    "model_api transcriber needs the transcribe endpoint — pass "
                    "TranscriberConfig(base_url=...) or set "
                    "MODEL_API_TRANSCRIBE_URL "
                    "(e.g. http://model-host/transcribe)."
                )
            audio_bytes, resolved_mime = self._resolve_audio(audio, mime_type)
            lang = language or self.config.language

            params = {"language": lang} if lang else None
            ext = _MIME_EXT.get((resolved_mime or "").lower(), "bin")
            files = {
                "file": (
                    f"audio.{ext}",
                    io.BytesIO(audio_bytes),
                    resolved_mime or "application/octet-stream",
                )
            }
            headers = {}
            token = self.config.api_key or os.getenv("MODEL_API_AUTH_TOKEN")
            if token:
                headers["Authorization"] = f"Bearer {token}"

            response = await self.client.post(
                self.endpoint, params=params, files=files, headers=headers
            )
            return self._parse_response(response, lang)

        except Exception as e:
            logger.error("ModelApiTranscriber transcribe error: %s", e, exc_info=True)
            return TranscriptionResponse(text="", error=str(e))

    def _resolve_audio(
        self, audio: AudioInput, mime_type: Optional[str]
    ) -> Tuple[bytes, Optional[str]]:
        if isinstance(audio, bytes):
            return audio, mime_type
        raise TypeError(
            "model_api transcriber accepts raw audio bytes only; "
            f"got {type(audio).__name__}."
        )

    def _parse_response(
        self, response, language: Optional[str]
    ) -> TranscriptionResponse:
        if response.status_code != 200:
            body_preview = (response.text or "")[:200]
            return TranscriptionResponse(
                text="",
                error=f"model API HTTP {response.status_code}: {body_preview}",
            )

        try:
            data = orjson.loads(response.content)
        except Exception:
            return TranscriptionResponse(
                text=(response.text or "").strip(), language_detected=language
            )

        if isinstance(data, str):
            return TranscriptionResponse(text=data.strip(), language_detected=language)

        if isinstance(data, dict):
            inner = data.get("data") if isinstance(data.get("data"), dict) else data
            for key in ("text", "transcript", "transcription"):
                value = inner.get(key)
                if isinstance(value, str):
                    detected = inner.get("language") or inner.get("language_detected")
                    return TranscriptionResponse(
                        text=value.strip(),
                        language_detected=detected
                        if isinstance(detected, str)
                        else language,
                        duration_s=inner.get("duration")
                        or inner.get("audio_duration"),
                    )

        return TranscriptionResponse(
            text="",
            error=(
                "model API returned an unrecognized transcription payload: "
                f"{str(data)[:200]}"
            ),
        )
