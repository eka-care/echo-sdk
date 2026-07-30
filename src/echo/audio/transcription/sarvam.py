"""Sarvam AI speech-to-text provider (on-prem default STT — plan decision #14).

Uses Sarvam's Speech-to-Text API (saarika models) via httpx. Strong Indic
language coverage. Configure with SARVAM_API_KEY; base URL overridable for
proxies via config.base_url / SARVAM_BASE_URL.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

from .base import BaseTranscriber
from .config import TranscriberConfig
from .schemas import AudioInput, TranscriptionResponse

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.sarvam.ai"
DEFAULT_MODEL = "saarika:v2.5"

# echo language codes → Sarvam language_code (BCP-47-ish)
_LANGUAGE_MAP = {
    "en": "en-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "od": "od-IN",
    "or": "od-IN",
    "pa": "pa-IN",
    "ta": "ta-IN",
    "te": "te-IN",
}

_EXT_BY_MIME = {
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
}


class SarvamTranscriber(BaseTranscriber):
    """Transcriber backed by Sarvam's /speech-to-text endpoint."""

    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("SARVAM_API_KEY")
        self.base_url = (
            getattr(config, "base_url", None)
            or os.getenv("SARVAM_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        if not self.api_key:
            raise ValueError("Sarvam requires an API key (SARVAM_API_KEY)")

    def _language_code(self) -> str:
        lang = (self.config.language or "").strip()
        if not lang:
            return "unknown"  # Sarvam auto-detects
        return _LANGUAGE_MAP.get(lang.lower(), lang)

    async def transcribe(
        self,
        audio: AudioInput,
        prompt: Optional[str] = None,
        mime_type: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptionResponse:
        try:
            if isinstance(audio, str):
                if audio.startswith(("http://", "https://")):
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        resp = await client.get(audio)
                        resp.raise_for_status()
                        content = resp.content
                else:
                    with open(audio, "rb") as f:
                        content = f.read()
            else:
                content = audio

            mime = mime_type or "audio/mp4"
            ext = _EXT_BY_MIME.get(mime.split(";")[0].strip().lower(), "m4a")

            data = {
                "model": self.model or DEFAULT_MODEL,
                "language_code": self._language_code(),
            }
            files = {"file": (f"audio.{ext}", content, mime)}

            async with httpx.AsyncClient(
                timeout=self.config.request_timeout_s
            ) as client:
                resp = await client.post(
                    f"{self.base_url}/speech-to-text",
                    headers={"api-subscription-key": self.api_key},
                    data=data,
                    files=files,
                )

            if resp.status_code != 200:
                return TranscriptionResponse(
                    error=f"Sarvam API {resp.status_code}: {resp.text[:500]}"
                )

            payload = resp.json()
            return TranscriptionResponse(
                text=payload.get("transcript", "") or "",
                language_detected=payload.get("language_code"),
                details={"request_id": payload.get("request_id")},
            )
        except Exception as e:  # network, IO
            logger.exception("Sarvam transcription failed")
            return TranscriptionResponse(error=str(e))
