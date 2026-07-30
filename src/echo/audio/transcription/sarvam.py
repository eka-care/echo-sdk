"""Sarvam AI speech-to-text provider, built on the official ``sarvamai`` SDK.

Strong Indic language coverage (saarika models). Configure with
SARVAM_API_KEY; base URL overridable for proxies/self-hosted gateways via
``TranscriberConfig.base_url`` or SARVAM_BASE_URL.

Install: pip install 'echo-sdk[sarvam]'  (extra pulls ``sarvamai``)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .base import BaseTranscriber
from .config import TranscriberConfig
from .schemas import AudioInput, TranscriptionResponse

logger = logging.getLogger(__name__)

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
    "as": "as-IN",
    "ur": "ur-IN",
    "ne": "ne-IN",
}

_EXT_BY_MIME = {
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/aac": "aac",
    "audio/flac": "flac",
}


class SarvamTranscriber(BaseTranscriber):
    """Transcriber backed by Sarvam's speech-to-text via the official SDK."""

    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("SARVAM_API_KEY")
        self.base_url = (
            getattr(config, "base_url", None) or os.getenv("SARVAM_BASE_URL") or None
        )
        if not self.api_key:
            raise ValueError("Sarvam requires an API key (SARVAM_API_KEY)")
        self._client = None

    @property
    def client(self):
        """Lazy AsyncSarvamAI, with optional endpoint override."""
        if self._client is None:
            from sarvamai import AsyncSarvamAI

            kwargs: dict = {"api_subscription_key": self.api_key}
            if self.base_url:
                from sarvamai.environment import SarvamAIEnvironment

                base = self.base_url.rstrip("/")
                kwargs["environment"] = SarvamAIEnvironment(
                    base=base,
                    production=base.replace("https://", "wss://").replace(
                        "http://", "ws://"
                    ),
                )
            self._client = AsyncSarvamAI(**kwargs)
        return self._client

    def _language_code(self) -> str:
        lang = (self.config.language or "").strip()
        if not lang:
            return "unknown"  # Sarvam auto-detects
        return _LANGUAGE_MAP.get(lang.lower(), lang)

    @staticmethod
    async def _load_audio(audio: AudioInput) -> bytes:
        """Materialize the AudioInput contract (bytes | file path | http(s) URL)
        into bytes.

        The sarvamai SDK only accepts file bytes/handles — it does not fetch
        remote URLs — so URL inputs are downloaded here first. httpx is used
        because it is already a dependency of sarvamai itself.
        """
        if isinstance(audio, bytes):
            return audio
        if audio.startswith(("http://", "https://")):
            import httpx

            async with httpx.AsyncClient(timeout=60.0) as http:
                resp = await http.get(audio)
                resp.raise_for_status()
                return resp.content
        with open(audio, "rb") as f:
            return f.read()

    async def transcribe(
        self,
        audio: AudioInput,
        prompt: Optional[str] = None,
        mime_type: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptionResponse:
        try:
            content = await self._load_audio(audio)

            mime = (mime_type or "audio/mp4").split(";")[0].strip().lower()
            ext = _EXT_BY_MIME.get(mime, "m4a")

            result = await self.client.speech_to_text.transcribe(
                file=(f"audio.{ext}", content, mime),
                model=self.model or DEFAULT_MODEL,
                language_code=self._language_code(),
            )

            return TranscriptionResponse(
                text=result.transcript or "",
                language_detected=result.language_code,
                details={
                    "request_id": result.request_id,
                    "language_probability": result.language_probability,
                },
            )
        except Exception as e:  # SDK/network/IO errors
            logger.exception("Sarvam transcription failed")
            return TranscriptionResponse(error=str(e))
