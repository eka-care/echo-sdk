"""Sarvam AI speech-to-text provider, built on the official ``sarvamai`` SDK.

Strong Indic language coverage (saarika models). Configure with
SARVAM_API_KEY; base URL overridable for proxies/self-hosted gateways via
``TranscriberConfig.base_url`` or SARVAM_BASE_URL.

Two entry points:

  ``SarvamTranscriber``     one-shot REST ASR behind ``BaseTranscriber``.
                            Reachable via ``get_transcriber()``.

  ``SarvamRealtimeClient``  Sarvam's streaming socket. Deliberately not a
                            ``BaseTranscriber`` — that ABC is a single
                            request/response call with no streaming shape to
                            satisfy. Plain async, no web framework, so any
                            caller (a websocket relay, a worker, a script) can
                            drive it. Uses ``websockets``, not ``sarvamai``.

Install: pip install 'echo-sdk[sarvam]'  (extra pulls ``sarvamai`` + ``websockets``)
"""

from __future__ import annotations

import base64
import logging
import os
import struct
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlencode

import orjson

from .base import BaseTranscriber
from .config import (
    SARVAM_REALTIME_LANGUAGES,
    SARVAM_REALTIME_MODEL,
    TranscriberConfig,
)
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


# ==========================================================================
# Streaming — driven directly, not via get_transcriber()
# ==========================================================================

# NOT /speech-to-text-realtime/ws. That endpoint still accepts a websocket
# handshake and then immediately closes every session with
# {"code": "quota_exceeded", "status_code": 402} regardless of account balance
# — a dead beta lane behind a different server, speaking an incompatible
# protocol. This is the live one.
DEFAULT_REALTIME_URL = os.getenv(
    "SARVAM_REALTIME_URL", "wss://api.sarvam.ai/speech-to-text/ws"
)

SUPPORTED_SAMPLE_RATES = (8000, 16000)


def wav_stream_header(sample_rate: int) -> bytes:
    """A 44-byte WAV header for a stream of unknown length.

    The socket wants ``encoding: audio/wav``, and Sarvam's own reference client
    satisfies that by chunking a finished file — so the header lands in chunk 1
    and raw PCM follows it. A live mic has no finished file and no known length,
    so declare the maximum and let the socket's end be the end. Verified against
    the live endpoint: byte-identical transcript and audio_duration to sending a
    real file's header.
    """
    unknown = 0xFFFFFFFF
    return (
        b"RIFF"
        + struct.pack("<I", unknown)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", unknown)
    )


class SarvamRealtimeClient:
    """Thin wrapper over Sarvam's streaming STT socket.

        async with SarvamRealtimeClient(api_key, language_code="hi-IN") as stt:
            asyncio.create_task(pump_mic_into(stt))
            async for event in stt.events():
                ...

    Protocol notes, all confirmed against the live endpoint:
      * the language query param is ``language-code`` (hyphen); REST uses
        ``language_code`` (underscore). Getting this wrong is a hard error.
      * audio is base64 inside {"audio": {"data", "sample_rate", "encoding"}}.
      * the server does its own endpointing and returns one {"type": "data"}
        frame per utterance, mid-stream. There are no partials — nothing
        arrives before an utterance closes.
      * it never closes the socket and never sends a session-end frame. The
        caller owns teardown, and must drain on an idle timeout rather than
        waiting for a signal that does not come.
    """

    def __init__(
        self,
        api_key: str,
        language_code: str = "unknown",
        mode: str = "transcribe",
        sample_rate: int = 16000,
        model: str = SARVAM_REALTIME_MODEL,
        high_vad_sensitivity: bool = False,
        base_url: str = DEFAULT_REALTIME_URL,
    ):
        if not api_key:
            raise ValueError("Sarvam requires an API key (SARVAM_API_KEY)")
        if language_code not in SARVAM_REALTIME_LANGUAGES:
            raise ValueError(
                f"language_code {language_code!r} not supported. "
                f"Supported: {SARVAM_REALTIME_LANGUAGES}"
            )
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}, got {sample_rate}."
            )

        self.api_key = api_key
        self.sample_rate = sample_rate
        self.base_url = base_url
        self.params = {
            "model": model,
            "language-code": language_code,
            "mode": mode,
            "sample_rate": str(sample_rate),
            # Turn signals cost nothing and are the only thing that arrives
            # while someone is still talking — without them a UI has no way to
            # show it is hearing anything until the utterance closes.
            "vad_signals": "true",
        }
        if high_vad_sensitivity:
            # Only sent when asked for. Measured effect: cuts turns roughly
            # twice as early, at the price of splitting mid-phrase and mangling
            # the tail of what it split. Off is the accurate default.
            self.params["high_vad_sensitivity"] = "true"

        self._ws: Optional[Any] = None
        self._header_sent = False

    @property
    def url(self) -> str:
        # safe=":" keeps the colon in `saaras:v4` literal, matching what the
        # endpoint was verified against.
        return f"{self.base_url}?{urlencode(self.params, safe=':')}"

    async def connect(self) -> SarvamRealtimeClient:
        try:
            import websockets
        except ImportError as e:
            raise ImportError(
                "websockets is required for Sarvam streaming transcription. "
                "Install with: pip install 'echo-sdk[sarvam]'"
            ) from e

        headers = {"api-subscription-key": self.api_key}
        try:
            self._ws = await websockets.connect(self.url, additional_headers=headers)
        except TypeError:
            # websockets < 14 spells it extra_headers.
            self._ws = await websockets.connect(self.url, extra_headers=headers)
        return self

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def __aenter__(self) -> SarvamRealtimeClient:
        return await self.connect()

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def send_audio(self, pcm: bytes) -> None:
        """Raw PCM16 mono in. The WAV header is injected ahead of the first chunk."""
        if not self._header_sent:
            self._header_sent = True
            await self._send_bytes(wav_stream_header(self.sample_rate))
        await self._send_bytes(pcm)

    async def _send_bytes(self, payload: bytes) -> None:
        await self._ws.send(
            orjson.dumps(
                {
                    "audio": {
                        "data": base64.b64encode(payload).decode("ascii"),
                        "sample_rate": str(self.sample_rate),
                        "encoding": "audio/wav",
                    }
                }
            ).decode()
        )

    async def flush(self) -> None:
        """Ask the server to close the current utterance and emit its transcript."""
        await self._ws.send(orjson.dumps({"type": "flush"}).decode())

    async def events(self) -> AsyncIterator[dict]:
        """Yield upstream frames as parsed dicts, verbatim.

        Frames seen in practice: {"type": "events", "data": {"signal_type": ...}}
        for VAD turn signals, {"type": "data", "data": {"transcript", ...}} per
        utterance, and {"type": "error", "data": {...}}. Interpretation is left
        to the caller — this client does not filter or reshape.
        """
        async for raw in self._ws:
            try:
                yield orjson.loads(raw)
            except orjson.JSONDecodeError:
                logger.warning("Non-JSON frame from Sarvam: %r", raw[:120])
