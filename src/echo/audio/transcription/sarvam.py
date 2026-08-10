"""Sarvam AI speech-to-text provider, built on the official ``sarvamai`` SDK.

Strong Indic language coverage (saarika models). Configure with
SARVAM_API_KEY; base URL overridable for proxies/self-hosted gateways via
``TranscriberConfig.base_url`` or SARVAM_BASE_URL.

Two entry points:

  ``SarvamTranscriber``     one-shot REST ASR behind ``BaseTranscriber``.
                            Reachable via ``get_transcriber()``.

  ``SarvamRealtimeClient``  Sarvam's realtime streaming socket. Deliberately
                            not a ``BaseTranscriber`` — that ABC is a single
                            request/response call with no streaming shape to
                            satisfy. Plain async, no web framework, so any
                            caller (a websocket relay, a worker, a script) can
                            drive it. Also ``sarvamai``, via
                            ``speech_to_text_realtime_streaming``.

Install: pip install 'echo-sdk[sarvam]'  (extra pulls ``sarvamai``)
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, AsyncIterator, Optional

from .base import BaseTranscriber
from .config import (
    SARVAM_REALTIME_LANGUAGES,
    SARVAM_REALTIME_MODEL,
    SARVAM_STREAM_TYPES,
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


def _environment_for(base_url: str):
    """Point every sarvamai surface at an override host (proxy / self-hosted).

    All three fields are required — omitting any is a TypeError. ``base`` is the
    REST host, ``production`` the websocket host, and ``creative`` the dubbing
    host, which nothing here uses but still has to be supplied.
    """
    from sarvamai.environment import SarvamAIEnvironment

    base = base_url.rstrip("/")
    return SarvamAIEnvironment(
        base=base,
        creative=f"{base}/dubbing",
        production=base.replace("https://", "wss://").replace("http://", "ws://"),
    )


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
                kwargs["environment"] = _environment_for(self.base_url)
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
# Realtime streaming — driven directly, not via get_transcriber()
# ==========================================================================

SUPPORTED_SAMPLE_RATES = (8000, 16000)


class SarvamRealtimeClient:
    """Thin wrapper over Sarvam's realtime STT socket.

        async with SarvamRealtimeClient(api_key, language_code="hi-IN") as stt:
            asyncio.create_task(pump_mic_into(stt))
            async for event in stt.events():
                ...

    Built on ``sarvamai``'s ``speech_to_text_realtime_streaming`` — the same
    endpoint and protocol as Sarvam's own published example, so URL building,
    auth headers, framing and message parsing are the vendor SDK's problem
    rather than ours.

    Protocol notes:
      * audio goes up as base64 in {"event": "audio_input", "audio": ...}, and
        ``encoding="linear16"`` means raw PCM16 — no WAV container, unlike the
        older /speech-to-text/ws socket.
      * this endpoint DOES emit partials: "transcript.partial" arrives while
        someone is still speaking, "transcript.final" when the turn closes.
      * auto-detect is spelled "auto" here and Odia is "or-IN" — both the
        opposite of the older socket. See SARVAM_REALTIME_LANGUAGES.
      * ``end`` tells the server to flush its tail. The caller still decides
        when to stop reading.
    """

    def __init__(
        self,
        api_key: str,
        language_code: str = "auto",
        mode: str = "transcribe",
        sample_rate: int = 16000,
        model: str = SARVAM_REALTIME_MODEL,
        stream_type: str = "balanced",
        prompt: Optional[str] = None,
        endpointing: Optional[str] = None,
        threshold: Optional[float] = None,
        prefix_padding_ms: Optional[int] = None,
        silence_duration_ms: Optional[int] = None,
        min_speech_duration_ms: Optional[int] = None,
        return_timestamps: bool = False,
        base_url: Optional[str] = None,
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
        if stream_type not in SARVAM_STREAM_TYPES:
            raise ValueError(
                f"stream_type {stream_type!r} not supported. "
                f"Supported: {SARVAM_STREAM_TYPES}"
            )

        self.api_key = api_key
        self.sample_rate = sample_rate
        self.base_url = base_url or os.getenv("SARVAM_BASE_URL") or None

        # Only non-None values are forwarded. The SDK omits unset query params,
        # so leaving a knob alone means "use Sarvam's default" rather than
        # sending an empty string.
        params: dict[str, Any] = {
            "language_code": language_code,
            "model": model,
            "mode": mode,
            "stream_type": stream_type,
            "encoding": "linear16",
            "sample_rate": str(sample_rate),
            "return_timestamps": "true" if return_timestamps else None,
            "prompt": prompt,
            "endpointing": endpointing,
            "threshold": None if threshold is None else str(threshold),
            "prefix_padding_ms": (
                None if prefix_padding_ms is None else str(prefix_padding_ms)
            ),
            "silence_duration_ms": (
                None if silence_duration_ms is None else str(silence_duration_ms)
            ),
            "min_speech_duration_ms": (
                None if min_speech_duration_ms is None else str(min_speech_duration_ms)
            ),
        }
        self.params = {k: v for k, v in params.items() if v is not None}

        self._cm: Optional[Any] = None
        self._sock: Optional[Any] = None

    def _build_client(self):
        try:
            from sarvamai import AsyncSarvamAI
        except ImportError as e:
            raise ImportError(
                "sarvamai is required for Sarvam streaming transcription. "
                "Install with: pip install 'echo-sdk[sarvam]'"
            ) from e

        kwargs: dict[str, Any] = {"api_subscription_key": self.api_key}
        if self.base_url:
            kwargs["environment"] = _environment_for(self.base_url)
        return AsyncSarvamAI(**kwargs)

    async def connect(self) -> SarvamRealtimeClient:
        # connect() is an @asynccontextmanager, so the context is entered here
        # and held open until close() — the session must outlive this call.
        self._cm = self._build_client().speech_to_text_realtime_streaming.connect(
            api_subscription_key=self.api_key, **self.params
        )
        self._sock = await self._cm.__aenter__()
        return self

    async def close(self) -> None:
        if self._cm is None:
            return
        cm, self._cm, self._sock = self._cm, None, None
        try:
            await cm.__aexit__(None, None, None)
        except Exception as e:
            logger.debug("Sarvam realtime socket close raised: %s", e)

    async def __aenter__(self) -> SarvamRealtimeClient:
        return await self.connect()

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def send_audio(self, pcm: bytes) -> None:
        """Raw PCM16 mono in — base64 framing is handled here."""
        # These take pydantic models, not dicts: the SDK calls .dict() on what
        # it is given, so a plain dict raises AttributeError at send time.
        from sarvamai.types.realtime_audio_input import RealtimeAudioInput

        await self._sock.send_realtime_audio_input(
            RealtimeAudioInput(audio=base64.b64encode(pcm).decode("ascii"))
        )

    async def flush(self) -> None:
        """Ask the server to close the current utterance and emit its transcript."""
        from sarvamai.types.realtime_flush import RealtimeFlush

        await self._sock.send_realtime_flush(RealtimeFlush())

    async def end(self) -> None:
        """Tell the server no more audio is coming, so it flushes its tail."""
        from sarvamai.types.realtime_end import RealtimeEnd

        await self._sock.send_realtime_end(RealtimeEnd())

    async def events(self) -> AsyncIterator[dict]:
        """Yield upstream messages as plain dicts.

        Events seen in practice: "transcript.partial" and "transcript.final"
        (both carry ``text``), "vad.speech_start"/"vad.speech_end", and "error"
        (``code``, ``message``). Interpretation is left to the caller — this
        client does not filter or reshape.
        """
        async for message in self._sock:
            if isinstance(message, bytes):
                continue  # binary frames are not part of the STT protocol
            yield _as_dict(message)


def _as_dict(message: Any) -> dict:
    """Normalize a sarvamai response object to a plain dict.

    The SDK returns pydantic models for events it knows; anything it could not
    parse never reaches us (it logs and skips).
    """
    if isinstance(message, dict):
        return message
    for attr in ("model_dump", "dict"):
        dump = getattr(message, attr, None)
        if callable(dump):
            try:
                return dump()
            except Exception:
                break
    return {"event": "unknown", "raw": repr(message)}
