"""Sarvam AI speech-to-text provider, built on the official ``sarvamai`` SDK.

Strong Indic language coverage (saarika models). Configure with
SARVAM_API_KEY; base URL overridable for proxies/self-hosted gateways via
``TranscriberConfig.base_url`` or SARVAM_BASE_URL.

Install: pip install 'echo-sdk[sarvam]'  (extra pulls ``sarvamai``)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from .base import (
    BaseStreamingTranscriber,
    BaseTranscriber,
    StreamingTranscriptionSession,
)
from .config import (
    SARVAM_BATCH_DEFAULT_MODEL,
    SARVAM_STREAM_DEFAULT_MODEL,
    TranscriberConfig,
)
from .schemas import (
    AudioInput,
    StreamingEventType,
    StreamingTranscriptEvent,
    TranscriptionResponse,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = SARVAM_BATCH_DEFAULT_MODEL
STREAMING_MODELS = ("saaras:v3", "saarika:v2.5")
# Sarvam error codes after which the socket is not worth keeping open.
_FATAL_ERROR_CODES = {
    "invalid_api_key",
    "unauthorized",
    "forbidden",
    "quota_exceeded",
    "insufficient_credits",
    "invalid_request",
    "invalid_language_code",
    "invalid_model",
}

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


class _SarvamClientMixin:
    """Shared credential/client/language plumbing for batch + streaming."""

    config: TranscriberConfig

    def _init_sarvam(self, config: TranscriberConfig) -> None:
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


class SarvamTranscriber(_SarvamClientMixin, BaseTranscriber):
    """Transcriber backed by Sarvam's speech-to-text via the official SDK."""

    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
        self._init_sarvam(config)

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


# --------------------------------------------------------------------------- #
# Streaming (real-time websocket)
# --------------------------------------------------------------------------- #


class _SarvamSession(StreamingTranscriptionSession):
    """Wraps ``AsyncSpeechToTextStreamingSocketClient``.

    Sarvam segments on its own VAD and emits one ``data`` message per
    utterance — there are no interim partials, so only FINAL events carry text.
    """

    def __init__(self, socket: Any, config: TranscriberConfig):
        self._socket = socket
        self._config = config
        self._closed = False
        self._segment_seq = 0

    async def send_audio(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        await self._socket.transcribe(
            audio=base64.b64encode(pcm).decode("ascii"),
            # The SDK's AudioData.encoding literal is "audio/wav" even for raw
            # PCM; the actual codec is declared once at connect time via
            # input_audio_codec.
            encoding="audio/wav",
            sample_rate=self._config.sample_rate,
        )

    async def flush(self) -> None:
        if self._closed:
            return
        await self._socket.flush()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        ws = getattr(self._socket, "_websocket", None)
        if ws is not None:
            try:
                await ws.close()
            except Exception:  # already closed / transport gone
                logger.debug("Sarvam websocket close raised", exc_info=True)

    async def events(self) -> AsyncIterator[StreamingTranscriptEvent]:
        try:
            async for msg in self._socket:
                event = self._map_message(msg)
                if event is None:
                    continue
                yield event
                if event.type is StreamingEventType.ERROR and not event.recoverable:
                    return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if _is_clean_close(e):
                yield StreamingTranscriptEvent(type=StreamingEventType.DONE)
                return
            logger.exception("Sarvam streaming socket failed")
            yield StreamingTranscriptEvent(
                type=StreamingEventType.ERROR, error=str(e), recoverable=False
            )
            return
        yield StreamingTranscriptEvent(type=StreamingEventType.DONE)

    def _map_message(self, msg: Any) -> Optional[StreamingTranscriptEvent]:
        msg_type = getattr(msg, "type", None)
        data = getattr(msg, "data", None)

        if msg_type == "data":
            text = (getattr(data, "transcript", None) or "").strip()
            if not text:
                return None
            self._segment_seq += 1
            metrics = getattr(data, "metrics", None)
            return StreamingTranscriptEvent(
                type=StreamingEventType.FINAL,
                text=text,
                segment_id=getattr(data, "request_id", None) or str(self._segment_seq),
                language_detected=getattr(data, "language_code", None),
                timestamps=getattr(data, "timestamps", None),
                details={
                    "language_probability": getattr(data, "language_probability", None),
                    "metrics": _as_dict(metrics),
                },
            )

        if msg_type == "events":
            signal = str(getattr(data, "signal_type", "") or "").lower()
            details = {
                "event_type": getattr(data, "event_type", None),
                "signal_type": signal or None,
                "occured_at": getattr(data, "occured_at", None),
            }
            if "start" in signal:
                return StreamingTranscriptEvent(
                    type=StreamingEventType.SPEECH_STARTED, details=details
                )
            if "end" in signal:
                return StreamingTranscriptEvent(
                    type=StreamingEventType.SPEECH_ENDED, details=details
                )
            return None

        if msg_type == "error":
            code = str(getattr(data, "code", "") or "")
            message = getattr(data, "error", None) or "Sarvam streaming error"
            return StreamingTranscriptEvent(
                type=StreamingEventType.ERROR,
                error=f"{code}: {message}" if code else str(message),
                details={"code": code or None},
                recoverable=code.lower() not in _FATAL_ERROR_CODES,
            )

        logger.debug("Ignoring unknown Sarvam streaming message type %r", msg_type)
        return None


def _is_clean_close(exc: BaseException) -> bool:
    try:
        from websockets.exceptions import ConnectionClosedOK

        return isinstance(exc, ConnectionClosedOK)
    except Exception:
        return False


def _as_dict(obj: Any) -> Optional[dict]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None) or getattr(obj, "dict", None)
    try:
        return dump() if dump else None
    except Exception:
        return None


class SarvamStreamingTranscriber(_SarvamClientMixin, BaseStreamingTranscriber):
    """Real-time transcriber on Sarvam's websocket STT (saaras:v3 / saarika:v2.5).

    Constraints from the provider: connection-level ``sample_rate`` must be
    16000 or 8000; input is raw ``pcm_s16le``; segments are VAD-committed on
    the server (``high_vad_sensitivity`` shortens pauses), and ``flush()``
    forces a commit of buffered audio.
    """

    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
        self._init_sarvam(config)
        if config.sample_rate not in (8000, 16000):
            raise ValueError(
                "Sarvam streaming supports sample_rate 16000 or 8000 only "
                f"(got {config.sample_rate})"
            )

    def _stream_model(self) -> str:
        """Pick the streaming model.

        ``SARVAM_STT_STREAM_MODEL`` wins. The batch default (``saarika:v2.5``)
        is what ``TranscriberConfig`` fills in when nothing was asked for, so
        it is treated as "unset" here and replaced with ``saaras:v3``; pass any
        other streaming model explicitly to use it.
        """
        env_model = os.getenv("SARVAM_STT_STREAM_MODEL")
        if env_model:
            return env_model
        if self.model in STREAMING_MODELS and self.model != SARVAM_BATCH_DEFAULT_MODEL:
            return self.model
        return SARVAM_STREAM_DEFAULT_MODEL

    def _connect_kwargs(self) -> dict:
        cfg = self.config
        model = self._stream_model()
        kwargs: dict = {
            "language_code": self._language_code(),
            "model": model,
            "sample_rate": str(cfg.sample_rate),
            "input_audio_codec": "pcm_s16le",
            "vad_signals": "true",
            "flush_signal": "true",
        }
        if cfg.high_vad_sensitivity:
            kwargs["high_vad_sensitivity"] = "true"
        if cfg.mode and model.startswith("saaras"):
            kwargs["mode"] = cfg.mode
        return kwargs

    @asynccontextmanager
    async def stream(self) -> AsyncIterator[StreamingTranscriptionSession]:
        kwargs = self._connect_kwargs()
        logger.info(
            "Opening Sarvam streaming session model=%s language=%s sample_rate=%s",
            kwargs["model"],
            kwargs["language_code"],
            kwargs["sample_rate"],
        )
        async with self.client.speech_to_text_streaming.connect(**kwargs) as socket:
            session = _SarvamSession(socket, self.config)
            try:
                yield session
            finally:
                await session.close()
