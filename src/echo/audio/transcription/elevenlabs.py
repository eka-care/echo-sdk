"""ElevenLabs real-time speech-to-text (Scribe v2 Realtime) via the official
``elevenlabs`` SDK.

Streaming-only for now (``get_streaming_transcriber``). Configure with
ELEVENLABS_API_KEY; model via ELEVENLABS_STT_MODEL (default
``scribe_v2_realtime``; ``scribe_v2_realtime_turbo`` / ``_lite`` also exist).

Unlike Sarvam, ElevenLabs emits interim ``partial_transcript`` messages before
each ``committed_transcript``; interim events are surfaced as PARTIAL unless
``TranscriberConfig.interim_results`` is False.

Install: pip install 'echo-sdk[elevenlabs]'
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from .base import BaseStreamingTranscriber, StreamingTranscriptionSession
from .config import TranscriberConfig
from .schemas import StreamingEventType, StreamingTranscriptEvent

logger = logging.getLogger(__name__)

SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)

# message_type values from the realtime API → event type
_PARTIAL_TYPES = {"partial_transcript"}
_FINAL_TYPES = {
    "committed_transcript",
    "committed_transcript_with_timestamps",
    "final_transcript",
    "final_transcript_with_timestamps",
}
_RECOVERABLE_ERROR_TYPES = {"rate_limited", "error", "scribe_error"}
_FATAL_ERROR_TYPES = {
    "auth_error",
    "scribe_auth_error",
    "quota_exceeded",
    "scribe_quota_exceeded_error",
    "invalid_request",
    "scribe_invalid_request_error",
    "resource_exhausted",
    "scribe_resource_exhausted_error",
    "transcriber_error",
}
_IGNORED_TYPES = {"committed_transcript_entities"}


def _iso_language(lang: Optional[str]) -> Optional[str]:
    """ElevenLabs takes ISO-639-1/-3 codes; strip any region suffix (hi-IN → hi)."""
    if not lang:
        return None
    lang = lang.strip()
    if not lang or lang.lower() in ("auto", "unknown", "auto_detect"):
        return None
    return lang.split("-")[0].split("_")[0].lower()


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _normalise_type(raw: Any) -> str:
    """Accept ``message_type``/``type`` values in snake_case, camelCase or enum form."""
    if raw is None:
        return ""
    value = getattr(raw, "value", raw)
    s = str(value)
    out = []
    for i, ch in enumerate(s):
        if ch.isupper() and i and (s[i - 1].islower() or s[i - 1].isdigit()):
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


class _ElevenLabsSession(StreamingTranscriptionSession):
    """Wraps the SDK's realtime connection object."""

    def __init__(self, connection: Any, config: TranscriberConfig):
        self._conn = connection
        self._config = config
        self._closed = False
        self._segment_seq = 0

    async def send_audio(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        await self._conn.send(
            {
                "audio_base_64": base64.b64encode(pcm).decode("ascii"),
                "sample_rate": self._config.sample_rate,
            }
        )

    async def flush(self) -> None:
        if self._closed:
            return
        await self._conn.commit()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._conn, "close", None)
        if close is None:
            return
        try:
            result = close()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.debug("ElevenLabs realtime close raised", exc_info=True)

    async def events(self) -> AsyncIterator[StreamingTranscriptEvent]:
        try:
            async for msg in self._conn:
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
            logger.exception("ElevenLabs realtime socket failed")
            yield StreamingTranscriptEvent(
                type=StreamingEventType.ERROR, error=str(e), recoverable=False
            )
            return
        yield StreamingTranscriptEvent(type=StreamingEventType.DONE)

    def _map_message(self, msg: Any) -> Optional[StreamingTranscriptEvent]:
        msg_type = _normalise_type(_field(msg, "message_type") or _field(msg, "type"))

        if msg_type == "session_started":
            return StreamingTranscriptEvent(
                type=StreamingEventType.SESSION_STARTED,
                details={
                    "session_id": _field(msg, "session_id"),
                    "config": _as_plain(_field(msg, "config")),
                },
            )

        if msg_type in _PARTIAL_TYPES:
            if not self._config.interim_results:
                return None
            text = (_field(msg, "text") or "").strip()
            if not text:
                return None
            return StreamingTranscriptEvent(type=StreamingEventType.PARTIAL, text=text)

        if msg_type in _FINAL_TYPES:
            text = (_field(msg, "text") or "").strip()
            if not text:
                return None
            self._segment_seq += 1
            timestamps = None
            words = _field(msg, "words")
            if words is not None:
                timestamps = {"words": _as_plain(words)}
            return StreamingTranscriptEvent(
                type=StreamingEventType.FINAL,
                text=text,
                segment_id=str(self._segment_seq),
                language_detected=_field(msg, "language_code"),
                timestamps=timestamps,
                details={"message_type": msg_type},
            )

        if msg_type in _FATAL_ERROR_TYPES or msg_type in _RECOVERABLE_ERROR_TYPES:
            message = (
                _field(msg, "error")
                or _field(msg, "message")
                or _field(msg, "detail")
                or msg_type
            )
            return StreamingTranscriptEvent(
                type=StreamingEventType.ERROR,
                error=f"{msg_type}: {message}",
                details={"message_type": msg_type},
                recoverable=msg_type not in _FATAL_ERROR_TYPES,
            )

        if msg_type in _IGNORED_TYPES or not msg_type:
            return None

        logger.debug("Ignoring unknown ElevenLabs realtime message type %r", msg_type)
        return None


def _is_clean_close(exc: BaseException) -> bool:
    try:
        from websockets.exceptions import ConnectionClosedOK

        return isinstance(exc, ConnectionClosedOK)
    except Exception:
        return False


def _as_plain(obj: Any) -> Any:
    if obj is None or isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    dump = getattr(obj, "model_dump", None) or getattr(obj, "dict", None)
    try:
        return dump() if dump else str(obj)
    except Exception:
        return str(obj)


class ElevenLabsStreamingTranscriber(BaseStreamingTranscriber):
    """Real-time transcriber on ElevenLabs Scribe v2 Realtime."""

    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("ELEVENLABS_API_KEY")
        self.base_url = config.base_url or os.getenv("ELEVENLABS_BASE_URL") or None
        if not self.api_key:
            raise ValueError("ElevenLabs requires an API key (ELEVENLABS_API_KEY)")
        if config.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                "ElevenLabs realtime supports sample rates "
                f"{SUPPORTED_SAMPLE_RATES} (got {config.sample_rate})"
            )
        self._client = None

    @property
    def client(self):
        """Lazy AsyncElevenLabs client."""
        if self._client is None:
            try:
                from elevenlabs import AsyncElevenLabs
            except ImportError as e:
                raise ImportError(
                    "elevenlabs is required for ElevenLabsStreamingTranscriber. "
                    "Install with: pip install 'echo-sdk[elevenlabs]'"
                ) from e

            kwargs: dict = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url.rstrip("/")
            self._client = AsyncElevenLabs(**kwargs)
        return self._client

    def _connect_kwargs(self) -> dict:
        cfg = self.config
        kwargs: dict = {
            "model_id": self.model,
            "audio_format": f"pcm_{cfg.sample_rate}",
            "commit_strategy": cfg.commit_strategy,
        }
        lang = _iso_language(cfg.language)
        if lang:
            kwargs["language_code"] = lang
        if cfg.vad_silence_threshold_s is not None:
            kwargs["vad_silence_threshold_secs"] = cfg.vad_silence_threshold_s
        if cfg.keyterms:
            kwargs["keyterms"] = list(cfg.keyterms)
        return kwargs

    @asynccontextmanager
    async def stream(self) -> AsyncIterator[StreamingTranscriptionSession]:
        kwargs = self._connect_kwargs()
        logger.info(
            "Opening ElevenLabs realtime session model=%s audio_format=%s language=%s",
            kwargs["model_id"],
            kwargs["audio_format"],
            kwargs.get("language_code", "auto"),
        )
        async with self.client.speech_to_text.realtime.connect(**kwargs) as conn:
            session = _ElevenLabsSession(conn, self.config)
            try:
                yield session
            finally:
                await session.close()
