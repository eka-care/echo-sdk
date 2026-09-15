"""Schemas for the audio transcription module."""

from enum import Enum
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel

# bytes => raw audio payload (requires explicit mime_type)
# str   => Files API URI ("files/abc123" or full URI) or http(s):// URL
AudioInput = Union[bytes, str]


class TokenUsage(BaseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class TranscriptionResponse(BaseModel):
    text: str = ""
    language_detected: Optional[str] = None
    duration_s: Optional[float] = None
    usage: Optional[TokenUsage] = None
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class StreamingEventType(str, Enum):
    """Event types emitted by a streaming transcription session.

    Mirrors ``echo.llm.schemas.StreamEventType`` so consumers deal with one
    familiar shape across LLM and STT streams.
    """

    SESSION_STARTED = "session_started"  # provider accepted the stream
    PARTIAL = "partial"  # interim text, may be revised by the next event
    FINAL = "final"  # committed segment; will not change
    SPEECH_STARTED = "speech_started"  # provider VAD signal (where supported)
    SPEECH_ENDED = "speech_ended"
    ERROR = "error"
    DONE = "done"  # provider closed the stream; no more events follow


class StreamingTranscriptEvent(BaseModel):
    """One event from ``StreamingTranscriptionSession.events()``."""

    type: StreamingEventType
    text: Optional[str] = None  # PARTIAL / FINAL
    segment_id: Optional[str] = None  # provider request/segment id, FINAL
    language_detected: Optional[str] = None
    timestamps: Optional[Dict[str, Any]] = None  # word/segment timing when enabled
    details: Optional[Dict[str, Any]] = None  # raw provider extras (metrics, probability)
    error: Optional[str] = None  # ERROR
    # False => the provider stream is unusable (auth, quota, invalid request);
    # the caller should stop sending audio and end the session.
    recoverable: bool = True
