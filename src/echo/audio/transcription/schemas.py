"""Schemas for the audio transcription module."""

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
