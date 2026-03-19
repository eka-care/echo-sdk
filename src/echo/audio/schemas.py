"""
Audio transcription response models.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel


class TranscriptionSegment(BaseModel):
    """A segment of transcribed audio."""

    text: str
    start: Optional[float] = None
    end: Optional[float] = None
    speaker: Optional[str] = None


class TranscriptionResult(BaseModel):
    """Result of an audio transcription."""

    text: str
    segments: Optional[List[TranscriptionSegment]] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    raw_response: Optional[Dict] = None
