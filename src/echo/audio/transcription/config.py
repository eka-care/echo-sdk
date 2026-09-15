"""Configuration for the audio transcription module."""

import os
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

GEMINI_AUDIO_MODELS = [
    "models/gemini-2.5-pro",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
]

EKACARE_LANGUAGES = ["en-IN", "en-US", "hi"]

# Providers that implement BaseStreamingTranscriber (real-time websocket STT).
STREAMING_PROVIDERS = ("sarvam", "elevenlabs")

SARVAM_BATCH_DEFAULT_MODEL = "saarika:v2.5"
SARVAM_STREAM_DEFAULT_MODEL = "saaras:v3"
ELEVENLABS_STREAM_DEFAULT_MODEL = "scribe_v2_realtime"


class TranscriberConfig(BaseModel):
    provider: Literal["gemini", "ekacare", "sarvam", "elevenlabs"] = Field(
        default_factory=lambda: os.getenv("ECHO_DEFAULT_TRANSCRIBER_PROVIDER", "gemini")
    )
    model: str = Field(
        default_factory=lambda: os.getenv(
            "ECHO_DEFAULT_TRANSCRIBER_MODEL", "models/gemini-2.5-flash"
        )
    )
    api_key: Optional[str] = None
    base_url: Optional[str] = None  # provider endpoint override (proxies, self-hosted)
    language: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: int = 8192
    request_timeout_s: float = 60.0

    # --- streaming (BaseStreamingTranscriber) -------------------------------
    # Input is always raw 16-bit little-endian mono PCM; callers resample
    # before sending. 16 kHz is the rate both Sarvam and ElevenLabs prefer.
    sample_rate: int = 16000
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    # "vad": provider commits a segment on detected silence (default).
    # "manual": segments are committed only on flush().
    commit_strategy: Literal["vad", "manual"] = "vad"
    vad_silence_threshold_s: Optional[float] = None
    high_vad_sensitivity: bool = False
    interim_results: bool = True  # emit PARTIAL events where the provider has them
    keyterms: Optional[List[str]] = None  # vocabulary biasing (ElevenLabs, <=50)
    # Sarvam saaras:v3 output mode; None => provider default ("transcribe").
    mode: Optional[
        Literal["transcribe", "verbatim", "translate", "translit", "codemix"]
    ] = None

    @model_validator(mode="after")
    def _validate_model(self):
        gemini_shaped = self.model.startswith("models/gemini")
        if self.provider == "sarvam" and gemini_shaped:
            # the env default model is gemini-shaped; swap in sarvam's default
            self.model = os.getenv("SARVAM_STT_MODEL", SARVAM_BATCH_DEFAULT_MODEL)
        if self.provider == "elevenlabs" and gemini_shaped:
            self.model = os.getenv("ELEVENLABS_STT_MODEL", ELEVENLABS_STREAM_DEFAULT_MODEL)
        if self.provider == "gemini" and self.model not in GEMINI_AUDIO_MODELS:
            raise ValueError(
                f"Model {self.model!r} not supported for provider 'gemini'. "
                f"Supported: {GEMINI_AUDIO_MODELS}"
            )
        if (
            self.provider == "ekacare"
            and self.language is not None
            and self.language not in EKACARE_LANGUAGES
        ):
            raise ValueError(
                f"Language {self.language!r} not supported for provider 'ekacare'. "
                f"Supported: {EKACARE_LANGUAGES}"
            )
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer (Hz)")
        if self.keyterms is not None and len(self.keyterms) > 50:
            raise ValueError("keyterms supports at most 50 entries")
        return self
