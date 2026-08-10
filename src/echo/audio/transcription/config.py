"""Configuration for the audio transcription module."""

import os
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

GEMINI_AUDIO_MODELS = [
    "models/gemini-2.5-pro",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
]

EKACARE_LANGUAGES = ["en-IN", "en-US", "hi"]

# Sarvam streaming socket (SarvamRealtimeClient). The REST transcriber uses the
# saarika models and its own language handling; these apply to the socket only.
SARVAM_REALTIME_MODEL = "saaras:v4"

# Taken from the socket's own validation error, which enumerates the set when
# sent something invalid. Auto-detect is spelled "unknown" ("auto" is rejected)
# and Odia is "od-IN". Wider than what REST accepts.
SARVAM_REALTIME_LANGUAGES = [
    "unknown", "en-IN", "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN",
    "pa-IN", "ta-IN", "te-IN", "gu-IN", "as-IN", "ur-IN", "ne-IN", "kok-IN",
    "ks-IN", "sd-IN", "sa-IN", "sat-IN", "mni-IN", "brx-IN", "mai-IN", "doi-IN",
]


class TranscriberConfig(BaseModel):
    provider: Literal["gemini", "ekacare", "sarvam"] = Field(
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

    @model_validator(mode="after")
    def _validate_model(self):
        if self.provider == "sarvam" and self.model.startswith("models/gemini"):
            # the env default model is gemini-shaped; swap in sarvam's default
            self.model = os.getenv("SARVAM_STT_MODEL", "saarika:v2.5")
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
        return self
