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


class TranscriberConfig(BaseModel):
    provider: Literal[
        "gemini", "ekacare", "sarvam", "openai_compatible", "model_api"
    ] = Field(
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
        if self.provider == "openai_compatible" and self.model.startswith(
            "models/gemini"
        ):
            # the env default model is gemini-shaped; swap in the OpenAI default
            self.model = os.getenv("OPENAI_COMPAT_STT_MODEL", "whisper-1")
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
