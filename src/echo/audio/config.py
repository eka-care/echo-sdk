"""
Audio transcription configuration.
"""

from typing import Literal, Optional, Set

from pydantic import BaseModel, model_validator


class TranscriptionConfig(BaseModel):
    """Transcription provider configuration."""

    provider: Literal["openai", "gemini"]
    model: str
    language: Optional[str] = None
    response_format: Optional[str] = None
    api_key: Optional[str] = None
    prompt: Optional[str] = None  # Gemini custom transcription instruction

    def get_provider_supported_model_ids(self) -> Set[str]:
        """Get the supported model IDs for the provider."""
        if self.provider == "openai":
            return {
                "whisper-1",
                "gpt-4o-transcribe",
                "gpt-4o-mini-transcribe",
                "gpt-4o-transcribe-diarize",
            }
        elif self.provider == "gemini":
            return {
                "models/gemini-3-pro-preview",
                "models/gemini-3-flash-preview",
                "models/gemini-pro-latest",
                "models/gemini-2.5-pro",
                "models/gemini-2.5-pro-preview-06-05",
                "models/gemini-flash-latest",
                "models/gemini-2.5-flash",
                "models/gemini-2.5-flash-preview-09-2025",
                "models/gemini-2.5-flash-lite-preview-09-2025",
                "models/gemini-2.5-flash-lite",
                "models/gemini-2.0-flash",
                "models/gemini-2.0-flash-lite",
            }
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    @model_validator(mode="after")
    def validate_model(self):
        """Validate the model is supported for the given provider."""
        supported_model_ids = self.get_provider_supported_model_ids()
        if self.model not in supported_model_ids:
            raise ValueError(
                f"Unsupported model: {self.model} for provider: {self.provider}. "
                f"Supported: {supported_model_ids}"
            )
        return self
