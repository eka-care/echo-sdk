"""
LLM configuration for Echo SDK.

Pydantic models for LLM provider configuration.
"""

import os
from typing import Any, Literal, Optional, Set

from pydantic import BaseModel, model_validator


class LLMConfig(BaseModel):
    """LLM provider configuration. Defaults to Bedrock Haiku."""

    provider: Literal["bedrock", "openai", "anthropic", "gemini"] = os.getenv(
        "ECHO_DEFAULT_LLM_PROVIDER", "bedrock"
    )
    model: str = os.getenv(
        "ECHO_DEFAULT_LLM_MODEL", "anthropic.claude-3-haiku-20240307-v1:0"
    )
    temperature: float = float(os.getenv("ECHO_DEFAULT_LLM_TEMPERATURE", 0.2))
    region: Optional[str] = os.getenv("AWS_DEFAULT_REGION") or None  # For Bedrock
    max_tokens: int = int(os.getenv("ECHO_DEFAULT_LLM_MAX_TOKENS", 4096))
    # make this max of 10 calls not more than that
    max_iterations: int = int(os.getenv("ECHO_DEFAULT_LLM_MAX_ITERATIONS", 10))

    # Optional user-provided API keys (falls back to env vars if None)
    api_key: Optional[str] = None  # For OpenAI, Anthropic, Gemini
    aws_access_key_id: Optional[str] = None  # For Bedrock
    aws_secret_access_key: Optional[str] = None  # For Bedrock

    def to_crewai_llm(self) -> Any:
        """Convert to CrewAI LLM instance."""
        try:
            from crewai import LLM
        except ImportError:
            raise ImportError("crewai is required. Install with: pip install crewai")

        if self.provider == "bedrock":
            return LLM(model=f"bedrock/{self.model}", temperature=self.temperature)
        elif self.provider == "openai":
            return LLM(model=self.model, temperature=self.temperature)
        elif self.provider == "anthropic":
            return LLM(model=self.model, temperature=self.temperature)
        else:
            raise ValueError(f"Unsupported provider for CrewAI: {self.provider}")

    def get_provider_supported_model_ids(self) -> Set[str]:
        """Get the supported model IDs for the LLM."""
        if self.provider == "bedrock":
            return set(["anthropic.claude-3-haiku-20240307-v1:0"])
        elif self.provider == "openai":
            return set(
                [
                    "gpt-4.1-2025-04-14",  # $2, $8
                    "gpt-4.1-mini-2025-04-14",  # $0.40, $1.60
                    "gpt-4.1-nano-2025-04-14",  # $0.10, $0.40
                    "gpt-4o",  # $2.50, $10
                    "gpt-4o-mini",  # $0.15, $0.60
                    "gpt-4-turbo",  # $10, $30,
                    "o1",  # $15, $60
                    "o1-mini",  # $1.10, $4.40
                    "o3-mini",  # $1.10, $4.40
                    "o4-mini",  # $1.10, $4.40
                    "o3",  # $10, $40
                ]
            )
        elif self.provider == "anthropic":
            return set(
                [
                    "claude-opus-4-20250514",  # IN: $15, OP: $75
                    "claude-sonnet-4-20250514",  # IN: $3, OP: $15
                    "claude-sonnet-4-5-20250929",  # IN: $0.30, OP: $1.50
                    "claude-haiku-4-5-20251001",  # IN: $0.08, OP: $0.40
                    "claude-opus-4-5-20251101",  # IN: $0.15, OP: $0.75
                    "claude-3-haiku-20240307",  # IN: $0.025, OP: $0.125
                ]
            )
        elif self.provider == "gemini":
            return set(
                [
                    "models/gemini-2.5-flash",
                    "models/gemini-flash-latest",
                    "models/gemini-pro-latest",
                    "models/gemini-2.5-pro",  # $1.25 / $10 - Most capable, reasoning
                    "models/gemini-2.5-pro-preview-06-05",  # $1.25 / $10 - Most capable, reasoning
                    "models/gemini-2.0-flash",  # $0.10 / $0.40 - Fast, multimodal
                    "models/gemini-2.0-flash-lite",  # $0.075 / $0.30 - Budget option
                ]
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    @model_validator(mode="after")
    def validate_model(self):
        """Validate the model."""
        supported_model_ids = self.get_provider_supported_model_ids()
        if self.model not in supported_model_ids:
            raise ValueError(
                f"Unsupported model: {self.model} for provider: {self.provider}. Supported: {supported_model_ids}"
            )
        return self


""" 
curl "https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_API_KEY" | jq '.models[].name' 
"models/gemini-2.5-flash-image"
"models/gemini-2.5-flash-preview-09-2025"
"models/gemini-2.5-flash-lite-preview-09-2025"  
"models/gemini-2.5-flash-lite"
"models/gemini-2.5-flash"
"models/gemini-flash-latest"
"models/gemini-pro-latest"
"models/gemini-2.5-pro"
"models/gemini-2.0-flash"
"models/gemini-2.0-flash-lite"
"models/gemini-2.0-flash-001"
"models/gemini-2.0-flash-exp-image-generation"
"models/gemini-2.0-flash-lite-001"
"models/gemini-exp-1206"
"models/gemini-2.5-flash-preview-tts"
"models/gemini-2.5-pro-preview-tts"
"models/gemma-3-1b-it"
"models/gemma-3-4b-it"
"models/gemma-3-12b-it"
"models/gemma-3-27b-it"
"models/gemma-3n-e4b-it"
"models/gemma-3n-e2b-it"
"models/gemini-flash-lite-latest"
"models/gemini-3-pro-preview"
"models/gemini-3-flash-preview"
"models/gemini-3-pro-image-preview"
"models/nano-banana-pro-preview"
"models/gemini-robotics-er-1.5-preview"
"models/gemini-2.5-computer-use-preview-10-2025"
"models/deep-research-pro-preview-12-2025"
"models/embedding-001"
"models/text-embedding-004"
"models/gemini-embedding-001"
"models/aqa"
"models/imagen-4.0-generate-preview-06-06"
"models/imagen-4.0-ultra-generate-preview-06-06"
"models/imagen-4.0-generate-001"
"models/imagen-4.0-ultra-generate-001"
"models/imagen-4.0-fast-generate-001"
"models/veo-2.0-generate-001"
"models/veo-3.0-generate-001"
"models/veo-3.0-fast-generate-001"
"models/veo-3.1-generate-preview"
"models/veo-3.1-fast-generate-preview"
"models/gemini-2.5-flash-native-audio-latest"
"models/gemini-2.5-flash-native-audio-preview-09-2025"
"models/gemini-2.5-flash-native-audio-preview-12-2025"
"""
