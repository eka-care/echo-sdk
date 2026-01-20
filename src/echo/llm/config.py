"""
LLM configuration for Echo SDK.

Pydantic models for LLM provider configuration.
"""

import os
from typing import Any, Literal, Optional

from pydantic import BaseModel

class LLMConfig(BaseModel):
    """LLM provider configuration. Defaults to Bedrock Haiku."""

    provider: Literal["bedrock", "openai", "anthropic"] = os.getenv(
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
