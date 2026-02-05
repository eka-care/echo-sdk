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
                    "gpt-5.2",
                    "gpt-5.1",
                    "gpt-5",
                    "gpt-5-mini",
                    "gpt-5-nano",
                    "gpt-4.1",
                    "gpt-4.1-mini",
                    "gpt-4.1-nano",
                    "gpt-4o",
                    "gpt-4o-mini",
                    "o3",
                    "o4-mini",
                    "o3-mini",
                    "o1",
                    "o3-pro",
                    "gpt-5-pro",
                    "gpt-5.2-pro",
                    "o1-pro",
                    "codex-mini-latest",
                    "computer-use-preview",
                    "o3-deep-research",
                    "o4-mini-deep-research",
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
                    "models/gemini-3-pro-preview",
                    "models/gemini-3-flash-preview",
                    "models/gemini-pro-latest",
                    "models/gemini-2.5-pro",  # $1.25 / $10 - Most capable, reasoning
                    "models/gemini-2.5-pro-preview-06-05",  # $1.25 / $10 - Most capable, reasoning
                    "models/gemini-flash-latest",
                    "models/gemini-2.5-flash",
                    "models/gemini-2.5-flash-preview-09-2025",
                    "models/gemini-2.5-flash-lite-preview-09-2025",
                    "models/gemini-2.5-flash-lite",
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

""" 
OPENAI CHAT API MODELS AND PRICING (as of Feb 2025)
====================================================

All prices are per 1 Million tokens.
1000 tokens is roughly 750 words.


FLAGSHIP MODELS (GPT-5 FAMILY)
------------------------------

gpt-5.2
  Input: $1.75  |  Output: $14.00  |  Context: 400K tokens
  Best overall model for coding and agentic tasks

gpt-5.1
  Input: $1.25  |  Output: $10.00  |  Context: 400K tokens
  Near-flagship, slightly cheaper than 5.2

gpt-5
  Input: $1.25  |  Output: $10.00  |  Context: 400K tokens
  General purpose flagship

gpt-5-mini
  Input: $0.25  |  Output: $2.00  |  Context: 400K tokens
  Balanced performance and cost

gpt-5-nano
  Input: $0.05  |  Output: $0.40  |  Context: 400K tokens
  Ultra cheap, good for simple tasks


PREVIOUS GENERATION (still available)
-------------------------------------

gpt-4.1
  Input: $2.00  |  Output: $8.00  |  Context: 1M tokens
  Biggest context window available

gpt-4.1-mini
  Input: $0.40  |  Output: $1.60  |  Context: 1M tokens
  Cheap with massive context

gpt-4.1-nano
  Input: $0.10  |  Output: $0.40  |  Context: 1M tokens
  Cheapest model with 1M context

gpt-4o
  Input: $2.50  |  Output: $10.00  |  Context: 128K tokens
  Multimodal (supports vision/images)

gpt-4o-mini
  Input: $0.15  |  Output: $0.60  |  Context: 128K tokens
  Cheap multimodal option


REASONING MODELS (O-SERIES)
---------------------------

Note: O-series models use hidden "reasoning tokens" that are billed
as output tokens but not visible in the API response. A 500 token
visible response may actually consume 2000+ tokens. Actual cost can
be 2-4x what you expect from visible output alone.

o3
  Input: $2.00  |  Output: $8.00  |  Context: 200K tokens
  Complex reasoning tasks

o4-mini
  Input: $1.10  |  Output: $4.40  |  Context: 200K tokens
  Cost-efficient reasoning

o3-mini
  Input: $1.10  |  Output: $4.40  |  Context: 200K tokens
  Budget reasoning option

o1
  Input: $15.00  |  Output: $60.00  |  Context: 128K tokens
  Deep reasoning, expensive

o3-pro
  Input: $20.00  |  Output: $80.00  |  Context: 200K tokens
  Maximum reasoning power


PRO / PREMIUM TIER
------------------

gpt-5-pro
  Input: $15.00  |  Output: $120.00  |  Context: 400K tokens

gpt-5.2-pro
  Input: $21.00  |  Output: $168.00  |  Context: 400K tokens

o1-pro
  Input: $150.00  |  Output: $600.00  |  Context: 200K tokens


SPECIALTY MODELS
----------------

codex-mini-latest
  Input: $1.50  |  Output: $6.00  |  Context: 128K tokens
  Optimized for coding tasks

computer-use-preview
  Input: $3.00  |  Output: $12.00  |  Context: 128K tokens
  Specialized for computer use

o3-deep-research
  Input: $10.00  |  Output: $40.00  |  Context: 200K tokens
  Deep research tasks

o4-mini-deep-research
  Input: $2.00  |  Output: $8.00  |  Context: 200K tokens
  Cheaper deep research option


COST SAVING OPTIONS
-------------------

1. Batch API: 50% discount, results delivered within 24 hours
2. Prompt Caching: Save 50-90% on repeated content
   - GPT-5 family: 90% off cached input
   - GPT-4.1 family: 75% off cached input
   - GPT-4o / O-series: 50% off cached input
3. Free credits: $5 for new accounts, no credit card needed


Source: OpenAI official pricing page (platform.openai.com/docs/pricing)
"""

""" 
ANTHROPIC CLAUDE API MODELS AND PRICING (as of Feb 2026)
=========================================================

All prices are per 1 Million tokens.
1000 tokens is roughly 750 words.


CLAUDE 4.5 SERIES (Latest)
---------------------------

claude-opus-4-5-20251101 (Claude Opus 4.5)
  Input: $5.00  |  Output: $25.00  |  Context: 200K tokens
  Most capable model, best reasoning and coding

claude-sonnet-4-5-20250929 (Claude Sonnet 4.5)
  Input: $3.00  |  Output: $15.00  |  Context: 200K tokens (up to 1M in beta)
  Best balance of performance and cost, great for coding
  Long context (over 200K): $6.00 input / $22.50 output

claude-haiku-4-5-20251001 (Claude Haiku 4.5)
  Input: $1.00  |  Output: $5.00  |  Context: 200K tokens
  Near-frontier performance, fast and affordable


PREVIOUS GENERATION (still available)
--------------------------------------

claude-opus-4-20250514 (Claude Opus 4)
  Input: $15.00  |  Output: $75.00  |  Context: 200K tokens

claude-sonnet-4-20250514 (Claude Sonnet 4)
  Input: $3.00  |  Output: $15.00  |  Context: 200K tokens (up to 1M in beta)

claude-haiku-3-5-20241022 (Claude Haiku 3.5)
  Input: $0.80  |  Output: $4.00  |  Context: 200K tokens

claude-3-haiku-20240307 (Claude Haiku 3)
  Input: $0.25  |  Output: $1.25  |  Context: 200K tokens
  Cheapest Claude model


COST SAVING OPTIONS
-------------------

1. Batch API: 50% discount on all models, results within 24 hours
2. Prompt Caching:
   - Cache reads: 90% off (0.1x base input price)
   - 5-min cache writes: 1.25x base input price
   - 1-hour cache writes: 2x base input price
3. Free credits: $5 for new accounts


========================================================================


GOOGLE GEMINI API MODELS AND PRICING (as of Feb 2026)
======================================================

All prices are per 1 Million tokens.
Note: Many Gemini models have a FREE tier with rate limits.


GEMINI 3 SERIES (Latest)
--------------------------

gemini-3-pro-preview
  Input: $2.00  |  Output: $12.00  |  Context: 1M tokens
  Long context (over 200K): $4.00 input / $18.00 output
  Most powerful Gemini model, deep reasoning
  No free tier currently

gemini-3-flash (expected)
  Input: $0.50  |  Output: $3.00  |  Context: 1M tokens
  Fast and affordable latest gen


GEMINI 2.5 SERIES
------------------

gemini-2.5-pro
  Input: $1.25  |  Output: $10.00  |  Context: 1M tokens
  Long context (over 200K): $2.50 input / $10.00 output (2x for input only)
  Best for coding and complex reasoning
  Free tier: 5 RPM, 250K TPM

gemini-2.5-flash
  Input: $0.30  |  Output: $2.50  |  Context: 1M tokens
  Hybrid reasoning model with thinking budgets
  Free tier: 10 RPM, 250K TPM

gemini-2.5-flash-lite
  Input: $0.10  |  Output: $0.40  |  Context: 1M tokens
  Most cost-effective option


GEMINI 2.0 SERIES
------------------

gemini-2.0-flash
  Input: $0.10  |  Output: $0.40  |  Context: 1M tokens
  Fast, balanced model
  Free tier: 15 RPM, 250K TPM

gemini-2.0-flash-lite
  Input: $0.10  |  Output: $0.40  |  Context: 1M tokens
  Fastest and cheapest


GEMINI 1.5 SERIES (older)
--------------------------

gemini-1.5-pro
  Input: $1.25  |  Output: $5.00  |  Context: 1M tokens
  Long context (over 200K): $2.50 input / $10.00 output

gemini-1.5-flash
  Input: $0.15  |  Output: $0.60  |  Context: 1M tokens


COST SAVING OPTIONS
-------------------

1. Free tier: Available for most models with daily rate limits
2. Batch API: 50% discount for async processing
3. Context caching: Cache reads at 10% of base input price
4. Grounding with Google Search: 1500 free queries/day, then $35/1000 queries


========================================================================

# Anthropic Claude Models
claude_models = [
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-haiku-3-5-20241022",
    "claude-3-haiku-20240307",
]

# Google Gemini Models

"""
