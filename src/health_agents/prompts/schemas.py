from __future__ import annotations

from pydantic import BaseModel, Field


class UserPrompt(BaseModel):
    name: str = Field(description="Stable identifier; matches the YAML filename stem.")
    description: str = Field(
        default="", description="One-line human summary of what this template captures."
    )
    content: str = Field(description="The template body, appended to the system prompt.")
