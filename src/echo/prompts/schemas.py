"""Prompt schemas for Echo SDK.

Pydantic models describing an agent's prompt (persona + task) and the
descriptor used to fetch a prompt from a provider.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class PromptPersona(BaseModel):
    """Agent persona (role / goal / backstory)."""

    role: Optional[str] = None
    goal: Optional[str] = None
    backstory: Optional[str] = None


class PromptTask(BaseModel):
    """Task description and expected output."""

    description: str
    expected_output: Optional[str] = None


class AgentPrompt(BaseModel):
    """Combined persona + task — what the agent should be and do.

    Previously named ``AgentConfig``; renamed because it describes the
    prompt (persona + task), not runtime configuration of the agent.

    ``persona`` + ``task`` are the *stable* half of the system prompt: the same
    bytes for every session of an agent, so providers put the prompt-cache
    breakpoint at their end. ``context`` is the *volatile* half — anything that
    varies per user, per session, or per turn — and is sent after that
    breakpoint, so it never invalidates the cached prefix.
    """

    persona: PromptPersona = PromptPersona()
    task: PromptTask
    #: Per-request context (user, session, current time, …). Rendered after the
    #: cache breakpoint. Agent instructions do not belong here — they go in
    #: ``task``, where they get cached.
    context: Optional[str] = None
    #: Reference instant for the "Current time" line the agent appends to the
    #: uncached context. None → now(UTC) at each LLM call; pin it only for
    #: tests/replays.
    datetime_utc: Optional[datetime] = None
    #: IANA timezone the reference instant is rendered in (e.g.
    #: "Asia/Kolkata"). None or unrecognized → UTC.
    timezone: Optional[str] = None


class PromptConfig(BaseModel):
    """Descriptor for fetching a prompt from a provider."""

    provider: Literal["langfuse"] = "langfuse"
    name: str = Field(..., description="The name of the prompt")
    prompt_variables: Optional[dict[str, str]] = None
    version: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def strip_name(cls, data):
        if isinstance(data, dict) and data.get("name"):
            data["name"] = data["name"].strip()
        return data
