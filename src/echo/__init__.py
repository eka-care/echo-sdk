"""
echo - Base Medical Agents for Eka Care

This package provides reusable medical AI agents for healthcare applications.
Framework-agnostic design with adapters for CrewAI, LangGraph, and standalone use.

Key components:
- Agents: Intent classification agents with to_crewai_agent() adapters
- Config: Type-safe configuration with LLMConfig, AgentPrompt, PromptTask
- LLM: Unified LLM interface supporting Bedrock, OpenAI, Anthropic
- Tools: Framework-agnostic tools with to_crewai_tool() adapters
"""

__version__ = "0.1.3"

from .agents import *
from .llm import *
from .models import *
from .tools import *

__all__ = [
    "agents",
    "models",
    "llm",
    "tools",
]

# --- Backwards-compatible aliases (pre-0.3.7 names) --------------------------
# AgentConfig/PersonaConfig/TaskConfig were renamed to AgentPrompt/
# PromptPersona/PromptTask. Keep the old names importable so consumers pinned
# to the 0.3.x API (e.g. voice2rx-based backends) upgrade without changes.
from .prompts.schemas import (  # noqa: E402
    AgentPrompt as AgentConfig,
    PromptPersona as PersonaConfig,
    PromptTask as TaskConfig,
)

__all__ += ["AgentConfig", "PersonaConfig", "TaskConfig"]
