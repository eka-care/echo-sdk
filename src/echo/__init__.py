"""
echo - Base Medical Agents for Eka Care

This package provides reusable medical AI agents for healthcare applications.
Framework-agnostic design with adapters for CrewAI, LangGraph, and standalone use.

Key components:
- Agents: Intent classification agents with to_crewai_agent() adapters
- Config: Type-safe configuration with LLMConfig, AgentConfig, TaskConfig
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
