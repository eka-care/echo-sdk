"""
Configuration schemas for Echo SDK.

Pydantic models for YAML-based agent/task configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class PersonaConfig(BaseModel):
    """Agent configuration."""

    role: Optional[str] = None
    goal: Optional[str] = None
    backstory: Optional[str] = None


class TaskConfig(BaseModel):
    """Task configuration."""

    description: str
    expected_output: Optional[str] = None


class AgentConfig(BaseModel):
    """Combined role + task configuration."""

    persona: PersonaConfig = PersonaConfig()
    task: TaskConfig


"""
Configuration loader for Echo SDK.

Loads YAML configs and returns typed Pydantic models.
"""


def load_agent_config(config_path: Path) -> AgentConfig:
    """
    Load agent configuration from YAML.

    Args:
        name: Name of the agent (e.g., "core_intent_classifier")
        config_path: Optional custom path. Defaults to config/{name}.yaml

    Returns:
        Config with .agent and .task attributes
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return AgentConfig(**data)
