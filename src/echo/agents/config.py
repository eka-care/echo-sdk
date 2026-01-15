"""
Configuration schemas for Echo SDK.

Pydantic models for YAML-based agent/task configuration.
"""
from pydantic import BaseModel
from pathlib import Path

import yaml


class PersonaConfig(BaseModel):
    """Agent configuration."""

    role: str
    goal: str
    backstory: str


class TaskConfig(BaseModel):
    """Task configuration."""

    description: str
    expected_output: str


class AgentConfig(BaseModel):
    """Combined role + task configuration."""

    persona: PersonaConfig
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

