"""YAML loader for AgentPrompt."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .schemas import AgentPrompt

logger = logging.getLogger(__name__)


def load_agent_prompt(config_path: Path) -> AgentPrompt:
    """Load an AgentPrompt from a YAML file."""
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return AgentPrompt(**data)
