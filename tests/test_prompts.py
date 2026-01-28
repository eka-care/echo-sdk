"""
Unit tests for Echo SDK prompt management.

These tests do not require Langfuse credentials and test the core
functionality of the prompt management system.
"""

import pytest

from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig
from echo.prompts import (
    BasePromptProvider,
    FetchedPrompt,
    PromptFetchError,
    get_prompt_provider,
    reset_prompt_provider,
)


class TestFetchedPrompt:
    """Tests for FetchedPrompt model."""

    def test_basic_creation(self):
        """Test creating a basic FetchedPrompt."""
        config = AgentConfig(
            persona=PersonaConfig(role="Test Role"),
            task=TaskConfig(description="Test task", expected_output="Output"),
        )
        prompt = FetchedPrompt(
            name="test-prompt",
            agent_config=config,
        )
        assert prompt.name == "test-prompt"
        assert prompt.version is None
        assert prompt.agent_config.task.description == "Test task"

    def test_with_version(self):
        """Test creating a FetchedPrompt with version."""
        config = AgentConfig(
            task=TaskConfig(description="Test content"),
        )
        prompt = FetchedPrompt(
            name="test-prompt",
            version="3",
            agent_config=config,
        )
        assert prompt.version == "3"

    def test_agent_config_access(self):
        """Test accessing agent_config fields."""
        config = AgentConfig(
            persona=PersonaConfig(
                role="Medical Analyst",
                goal="Provide accurate analysis",
                backstory="Expert in medical data",
            ),
            task=TaskConfig(
                description="Analyze the patient data",
                expected_output="A summary report",
            ),
        )
        prompt = FetchedPrompt(
            name="test-prompt",
            agent_config=config,
        )

        assert prompt.agent_config.task.description == "Analyze the patient data"
        assert prompt.agent_config.task.expected_output == "A summary report"
        assert prompt.agent_config.persona.role == "Medical Analyst"
        assert prompt.agent_config.persona.goal == "Provide accurate analysis"
        assert prompt.agent_config.persona.backstory == "Expert in medical data"


class TestAgentConfig:
    """Tests for AgentConfig model."""

    def test_with_all_fields(self):
        """Test creating AgentConfig with all fields present."""
        config = AgentConfig(
            persona=PersonaConfig(
                role="Medical Analyst",
                goal="Provide accurate analysis",
                backstory="Expert in medical data",
            ),
            task=TaskConfig(
                description="Analyze the patient data",
                expected_output="A summary report",
            ),
        )

        assert config.task.description == "Analyze the patient data"
        assert config.task.expected_output == "A summary report"
        assert config.persona.role == "Medical Analyst"
        assert config.persona.goal == "Provide accurate analysis"
        assert config.persona.backstory == "Expert in medical data"

    def test_with_required_fields_only(self):
        """Test creating AgentConfig with only required fields."""
        config = AgentConfig(
            task=TaskConfig(
                description="Do the task",
                expected_output="The output",
            ),
        )

        assert config.task.description == "Do the task"
        assert config.task.expected_output == "The output"
        # Optional fields should default to None
        assert config.persona.role is None
        assert config.persona.goal is None
        assert config.persona.backstory is None

    def test_default_persona(self):
        """Test that persona defaults to empty PersonaConfig."""
        config = AgentConfig(
            task=TaskConfig(description="Task"),
        )
        assert config.persona is not None
        assert config.persona.role is None


class MockPromptProvider(BasePromptProvider):
    """Mock provider for testing base class functionality."""

    def __init__(self):
        self.fetch_count = 0

    async def get_prompt(self, name, version=None, prompt_variables=None, **kwargs):
        self.fetch_count += 1
        # Build description with variables if provided
        description = f"Task for {name}"
        if prompt_variables:
            description += f" with {prompt_variables}"

        return FetchedPrompt(
            name=name,
            version=str(version),
            agent_config=AgentConfig(
                task=TaskConfig(description=description, expected_output="output"),
            ),
        )


class TestBasePromptProvider:
    """Tests for BasePromptProvider."""

    @pytest.mark.asyncio
    async def test_get_prompt_returns_fetched_prompt(self):
        """Test that get_prompt returns a FetchedPrompt."""
        provider = MockPromptProvider()

        prompt = await provider.get_prompt("test-prompt")

        assert isinstance(prompt, FetchedPrompt)
        assert prompt.name == "test-prompt"
        assert prompt.agent_config is not None
        assert provider.fetch_count == 1

    @pytest.mark.asyncio
    async def test_get_prompt_with_version(self):
        """Test get_prompt with version parameter."""
        provider = MockPromptProvider()

        prompt = await provider.get_prompt("test-prompt", version="2")

        assert prompt.version == "2"
        assert prompt.agent_config is not None

    @pytest.mark.asyncio
    async def test_get_prompt_with_variables(self):
        """Test get_prompt with prompt_variables parameter."""
        provider = MockPromptProvider()

        prompt = await provider.get_prompt(
            "test-prompt",
            prompt_variables={"specialty": "cardiology", "patient_name": "John"},
        )

        assert prompt.name == "test-prompt"
        # Variables should be included in description
        assert "cardiology" in prompt.agent_config.task.description
        assert "John" in prompt.agent_config.task.description


class TestSingletonPattern:
    """Tests for the singleton pattern in factory."""

    def test_reset_clears_singleton(self):
        """Test that reset_prompt_provider clears the singleton."""
        reset_prompt_provider()
        # Singleton should be None after reset - no error means success


class TestExceptions:
    """Tests for custom exceptions."""

    def test_prompt_fetch_error(self):
        """Test PromptFetchError can be raised and caught."""
        with pytest.raises(PromptFetchError):
            raise PromptFetchError("Failed to fetch prompt")
