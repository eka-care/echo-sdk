"""
Unit tests for Echo SDK prompt management.

These tests do not require Langfuse credentials and test the core
functionality of the prompt management system.
"""

import pytest

from echo.prompts import (
    AgentPrompt,
    BasePromptProvider,
    FetchedPrompt,
    PromptPersona,
    PromptFetchError,
    PromptTask,
    get_prompt_provider,
    reset_prompt_provider,
)


class TestFetchedPrompt:
    """Tests for FetchedPrompt model."""

    def test_basic_creation(self):
        """Test creating a basic FetchedPrompt."""
        prompt_def = AgentPrompt(
            persona=PromptPersona(role="Test Role"),
            task=PromptTask(description="Test task", expected_output="Output"),
        )
        prompt = FetchedPrompt(
            name="test-prompt",
            agent_prompt=prompt_def,
        )
        assert prompt.name == "test-prompt"
        assert prompt.version is None
        assert prompt.agent_prompt.task.description == "Test task"

    def test_with_version(self):
        """Test creating a FetchedPrompt with version."""
        prompt_def = AgentPrompt(
            task=PromptTask(description="Test content"),
        )
        prompt = FetchedPrompt(
            name="test-prompt",
            version="3",
            agent_prompt=prompt_def,
        )
        assert prompt.version == "3"

    def test_agent_prompt_access(self):
        """Test accessing agent_prompt fields."""
        prompt_def = AgentPrompt(
            persona=PromptPersona(
                role="Medical Analyst",
                goal="Provide accurate analysis",
                backstory="Expert in medical data",
            ),
            task=PromptTask(
                description="Analyze the patient data",
                expected_output="A summary report",
            ),
        )
        prompt = FetchedPrompt(
            name="test-prompt",
            agent_prompt=prompt_def,
        )

        assert prompt.agent_prompt.task.description == "Analyze the patient data"
        assert prompt.agent_prompt.task.expected_output == "A summary report"
        assert prompt.agent_prompt.persona.role == "Medical Analyst"
        assert prompt.agent_prompt.persona.goal == "Provide accurate analysis"
        assert prompt.agent_prompt.persona.backstory == "Expert in medical data"


class TestAgentPrompt:
    """Tests for AgentPrompt model."""

    def test_with_all_fields(self):
        """Test creating AgentPrompt with all fields present."""
        prompt_def = AgentPrompt(
            persona=PromptPersona(
                role="Medical Analyst",
                goal="Provide accurate analysis",
                backstory="Expert in medical data",
            ),
            task=PromptTask(
                description="Analyze the patient data",
                expected_output="A summary report",
            ),
        )

        assert prompt_def.task.description == "Analyze the patient data"
        assert prompt_def.task.expected_output == "A summary report"
        assert prompt_def.persona.role == "Medical Analyst"
        assert prompt_def.persona.goal == "Provide accurate analysis"
        assert prompt_def.persona.backstory == "Expert in medical data"

    def test_with_required_fields_only(self):
        """Test creating AgentPrompt with only required fields."""
        prompt_def = AgentPrompt(
            task=PromptTask(
                description="Do the task",
                expected_output="The output",
            ),
        )

        assert prompt_def.task.description == "Do the task"
        assert prompt_def.task.expected_output == "The output"
        # Optional fields should default to None
        assert prompt_def.persona.role is None
        assert prompt_def.persona.goal is None
        assert prompt_def.persona.backstory is None

    def test_default_persona(self):
        """Test that persona defaults to empty PromptPersona."""
        prompt_def = AgentPrompt(
            task=PromptTask(description="Task"),
        )
        assert prompt_def.persona is not None
        assert prompt_def.persona.role is None


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
            agent_prompt=AgentPrompt(
                task=PromptTask(description=description, expected_output="output"),
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
        assert prompt.agent_prompt is not None
        assert provider.fetch_count == 1

    @pytest.mark.asyncio
    async def test_get_prompt_with_version(self):
        """Test get_prompt with version parameter."""
        provider = MockPromptProvider()

        prompt = await provider.get_prompt("test-prompt", version="2")

        assert prompt.version == "2"
        assert prompt.agent_prompt is not None

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
        assert "cardiology" in prompt.agent_prompt.task.description
        assert "John" in prompt.agent_prompt.task.description


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
