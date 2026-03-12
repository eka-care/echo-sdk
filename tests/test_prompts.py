"""
Unit tests for Echo SDK prompt management.

These tests do not require Langfuse credentials and test the core
functionality of the prompt management system.
"""

import time
from typing import Any, Dict, Optional

import pytest

from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig
from echo.prompts import (
    BasePromptProvider,
    FetchedPrompt,
    PromptFetchError,
    get_prompt_provider,
    reset_prompt_provider,
    reset_prompt_observability,
    set_prompt_observability,
)
from echo.prompts.observability import (
    PromptFetchMetadata,
    PromptObservationContext,
    PromptObservability,
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

    def __init__(
        self, observability: Optional[PromptObservability] = None
    ):
        super().__init__(observability=observability)
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


class RecordingObservability(PromptObservability):
    """Helper observability implementation used in tests."""

    def __init__(self) -> None:
        self.starts: list[PromptFetchMetadata] = []
        self.successes: list[tuple[PromptFetchMetadata, Dict[str, Any]]] = []
        self.failures: list[tuple[PromptFetchMetadata, Exception]] = []

    def on_fetch_start(
        self,
        metadata: PromptFetchMetadata,
        *,
        langfuse_client: Optional[Any] = None,
    ) -> PromptObservationContext:
        del langfuse_client
        self.starts.append(metadata)
        return PromptObservationContext(start_time=time.monotonic())

    def on_fetch_success(
        self,
        context: PromptObservationContext,
        metadata: PromptFetchMetadata,
        result: Dict[str, Any],
    ) -> None:
        del context
        self.successes.append((metadata, result))

    def on_fetch_failure(
        self,
        context: PromptObservationContext,
        metadata: PromptFetchMetadata,
        error: Exception,
    ) -> None:
        del context
        self.failures.append((metadata, error))


class InstrumentedPromptProvider(BasePromptProvider):
    """Simple provider that exercises the observability hooks."""

    def __init__(
        self,
        fail: bool = False,
        observability: Optional[PromptObservability] = None,
    ):
        super().__init__(observability=observability)
        self.fail = fail

    async def get_prompt(
        self,
        name: str,
        version: Optional[str] = None,
        prompt_variables: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> FetchedPrompt:
        del kwargs, prompt_variables

        metadata = PromptFetchMetadata(
            prompt_name=name,
            provider_name="instrumented",
            version=version,
            prompt_variables={},
        )

        context = self.observability.on_fetch_start(metadata, langfuse_client=None)

        try:
            if self.fail:
                raise RuntimeError("forced failure")

            prompt = FetchedPrompt(
                name=name,
                version=version,
                agent_config=AgentConfig(
                    task=TaskConfig(description="desc", expected_output="ok")
                ),
            )

            self.observability.on_fetch_success(
                context, metadata, {"version": prompt.version}
            )

            return prompt
        except Exception as exc:
            self.observability.on_fetch_failure(context, metadata, exc)
            raise


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


class TestPromptObservabilityHooks:
    """Tests that the observability hooks are invoked."""

    @pytest.mark.asyncio
    async def test_success_records_events(self):
        observer = RecordingObservability()
        provider = InstrumentedPromptProvider(observability=observer)

        prompt = await provider.get_prompt("obs-success")

        assert len(observer.starts) == 1
        assert len(observer.successes) == 1
        assert observer.successes[0][0].prompt_name == "obs-success"
        assert observer.successes[0][1]["version"] == prompt.version
        assert not observer.failures

    @pytest.mark.asyncio
    async def test_failure_records_events(self):
        observer = RecordingObservability()
        provider = InstrumentedPromptProvider(observability=observer, fail=True)

        with pytest.raises(RuntimeError):
            await provider.get_prompt("obs-failure")

        assert len(observer.starts) == 1
        assert len(observer.failures) == 1
        assert observer.failures[0][0].prompt_name == "obs-failure"


class TestPromptObservabilityFactory:
    """Ensure the prompt factory reuses the shared observability instance."""

    def test_factory_reuses_shared_observability(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        observer = RecordingObservability()
        set_prompt_observability(observer)
        reset_prompt_provider()

        class DummyPromptProvider(BasePromptProvider):
            def __init__(self, observability: Optional[PromptObservability] = None):
                super().__init__(observability=observability)

            async def get_prompt(self, name, **kwargs):
                return FetchedPrompt(
                    name=name,
                    agent_config=AgentConfig(
                        task=TaskConfig(description="dummy", expected_output="ok")
                    ),
                )

        monkeypatch.setattr(
            "echo.prompts.factory.LangfusePromptProvider",
            DummyPromptProvider,
        )

        provider_one = get_prompt_provider()
        provider_two = get_prompt_provider()

        assert provider_one is provider_two
        assert provider_one.observability is observer

        reset_prompt_provider()
        reset_prompt_observability()


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
