"""Observability helpers for prompt providers."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PromptFetchMetadata:
    """Information emitted to observability hooks for a prompt fetch."""

    prompt_name: str
    provider_name: str
    version: Optional[str] = None
    prompt_variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptObservationContext:
    """Captures runtime state for a prompt observability session."""

    start_time: float
    langfuse_observation: Optional[Any] = None


@dataclass
class PromptTelemetryConfig:
    """Configuration applied to prompt observability helpers."""

    logger: logging.Logger = logging.getLogger("echo.prompts.observability")
    log_on_success: bool = True
    log_on_failure: bool = True
    success_log_level: int = logging.INFO
    failure_log_level: int = logging.ERROR
    span_name_template: str = "prompt.fetch.{name}"
    span_type: str = "generation"


class PromptObservability(ABC):
    """Abstract observability surface for prompt providers."""

    @abstractmethod
    def on_fetch_start(
        self,
        metadata: PromptFetchMetadata,
        *,
        langfuse_client: Optional[Any] = None,
    ) -> PromptObservationContext:
        """Begin a prompt fetch observation."""

    @abstractmethod
    def on_fetch_success(
        self,
        context: PromptObservationContext,
        metadata: PromptFetchMetadata,
        result: Dict[str, Any],
    ) -> None:
        """Record a successful prompt fetch."""

    @abstractmethod
    def on_fetch_failure(
        self,
        context: PromptObservationContext,
        metadata: PromptFetchMetadata,
        error: Exception,
    ) -> None:
        """Record a failed prompt fetch."""


class NoopPromptObservability(PromptObservability):
    """No-op implementation used when instrumentation is disabled."""

    def on_fetch_start(
        self,
        metadata: PromptFetchMetadata,
        *,
        langfuse_client: Optional[Any] = None,
    ) -> PromptObservationContext:
        del metadata, langfuse_client
        return PromptObservationContext(start_time=time.monotonic())

    def on_fetch_success(
        self,
        context: PromptObservationContext,
        metadata: PromptFetchMetadata,
        result: Dict[str, Any],
    ) -> None:
        del context, metadata, result

    def on_fetch_failure(
        self,
        context: PromptObservationContext,
        metadata: PromptFetchMetadata,
        error: Exception,
    ) -> None:
        del context, metadata, error


class LangfusePromptObservability(PromptObservability):
    """Langfuse-aware observability implementation."""

    def __init__(self, config: Optional[PromptTelemetryConfig] = None) -> None:
        self.config = config or PromptTelemetryConfig()

    def _span_name(self, metadata: PromptFetchMetadata) -> str:
        return self.config.span_name_template.format(
            name=metadata.prompt_name,
            provider=metadata.provider_name,
        )

    def _start_langfuse_observation(
        self,
        client: Any,
        metadata: PromptFetchMetadata,
    ) -> Optional[Any]:
        if client is None:
            return None

        try:
            return client.start_observation(
                name=self._span_name(metadata),
                as_type=self.config.span_type,
                input={
                    "prompt_name": metadata.prompt_name,
                    "version": metadata.version,
                    "variables": metadata.prompt_variables,
                },
                metadata={"provider": metadata.provider_name},
            )
        except AttributeError as attribute_error:
            self.config.logger.debug(
                "Langfuse client missing observation helpers: %s", attribute_error
            )
        except Exception as error:
            self.config.logger.debug("Failed to start Langfuse span: %s", error)

        return None

    def on_fetch_start(
        self,
        metadata: PromptFetchMetadata,
        *,
        langfuse_client: Optional[Any] = None,
    ) -> PromptObservationContext:
        context = PromptObservationContext(start_time=time.monotonic())
        context.langfuse_observation = self._start_langfuse_observation(
            langfuse_client, metadata
        )
        return context

    def _update_span_for_success(
        self,
        context: PromptObservationContext,
        metadata: PromptFetchMetadata,
        duration_ms: float,
        version: Optional[str],
    ) -> None:
        span = context.langfuse_observation
        if not span:
            return

        try:
            span.update(
                output={"version": version},
                metadata={"status": "success", "duration_ms": duration_ms},
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            self.config.logger.debug("Failed to update Langfuse span: %s", exc)
        finally:
            span.end()
            context.langfuse_observation = None

    def _update_span_for_failure(
        self,
        context: PromptObservationContext,
        metadata: PromptFetchMetadata,
        duration_ms: float,
        error: Exception,
    ) -> None:
        span = context.langfuse_observation
        if not span:
            return

        try:
            span.update(
                metadata={
                    "status": "failure",
                    "duration_ms": duration_ms,
                    "error": str(error),
                },
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            self.config.logger.debug("Failed to update Langfuse span: %s", exc)
        finally:
            span.end()
            context.langfuse_observation = None

    def on_fetch_success(
        self,
        context: PromptObservationContext,
        metadata: PromptFetchMetadata,
        result: Dict[str, Any],
    ) -> None:
        duration_ms = (time.monotonic() - context.start_time) * 1e3
        version = result.get("version")
        self._update_span_for_success(context, metadata, duration_ms, version)

        if self.config.log_on_success:
            self.config.logger.log(
                self.config.success_log_level,
                "Prompt fetch succeeded: %s (version=%s, provider=%s) in %.2fms",
                metadata.prompt_name,
                version,
                metadata.provider_name,
                duration_ms,
            )

    def on_fetch_failure(
        self,
        context: PromptObservationContext,
        metadata: PromptFetchMetadata,
        error: Exception,
    ) -> None:
        duration_ms = (time.monotonic() - context.start_time) * 1e3
        self._update_span_for_failure(context, metadata, duration_ms, error)

        if self.config.log_on_failure:
            self.config.logger.log(
                self.config.failure_log_level,
                "Prompt fetch failed: %s (provider=%s) after %.2fms: %s",
                metadata.prompt_name,
                metadata.provider_name,
                duration_ms,
                error,
            )


_default_observability: Optional[PromptObservability] = None


def get_prompt_observability(reset: bool = False) -> PromptObservability:
    """Return the SDK default prompt observability implementation."""
    global _default_observability
    if _default_observability is None or reset:
        _default_observability = LangfusePromptObservability()
    return _default_observability


def set_prompt_observability(observability: PromptObservability) -> None:
    """Override the SDK default prompt observability singleton."""
    global _default_observability
    _default_observability = observability


def reset_prompt_observability() -> None:
    """Clear the cached SDK prompt observability instance (for tests)."""
    global _default_observability
    _default_observability = None


__all__ = [
    "LangfusePromptObservability",
    "NoopPromptObservability",
    "PromptFetchMetadata",
    "PromptObservationContext",
    "PromptObservability",
    "PromptTelemetryConfig",
    "get_prompt_observability",
    "reset_prompt_observability",
    "set_prompt_observability",
]
