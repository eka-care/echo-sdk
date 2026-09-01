"""Factory for knowledge base connectors."""

from .base import BaseKnowledgeBase
from .config import KnowledgeBaseConfig


def get_knowledge_base(config: KnowledgeBaseConfig) -> BaseKnowledgeBase:
    """Build a connector for the configured provider.

    Args:
        config: Which corpus to search. A provider-specific subclass
            (``WeaviateConfig``) carries connection details; a plain
            ``KnowledgeBaseConfig`` is upgraded to one using environment
            defaults.

    Raises:
        ValueError: Provider is not supported.
        ImportError: The provider's client library is not installed.
    """
    provider = config.provider.lower()

    if provider == "weaviate":
        from .weaviate_kb import WeaviateConfig, WeaviateKnowledgeBase

        if not isinstance(config, WeaviateConfig):
            config = WeaviateConfig(**config.model_dump())
        return WeaviateKnowledgeBase(config)

    raise ValueError(f"Unsupported knowledge base provider: {config.provider}")
