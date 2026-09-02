"""
echo.knowledge - Retrieval from indexed document corpora.

Read-only: a connector searches an index that a separate pipeline built.

    from echo.knowledge import KnowledgeBaseConfig, get_knowledge_base

    config = KnowledgeBaseConfig(
        provider="weaviate",
        collection="Documents",
        tenant=workspace_id,          # whose data — isolated shard
        kb_id="handbooks",            # which corpus — always filtered on
        top_k=8,
    )
    async with get_knowledge_base(config) as kb:
        for r in await kb.retrieve("what is the refund window?"):
            print(r.filename, r.page_start, r.similarity)

Connection details default from the environment (WEAVIATE_HTTP_HOST and
friends); pass a ``WeaviateConfig`` to set them explicitly.

On judging results: use ``similarity``, never ``score`` — see RetrievalResult.
"""

from .base import BaseKnowledgeBase, KnowledgeBaseError, RetrievalResult
from .config import KnowledgeBaseConfig
from .factory import get_knowledge_base

__all__ = [
    "BaseKnowledgeBase",
    "KnowledgeBaseConfig",
    "KnowledgeBaseError",
    "RetrievalResult",
    "get_knowledge_base",
]
