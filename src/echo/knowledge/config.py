"""Knowledge base configuration.

Only what every knowledge base needs: which provider, which corpus, how much to
return. Connection details are provider-specific and live with their provider —
`WeaviateConfig` in `weaviate_kb.py` — so this stays the thing a caller can
build without knowing what is behind it.

Callers build this from their own system of record. In matrix that is the
`knowledge_base` row: `provider` -> provider, `provider_id` -> collection,
`workspace_id` -> tenant, `num_results` -> top_k.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class KnowledgeBaseConfig(BaseModel):
    """Which corpus to search, and how much of it to return."""

    provider: Literal["weaviate"] = "weaviate"

    # The index at the provider — a Weaviate collection, a Bedrock
    # knowledgeBaseId, whatever the provider calls it.
    collection: str

    # Where the corpus is partitioned per customer, the partition to read.
    # A multi-tenant Weaviate collection rejects any read that omits it.
    tenant: Optional[str] = None

    top_k: int = Field(default=8, ge=1)
