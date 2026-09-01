"""Base classes for knowledge base connectors.

A knowledge base here is a *retrieval* surface: text in, ranked passages out.
Indexing, chunking and embedding happen elsewhere — a connector never writes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KnowledgeBaseError(Exception):
    """Retrieval or connection failed.

    Raised rather than swallowed: a caller that silently returns no passages
    looks identical to a corpus with no answer, and the agent then answers from
    the model's own knowledge with nothing to say it did.
    """


class RetrievalResult(BaseModel):
    """One retrieved passage.

    `score` and `similarity` are NOT interchangeable, and the difference
    matters:

    - `score` is the provider's ranking score. Under Weaviate's hybrid search
      it is normalised *within a single result set*, so the best passage of any
      query scores near the top of the range whether it is genuinely relevant
      or merely the least-wrong of ten thousand. Measured on this corpus,
      "what is the capital of France" scores 0.800 — indistinguishable from a
      real clinical question. Use it to order results. Never to judge them.

    - `similarity` is raw cosine similarity, 0-1, comparable across queries.
      The same three queries score 0.591 (answerable), 0.340 and 0.083. This
      is the field to threshold on when deciding whether the corpus can answer
      at all. It is Optional because not every provider exposes it.
    """

    content: str
    score: float
    similarity: Optional[float] = None

    # Citation: a link is built from the KB's configured prefix plus filename,
    # with page_start as the page anchor.
    source_uri: Optional[str] = None
    filename: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    doc_title: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    source_org: Optional[str] = None
    heading_path: List[str] = Field(default_factory=list)

    source_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseKnowledgeBase(ABC):
    """A read-only retrieval surface over an indexed corpus.

    Implementations hold a network client, so they are async and closeable.
    Prefer the async context manager when the lifetime is scoped::

        async with get_knowledge_base(config) as kb:
            results = await kb.retrieve("...")
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Return the best passages for `query`, most relevant first.

        Args:
            query: Natural-language search text.
            top_k: Passages to return. Falls back to the configured default.
            filters: Equality filters on metadata, e.g.
                ``{"category": "paediatric_medical"}``. Multiple keys are
                ANDed. Unknown keys raise rather than matching nothing.

        Raises:
            KnowledgeBaseError: The query could not be executed.
        """

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Report reachability and what this connector is pointed at."""

    @abstractmethod
    async def close(self) -> None:
        """Release the underlying client."""

    async def __aenter__(self) -> "BaseKnowledgeBase":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()
