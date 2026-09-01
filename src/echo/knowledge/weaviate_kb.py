"""Weaviate knowledge base connector.

Read-only hybrid search over a multi-tenant collection. The corpus is indexed
by a separate pipeline; nothing here writes.

Clients are shared per cluster across the process. A gRPC handshake costs
~450ms against a ~130ms query, so a client per request would nearly quadruple
retrieval latency, and connectors are commonly built per request.
"""

import asyncio
import os
import re
from typing import Any, Dict, List, Optional

from pydantic import Field

from .base import BaseKnowledgeBase, KnowledgeBaseError, RetrievalResult
from .config import KnowledgeBaseConfig

# Properties read back on every query. Fixed by the schema the indexing
# pipeline creates, not by configuration.
_PROPERTIES = [
    "text", "source_uri", "filename", "doc_title",
    "category", "subcategory", "source_org", "heading_path",
    "page_start", "page_end",
]

# A STRICT SUBSET: only these carry index_filterable. Filtering on a property
# without that index matches nothing and returns no error, so `text`,
# `source_uri` and `heading_path` must never appear here even though they come
# back on every result.
_FILTERABLE = [
    "category", "subcategory", "source_org", "filename", "doc_title",
    "page_start", "page_end",
]

# Weaviate reports the vector half of a hybrid score inside a human-readable
# debug string, and its "original score" there is the raw cosine similarity —
# the only route to a comparable, thresholdable number, since hybrid queries
# return distance=None. Parsing prose is fragile across versions, so a miss
# degrades `similarity` to None rather than failing the query.
_COSINE_RE = re.compile(r"vector,hybridVector\).*?original score ([0-9.]+)")

_clients: Dict[tuple, Any] = {}
_clients_lock = asyncio.Lock()


class WeaviateConfig(KnowledgeBaseConfig):
    """Weaviate connection settings, defaulted from the environment."""

    # 1.0 is pure vector, 0.0 pure keyword. 0.8 measured best on the clinical
    # corpus; a different corpus should be measured rather than inherit it.
    alpha: float = Field(default=0.8, ge=0.0, le=1.0)

    # Ports and TLS are explicit rather than inferred from the port number:
    # clusters do serve gRPC plaintext on non-standard ports, and a wrong guess
    # surfaces as a connection timeout that reads like a network fault.
    http_host: str = os.getenv("WEAVIATE_HTTP_HOST", "")
    http_port: int = int(os.getenv("WEAVIATE_HTTP_PORT", "443"))
    http_secure: bool = os.getenv("WEAVIATE_HTTP_SECURE", "true").lower() == "true"
    grpc_host: Optional[str] = os.getenv("WEAVIATE_GRPC_HOST") or None
    grpc_port: int = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    grpc_secure: bool = os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true"

    api_key: Optional[str] = os.getenv("WEAVIATE_API_KEY") or None
    # Forwarded so Weaviate can vectorize the query text. Optional: a cluster
    # configured with its own key does not need it.
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY") or None

    @property
    def _host(self) -> str:
        # Config UIs habitually store the scheme with the host
        # ("https://weaviate.example.com"); the client will not accept it.
        return self.http_host.split("://", 1)[-1].strip("/")

    @property
    def connection_key(self) -> tuple:
        """Connectors pointed at the same cluster share one client."""
        return (self._host, self.http_port, self.http_secure,
                self.grpc_host or self._host, self.grpc_port, self.grpc_secure,
                self.api_key)


class WeaviateKnowledgeBase(BaseKnowledgeBase):
    """Hybrid retrieval from one Weaviate collection and tenant."""

    def __init__(self, config: WeaviateConfig) -> None:
        try:
            import weaviate  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "weaviate-client is required for WeaviateKnowledgeBase. "
                "Install with: pip install 'echo[weaviate]'"
            ) from e

        if not config.http_host:
            raise ValueError(
                "http_host is required — set it on the config or via "
                "WEAVIATE_HTTP_HOST"
            )
        self.config = config
        self._client: Any = None

    # -- connection ---------------------------------------------------------

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        import weaviate
        from weaviate.classes.init import Auth

        key = self.config.connection_key
        async with _clients_lock:
            client = _clients.get(key)
            if client is None:
                c = self.config
                headers = {}
                if c.openai_api_key:
                    headers["X-OpenAI-Api-Key"] = c.openai_api_key
                try:
                    client = weaviate.use_async_with_custom(
                        http_host=c._host, http_port=c.http_port,
                        http_secure=c.http_secure,
                        grpc_host=c.grpc_host or c._host, grpc_port=c.grpc_port,
                        grpc_secure=c.grpc_secure,
                        auth_credentials=Auth.api_key(c.api_key) if c.api_key else None,
                        headers=headers or None,
                    )
                    await client.connect()
                except Exception as e:
                    raise KnowledgeBaseError(
                        f"could not connect to Weaviate at {c._host}:{c.http_port}: {e}"
                    ) from e
                _clients[key] = client
        self._client = client
        return client

    async def _collection(self) -> Any:
        client = await self._get_client()
        coll = client.collections.get(self.config.collection)
        # A multi-tenant collection errors on any read that omits the tenant,
        # which is the right failure — it cannot silently read another
        # workspace's shard.
        return coll.with_tenant(self.config.tenant) if self.config.tenant else coll

    # -- retrieval ----------------------------------------------------------

    def _build_filters(self, filters: Dict[str, Any]) -> Any:
        from weaviate.classes.query import Filter

        unknown = set(filters) - set(_FILTERABLE)
        if unknown:
            # Weaviate matches nothing for an unknown property rather than
            # complaining, which reads as "no results" — a typo in a category
            # name would look like an empty corpus.
            raise KnowledgeBaseError(
                f"unknown filter field(s): {sorted(unknown)}; "
                f"filterable fields are {sorted(_FILTERABLE)}"
            )
        clauses = [Filter.by_property(k).equal(v) for k, v in filters.items()]
        return clauses[0] if len(clauses) == 1 else Filter.all_of(clauses)

    async def retrieve(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        if not query or not query.strip():
            return []

        from weaviate.classes.query import MetadataQuery

        coll = await self._collection()
        try:
            res = await coll.query.hybrid(
                query=query,
                alpha=self.config.alpha,
                limit=top_k or self.config.top_k,
                filters=self._build_filters(filters) if filters else None,
                return_properties=_PROPERTIES,
                # explain_score is not diagnostics here: it carries the cosine
                # similarity that `score` cannot provide.
                return_metadata=MetadataQuery(score=True, explain_score=True),
            )
        except KnowledgeBaseError:
            raise
        except Exception as e:
            raise KnowledgeBaseError(f"Weaviate query failed: {e}") from e

        return [self._to_result(o) for o in res.objects]

    @staticmethod
    def _to_result(obj: Any) -> RetrievalResult:
        p = obj.properties or {}
        m = getattr(obj, "metadata", None)

        similarity = None
        explain = getattr(m, "explain_score", None) if m else None
        if explain:
            match = _COSINE_RE.search(explain)
            if match:
                try:
                    similarity = float(match.group(1))
                except ValueError:
                    similarity = None

        return RetrievalResult(
            content=p.get("text") or "",
            score=float(getattr(m, "score", 0.0) or 0.0) if m else 0.0,
            similarity=similarity,
            source_uri=p.get("source_uri"),
            filename=p.get("filename"),
            page_start=p.get("page_start"),
            page_end=p.get("page_end"),
            doc_title=p.get("doc_title"),
            category=p.get("category"),
            subcategory=p.get("subcategory"),
            source_org=p.get("source_org"),
            heading_path=list(p.get("heading_path") or []),
            source_id=str(obj.uuid) if getattr(obj, "uuid", None) else None,
        )

    # -- lifecycle ----------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "provider": "weaviate",
            "collection": self.config.collection,
            "tenant": self.config.tenant,
            "host": self.config._host,
        }
        try:
            client = await self._get_client()
            info["ready"] = await client.is_ready()
            info["healthy"] = bool(info["ready"])
        except Exception as e:
            info["healthy"] = False
            info["error"] = str(e)
        return info

    async def close(self) -> None:
        """Detach from the shared client.

        Deliberately does NOT close it — other connectors in the process are
        using the same connection. Use `close_all()` on shutdown.
        """
        self._client = None


async def close_all() -> None:
    """Close every shared Weaviate client. Call once, on service shutdown."""
    async with _clients_lock:
        for client in _clients.values():
            try:
                await client.close()
            except Exception:
                # Shutdown path: a client that is already gone is not a failure
                # worth propagating over the ones still to be closed.
                pass
        _clients.clear()
