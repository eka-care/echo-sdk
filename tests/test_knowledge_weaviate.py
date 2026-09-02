"""WeaviateKnowledgeBase — result mapping, filters, lifecycle.

The client is faked throughout: CI has no cluster, and the behaviour worth
pinning is how a Weaviate response becomes a RetrievalResult, not whether
Weaviate works.
"""

import os

# Set BEFORE importing echo.knowledge: WeaviateConfig resolves its env defaults
# when the class is defined, so patching os.environ inside a test is too late.
# Production does not depend on this — matrix passes connection details
# explicitly from its MATRIX config blob.
os.environ.setdefault("WEAVIATE_HTTP_HOST", "weaviate.test.invalid")

from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

import pytest  # noqa: E402

from echo.knowledge import (  # noqa: E402
    KnowledgeBaseConfig,
    KnowledgeBaseError,
    RetrievalResult,
    get_knowledge_base,
)
from echo.knowledge.weaviate_kb import (  # noqa: E402
    WeaviateConfig,
    WeaviateKnowledgeBase,
    close_all,
)

pytest.importorskip("weaviate")

EXPLAIN = (
    "\nHybrid (Result Set keyword,bm25) Document abc: original score 5.725466, "
    "normalized score: 0.06908903 - \n"
    "Hybrid (Result Set vector,hybridVector) Document abc: original score "
    "0.5909128, normalized score: 0.8"
)


def _obj(**overrides):
    props = {
        "text": "Lorazepam, IV, 4 mg.",
        "kb_id": "clinical-guidelines",
        "source_uri": "s3://bucket/prefix/AHL_Ch14.pdf",
        "filename": "AHL_Ch14.pdf",
        "doc_title": "AHL Ch14 Neurological Disorders",
        "category": "adult_general_medical",
        "subcategory": "neurological_disorders",
        "heading_path": ["CHAPTER 14", "14.2 SEIZURES"],
        "page_start": 16,
        "page_end": 16,
    }
    props.update(overrides.pop("properties", {}))
    return SimpleNamespace(
        properties=props,
        uuid="b3d7f62c-771d-5e7b-9feb-d8346c26919e",
        metadata=SimpleNamespace(score=0.8690, explain_score=overrides.get("explain", EXPLAIN)),
    )


def _kb(objects=None, kb_id=None):
    kb = WeaviateKnowledgeBase(
        WeaviateConfig(collection="ClinicalGuidelines", tenant="ws-1",
                       kb_id=kb_id, http_host="weaviate.example.com")
    )
    coll = MagicMock()
    coll.query.hybrid = AsyncMock(
        return_value=SimpleNamespace(objects=objects if objects is not None else [_obj()])
    )
    kb._collection = AsyncMock(return_value=coll)
    return kb, coll


async def test_maps_properties_onto_result():
    kb, _ = _kb()
    (r,) = await kb.retrieve("status epilepticus")
    assert isinstance(r, RetrievalResult)
    assert r.content == "Lorazepam, IV, 4 mg."
    assert r.filename == "AHL_Ch14.pdf"
    assert r.page_start == 16
    assert r.heading_path == ["CHAPTER 14", "14.2 SEIZURES"]
    assert r.source_id == "b3d7f62c-771d-5e7b-9feb-d8346c26919e"


async def test_similarity_comes_from_explain_score_not_score():
    """The vector half's "original score" is the raw cosine. `score` is the
    fused ranking number and is not comparable across queries — the two must
    not be conflated."""
    kb, _ = _kb()
    (r,) = await kb.retrieve("status epilepticus")
    assert r.similarity == pytest.approx(0.5909128)
    assert r.score == pytest.approx(0.8690)


async def test_similarity_is_none_when_explain_score_unparseable():
    """Weaviate may reword that debug string. A retrieval that still returns
    passages must not fail because the similarity could not be read."""
    kb, _ = _kb(objects=[_obj(explain="something else entirely")])
    (r,) = await kb.retrieve("status epilepticus")
    assert r.similarity is None
    assert r.content


async def test_top_k_overrides_config_default():
    kb, coll = _kb()
    await kb.retrieve("q", top_k=15)
    assert coll.query.hybrid.await_args.kwargs["limit"] == 15
    await kb.retrieve("q")
    assert coll.query.hybrid.await_args.kwargs["limit"] == 8


async def test_unknown_filter_field_raises():
    """Weaviate matches nothing for an unknown property rather than
    complaining, so a mistyped category would read as an empty corpus."""
    kb, _ = _kb()
    with pytest.raises(KnowledgeBaseError, match="unknown filter field"):
        await kb.retrieve("q", filters={"catgory": "paediatric_medical"})


async def test_known_filter_is_passed_through():
    kb, coll = _kb()
    await kb.retrieve("q", filters={"category": "paediatric_medical"})
    assert coll.query.hybrid.await_args.kwargs["filters"] is not None


async def test_no_filters_and_no_kb_id_sends_none():
    kb, coll = _kb()
    await kb.retrieve("q")
    assert coll.query.hybrid.await_args.kwargs["filters"] is None


async def test_configured_kb_id_is_always_applied():
    """One workspace can hold several knowledge bases in one collection, and
    tenant does not separate them. A retrieval that forgets the kb_id clause
    answers from a neighbouring corpus and looks perfectly well-formed, so the
    connector adds it rather than trusting every caller to."""
    kb, coll = _kb(kb_id="hr-policies")
    await kb.retrieve("q")
    assert coll.query.hybrid.await_args.kwargs["filters"] is not None


async def test_caller_filters_combine_with_kb_id():
    kb, coll = _kb(kb_id="hr-policies")
    await kb.retrieve("q", filters={"category": "paediatric_medical"})
    assert coll.query.hybrid.await_args.kwargs["filters"] is not None


async def test_kb_id_is_mapped_onto_the_result():
    kb, _ = _kb()
    (r,) = await kb.retrieve("q")
    assert r.kb_id == "clinical-guidelines"


async def test_empty_query_short_circuits():
    kb, coll = _kb()
    assert await kb.retrieve("   ") == []
    coll.query.hybrid.assert_not_awaited()


async def test_query_failure_becomes_knowledge_base_error():
    kb, coll = _kb()
    coll.query.hybrid = AsyncMock(side_effect=RuntimeError("grpc unavailable"))
    with pytest.raises(KnowledgeBaseError, match="Weaviate query failed"):
        await kb.retrieve("q")


async def test_close_detaches_without_closing_the_shared_client():
    """close() must not tear down a connection other connectors are using;
    close_all() owns that."""
    kb, _ = _kb()
    client = AsyncMock()
    kb._client = client
    await kb.close()
    assert kb._client is None
    client.close.assert_not_awaited()


async def test_close_all_closes_shared_clients():
    import echo.knowledge.weaviate_kb as mod

    client = AsyncMock()
    mod._clients[("host", 443, True, "host", 50051, False, None)] = client
    await close_all()
    client.close.assert_awaited_once()
    assert mod._clients == {}


def test_factory_upgrades_plain_config():
    """A caller may supply the generic config; the factory fills in the
    provider-specific one.

    http_host is passed rather than set in the environment: WeaviateConfig
    reads env defaults when the class is DEFINED, so patching os.environ here
    would come too late. Production passes connection details explicitly too —
    matrix takes them from its MATRIX config blob, not from the environment.
    """
    cfg = KnowledgeBaseConfig(collection="C", tenant="ws-1")
    kb = get_knowledge_base(cfg)
    assert isinstance(kb, WeaviateKnowledgeBase)
    assert isinstance(kb.config, WeaviateConfig)
    assert kb.config.collection == "C"
    assert kb.config.tenant == "ws-1"


def test_factory_rejects_unknown_provider():
    cfg = KnowledgeBaseConfig(collection="C")
    cfg.provider = "pinecone"  # bypass the Literal to reach the factory branch
    with pytest.raises(ValueError, match="Unsupported knowledge base provider"):
        get_knowledge_base(cfg)


def test_host_strips_scheme():
    """Config UIs store "https://host"; the client rejects it."""
    cfg = WeaviateConfig(collection="C", http_host="https://weaviate.example.com/")
    assert cfg._host == "weaviate.example.com"


def test_missing_host_is_rejected_at_construction():
    with pytest.raises(ValueError, match="http_host is required"):
        WeaviateKnowledgeBase(WeaviateConfig(collection="C", http_host=""))
