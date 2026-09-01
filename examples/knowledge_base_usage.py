"""Retrieving from a Weaviate knowledge base.

    WEAVIATE_HTTP_HOST=weaviate.example.com \
    WEAVIATE_GRPC_PORT=50051 \
    OPENAI_API_KEY=sk-... \
    python examples/knowledge_base_usage.py

The corpus is indexed by a separate pipeline; a connector only reads.
"""

import asyncio
import os

from echo.knowledge import KnowledgeBaseConfig, KnowledgeBaseError, get_knowledge_base
from echo.knowledge.weaviate_kb import close_all

# Below this, the corpus has nothing useful to say and the agent should decline
# rather than answer from the retrieved passages. Measured on a clinical
# corpus: answerable questions score 0.55-0.70, unrelated ones under 0.35.
# Re-measure for a different corpus.
MIN_SIMILARITY = 0.45


async def main() -> None:
    config = KnowledgeBaseConfig(
        provider="weaviate",
        collection=os.getenv("KB_COLLECTION", "ClinicalGuidelines"),
        tenant=os.getenv("KB_TENANT"),          # the workspace's shard
        top_k=8,
    )

    async with get_knowledge_base(config) as kb:
        health = await kb.health_check()
        print(f"healthy={health['healthy']}  {health['collection']}/{health['tenant']}\n")

        for question in (
            "What is the first line treatment for status epilepticus?",
            "How do I change a car tyre?",          # nothing in a clinical corpus
        ):
            results = await kb.retrieve(question)
            print(question)

            # Judge on `similarity`, never on `score`: hybrid scores are
            # normalised within one result set, so the top passage of ANY query
            # scores near the top of the range — including nonsense ones.
            best = results[0].similarity if results else None
            if best is None or best < MIN_SIMILARITY:
                print(f"  -> no usable match (best similarity {best})\n")
                continue

            for r in results[:3]:
                where = f"{r.filename} p{r.page_start}"
                print(f"  [{r.similarity:.3f}] {where}")
                print(f"          {r.content[:90].strip()}...")
            print()

        # Filters narrow to indexed metadata. An unknown field raises rather
        # than quietly matching nothing.
        paeds = await kb.retrieve(
            "dehydration management", top_k=3,
            filters={"category": "paediatric_medical"},
        )
        print(f"filtered to paediatrics: {len(paeds)} results")
        for r in paeds:
            print(f"  {r.category} — {r.filename} p{r.page_start}")

        try:
            await kb.retrieve("anything", filters={"catgory": "typo"})
        except KnowledgeBaseError as e:
            print(f"\nbad filter rejected: {e}")

    # close() detaches this connector; the client is shared per cluster across
    # the process, so a service closes it once on shutdown.
    await close_all()


if __name__ == "__main__":
    asyncio.run(main())
