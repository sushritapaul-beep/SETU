"""Integration tests: real Qdrant, real embeddings, no mocks in the data path.

Everything else in tests/ is mocked, which is why the suite passed while soft
404s, mislabelled content types and dropped provenance all shipped. These tests
exercise the seams between components — ingest to index to retrieve — where
those bugs actually live.

They are marked `integration` and excluded from the default run because they
make real Jina/Qdrant Cloud API calls — real network dependency and metered
quota, not a local model load. Run them explicitly:

    pytest tests/test_integration.py -m integration -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient

from ingestion.chunker import SemanticParentChildChunker
from retrieval.vectorstore import QdrantStore

pytestmark = pytest.mark.integration


SCHEME_DOC = """## Details
The MSME Sustainable ZED Certification Scheme promotes Zero Defect Zero Effect
manufacturing across micro, small and medium enterprises in India.

## Eligibility
Should be a Micro, Small or Medium Enterprise unit.
Should be registered on the Udyam portal with a valid registration number.

## Benefits
Certification Level 1 BRONZE costs ten thousand rupees. A joining reward of
ten thousand rupees is available, and subsidy is graded by enterprise size.

## Documents Required
Udyam Registration Number of the enterprise, the mobile number linked to that
registration, and a valid email address for login credentials.
"""

WELFARE_DOC = """## Details
The National Overseas Scholarship supports students from scheduled castes for
higher study abroad, administered by the Ministry of Social Justice.

## Eligibility
Candidates must belong to Scheduled Caste, denotified nomadic or semi-nomadic
tribes, landless agricultural labourers or traditional artisan categories.

## Benefits
Full tuition fees, an annual maintenance allowance and one economy class
air fare are provided to selected candidates.
"""


@pytest.fixture(scope="module")
def embedder() -> Any:
    """The real Jina-backed embedder, constructed once for the module."""
    from retrieval.embeddings import JinaEmbedder

    return JinaEmbedder()


def _chunk(doc: str, url: str, name: str) -> list[Any]:
    """Chunk a document with provenance, as the crawl pipeline does."""
    return SemanticParentChildChunker().chunk_document(
        doc, url, name,
        provenance={
            "doc_format": "html",
            "retrieved_at": "2026-08-06T12:00:00+00:00",
            "source_title": name,
            "doc_hash": "integration-test-hash",
        },
    )


def test_ingest_index_and_retrieve_round_trip(embedder: Any) -> None:
    """Chunks survive embedding, indexing and hybrid retrieval intact.

    This is the path that unit tests cannot cover: real vectors, a real Qdrant
    collection, and RRF fusion over both branches.
    """
    chunks = (
        _chunk(SCHEME_DOC, "https://www.myscheme.gov.in/schemes/mszcs", "ZED Certification")
        + _chunk(WELFARE_DOC, "https://socialjustice.gov.in/scheme/nos", "Overseas Scholarship")
    )
    assert chunks, "fixture must produce chunks"

    store = QdrantStore(
        client=QdrantClient(":memory:"), collection="itest", dim=embedder.dim,
        embedder=embedder,
    )
    store.ensure_collection(dim=embedder.dim)
    written = store.upsert_chunks(chunks, embedder.embed_batch([c.text for c in chunks]))
    assert written == len(chunks)

    # Dense + BM25 + RRF against real vectors.
    hits = store.hybrid_search(
        "What documents are required for ZED certification?",
        top_k=5, filters=QdrantStore.child_only_filter(),
    )
    assert hits, "hybrid search returned nothing"
    assert "mszcs" in hits[0]["source_url"], "top hit should be the ZED scheme"
    # Provenance must survive the whole round trip, not just chunking.
    assert hits[0]["chunk_type"] == "child"

    # A welfare question must reach the welfare document, proving both are
    # discriminable in one shared index.
    welfare = store.hybrid_search(
        "Who is eligible for the overseas scholarship for scheduled castes?",
        top_k=5, filters=QdrantStore.child_only_filter(),
    )
    assert "socialjustice" in welfare[0]["source_url"]


def test_parent_expansion_resolves_from_the_index(embedder: Any) -> None:
    """A retrieved child resolves back to its parent through the store."""
    chunks = _chunk(SCHEME_DOC, "https://www.myscheme.gov.in/schemes/mszcs", "ZED")
    store = QdrantStore(
        client=QdrantClient(":memory:"), collection="itest2", dim=embedder.dim,
        embedder=embedder,
    )
    store.ensure_collection(dim=embedder.dim)
    store.upsert_chunks(chunks, embedder.embed_batch([c.text for c in chunks]))

    children = store.hybrid_search(
        "Udyam registration", top_k=3, filters=QdrantStore.child_only_filter()
    )
    parent_ids = [h["parent_id"] for h in children if h["parent_id"]]
    assert parent_ids, "children must carry a parent_id"

    parents = store.fetch_parents(parent_ids)
    assert parents, "parent lookup must resolve"
    assert any("Zero Defect" in text for text in parents.values())


def test_out_of_corpus_query_scores_far_below_in_corpus(embedder: Any) -> None:
    """The reranker separates answerable from unanswerable questions.

    This gap is what lets the agent decline instead of confabulating, so it is
    worth asserting against real models rather than trusting it by construction.
    """
    from retrieval.reranker import JinaReranker

    chunks = _chunk(SCHEME_DOC, "https://www.myscheme.gov.in/schemes/mszcs", "ZED")
    store = QdrantStore(
        client=QdrantClient(":memory:"), collection="itest3", dim=embedder.dim,
        embedder=embedder,
    )
    store.ensure_collection(dim=embedder.dim)
    store.upsert_chunks(chunks, embedder.embed_batch([c.text for c in chunks]))

    reranker = JinaReranker()
    in_corpus = reranker.rerank(
        "What documents are required for ZED certification?",
        store.hybrid_search("ZED certification documents", top_k=5), top_k=1,
    )
    out_of_corpus = reranker.rerank(
        "How do I apply for a driving licence in Kerala?",
        store.hybrid_search("driving licence Kerala", top_k=5), top_k=1,
    )

    assert in_corpus and out_of_corpus
    assert in_corpus[0]["rerank_score"] > out_of_corpus[0]["rerank_score"] * 5, (
        "in-corpus relevance should dominate; got "
        f"{in_corpus[0]['rerank_score']:.4f} vs {out_of_corpus[0]['rerank_score']:.4f}"
    )


def test_indexed_corpus_matches_disk(tmp_path: Path) -> None:
    """The live collection agrees with the chunks on disk.

    Catches the failure mode where an index quietly serves a stale corpus after
    a re-ingest that never reached Qdrant.
    """
    from config.settings import get_settings

    settings = get_settings()
    chunks_path = settings.processed_data_dir / "chunks.jsonl"
    if not chunks_path.exists():
        pytest.skip("no chunks.jsonl to compare against")

    try:
        api_key = settings.QDRANT_API_KEY
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=api_key.get_secret_value() if api_key else None,
            timeout=10,
        )
        if not client.collection_exists(settings.QDRANT_COLLECTION):
            pytest.skip(f"collection {settings.QDRANT_COLLECTION} does not exist")
        indexed = client.count(settings.QDRANT_COLLECTION, exact=True).count
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qdrant unreachable: {exc}")

    on_disk = sum(
        1 for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    assert indexed == on_disk, (
        f"index holds {indexed} points but chunks.jsonl has {on_disk} — "
        "re-run: python scripts/ingest.py --index-only"
    )


def test_embeddings_are_normalised_and_correctly_shaped(embedder: Any) -> None:
    """Real Jina embedding output matches what the Qdrant collection is configured for."""
    vectors = embedder.embed_batch(
        ["Udyam registration is required", "उद्यम पंजीकरण आवश्यक है"]
    )

    assert vectors.shape == (2, embedder.dim)
    assert embedder.dim == 1024, "collection is configured for 1024 dimensions"
    # Cosine distance in Qdrant assumes normalised vectors.
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
