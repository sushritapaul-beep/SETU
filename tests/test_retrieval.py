"""Offline tests for the retrieval layer.

No model downloads and no Qdrant server: the embedder and reranker are stubbed,
and Qdrant runs in its in-memory mode, which supports sparse vectors and RRF
fusion exactly as the real server does.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient

from ingestion.chunker import Chunk, ChunkMetadata
from retrieval.embeddings import BGEM3Embedder
from retrieval.reranker import RERANK_SCORE_KEY, BGEReranker
from retrieval.vectorstore import DENSE_VECTOR, SPARSE_VECTOR, QdrantStore

DIM = 8


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------

class _FakeSentenceTransformer:
    """Deterministic stand-in for SentenceTransformer."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.calls.append(list(texts))
        # Seeded by text hash so the same string always yields the same vector.
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.random(self.dim, dtype=np.float32)
            if kwargs.get("normalize_embeddings"):
                vec = vec / np.linalg.norm(vec)
            rows.append(vec)
        return np.vstack(rows)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


class _FakeCrossEncoder:
    """Cross-encoder stub that scores by keyword overlap instead of inference."""

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        scores = []
        for query, doc in pairs:
            query_terms = set(query.lower().split())
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            scores.append(float(overlap))
        return scores


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> BGEM3Embedder:
    """An embedder whose model property returns the deterministic stub."""
    BGEM3Embedder.clear_cache()
    embedder = BGEM3Embedder(model_name="stub-model")
    monkeypatch.setattr(
        BGEM3Embedder, "model", property(lambda self: _FakeSentenceTransformer(DIM))
    )
    return embedder


def _chunk(chunk_id: str, text: str, parent_id: str | None, chunk_type: str) -> Chunk:
    """Build a Chunk with a valid 32-hex id."""
    return Chunk(
        id=chunk_id,
        parent_id=parent_id,
        text=text,
        metadata=ChunkMetadata(
            source_url="https://www.myscheme.gov.in/schemes/mszcs",
            scheme_name="MSME Sustainable (ZED) Certification Scheme",
            chunk_type=chunk_type,
        ),
    )


PARENT_ID = f"{99:032x}"
CORPUS = [
    _chunk(f"{1:032x}", "Should be registered in the Udyam portal to apply", PARENT_ID, "child"),
    _chunk(f"{2:032x}", "Credit guarantee covers eighty five percent", PARENT_ID, "child"),
    _chunk(f"{3:032x}", "Tamil Nadu handloom weaver subsidy for textiles", PARENT_ID, "child"),
    _chunk(PARENT_ID, "Full eligibility section including Udyam and credit terms", None, "parent"),
]


@pytest.fixture
def populated_store(fake_embedder: BGEM3Embedder) -> QdrantStore:
    """An in-memory Qdrant store preloaded with the small corpus."""
    store = QdrantStore(
        client=QdrantClient(":memory:"),
        collection="test_schemes",
        dim=DIM,
        embedder=fake_embedder,
    )
    store.ensure_collection(dim=DIM)
    vectors = fake_embedder.embed_batch([c.text for c in CORPUS])
    store.upsert_chunks(CORPUS, vectors)
    return store


# --------------------------------------------------------------------------
# 1. Embedder
# --------------------------------------------------------------------------

def test_embedder_returns_correct_shape_for_two_strings(fake_embedder: BGEM3Embedder) -> None:
    """Two sample strings produce a (2, dim) float32 matrix."""
    vectors = fake_embedder.embed_batch(
        ["Who is eligible for the ZED certification scheme?", "उद्यम पंजीकरण क्या है?"]
    )

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (2, DIM)
    assert vectors.dtype == np.float32
    # Normalisation is what makes cosine distance equal dot product in Qdrant.
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_query_returns_one_dimensional_vector(fake_embedder: BGEM3Embedder) -> None:
    """A single query embeds to shape (dim,), not (1, dim)."""
    assert fake_embedder.embed_query("eligibility criteria").shape == (DIM,)


def test_embed_batch_handles_empty_input(fake_embedder: BGEM3Embedder) -> None:
    """Empty input yields an empty but correctly shaped array, not an error."""
    vectors = fake_embedder.embed_batch([])
    assert vectors.shape[0] == 0
    assert vectors.ndim == 2


def test_embedder_does_not_load_model_on_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing an embedder must not trigger a multi-GB download."""
    BGEM3Embedder.clear_cache()
    embedder = BGEM3Embedder(model_name="definitely-not-a-real-model")
    # Nothing cached yet means nothing was loaded.
    assert not BGEM3Embedder._model_cache
    assert embedder.dim == 1024  # falls back to the configured dimension


# --------------------------------------------------------------------------
# 2. Reranker
# --------------------------------------------------------------------------

def test_reranker_reorders_hits_by_relevance() -> None:
    """The cross-encoder promotes the most relevant hit regardless of input order."""
    reranker = BGEReranker(model=_FakeCrossEncoder())
    hits = [
        {"chunk_id": "a", "text": "tamil nadu handloom weaver subsidy", "score": 0.9},
        {"chunk_id": "b", "text": "udyam portal registration is required", "score": 0.8},
        {"chunk_id": "c", "text": "unrelated boilerplate footer text", "score": 0.7},
    ]

    result = reranker.rerank("udyam portal registration", hits, top_k=2)

    assert len(result) == 2
    # "b" was second by fusion score but shares the most terms with the query.
    assert result[0]["chunk_id"] == "b"
    assert result[0][RERANK_SCORE_KEY] > result[1][RERANK_SCORE_KEY]
    # The original fusion score survives alongside the new one.
    assert result[0]["score"] == 0.8


def test_reranker_does_not_mutate_input_hits() -> None:
    """Reranking returns new dicts so the pre-rerank ordering stays inspectable."""
    reranker = BGEReranker(model=_FakeCrossEncoder())
    hits = [{"chunk_id": "a", "text": "udyam registration", "score": 0.5}]

    reranker.rerank("udyam", hits, top_k=1)

    assert RERANK_SCORE_KEY not in hits[0]


def test_reranker_handles_empty_hits() -> None:
    """An empty candidate list short-circuits without loading a model."""
    assert BGEReranker(model=_FakeCrossEncoder()).rerank("anything", [], top_k=5) == []


# --------------------------------------------------------------------------
# 3. QdrantStore against in-memory Qdrant
# --------------------------------------------------------------------------

def test_ensure_collection_is_idempotent(fake_embedder: BGEM3Embedder) -> None:
    """Calling ensure_collection twice is safe, so ingest can always call it."""
    store = QdrantStore(client=QdrantClient(":memory:"), collection="c", dim=DIM)
    store.ensure_collection(dim=DIM)
    store.ensure_collection(dim=DIM)

    info = store.client.get_collection("c")
    assert DENSE_VECTOR in info.config.params.vectors
    assert SPARSE_VECTOR in info.config.params.sparse_vectors


def test_upsert_chunks_writes_all_points(populated_store: QdrantStore) -> None:
    """Every chunk becomes a point, with its original hex id preserved."""
    assert populated_store.client.count("test_schemes", exact=True).count == len(CORPUS)

    hits = populated_store.sparse_search("Udyam portal", k=5)
    assert hits
    assert all(len(h["chunk_id"]) == 32 for h in hits)


def test_upsert_rejects_length_mismatch(populated_store: QdrantStore) -> None:
    """Misaligned chunks and vectors fail loudly instead of silently mispairing."""
    with pytest.raises(ValueError, match="length mismatch"):
        populated_store.upsert_chunks(CORPUS, np.zeros((2, DIM), dtype=np.float32))


def test_dense_search_returns_hits(
    populated_store: QdrantStore, fake_embedder: BGEM3Embedder
) -> None:
    """Dense search returns scored hits with payload fields promoted."""
    query_vec = fake_embedder.embed_query("Udyam portal registration")
    hits = populated_store.dense_search(query_vec, k=3)

    assert 0 < len(hits) <= 3
    assert set(hits[0]) >= {"id", "score", "text", "chunk_id", "source_url", "scheme_name"}
    assert hits[0]["source_url"].startswith("https://www.myscheme.gov.in/")


def test_sparse_search_matches_exact_keywords(populated_store: QdrantStore) -> None:
    """BM25 finds the chunk containing the query's rare term."""
    hits = populated_store.sparse_search("handloom weaver", k=3)

    assert hits
    assert "handloom" in hits[0]["text"].lower()


def test_hybrid_search_fuses_both_branches(populated_store: QdrantStore) -> None:
    """Hybrid search runs without a precomputed vector and returns fused hits."""
    hits = populated_store.hybrid_search("Udyam portal registration", top_k=3)

    assert hits
    assert all(h["score"] > 0 for h in hits), "RRF scores should be positive"


def test_hybrid_search_respects_child_only_filter(populated_store: QdrantStore) -> None:
    """The child filter keeps parent chunks out of the candidate set."""
    hits = populated_store.hybrid_search(
        "Udyam credit eligibility", top_k=10, filters=QdrantStore.child_only_filter()
    )

    assert hits
    assert all(h["chunk_type"] == "child" for h in hits)


def test_hybrid_search_requires_a_vector_source() -> None:
    """Without an embedder or query vector, the failure is explicit."""
    store = QdrantStore(client=QdrantClient(":memory:"), collection="c", dim=DIM)
    store.ensure_collection(dim=DIM)

    with pytest.raises(ValueError, match="query_vec or an embedder"):
        store.hybrid_search("anything")


def test_fetch_parents_expands_children_to_context(populated_store: QdrantStore) -> None:
    """A child's parent_id resolves back to the parent's full text."""
    children = populated_store.sparse_search("Udyam portal", k=5)
    parent_ids = [c["parent_id"] for c in children if c["parent_id"]]

    parents = populated_store.fetch_parents(parent_ids)

    assert parents
    assert any("Full eligibility section" in text for text in parents.values())


def test_point_id_conversion_is_a_stable_uuid() -> None:
    """Hex chunk ids map deterministically onto Qdrant-legal UUID point ids."""
    chunk_id = "0" * 31 + "1"
    first = QdrantStore._to_point_id(chunk_id)

    assert first == QdrantStore._to_point_id(chunk_id)
    assert first == "00000000-0000-0000-0000-000000000001"
