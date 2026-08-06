"""Standalone retrieval pipeline for Hugging Face Spaces.

A Space runs one container with no Qdrant server and no room for the full agent,
so this is a deliberately reduced version of the local system:

    included : snapshot index, hybrid search, cross-encoder rerank, cited answer
    excluded : IndicTrans2 translation, the LangGraph self-critique loop,
               multi-turn checkpointing

Queries are answered in the language they are asked in, relying on the LLM's own
multilingual ability rather than IndicTrans2 (~2GB of extra weights and quadratic
CPU decoding). The full agent stays local.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("setu.standalone")

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshot"

# Named to match retrieval/vectorstore.py so payload shapes stay identical.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
BM25_MODEL = "Qdrant/bm25"
COLLECTION = "setu_snapshot"

ANSWER_PROMPT = """You answer questions about Indian government MSME schemes.

Rules:
1. Use ONLY the numbered context passages below. If they do not contain the
   answer, say so plainly. Do not use outside knowledge about these schemes.
2. Cite the passage id in square brackets after each claim, like [a3f1c2d4].
3. Never invent figures, deadlines or URLs.
4. Reply in the same language the question was asked in.

Context passages:
{context}

Question: {query}

Answer:"""


def snapshot_available() -> bool:
    """Whether a usable snapshot is present on disk.

    Returns:
        True if both the chunks and vectors files exist.
    """
    return (SNAPSHOT_DIR / "chunks.jsonl").exists() and (SNAPSHOT_DIR / "vectors.npy").exists()


@lru_cache(maxsize=1)
def _load_index() -> tuple[Any, list[dict[str, Any]]]:
    """Build an in-memory Qdrant collection from the bundled snapshot.

    Returns:
        (client, chunk records). Built once and cached for the process lifetime.

    Raises:
        FileNotFoundError: If the snapshot is missing.
    """
    import uuid

    from qdrant_client import QdrantClient, models

    if not snapshot_available():
        raise FileNotFoundError(
            f"No snapshot at {SNAPSHOT_DIR}. Run: python scripts/export_snapshot.py"
        )

    records = [
        json.loads(line)
        for line in (SNAPSHOT_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vectors = np.load(SNAPSHOT_DIR / "vectors.npy")
    logger.info("Loading snapshot: %d chunks, vectors %s", len(records), vectors.shape)

    client = QdrantClient(":memory:")
    client.create_collection(
        COLLECTION,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=int(vectors.shape[1]), distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid.UUID(hex=record["id"])),
                vector={
                    DENSE_VECTOR: vectors[i].tolist(),
                    SPARSE_VECTOR: models.Document(text=record["text"], model=BM25_MODEL),
                },
                payload={
                    "text": record["text"],
                    "chunk_id": record["id"],
                    "source_url": record["metadata"]["source_url"],
                    "scheme_name": record["metadata"]["scheme_name"],
                },
            )
            for i, record in enumerate(records)
        ],
        wait=True,
    )
    return client, records


@lru_cache(maxsize=1)
def _get_embedder() -> Any:
    """The query embedder (BGE-M3), loaded once."""
    from retrieval.embeddings import BGEM3Embedder

    return BGEM3Embedder()


@lru_cache(maxsize=1)
def _get_reranker() -> Any | None:
    """The cross-encoder, or None if it cannot be loaded.

    Returns:
        A BGEReranker, or None so the Space degrades to fusion order rather than
        failing when memory is tight.
    """
    try:
        from retrieval.reranker import BGEReranker

        return BGEReranker()
    except Exception as exc:  # noqa: BLE001 - reranking is an optimisation
        logger.warning("Reranker unavailable (%s); using fusion order", exc)
        return None


@lru_cache(maxsize=1)
def _get_llm() -> Any:
    """The answer-generation model.

    Returns:
        A LangChain chat model.

    Raises:
        RuntimeError: If no credentials are configured.
    """
    from agents.nodes import get_llm

    return get_llm()


def answer(query: str, top_k: int = 5, candidates: int = 20) -> dict[str, Any]:
    """Answer one question using the bundled snapshot.

    Args:
        query: The user's question, in any language.
        top_k: Passages kept after reranking.
        candidates: Passages fetched before reranking.

    Returns:
        {answer, citations, hits}. On failure, answer explains what went wrong
        and citations is empty.
    """
    from qdrant_client import models

    try:
        client, _ = _load_index()
        vector = _get_embedder().embed_query(query)
    except Exception as exc:  # noqa: BLE001 - show the user something useful
        logger.exception("Snapshot retrieval failed")
        return {"answer": f"Retrieval unavailable: {exc}", "citations": [], "hits": []}

    response = client.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=vector.tolist(), using=DENSE_VECTOR, limit=candidates),
            models.Prefetch(
                query=models.Document(text=query, model=BM25_MODEL),
                using=SPARSE_VECTOR,
                limit=candidates,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=candidates,
        with_payload=True,
    )
    hits = [
        {
            "chunk_id": p.payload.get("chunk_id", ""),
            "text": p.payload.get("text", ""),
            "source_url": p.payload.get("source_url", ""),
            "scheme_name": p.payload.get("scheme_name", ""),
            "score": float(p.score or 0.0),
        }
        for p in response.points
    ]
    if not hits:
        return {"answer": "I could not find anything about that in my sources.",
                "citations": [], "hits": []}

    reranker = _get_reranker()
    hits = reranker.rerank(query, hits, top_k=top_k) if reranker else hits[:top_k]

    context = "\n\n".join(
        f"[{h['chunk_id'][:8]}] ({h['scheme_name']}) {' '.join(h['text'].split())}" for h in hits
    )
    try:
        response_text = _get_llm().invoke(ANSWER_PROMPT.format(context=context, query=query))
        text = getattr(response_text, "content", str(response_text))
    except Exception as exc:  # noqa: BLE001 - never lose the retrieved sources
        logger.exception("Generation failed")
        text = f"Could not generate an answer ({exc}), but these passages look relevant."

    citations = [
        {
            "chunk_id": h["chunk_id"],
            "label": h["chunk_id"][:8],
            "source_url": h["source_url"],
            "scheme_name": h["scheme_name"],
        }
        for h in hits
    ]
    return {"answer": text, "citations": citations, "hits": hits}
