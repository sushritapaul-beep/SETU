"""Smoke test: run one English query through hybrid search + reranking.

Prints the top results with both scores so you can see what fusion retrieved and
what the cross-encoder promoted or demoted.

Usage:
    python scripts/smoke_retrieval.py
    python scripts/smoke_retrieval.py --query "What documents are needed for ZED certification?"
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from retrieval.embeddings import BGEM3Embedder  # noqa: E402
from retrieval.reranker import RERANK_SCORE_KEY, BGEReranker  # noqa: E402
from retrieval.vectorstore import QdrantStore  # noqa: E402

logger = logging.getLogger("setu.smoke")

DEFAULT_QUERY = "Who is eligible for PMEGP?"


def parse_args() -> argparse.Namespace:
    """Read command-line options.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Smoke-test hybrid retrieval + reranking.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query to run.")
    parser.add_argument("--top-k", type=int, default=5, help="Results to display.")
    parser.add_argument("--candidates", type=int, default=20, help="Candidates before reranking.")
    return parser.parse_args()


def main() -> int:
    """Run the query end to end and print ranked results.

    Returns:
        Process exit code: 0 on success, 1 if the collection is missing or empty.
    """
    args = parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)-7s | %(name)s | %(message)s")
    settings = get_settings()

    embedder = BGEM3Embedder()
    store = QdrantStore(embedder=embedder)

    if not store.client.collection_exists(store.collection):
        logger.error(
            "Collection '%s' does not exist. Run: python scripts/ingest.py", store.collection
        )
        return 1

    count = store.client.count(store.collection, exact=True).count
    if count == 0:
        logger.error("Collection '%s' is empty. Run: python scripts/ingest.py", store.collection)
        return 1

    print(f"\nCollection : {store.collection} ({count} points) at {settings.QDRANT_URL}")
    print(f"Query      : {args.query}\n")

    # Restricted to children: they are the precise match targets, and their
    # parents can be expanded afterwards for context.
    hits = store.hybrid_search(
        query_text=args.query,
        k_dense=args.candidates,
        k_sparse=args.candidates,
        top_k=args.candidates,
        filters=QdrantStore.child_only_filter(),
    )
    print(f"Hybrid search (dense + BM25 + RRF) returned {len(hits)} candidates.")
    if not hits:
        logger.error("No candidates returned; is the collection populated with child chunks?")
        return 1

    reranked = BGEReranker().rerank(args.query, hits, top_k=args.top_k)

    # Fusion rank before reranking, so the reordering is visible.
    fusion_rank = {hit["chunk_id"]: i + 1 for i, hit in enumerate(hits)}

    print(f"\nTop {len(reranked)} after cross-encoder reranking:\n" + "=" * 78)
    for rank, hit in enumerate(reranked, start=1):
        moved = fusion_rank.get(hit["chunk_id"], "?")
        print(f"\n[{rank}] rerank={hit[RERANK_SCORE_KEY]:+.4f}   rrf={hit['score']:.5f}   "
              f"(was #{moved} after fusion)")
        print(f"    scheme : {hit['scheme_name']}")
        print(f"    source : {hit['source_url']}")
        snippet = " ".join(hit["text"].split())[:300]
        print(textwrap.fill(snippet, width=74, initial_indent="    ", subsequent_indent="    "))
    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
