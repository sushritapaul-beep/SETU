"""Cross-encoder reranking via Jina AI's rerank API.

Retrieval and reranking answer different questions. Bi-encoders embed the query
and document separately, so they are fast but approximate. A cross-encoder reads
query and document together and scores the pair directly — far too slow to run
over a whole corpus, but ideal for reordering the top ~20 candidates. Jina's
jina-reranker-v2-base-multilingual runs this as an API call instead of a
~2GB local model.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)

Hit = dict[str, Any]

# Key added to each hit; the original fusion score is preserved separately so
# both can be inspected when debugging a bad answer.
RERANK_SCORE_KEY = "rerank_score"

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


class JinaReranker:
    """Reorder retrieved hits by cross-encoder relevance to the query."""

    def __init__(self, model_name: str | None = None) -> None:
        """Configure the reranker.

        Args:
            model_name: Jina reranker model id. Defaults to settings.RERANKER_MODEL.
        """
        settings = get_settings()
        self.model_name = model_name or settings.RERANKER_MODEL
        self._api_key = settings.JINA_API_KEY

    def rerank(self, query: str, hits: list[Hit], top_k: int = 5) -> list[Hit]:
        """Score every hit against the query and return the best ones.

        Args:
            query: The user's query, in the same language as the hit texts.
            hits: Candidate hits, each with a "text" key.
            top_k: Maximum hits to return.

        Returns:
            A new list of the top_k hits, each with RERANK_SCORE_KEY added,
            ordered by descending relevance score. Input hits are not mutated.
        """
        if not hits:
            return []

        scores = self._request(query, [hit.get("text", "") for hit in hits])

        # Copy rather than mutate, so callers can still inspect the original
        # fusion ordering after reranking.
        scored = [
            {**hit, RERANK_SCORE_KEY: score} for hit, score in zip(hits, scores, strict=True)
        ]
        scored.sort(key=lambda h: h[RERANK_SCORE_KEY], reverse=True)
        return scored[:top_k]

    def _request(self, query: str, documents: list[str]) -> list[float]:
        """Call Jina's /v1/rerank and return scores in input order."""
        response = httpx.post(
            JINA_RERANK_URL,
            headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
            if self._api_key
            else {},
            json={
                "model": self.model_name,
                "query": query,
                "documents": documents,
                "return_documents": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        results = response.json()["results"]
        return [r["relevance_score"] for r in sorted(results, key=lambda r: r["index"])]
