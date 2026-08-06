"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3.

Retrieval and reranking answer different questions. Bi-encoders embed the query
and document separately, so they are fast but approximate. A cross-encoder reads
query and document together and scores the pair directly — far too slow to run
over a whole corpus, but ideal for reordering the top ~20 candidates.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from config.settings import get_settings

logger = logging.getLogger(__name__)

Hit = dict[str, Any]

# Key added to each hit; the original fusion score is preserved separately so
# both can be inspected when debugging a bad answer.
RERANK_SCORE_KEY = "rerank_score"


class BGEReranker:
    """Reorder retrieved hits by cross-encoder relevance to the query."""

    # Shared across instances: the cross-encoder is ~2GB and must not reload
    # per request.
    _model_cache: ClassVar[dict[tuple[str, str | None], Any]] = {}

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
        model: Any | None = None,
    ) -> None:
        """Configure the reranker without loading the model yet.

        Args:
            model_name: Hugging Face model id. Defaults to settings.RERANKER_MODEL.
            device: Torch device such as "cpu" or "cuda".
            batch_size: Query/document pairs scored per forward pass.
            model: Preloaded cross-encoder, injected by tests to avoid any
                download or inference.
        """
        settings = get_settings()
        self.model_name = model_name or settings.RERANKER_MODEL
        self.device = device
        self.batch_size = batch_size
        self._injected_model = model

    @property
    def model(self) -> Any:
        """The underlying CrossEncoder, loaded and cached on first access.

        Returns:
            A ready CrossEncoder instance.
        """
        if self._injected_model is not None:
            return self._injected_model

        key = (self.model_name, self.device)
        if key not in self._model_cache:
            # Lazy import keeps torch off the import path for callers that never
            # rerank (for example the ingestion script).
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker %s (first use may download ~2GB)", self.model_name)
            self._model_cache[key] = CrossEncoder(self.model_name, device=self.device)
        return self._model_cache[key]

    def rerank(self, query: str, hits: list[Hit], top_k: int = 5) -> list[Hit]:
        """Score every hit against the query and return the best ones.

        Args:
            query: The user's query, in the same language as the hit texts.
            hits: Candidate hits, each with a "text" key.
            top_k: Maximum hits to return.

        Returns:
            A new list of the top_k hits, each with RERANK_SCORE_KEY added,
            ordered by descending cross-encoder score. Input hits are not
            mutated.
        """
        if not hits:
            return []

        pairs = [(query, hit.get("text", "")) for hit in hits]
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        # Copy rather than mutate, so callers can still inspect the original
        # fusion ordering after reranking.
        scored = [
            {**hit, RERANK_SCORE_KEY: float(score)}
            for hit, score in zip(hits, scores, strict=True)
        ]
        scored.sort(key=lambda h: h[RERANK_SCORE_KEY], reverse=True)
        return scored[:top_k]

    @classmethod
    def clear_cache(cls) -> None:
        """Drop cached models so tests and long-running processes can free memory."""
        cls._model_cache.clear()
