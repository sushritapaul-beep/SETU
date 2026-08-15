"""Dense embeddings for SETU, backed by Jina AI's embeddings API.

Jina is used because jina-embeddings-v3 is genuinely multilingual — one model
covers English, Hindi and Tamil in a shared vector space — and because it's an
API call rather than a ~2.2GB local model, which keeps the pipeline off a
machine that is already short of RAM.
"""

from __future__ import annotations

import logging

import httpx
import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)

JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"


class JinaEmbedder:
    """Encode text into normalised dense vectors via Jina's embeddings API."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> None:
        """Configure the embedder.

        Args:
            model_name: Jina model id. Defaults to settings.EMBEDDING_MODEL.
            batch_size: Texts sent per request. Defaults to settings.EMBED_BATCH_SIZE.
            normalize: L2-normalise outputs so cosine similarity equals dot
                product, which is what the Qdrant collection is configured for.
        """
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size if batch_size is not None else settings.EMBED_BATCH_SIZE
        self.normalize = normalize
        self._dim = settings.EMBEDDING_DIM
        self._api_key = settings.JINA_API_KEY

    @property
    def dim(self) -> int:
        """Vector width this embedder produces.

        Returns:
            The configured embedding dimension.
        """
        return self._dim

    def embed_batch(self, texts: list[str], task: str = "retrieval.passage") -> np.ndarray:
        """Encode a list of texts into a matrix of dense vectors.

        Args:
            texts: Texts to encode. May be empty.
            task: Jina's asymmetric-retrieval task hint — "retrieval.passage"
                for indexed documents, "retrieval.query" for a search query.

        Returns:
            Array of shape (len(texts), dim), dtype float32. An empty input
            returns shape (0, dim) so callers can index it unconditionally.
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            rows.extend(self._request(texts[start : start + self.batch_size], task))
        return np.asarray(rows, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a single query string.

        Args:
            text: The query.

        Returns:
            A 1-D array of shape (dim,), dtype float32.
        """
        return self.embed_batch([text], task="retrieval.query")[0]

    def _request(self, texts: list[str], task: str) -> list[list[float]]:
        """Call Jina's /v1/embeddings for one batch and return ordered vectors."""
        response = httpx.post(
            JINA_EMBEDDINGS_URL,
            headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
            if self._api_key
            else {},
            json={
                "model": self.model_name,
                "input": texts,
                "task": task,
                "dimensions": self._dim,
                "normalized": self.normalize,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [row["embedding"] for row in sorted(data, key=lambda row: row["index"])]
