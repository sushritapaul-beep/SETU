"""Dense embeddings for SETU, backed by Jina AI's embeddings API.

Jina is used because jina-embeddings-v3 is genuinely multilingual — one model
covers English, Hindi and Tamil in a shared vector space — and because it's an
API call rather than a ~2.2GB local model, which keeps the pipeline off a
machine that is already short of RAM.
"""

from __future__ import annotations

import logging
import time

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
        max_retries: int = 0,
    ) -> None:
        """Configure the embedder.

        Args:
            model_name: Jina model id. Defaults to settings.EMBEDDING_MODEL.
            batch_size: Texts sent per request. Defaults to settings.EMBED_BATCH_SIZE.
            normalize: L2-normalise outputs so cosine similarity equals dot
                product, which is what the Qdrant collection is configured for.
            max_retries: Retries on 429 / 5xx, with backoff. 0 by default so a
                live query fails fast; bulk indexing opts in, because Jina's
                per-minute rate limit is hit within two full batches.
        """
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = batch_size if batch_size is not None else settings.EMBED_BATCH_SIZE
        self.normalize = normalize
        self.max_retries = max_retries
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
        for attempt in range(self.max_retries + 1):
            try:
                response = self._post(texts, task)
            except httpx.TransportError as exc:
                # A dropped connection or timeout is as transient as a 429.
                if attempt == self.max_retries:
                    raise
                wait, reason = 5.0 * 2**attempt, type(exc).__name__
            else:
                retryable = response.status_code == 429 or response.status_code >= 500
                if not retryable or attempt == self.max_retries:
                    break
                try:
                    wait = float(response.headers.get("Retry-After", ""))
                except ValueError:
                    wait = 5.0 * 2**attempt
                reason = str(response.status_code)
            logger.warning("Jina request failed (%s); retrying in %.0fs (%d/%d)",
                           reason, wait, attempt + 1, self.max_retries)
            time.sleep(wait)
        response.raise_for_status()
        data = response.json()["data"]
        return [row["embedding"] for row in sorted(data, key=lambda row: row["index"])]

    def _post(self, texts: list[str], task: str) -> httpx.Response:
        """Send one embeddings request."""
        return httpx.post(
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
