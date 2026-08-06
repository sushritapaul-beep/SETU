"""Dense embeddings for SETU, backed by BAAI/bge-m3 via sentence-transformers.

BGE-M3 is used because it is genuinely multilingual: one model covers English,
Hindi and Tamil in a shared vector space, so a Hindi query can match an English
scheme document without a separate per-language index.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

from config.settings import get_settings

logger = logging.getLogger(__name__)


class BGEM3Embedder:
    """Encode text into normalised dense vectors, loading the model on first use."""

    # Keyed by (model_name, device) and held on the class, so constructing a new
    # embedder in a request handler never re-downloads or re-loads ~2GB of weights.
    _model_cache: ClassVar[dict[tuple[str, str | None], Any]] = {}

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> None:
        """Configure the embedder without loading anything yet.

        Args:
            model_name: Hugging Face model id. Defaults to settings.EMBEDDING_MODEL.
            device: Torch device such as "cpu" or "cuda". None lets
                sentence-transformers choose.
            batch_size: Texts encoded per forward pass. Defaults to
                settings.EMBED_BATCH_SIZE, which can be lowered on hosts where
                a large batch triggers paging.
            normalize: L2-normalise outputs so cosine similarity equals dot
                product, which is what the Qdrant collection is configured for.
        """
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device
        self.batch_size = batch_size if batch_size is not None else settings.EMBED_BATCH_SIZE
        self.normalize = normalize
        self._expected_dim = settings.EMBEDDING_DIM

    @property
    def model(self) -> Any:
        """The underlying SentenceTransformer, loaded and cached on first access.

        Returns:
            A ready SentenceTransformer instance.
        """
        key = (self.model_name, self.device)
        if key not in self._model_cache:
            # Imported here rather than at module scope so that importing this
            # module stays cheap and torch is only paid for when encoding.
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s (first use may download ~2GB)", self.model_name)
            self._model_cache[key] = SentenceTransformer(self.model_name, device=self.device)
        return self._model_cache[key]

    @property
    def dim(self) -> int:
        """Vector width of this model.

        Returns:
            The embedding dimension, from the loaded model when available and
            the configured value otherwise.
        """
        cached = self._model_cache.get((self.model_name, self.device))
        if cached is not None:
            return int(cached.get_sentence_embedding_dimension())
        return self._expected_dim

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into a matrix of dense vectors.

        Args:
            texts: Texts to encode. May be empty.

        Returns:
            Array of shape (len(texts), dim), dtype float32. An empty input
            returns shape (0, dim) so callers can index it unconditionally.
        """
        if not texts:
            return np.empty((0, self._expected_dim), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 128,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Encode a single query string.

        Args:
            text: The query.

        Returns:
            A 1-D array of shape (dim,), dtype float32.
        """
        return self.embed_batch([text])[0]

    @classmethod
    def clear_cache(cls) -> None:
        """Drop cached models so tests and long-running processes can free memory."""
        cls._model_cache.clear()
