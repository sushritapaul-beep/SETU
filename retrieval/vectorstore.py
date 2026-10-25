"""Qdrant collection management and hybrid (dense + BM25 + RRF) search.

Fusion happens inside Qdrant rather than in Python: the server runs both the
dense and sparse branches and merges them with Reciprocal Rank Fusion in one
round trip, so there is no client-side score normalisation to get wrong.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from config.settings import get_settings
from ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Named vectors, so one collection holds both representations per point.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

# Qdrant computes the sparse vector itself from this FastEmbed model; combined
# with Modifier.IDF on the collection, that is Qdrant-native BM25 scoring.
BM25_MODEL = "Qdrant/bm25"

# A search hit: {id, score, text, chunk_id, parent_id, source_url, scheme_name,
# chunk_type}. Kept as a plain dict so it survives JSON serialisation to the API.
Hit = dict[str, Any]


class QdrantStore:
    """Create, populate and query the SETU scheme collection."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection: str | None = None,
        dim: int | None = None,
        embedder: Any | None = None,
        timeout: int | None = None,
    ) -> None:
        """Connect to Qdrant.

        Args:
            client: An existing client. Pass QdrantClient(":memory:") in tests.
                When None, one is built from settings.
            collection: Collection name. Defaults to settings.QDRANT_COLLECTION.
            dim: Dense vector width. Defaults to settings.EMBEDDING_DIM.
            embedder: Optional JinaEmbedder used by hybrid_search when the
                caller does not supply a precomputed query vector.
            timeout: Request timeout in seconds for a client built here. None
                keeps the library default (5s), so live queries fail fast;
                bulk indexing raises it, because a ~1 MB upsert of 64 dense
                vectors can take longer than 5s to upload.
        """
        settings = get_settings()
        self.collection = collection or settings.QDRANT_COLLECTION
        self.dim = dim or settings.EMBEDDING_DIM
        self.embedder = embedder

        if client is not None:
            self.client = client
        else:
            api_key = settings.QDRANT_API_KEY
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                # QDRANT_API_KEY is optional for a local/dev Qdrant instance
                # run without auth; send no key at all rather than an empty
                # string, which the server would reject.
                api_key=api_key.get_secret_value() if api_key else None,
                timeout=timeout,
            )

    # ---------------------------------------------------------------- setup --

    def ensure_collection(self, name: str | None = None, dim: int = 1024) -> None:
        """Create the collection and payload indexes if they do not already exist.

        Idempotent, so ingestion can call it on every run.

        Args:
            name: Collection to create. Defaults to this store's collection.
            dim: Dense vector width.
        """
        collection = name or self.collection

        if self.client.collection_exists(collection):
            logger.info("Collection %s already exists", collection)
        else:
            self.client.create_collection(
                collection_name=collection,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=dim,
                        # Cosine pairs with the embedder's normalised output.
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    # IDF is what turns raw term frequencies into BM25 scoring;
                    # without it sparse search degenerates to term counting.
                    SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            logger.info("Created collection %s (dim=%d)", collection, dim)

        # Indexed so child-only filtering stays fast as the corpus grows.
        for field in ("chunk_type", "source_url", "level", "states", "ministry", "categories"):
            try:
                self.client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:  # noqa: BLE001 - index already present is fine
                logger.debug("Payload index on %s not created: %s", field, exc)

    # ------------------------------------------------------------- indexing --

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        batch_size: int = 64,
    ) -> int:
        """Write chunks and their vectors into the collection.

        Args:
            chunks: Chunks to index.
            embeddings: Dense vectors, row-aligned with chunks.
            batch_size: Points sent per request.

        Returns:
            Number of points upserted.

        Raises:
            ValueError: If the number of chunks and embedding rows disagree.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        if not chunks:
            return 0

        points = [
            models.PointStruct(
                id=self._to_point_id(chunk.id),
                vector={
                    DENSE_VECTOR: embeddings[i].tolist(),
                    # Qdrant runs BM25 inference server-side on this text.
                    SPARSE_VECTOR: models.Document(text=chunk.text, model=BM25_MODEL),
                },
                payload={
                    "text": chunk.text,
                    # Keep the original hex id: point ids must be UUID/int, so
                    # the chunk's own id would otherwise be lost.
                    "chunk_id": chunk.id,
                    "parent_id": chunk.parent_id,
                    # Built from the metadata dataclass rather than a hand-listed
                    # set of keys. An explicit list silently drops every field
                    # added later — which is exactly how provenance was lost
                    # here after it was added to ChunkMetadata.
                    **asdict(chunk.metadata),
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + batch_size],
                wait=True,
            )

        logger.info("Upserted %d points into %s", len(points), self.collection)
        return len(points)

    def upsert_sparse_only(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """Write chunks with a BM25 vector but no dense embedding.

        For parent chunks: retrieval only ever dense-searches children (the
        graph filters to chunk_type == "child"), and a parent is fetched by id.
        Embedding parents would roughly double the embedding bill for vectors
        nothing queries.

        Args:
            chunks: Chunks to index.
            batch_size: Points sent per request.

        Returns:
            Number of points upserted.
        """
        # Points are built one batch at a time: every parent in a full corpus
        # is tens of thousands of PointStructs if materialised up front.
        for start in range(0, len(chunks), batch_size):
            points = [
                models.PointStruct(
                    id=self._to_point_id(chunk.id),
                    vector={SPARSE_VECTOR: models.Document(text=chunk.text, model=BM25_MODEL)},
                    payload={"text": chunk.text, "chunk_id": chunk.id,
                             "parent_id": chunk.parent_id, **asdict(chunk.metadata)},
                )
                for chunk in chunks[start : start + batch_size]
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(chunks)

    def existing_chunk_ids(self, chunk_ids: list[str], batch_size: int = 256) -> set[str]:
        """Which of these chunk ids already have a point in the collection.

        Chunk ids hash the chunk's own text, so an existing id means identical
        content is already embedded — re-embedding it would cost tokens and
        change nothing.

        Args:
            chunk_ids: Hex chunk ids.
            batch_size: Ids looked up per request.

        Returns:
            The subset already indexed.
        """
        found: set[str] = set()
        for start in range(0, len(chunk_ids), batch_size):
            records = self.client.retrieve(
                collection_name=self.collection,
                ids=[self._to_point_id(c) for c in chunk_ids[start : start + batch_size]],
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            found.update(r.payload["chunk_id"] for r in records if r.payload)
        return found

    def delete_points_not_in(self, keep_chunk_ids: set[str]) -> int:
        """Remove points whose chunk is no longer in the corpus.

        When a source document changes, its chunks get new ids; without this
        the superseded text stays searchable next to the current text. The
        superseded document itself is preserved in the document store's
        history, so this removes it from retrieval, not from the record.

        Args:
            keep_chunk_ids: Every chunk id in the current corpus.

        Returns:
            Number of points deleted.
        """
        stale: list[Any] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection, limit=1000, offset=offset,
                with_payload=["chunk_id"], with_vectors=False,
            )
            stale += [p.id for p in points
                      if (p.payload or {}).get("chunk_id") not in keep_chunk_ids]
            if offset is None:
                break
        for start in range(0, len(stale), 500):
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.PointIdsList(points=stale[start : start + 500]),
                wait=True,
            )
        if stale:
            logger.info("Deleted %d stale points from %s", len(stale), self.collection)
        return len(stale)

    # -------------------------------------------------------------- queries --

    def dense_search(
        self,
        query_vec: np.ndarray | list[float],
        k: int = 20,
        filters: models.Filter | None = None,
    ) -> list[Hit]:
        """Semantic nearest-neighbour search over dense vectors.

        Args:
            query_vec: Query embedding.
            k: Maximum hits to return.
            filters: Optional Qdrant payload filter.

        Returns:
            Hits ordered by descending cosine similarity.
        """
        vector = query_vec.tolist() if isinstance(query_vec, np.ndarray) else list(query_vec)
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=DENSE_VECTOR,
            limit=k,
            query_filter=filters,
            with_payload=True,
        )
        return [self._to_hit(p) for p in response.points]

    def sparse_search(
        self,
        query_text: str,
        k: int = 20,
        filters: models.Filter | None = None,
    ) -> list[Hit]:
        """Keyword search using Qdrant's native BM25 sparse vectors.

        Complements dense search on exact tokens — scheme names, acronyms and
        figures like "Udyam" or "85%" that embeddings tend to blur.

        Args:
            query_text: Raw query string.
            k: Maximum hits to return.
            filters: Optional Qdrant payload filter.

        Returns:
            Hits ordered by descending BM25 score.
        """
        response = self.client.query_points(
            collection_name=self.collection,
            query=models.Document(text=query_text, model=BM25_MODEL),
            using=SPARSE_VECTOR,
            limit=k,
            query_filter=filters,
            with_payload=True,
        )
        return [self._to_hit(p) for p in response.points]

    def hybrid_search(
        self,
        query_text: str,
        k_dense: int = 20,
        k_sparse: int = 20,
        top_k: int = 20,
        query_vec: np.ndarray | list[float] | None = None,
        filters: models.Filter | None = None,
    ) -> list[Hit]:
        """Run dense and BM25 branches and fuse them with Qdrant-native RRF.

        RRF merges on rank rather than score, so the two branches need no
        comparable scale — the usual failure mode of hand-rolled hybrid search.

        Args:
            query_text: Raw query, used for the BM25 branch.
            k_dense: Candidates from the dense branch.
            k_sparse: Candidates from the sparse branch.
            top_k: Hits returned after fusion.
            query_vec: Precomputed query embedding. Computed via the injected
                embedder when omitted.
            filters: Optional Qdrant payload filter applied to both branches.

        Returns:
            Fused hits ordered by descending RRF score.

        Raises:
            ValueError: If no query vector is given and no embedder is available.
        """
        if query_vec is None:
            if self.embedder is None:
                raise ValueError(
                    "hybrid_search needs either query_vec or an embedder passed to QdrantStore()"
                )
            query_vec = self.embedder.embed_query(query_text)
        vector = query_vec.tolist() if isinstance(query_vec, np.ndarray) else list(query_vec)

        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=vector, using=DENSE_VECTOR, limit=k_dense, filter=filters
                ),
                models.Prefetch(
                    query=models.Document(text=query_text, model=BM25_MODEL),
                    using=SPARSE_VECTOR,
                    limit=k_sparse,
                    filter=filters,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [self._to_hit(p) for p in response.points]

    def fetch_parents(self, parent_ids: list[str]) -> dict[str, str]:
        """Look up parent chunk texts by their original chunk ids.

        This is the expansion half of the parent-document pattern: search
        matches small children, then the larger parent is what the LLM reads.

        Args:
            parent_ids: Original hex chunk ids of parents.

        Returns:
            Mapping of chunk id to parent text, omitting any not found.
        """
        if not parent_ids:
            return {}
        records = self.client.retrieve(
            collection_name=self.collection,
            ids=[self._to_point_id(pid) for pid in parent_ids],
            with_payload=True,
        )
        return {r.payload["chunk_id"]: r.payload["text"] for r in records if r.payload}

    # -------------------------------------------------------------- helpers --

    @staticmethod
    def child_only_filter() -> models.Filter:
        """Filter restricting search to child chunks.

        Returns:
            A Qdrant filter matching chunk_type == "child".
        """
        return models.Filter(
            must=[models.FieldCondition(key="chunk_type", match=models.MatchValue(value="child"))]
        )

    @staticmethod
    def _to_point_id(chunk_id: str) -> str:
        """Convert a 32-char hex chunk id into a Qdrant-acceptable point id.

        Qdrant only accepts unsigned integers or UUIDs as point ids, and a
        32-character hex digest is exactly a UUID's worth of bits.

        Args:
            chunk_id: 32-character hex string.

        Returns:
            The canonical UUID string form.
        """
        return str(uuid.UUID(hex=chunk_id))

    @staticmethod
    def _to_hit(point: Any) -> Hit:
        """Flatten a Qdrant ScoredPoint into a plain hit dict.

        Args:
            point: A Qdrant scored point.

        Returns:
            A hit dict with score and payload fields promoted to the top level.
        """
        payload = point.payload or {}
        return {
            "id": str(point.id),
            "score": float(point.score) if point.score is not None else 0.0,
            "text": payload.get("text", ""),
            "chunk_id": payload.get("chunk_id"),
            "parent_id": payload.get("parent_id"),
            "source_url": payload.get("source_url"),
            "scheme_name": payload.get("scheme_name"),
            "chunk_type": payload.get("chunk_type"),
            # Provenance, so a citation can state when the source was retrieved
            # rather than implying the information is current as of now.
            "doc_format": payload.get("doc_format"),
            "retrieved_at": payload.get("retrieved_at"),
            "published_at": payload.get("published_at"),
            "modified_at": payload.get("modified_at"),
            "source_title": payload.get("source_title"),
            "doc_hash": payload.get("doc_hash"),
            "ministry": payload.get("ministry"),
            "level": payload.get("level"),
            "states": payload.get("states"),
            "categories": payload.get("categories"),
            "discovery_method": payload.get("discovery_method"),
        }
