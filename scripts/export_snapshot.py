"""Export a self-contained retrieval snapshot for Hugging Face Spaces.

A Space has no Qdrant server, so the index has to travel with the code. This
writes the chunks and their precomputed BGE-M3 vectors to data/snapshot/, which
`ui/app.py` loads into an in-memory Qdrant at startup.

Vectors are precomputed rather than embedded on boot because re-embedding the
corpus on every Space restart would add minutes to cold start for no benefit.

Usage:
    python scripts/export_snapshot.py
    python scripts/export_snapshot.py --out data/snapshot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from ingestion.chunker import Chunk, ChunkMetadata  # noqa: E402

logger = logging.getLogger("setu.snapshot")

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "snapshot"


def parse_args() -> argparse.Namespace:
    """Read command-line options.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Export a retrieval snapshot for Spaces.")
    parser.add_argument("--chunks", type=Path, default=None, help="Path to chunks.jsonl.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory.")
    parser.add_argument(
        "--children-only",
        action="store_true",
        default=True,
        help="Export only child chunks; parents are context-only and are not searched.",
    )
    return parser.parse_args()


def load_chunks(path: Path) -> list[Chunk]:
    """Read chunks from JSONL.

    Args:
        path: Path to chunks.jsonl.

    Returns:
        The deserialised chunks.

    Raises:
        FileNotFoundError: If the file is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"No chunks at {path}. Run: python scripts/ingest.py")
    chunks: list[Chunk] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        chunks.append(
            Chunk(
                id=record["id"],
                parent_id=record["parent_id"],
                text=record["text"],
                metadata=ChunkMetadata(**record["metadata"]),
            )
        )
    return chunks


def main() -> int:
    """Embed the corpus and write the snapshot.

    Returns:
        Process exit code: 0 on success, 1 if there was nothing to export.
    """
    args = parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)-7s | %(name)s | %(message)s")

    settings = get_settings()
    chunks_path = args.chunks or (settings.processed_data_dir / "chunks.jsonl")
    chunks = load_chunks(chunks_path)
    if args.children_only:
        chunks = [c for c in chunks if c.metadata.chunk_type == "child"]
    if not chunks:
        logger.error("No chunks to export")
        return 1

    # Imported here so --help does not pay for torch.
    from retrieval.embeddings import BGEM3Embedder

    embedder = BGEM3Embedder()
    logger.info("Embedding %d chunks with %s", len(chunks), embedder.model_name)
    vectors = embedder.embed_batch([c.text for c in chunks])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks) + "\n",
        encoding="utf-8",
    )
    np.save(args.out / "vectors.npy", vectors)
    (args.out / "meta.json").write_text(
        json.dumps(
            {
                "created": datetime.now(UTC).isoformat(),
                "embedding_model": embedder.model_name,
                "dim": int(vectors.shape[1]),
                "count": len(chunks),
                "schemes": sorted({c.metadata.scheme_name for c in chunks}),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    size_mb = sum(f.stat().st_size for f in args.out.glob("*")) / 1024**2
    logger.info("Wrote %d chunks + vectors %s to %s (%.1f MB)",
                len(chunks), tuple(vectors.shape), args.out, size_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
