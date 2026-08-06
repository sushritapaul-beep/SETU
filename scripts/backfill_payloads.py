"""Refresh Qdrant payloads from chunks.jsonl without re-embedding.

Payload fields change more often than vectors do — a new provenance field, a
corrected scheme name — and re-embedding a corpus to carry a text change costs
tens of minutes for no benefit. This updates payloads in place, matching points
by their deterministic chunk id.

Usage:
    python scripts/backfill_payloads.py
    python scripts/backfill_payloads.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from retrieval.vectorstore import QdrantStore  # noqa: E402

logger = logging.getLogger("setu.backfill")


def parse_args() -> argparse.Namespace:
    """Read command-line options.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Refresh Qdrant payloads from chunks.jsonl.")
    parser.add_argument("--chunks", type=Path, default=None, help="Path to chunks.jsonl.")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing.")
    parser.add_argument("--batch-size", type=int, default=64, help="Points updated per request.")
    return parser.parse_args()


def main() -> int:
    """Update payloads for every chunk present in the index.

    Returns:
        Process exit code: 0 on success, 1 if the corpus or collection is missing.
    """
    args = parse_args()
    logging.basicConfig(level="INFO", format="%(levelname)-7s | %(name)s | %(message)s")

    settings = get_settings()
    chunks_path = args.chunks or (settings.processed_data_dir / "chunks.jsonl")
    if not chunks_path.exists():
        logger.error("No chunks at %s", chunks_path)
        return 1

    rows = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info("Loaded %d chunks from %s", len(rows), chunks_path)

    store = QdrantStore()
    if not store.client.collection_exists(store.collection):
        logger.error("Collection %s does not exist; run scripts/ingest.py first", store.collection)
        return 1

    indexed = store.client.count(store.collection, exact=True).count
    logger.info("Collection %s holds %d points", store.collection, indexed)

    if args.dry_run:
        sample = rows[0]["metadata"]
        print(f"\nWould update {len(rows)} payloads. Fields per point:")
        for key in sorted(sample):
            print(f"    {key}")
        return 0

    updated = 0
    for start in range(0, len(rows), args.batch_size):
        for row in rows[start : start + args.batch_size]:
            payload = {
                "text": row["text"],
                "chunk_id": row["id"],
                "parent_id": row["parent_id"],
                **row["metadata"],
            }
            # Matching on the deterministic point id means this is idempotent:
            # re-running it changes nothing.
            store.client.set_payload(
                collection_name=store.collection,
                payload=payload,
                points=[QdrantStore._to_point_id(row["id"])],
                wait=False,
            )
            updated += 1
        logger.info("Updated %d/%d", updated, len(rows))

    logger.info("Backfilled %d payloads without re-embedding", updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
