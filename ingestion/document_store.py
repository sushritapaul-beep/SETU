"""Durable store of every extracted document, so the corpus is the union of all runs.

The crawler only returns documents that are *new in this run* — unchanged pages
come back as 304s or content-hash duplicates and yield nothing. Writing chunks
from a single run's output therefore shrinks the corpus on every incremental
crawl. Persisting each extracted document here, and rebuilding chunks from the
whole store, makes repeated runs additive.

Versioning: one current file per URL. When a URL's content changes, the old
version moves to history/ instead of being overwritten, so superseded official
text (an amended guideline, a revised circular) remains auditable.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from ingestion.crawl_state import content_hash, normalise_url
from ingestion.fetchers import ExtractedDocument

logger = logging.getLogger(__name__)


class DocumentStore:
    """Current-plus-history JSON store of extracted documents, keyed by URL."""

    def __init__(self, root: Path) -> None:
        """Open (creating if needed) the store.

        Args:
            root: Directory holding current/ and history/.
        """
        self.current = root / "current"
        self.history = root / "history"
        self.current.mkdir(parents=True, exist_ok=True)
        self.history.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(url: str) -> str:
        """Filesystem-safe, collision-resistant name for a URL."""
        return hashlib.sha256(normalise_url(url).encode("utf-8")).hexdigest()[:32]

    def put(self, document: ExtractedDocument) -> str:
        """Store a document, archiving the previous version if its text changed.

        Args:
            document: The extracted document.

        Returns:
            "new", "updated" or "unchanged".
        """
        path = self.current / f"{self._key(document.url)}.json"
        new_hash = content_hash(document.text)
        status = "new"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            old_hash = content_hash(old.get("text", ""))
            if old_hash == new_hash:
                return "unchanged"
            archived = self.history / f"{path.stem}-{old_hash[:16]}.json"
            archived.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
            status = "updated"
        path.write_text(json.dumps(asdict(document), ensure_ascii=False), encoding="utf-8")
        return status

    def has(self, url: str) -> bool:
        """Whether a current version of this URL is stored."""
        return (self.current / f"{self._key(url)}.json").exists()

    def all(self) -> list[ExtractedDocument]:
        """Load every current document, in a stable order.

        Returns:
            Current documents sorted by URL.
        """
        return sorted(self.iter(), key=lambda d: d.url)

    def iter(self) -> Iterator[ExtractedDocument]:
        """Yield current documents one at a time, in a stable (key) order.

        For whole-corpus passes: only one document is in memory at a time,
        where all() holds the entire corpus at once.
        """
        for path in sorted(self.current.glob("*.json")):
            yield ExtractedDocument(**json.loads(path.read_text(encoding="utf-8")))

    def history_count(self) -> int:
        """Number of superseded versions kept."""
        return sum(1 for _ in self.history.glob("*.json"))
