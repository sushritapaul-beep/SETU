"""Durable crawl state: the frontier, what has been fetched, and what failed.

Backed by SQLite rather than memory so a crawl over thousands of government
pages survives a crash, a rate-limit ban or a laptop closing. Re-running the
crawler resumes from the frontier instead of starting over.

Two kinds of deduplication happen here:

* **URL identity** — a normalised URL is visited at most once.
* **Content identity** — a SHA-256 of the extracted text. The same scheme
  published on both myScheme and the ministry site produces one document, not
  two, which matters because duplicated chunks distort retrieval scores.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urlparse, urlunparse

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS urls (
    url            TEXT PRIMARY KEY,
    source_key     TEXT,
    depth          INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'pending',
    http_status    INTEGER,
    content_type   TEXT,
    content_hash   TEXT,
    etag           TEXT,
    last_modified  TEXT,
    title          TEXT,
    discovered_at  TEXT NOT NULL,
    fetched_at     TEXT,
    attempts       INTEGER NOT NULL DEFAULT 0,
    error          TEXT
);
CREATE INDEX IF NOT EXISTS idx_urls_status ON urls(status);
CREATE INDEX IF NOT EXISTS idx_urls_hash   ON urls(content_hash);

-- Content hashes seen anywhere, so cross-source duplicates collapse.
CREATE TABLE IF NOT EXISTS content (
    content_hash TEXT PRIMARY KEY,
    first_url    TEXT NOT NULL,
    bytes        INTEGER,
    seen_at      TEXT NOT NULL
);
"""

# Terminal states: never retried.
STATUS_DONE = "done"
STATUS_DUPLICATE = "duplicate"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
# Retryable.
STATUS_PENDING = "pending"

MAX_ATTEMPTS = 3


def normalise_url(url: str) -> str:
    """Canonicalise a URL so trivially different strings dedupe to one entry.

    Strips fragments, lowercases the host, drops a default port and removes a
    trailing slash on non-root paths. Query strings are preserved because
    government portals routinely encode the scheme id there.

    Args:
        url: Absolute URL.

    Returns:
        The canonical form.
    """
    url, _ = urldefrag(url.strip())
    parts = urlparse(url)
    host = (parts.hostname or "").lower()
    if parts.port and not ((parts.scheme == "http" and parts.port == 80) or
                           (parts.scheme == "https" and parts.port == 443)):
        host = f"{host}:{parts.port}"
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse((parts.scheme.lower(), host, path, "", parts.query, ""))


def content_hash(text: str) -> str:
    """Stable hash of extracted text, used for cross-source dedupe.

    Args:
        text: Extracted document text.

    Returns:
        Hex SHA-256 of the whitespace-normalised text, so cosmetic reformatting
        between two publications of the same document does not defeat dedupe.
    """
    normalised = " ".join(text.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CrawlTask:
    """One URL waiting to be fetched."""

    url: str
    depth: int
    source_key: str | None
    etag: str | None = None
    last_modified: str | None = None


class CrawlState:
    """SQLite-backed frontier and history for a crawl run."""

    def __init__(self, db_path: Path) -> None:
        """Open (creating if needed) the crawl database.

        Args:
            db_path: Where to store crawl_state.db.
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL keeps reads working while a write is in flight, which matters once
        # fetching runs concurrently.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> CrawlState:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------- frontier --

    def add_url(self, url: str, depth: int, source_key: str | None) -> bool:
        """Add a URL to the frontier if it has never been seen.

        Args:
            url: Absolute URL.
            depth: Link distance from a seed.
            source_key: Owning source, when known.

        Returns:
            True if newly added, False if already known.
        """
        canonical = normalise_url(url)
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT 1 FROM urls WHERE url = ?", (canonical,))
            if cur.fetchone():
                return False
            cur.execute(
                "INSERT INTO urls (url, source_key, depth, status, discovered_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (canonical, source_key, depth, STATUS_PENDING, datetime.now(UTC).isoformat()),
            )
        self._conn.commit()
        return True

    def next_batch(self, limit: int) -> list[CrawlTask]:
        """Claim the shallowest pending URLs.

        Ordering by depth makes the crawl breadth-first, so a depth limit yields
        broad coverage rather than one deep rabbit hole.

        Args:
            limit: Maximum tasks to return.

        Returns:
            Tasks to fetch; empty when the frontier is exhausted.
        """
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "SELECT url, depth, source_key, etag, last_modified FROM urls "
                "WHERE status = ? AND attempts < ? ORDER BY depth, discovered_at LIMIT ?",
                (STATUS_PENDING, MAX_ATTEMPTS, limit),
            )
            return [
                CrawlTask(r["url"], r["depth"], r["source_key"], r["etag"], r["last_modified"])
                for r in cur.fetchall()
            ]

    def done_urls(self) -> list[str]:
        """URLs recorded as successfully fetched with unique content."""
        with closing(self._conn.cursor()) as cur:
            # A "duplicate" of its own earlier fetch is still that URL's unique
            # content (older states recorded refetches that way), so it counts.
            cur.execute(
                "SELECT u.url FROM urls u LEFT JOIN content c ON c.content_hash = u.content_hash "
                "WHERE u.status = ? OR (u.status = ? AND c.first_url = u.url)",
                (STATUS_DONE, STATUS_DUPLICATE),
            )
            return [r["url"] for r in cur.fetchall()]

    def requeue(self, urls: list[str]) -> int:
        """Return URLs to the frontier so the next crawl fetches them again.

        Used when the frontier says "done" but the fetched text was never
        persisted (a crash between fetch and store); without this the crawler
        would treat those documents as captured forever.

        Args:
            urls: URLs to reset.

        Returns:
            Number of URLs requeued.
        """
        with closing(self._conn.cursor()) as cur:
            cur.executemany(
                "UPDATE urls SET status = ?, attempts = 0, error = NULL, etag = NULL, "
                "last_modified = NULL WHERE url = ?",
                [(STATUS_PENDING, normalise_url(u)) for u in urls],
            )
            changed = cur.rowcount
        self._conn.commit()
        return changed

    # -------------------------------------------------------------- outcomes --

    def mark_done(
        self,
        url: str,
        *,
        http_status: int,
        content_type: str,
        text_hash: str,
        title: str,
        etag: str | None,
        last_modified: str | None,
        size: int,
    ) -> bool:
        """Record a successful fetch, collapsing cross-source duplicates.

        Args:
            url: The fetched URL.
            http_status: Response status code.
            content_type: Response content type.
            text_hash: Hash of the extracted text.
            title: Document title.
            etag: Response ETag, for conditional refetch.
            last_modified: Response Last-Modified.
            size: Extracted text length in characters.

        Returns:
            True if this is new content, False if the same text was already
            captured from another URL (recorded as a duplicate).
        """
        canonical = normalise_url(url)
        now = datetime.now(UTC).isoformat()
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT first_url FROM content WHERE content_hash = ?", (text_hash,))
            existing = cur.fetchone()
            # Refetching a URL returns its own text; only text first captured
            # from a *different* URL is a duplicate.
            is_new = existing is None or existing["first_url"] == canonical

            if existing is None:
                cur.execute(
                    "INSERT INTO content (content_hash, first_url, bytes, seen_at) "
                    "VALUES (?, ?, ?, ?)",
                    (text_hash, canonical, size, now),
                )
            else:
                logger.debug("Duplicate content: %s matches %s", canonical, existing["first_url"])

            cur.execute(
                "UPDATE urls SET status = ?, http_status = ?, content_type = ?, "
                "content_hash = ?, title = ?, etag = ?, last_modified = ?, fetched_at = ?, "
                "attempts = attempts + 1, error = NULL WHERE url = ?",
                (
                    STATUS_DONE if is_new else STATUS_DUPLICATE,
                    http_status, content_type, text_hash, title, etag, last_modified,
                    now, canonical,
                ),
            )
        self._conn.commit()
        return is_new

    def mark_failed(self, url: str, error: str, *, permanent: bool = False) -> None:
        """Record a failed fetch, leaving it retryable unless permanent.

        Args:
            url: The URL that failed.
            error: Human-readable reason.
            permanent: True for 404/403 and similar, which must not be retried.
        """
        canonical = normalise_url(url)
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "UPDATE urls SET attempts = attempts + 1, error = ?, "
                "status = CASE WHEN ? OR attempts + 1 >= ? THEN ? ELSE ? END WHERE url = ?",
                (error[:500], permanent, MAX_ATTEMPTS, STATUS_FAILED, STATUS_PENDING, canonical),
            )
        self._conn.commit()

    def mark_skipped(self, url: str, reason: str) -> None:
        """Record a URL deliberately not fetched.

        Args:
            url: The URL.
            reason: Why it was skipped (robots, format, off-domain).
        """
        canonical = normalise_url(url)
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "UPDATE urls SET status = ?, error = ? WHERE url = ?",
                (STATUS_SKIPPED, reason[:500], canonical),
            )
        self._conn.commit()

    # ----------------------------------------------------------------- stats --

    def stats(self) -> dict[str, Any]:
        """Summarise crawl progress.

        Returns:
            Counts per status plus totals for unique content and errors.
        """
        with closing(self._conn.cursor()) as cur:
            cur.execute("SELECT status, COUNT(*) AS n FROM urls GROUP BY status")
            by_status = {r["status"]: r["n"] for r in cur.fetchall()}
            cur.execute("SELECT COUNT(*) AS n FROM content")
            unique_docs = cur.fetchone()["n"]
        return {
            "by_status": by_status,
            "total_urls": sum(by_status.values()),
            "unique_documents": unique_docs,
        }

    def failures(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent permanent failures for triage.

        Args:
            limit: Maximum rows.

        Returns:
            URL and error for each failure.
        """
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                "SELECT url, error, attempts FROM urls WHERE status = ? "
                "ORDER BY fetched_at DESC LIMIT ?",
                (STATUS_FAILED, limit),
            )
            return [dict(r) for r in cur.fetchall()]
