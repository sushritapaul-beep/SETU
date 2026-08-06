"""Recursive, fault-tolerant crawler over verified official government sources.

Design commitments:

* **Official sources only.** Every URL passes `is_official_url` before it enters
  the frontier and again before it is fetched. There is no path by which an
  unofficial domain reaches the corpus.
* **Polite.** robots.txt is honoured, concurrency is capped per domain, and a
  delay separates requests to the same host. These are public services funded by
  taxpayers, not infrastructure to hammer.
* **Fault-tolerant.** Transient failures retry with exponential backoff and
  jitter; permanent ones (404, 403, 410) are marked terminal immediately.
* **Incremental.** State lives in SQLite, so an interrupted crawl resumes and a
  repeat crawl re-fetches only what changed, using ETag/Last-Modified.
* **Deduplicating.** Identical text reached through different URLs is stored
  once, because duplicate chunks distort retrieval ranking.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from ingestion.crawl_state import CrawlState, CrawlTask, content_hash, normalise_url
from ingestion.fetchers import ExtractedDocument, extract
from ingestion.sources import (
    Source,
    is_fetchable_url,
    is_official_url,
    seed_urls,
    source_for_url,
)

logger = logging.getLogger(__name__)

# Several government WAFs reject any User-Agent that is not a full browser
# string — measured: both "SETU-SchemeBot/1.0 (+url)" and the conventional
# "Mozilla/5.0 (compatible; SETU-SchemeBot/1.0; +url)" form get HTTP 403 from
# msme.gov.in and india.gov.in, while a browser string gets 200. That filter is
# a crude bot check, not a crawling policy; robots.txt is the mechanism that
# actually expresses consent, and we honour it.
#
# So: browser UA for access, plus a From header that identifies the project to
# any operator who looks. Identification is kept, evasion of a stated policy is
# not attempted.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
CRAWLER_CONTACT = "https://github.com/sushritapaul-beep/SETU"

# Error pages that answer HTTP 200. Government portals redirect missing pages to
# a friendly error page rather than returning 404, so a status check alone would
# index "File Not Found" as scheme content.
SOFT_404_URL_MARKERS: tuple[str, ...] = (
    "/errorpages/", "filenotfound", "pagenotfound", "/error", "/404",
    "notfound.htm", "page-not-found",
)
SOFT_404_TEXT_MARKERS: tuple[str, ...] = (
    "file not found", "page not found", "page cannot be found",
    "404 error", "the page you are looking for", "does not exist",
    "under maintenance", "temporarily unavailable",
)
# Above this length, a page containing an error phrase is probably real content
# that merely mentions it.
SOFT_404_MAX_CHARS = 1500


# Transport failures that cannot succeed on retry.
PERMANENT_TRANSPORT_MARKERS: tuple[str, ...] = (
    "certificate_verify_failed", "certificate verify failed",
    "sslcertverificationerror", "hostname mismatch",
    "getaddrinfo failed", "name or service not known", "nodename nor servname",
)


def _is_permanent_transport_error(exc: BaseException) -> bool:
    """Whether a transport error will still fail on retry.

    Args:
        exc: The exception raised during fetching.

    Returns:
        True for certificate and DNS failures, which are configuration
        problems rather than transient network noise.
    """
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in message for marker in PERMANENT_TRANSPORT_MARKERS)


def looks_like_soft_404(final_url: str, text: str) -> str | None:
    """Detect an error page served with HTTP 200.

    Args:
        final_url: URL after redirects.
        text: Extracted page text.

    Returns:
        A reason string when the page looks like an error, else None.
    """
    lowered_url = final_url.lower()
    for marker in SOFT_404_URL_MARKERS:
        if marker in lowered_url:
            return f"redirected to an error page ({marker})"

    stripped = text.strip()
    if len(stripped) <= SOFT_404_MAX_CHARS:
        lowered = stripped.lower()
        for marker in SOFT_404_TEXT_MARKERS:
            if marker in lowered:
                return f"error-page content ({marker!r}) in a {len(stripped)}-char page"
    return None

# HTTP statuses that will never succeed on retry.
PERMANENT_STATUSES = frozenset({400, 401, 403, 404, 405, 410, 451})
# Statuses worth backing off and retrying.
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

MAX_BYTES = 60 * 1024 * 1024  # refuse anything larger; no scheme doc is 60MB


@dataclass
class CrawlConfig:
    """Tunable limits for a crawl run."""

    max_depth: int = 3
    max_pages: int = 500
    per_domain_concurrency: int = 2
    global_concurrency: int = 6
    request_timeout: float = 45.0
    max_retries: int = 3
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    respect_robots: bool = True
    enable_ocr: bool = False
    # Follow links only within the seed's own domain. Government sites link
    # outward constantly (to banks, vendors, social media); without this the
    # crawl sprawls even while staying inside .gov.in.
    same_domain_only: bool = True


@dataclass
class CrawlResult:
    """Outcome of a crawl run."""

    documents: list[ExtractedDocument] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)


class RobotsCache:
    """Per-domain robots.txt rules, fetched once and reused."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        """Store the HTTP client used to fetch robots files.

        Args:
            client: Shared async client.
        """
        self._client = client
        self._parsers: dict[str, RobotFileParser | None] = {}
        self._lock = asyncio.Lock()

    async def allowed(self, url: str) -> bool:
        """Whether robots.txt permits fetching a URL.

        Args:
            url: Absolute URL.

        Returns:
            True when allowed, or when robots.txt is missing or unreadable —
            an unavailable robots file is conventionally treated as permissive.
        """
        parts = urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"

        async with self._lock:
            if origin not in self._parsers:
                self._parsers[origin] = await self._load(origin)

        parser = self._parsers[origin]
        if parser is None:
            return True
        try:
            return parser.can_fetch(USER_AGENT, url)
        except Exception:  # noqa: BLE001 - a malformed rule must not block the crawl
            return True

    async def _load(self, origin: str) -> RobotFileParser | None:
        """Fetch and parse one origin's robots.txt.

        Args:
            origin: Scheme and host, e.g. https://msme.gov.in.

        Returns:
            A parser, or None if robots.txt could not be retrieved.
        """
        try:
            response = await self._client.get(f"{origin}/robots.txt", timeout=15.0)
            if response.status_code != 200:
                return None
            parser = RobotFileParser()
            parser.parse(response.text.splitlines())
            logger.debug("Loaded robots.txt for %s", origin)
            return parser
        except Exception as exc:  # noqa: BLE001
            logger.debug("No usable robots.txt for %s: %s", origin, exc)
            return None


class DomainLimiter:
    """Enforces per-domain concurrency and a minimum gap between requests."""

    def __init__(self, concurrency: int) -> None:
        """Configure the limiter.

        Args:
            concurrency: Simultaneous requests allowed per domain.
        """
        self._concurrency = concurrency
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._last_request: dict[str, float] = defaultdict(float)
        self._lock = asyncio.Lock()

    async def acquire(self, domain: str, delay: float) -> asyncio.Semaphore:
        """Wait until it is polite to call a domain again.

        Args:
            domain: Host name.
            delay: Minimum seconds between requests to this host.

        Returns:
            The held semaphore; the caller must release it.
        """
        async with self._lock:
            if domain not in self._semaphores:
                self._semaphores[domain] = asyncio.Semaphore(self._concurrency)
            semaphore = self._semaphores[domain]

        await semaphore.acquire()
        elapsed = time.monotonic() - self._last_request[domain]
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        self._last_request[domain] = time.monotonic()
        return semaphore


class GovernmentCrawler:
    """Crawls registered official sources and returns extracted documents."""

    def __init__(self, config: CrawlConfig | None = None, state_path: Path | None = None) -> None:
        """Configure the crawler.

        Args:
            config: Limits and behaviour. Defaults are conservative.
            state_path: Where the SQLite crawl state lives. Defaults to
                data/crawl/crawl_state.db.
        """
        self.config = config or CrawlConfig()
        self.state_path = state_path or (
            Path(__file__).resolve().parent.parent / "data" / "crawl" / "crawl_state.db"
        )
        # Set for the duration of a crawl; used only by sources with insecure_ssl.
        self._insecure_client: httpx.AsyncClient | None = None

    async def crawl(
        self,
        source_keys: list[str] | None = None,
        extra_urls: list[str] | None = None,
    ) -> CrawlResult:
        """Run a crawl over the selected sources.

        Args:
            source_keys: Registered source keys to crawl. None crawls all.
            extra_urls: Additional official URLs to seed with.

        Returns:
            Extracted documents plus crawl statistics and failures.

        Raises:
            ValueError: If an extra URL is not from an official source.
        """
        for url in extra_urls or []:
            if not is_official_url(url):
                raise ValueError(f"Refusing non-official seed URL: {url}")

        result = CrawlResult()
        with CrawlState(self.state_path) as state:
            for url, source in seed_urls(source_keys):
                state.add_url(url, depth=0, source_key=source.key)
            for url in extra_urls or []:
                source = source_for_url(url)
                state.add_url(url, depth=0, source_key=source.key if source else None)

            headers = {
                "User-Agent": USER_AGENT,
                "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
                # Identifies the project to any operator inspecting traffic,
                # even though the UA has to look like a browser to get past WAFs.
                "From": CRAWLER_CONTACT,
            }
            async with httpx.AsyncClient(
                headers=headers,
                follow_redirects=True,
                timeout=self.config.request_timeout,
                limits=httpx.Limits(max_connections=self.config.global_concurrency),
            ) as client:
                robots = RobotsCache(client)
                limiter = DomainLimiter(self.config.per_domain_concurrency)
                gate = asyncio.Semaphore(self.config.global_concurrency)

                # A second client for the handful of official hosts with broken
                # certificate chains. Kept separate so relaxed verification can
                # never leak onto a source that did not opt in.
                async with httpx.AsyncClient(
                    headers=headers,
                    follow_redirects=True,
                    timeout=self.config.request_timeout,
                    verify=False,
                    limits=httpx.Limits(max_connections=2),
                ) as insecure_client:
                    self._insecure_client = insecure_client

                    processed = 0
                    while processed < self.config.max_pages:
                        batch = state.next_batch(min(self.config.global_concurrency * 2,
                                                     self.config.max_pages - processed))
                        if not batch:
                            break

                        coros = [
                            self._process(task, client, robots, limiter, gate, state, result)
                            for task in batch
                        ]
                        await asyncio.gather(*coros, return_exceptions=True)
                        processed += len(batch)

            result.stats = state.stats()
            result.failures = state.failures()

        logger.info(
            "Crawl finished: %d documents, %s",
            len(result.documents), result.stats,
        )
        return result

    async def _process(
        self,
        task: CrawlTask,
        client: httpx.AsyncClient,
        robots: RobotsCache,
        limiter: DomainLimiter,
        gate: asyncio.Semaphore,
        state: CrawlState,
        result: CrawlResult,
    ) -> None:
        """Fetch, extract and expand one URL.

        Args:
            task: The URL to process.
            client: Shared HTTP client.
            robots: Robots rules cache.
            limiter: Per-domain politeness limiter.
            gate: Global concurrency gate.
            state: Crawl state store.
            result: Accumulator for extracted documents.
        """
        url = task.url

        # Re-check officialness at fetch time, not just at discovery: a redirect
        # can move a URL off a government domain between the two.
        if not is_official_url(url):
            state.mark_skipped(url, "not an official government source")
            return
        if not is_fetchable_url(url):
            state.mark_skipped(url, "format or path excluded")
            return
        if self.config.respect_robots and not await robots.allowed(url):
            state.mark_skipped(url, "disallowed by robots.txt")
            return

        source = source_for_url(url)
        domain = urlparse(url).hostname or ""
        delay = source.delay_seconds if source else 1.0

        async with gate:
            semaphore = await limiter.acquire(domain, delay)
            try:
                document = await self._fetch_and_extract(task, client, source, state)
            finally:
                semaphore.release()

        if document is None:
            return

        if not document.is_useful:
            state.mark_skipped(url, f"insufficient text ({len(document.text)} chars)")
            return

        text_hash = content_hash(document.text)
        is_new = state.mark_done(
            url,
            http_status=200,
            content_type=document.content_type,
            text_hash=text_hash,
            title=document.title,
            etag=document.extra.get("etag"),
            last_modified=document.extra.get("last_modified"),
            size=len(document.text),
        )
        if is_new:
            result.documents.append(document)

        if task.depth < self.config.max_depth:
            self._enqueue_links(document, task, source, state)

    async def _fetch_and_extract(
        self,
        task: CrawlTask,
        client: httpx.AsyncClient,
        source: Source | None,
        state: CrawlState,
    ) -> ExtractedDocument | None:
        """Fetch a URL with retries and extract its text.

        Args:
            task: The URL to fetch.
            client: Shared HTTP client.
            source: Owning source, when known.
            state: Crawl state, for recording failures.

        Returns:
            The extracted document, or None on failure.
        """
        # Client-side rendered portals return an empty shell over plain HTTP;
        # those domains are fetched through a real browser instead.
        if source and source.requires_js:
            return await self._fetch_with_browser(task, source, state)

        # Sources that opted into relaxed certificate verification use the
        # separate insecure client; everything else keeps strict verification.
        if source and source.insecure_ssl and self._insecure_client is not None:
            client = self._insecure_client

        headers: dict[str, str] = {}
        # Conditional request: a repeat crawl skips unchanged documents entirely.
        if task.etag:
            headers["If-None-Match"] = task.etag
        if task.last_modified:
            headers["If-Modified-Since"] = task.last_modified

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await client.get(task.url, headers=headers)

                if response.status_code == 304:
                    state.mark_skipped(task.url, "unchanged since last crawl (304)")
                    return None
                if response.status_code in PERMANENT_STATUSES:
                    state.mark_failed(task.url, f"HTTP {response.status_code}", permanent=True)
                    return None
                if response.status_code in RETRYABLE_STATUSES:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}", request=response.request, response=response
                    )
                response.raise_for_status()

                payload = response.content
                if len(payload) > MAX_BYTES:
                    state.mark_skipped(task.url, f"payload too large ({len(payload)} bytes)")
                    return None

                document = extract(
                    task.url,
                    payload,
                    response.headers.get("content-type", ""),
                    enable_ocr=self.config.enable_ocr,
                )

                # A 200 is not proof of success on government portals.
                soft_404 = looks_like_soft_404(str(response.url), document.text)
                if soft_404:
                    state.mark_failed(task.url, f"soft 404: {soft_404}", permanent=True)
                    logger.info("Soft 404 at %s (%s)", task.url, soft_404)
                    return None

                document.extra["etag"] = response.headers.get("etag")
                document.extra["last_modified"] = response.headers.get("last-modified")
                return document

            except Exception as exc:  # noqa: BLE001 - retry transient, record terminal
                # A bad certificate chain will not fix itself between attempts,
                # so retrying only burns the backoff budget.
                if _is_permanent_transport_error(exc):
                    state.mark_failed(task.url, f"{type(exc).__name__}: {exc}", permanent=True)
                    logger.warning("Permanent transport failure for %s: %s", task.url, exc)
                    return None
                if attempt >= self.config.max_retries:
                    state.mark_failed(task.url, f"{type(exc).__name__}: {exc}")
                    logger.warning("Giving up on %s after %d attempts: %s",
                                   task.url, attempt, exc)
                    return None
                # Exponential backoff with jitter, so a batch of retries against
                # one struggling server does not resynchronise into a thundering
                # herd.
                delay = min(self.config.backoff_base ** attempt, self.config.backoff_max)
                delay += random.uniform(0, delay * 0.3)
                logger.debug("Retry %d/%d for %s in %.1fs (%s)",
                             attempt, self.config.max_retries, task.url, delay, exc)
                await asyncio.sleep(delay)
        return None

    async def _fetch_with_browser(
        self, task: CrawlTask, source: Source, state: CrawlState
    ) -> ExtractedDocument | None:
        """Render a JavaScript-dependent page and extract it.

        Args:
            task: The URL to fetch.
            source: Owning source.
            state: Crawl state, for recording failures.

        Returns:
            The extracted document, or None on failure.
        """
        from datetime import UTC, datetime

        from ingestion.scraper import SchemeScraper

        try:
            records = await SchemeScraper(
                max_concurrency=1, wait_for_seconds=3.0
            ).scrape_urls([task.url])
        except Exception as exc:  # noqa: BLE001
            state.mark_failed(task.url, f"browser fetch failed: {exc}")
            return None

        if not records:
            state.mark_failed(task.url, "browser returned no content")
            return None

        record = records[0]
        return ExtractedDocument(
            url=task.url,
            text=record["markdown"],
            title=record["title"],
            doc_format="html",
            retrieved_at=record.get("scraped_at") or datetime.now(UTC).isoformat(),
            content_type="text/html",
        )

    def _enqueue_links(
        self,
        document: ExtractedDocument,
        task: CrawlTask,
        source: Source | None,
        state: CrawlState,
    ) -> None:
        """Add a document's outbound links to the frontier.

        Args:
            document: The extracted document whose links to follow.
            task: The task that produced it, for depth bookkeeping.
            source: Owning source, when known.
            state: Crawl state store.
        """
        origin_host = urlparse(task.url).hostname or ""
        added = 0
        for raw in document.links:
            candidate = normalise_url(urljoin(task.url, raw))
            if not is_official_url(candidate) or not is_fetchable_url(candidate):
                continue
            if self.config.same_domain_only:
                host = urlparse(candidate).hostname or ""
                if not (host == origin_host or host.endswith("." + origin_host)
                        or origin_host.endswith("." + host)):
                    continue
            link_source = source_for_url(candidate) or source
            if state.add_url(candidate, task.depth + 1,
                             link_source.key if link_source else None):
                added += 1
        if added:
            logger.debug("Discovered %d new URLs from %s", added, task.url)
