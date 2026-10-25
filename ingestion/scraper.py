"""Fetch government scheme pages with Crawl4AI and return them as clean markdown.

myScheme renders its scheme pages client-side, so a plain HTTP GET returns an
empty application shell. Crawl4AI drives a real headless browser, which is why it
is used here instead of requests/httpx.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Crawl4AI drags in Playwright and a browser binary. Import defensively so that
# unit tests, linting and CI containers without browsers can still import this
# module; the failure is raised only when a scrape is actually attempted.
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

    CRAWL4AI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in bare environments
    AsyncWebCrawler = None  # type: ignore[assignment]
    BrowserConfig = CacheMode = CrawlerRunConfig = None  # type: ignore[assignment]
    CRAWL4AI_AVAILABLE = False


# Starter corpus: five MSME scheme pages on myScheme. Slugs are short opaque
# codes rather than readable names, so these were taken from indexed search
# results instead of being guessed. See the Phase 2 notes on verification.
STARTER_SCHEME_URLS: list[str] = [
    # MSME Sustainable (ZED) Certification Scheme — Ministry of MSME
    "https://www.myscheme.gov.in/schemes/mszcs",
    # Design Scheme under MSME Innovative Scheme — Ministry of MSME
    "https://www.myscheme.gov.in/schemes/ds",
    # MSME Skill Training and Employment Scheme
    "https://www.myscheme.gov.in/schemes/msme-stes",
    # Assistance to MSME: Assistance for Technology Acquisition (Gujarat)
    "https://www.myscheme.gov.in/schemes/samsmeata",
    # Investment Promotion Scheme (IPS) for MSME Sector
    "https://www.myscheme.gov.in/schemes/ipsasmerctse",
]


class SchemeScraper:
    """Crawl scheme pages concurrently and normalise them to markdown records."""

    def __init__(
        self,
        max_concurrency: int = 3,
        timeout_seconds: int = 60,
        headless: bool = True,
        wait_for_seconds: float = 2.0,
    ) -> None:
        """Configure crawl behaviour.

        Args:
            max_concurrency: Simultaneous page loads. Kept low deliberately —
                these are public government servers, not a CDN.
            timeout_seconds: Per-page budget before the load is abandoned.
            headless: Run the browser without a visible window.
            wait_for_seconds: Extra settle time after load, so client-side
                rendered scheme content has appeared before markdown is taken.
        """
        self.max_concurrency = max_concurrency
        self.timeout_seconds = timeout_seconds
        self.headless = headless
        self.wait_for_seconds = wait_for_seconds
        # url -> the crawler's own error message, so callers can tell an
        # access block apart from an empty render.
        self.errors: dict[str, str] = {}

    async def scrape_urls(self, urls: list[str]) -> list[dict[str, Any]]:
        """Scrape every URL and return one record per page that succeeded.

        Individual failures are logged and skipped rather than aborting the
        batch, so one dead government URL cannot fail an entire ingest run.

        Args:
            urls: Absolute URLs of scheme pages to fetch.

        Returns:
            Records shaped {url, title, markdown, scraped_at}, in completion
            order. Pages that failed or returned no text are omitted.

        Raises:
            RuntimeError: If Crawl4AI is not installed.
        """
        if not CRAWL4AI_AVAILABLE:
            raise RuntimeError(
                "crawl4ai is not installed. Run: pip install -r requirements.txt "
                "&& python -m playwright install chromium"
            )
        if not urls:
            return []

        semaphore = asyncio.Semaphore(self.max_concurrency)
        browser_config = BrowserConfig(headless=self.headless, verbose=False)

        async with AsyncWebCrawler(config=browser_config) as crawler:
            tasks = [self._scrape_one(crawler, url, semaphore) for url in urls]
            # return_exceptions=True keeps one crashed page from cancelling the rest.
            settled = await asyncio.gather(*tasks, return_exceptions=True)

        records: list[dict[str, Any]] = []
        for url, outcome in zip(urls, settled, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning("Scrape raised for %s: %s", url, outcome)
            elif outcome is not None:
                records.append(outcome)

        logger.info("Scraped %d/%d pages successfully", len(records), len(urls))
        return records

    async def _scrape_one(
        self,
        crawler: Any,
        url: str,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any] | None:
        """Fetch a single page under the concurrency limit.

        Args:
            crawler: An open AsyncWebCrawler instance.
            url: The page to fetch.
            semaphore: Shared limiter guarding concurrent page loads.

        Returns:
            A scrape record, or None if the page failed or had no usable text.
        """
        async with semaphore:
            run_config = CrawlerRunConfig(
                # BYPASS so a re-run always reflects the live page; scheme rules
                # change and a stale cache would silently poison the index.
                cache_mode=CacheMode.BYPASS,
                page_timeout=self.timeout_seconds * 1000,  # Crawl4AI expects ms
                delay_before_return_html=self.wait_for_seconds,
                # Strip chrome that would otherwise dominate every chunk.
                excluded_tags=["nav", "footer", "header", "script", "style"],
                word_count_threshold=10,
            )
            result = await crawler.arun(url=url, config=run_config)

        if not getattr(result, "success", False):
            error = str(getattr(result, "error_message", "") or "unknown")
            self.errors[url] = error
            logger.warning("Scrape failed for %s: %s", url, error)
            return None

        markdown = self._extract_markdown(result)
        if not markdown.strip():
            # A blank body usually means the SPA never hydrated — surface it
            # rather than writing an empty document into the corpus.
            logger.warning(
                "Scrape returned empty markdown for %s (page may not have rendered)", url
            )
            return None

        return {
            "url": url,
            "title": self._extract_title(result, url),
            "markdown": markdown,
            "scraped_at": datetime.now(UTC).isoformat(),
            "links": self._extract_links(result),
        }

    @staticmethod
    def _extract_links(result: Any) -> list[str]:
        """Collect hrefs from the rendered page.

        Crawl4AI reports links as {"internal": [...], "external": [...]}, each
        entry a dict with an "href" (or a bare string in older versions).

        Args:
            result: A Crawl4AI CrawlResult.

        Returns:
            Absolute http(s) URLs in page order, de-duplicated.
        """
        links = getattr(result, "links", None) or {}
        hrefs: list[str] = []
        for group in ("internal", "external"):
            for item in links.get(group, []) if isinstance(links, dict) else []:
                href = item.get("href") if isinstance(item, dict) else item
                if isinstance(href, str) and href.startswith("http"):
                    hrefs.append(href)
        return list(dict.fromkeys(hrefs))

    @staticmethod
    def _extract_markdown(result: Any) -> str:
        """Pull markdown text out of a Crawl4AI result across API variants.

        Args:
            result: A Crawl4AI CrawlResult.

        Returns:
            The markdown body, or an empty string if none was produced.
        """
        markdown = getattr(result, "markdown", None)
        if markdown is None:
            return ""
        # Newer Crawl4AI returns a MarkdownGenerationResult object; older
        # versions return a plain string. Support both.
        if isinstance(markdown, str):
            return markdown
        return getattr(markdown, "raw_markdown", "") or ""

    @staticmethod
    def _extract_title(result: Any, url: str) -> str:
        """Determine a human-readable page title.

        Args:
            result: A Crawl4AI CrawlResult.
            url: Fallback identifier if the page exposes no title.

        Returns:
            The page title, or the URL's last path segment as a fallback.
        """
        metadata = getattr(result, "metadata", None) or {}
        title = metadata.get("title") if isinstance(metadata, dict) else None
        if title:
            return str(title).strip()
        return url.rstrip("/").rsplit("/", 1)[-1]
