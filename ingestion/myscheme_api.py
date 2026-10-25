"""Enumerate and fetch every scheme on myScheme through the portal's own public API.

myScheme renders scheme pages client-side and its sitemap lists none of them
(measured: 38 static URLs, zero /schemes/ pages). The page you see is built from
same-origin JSON routes the frontend calls without any key:

    /api/apisetu/search/schemes   paginated listing of every scheme (5000+)
    /api/apisetu/schemes?slug=    full structured detail for one scheme
    /api/apisetu/schemes/{id}/documents, /faqs

Consuming those is both more complete and less fragile than rendering pages in
a headless browser and scraping the DOM: the listing is exhaustive by
construction, and the detail payload already carries markdown for each section
plus structured facts (ministry, level, states, category) that HTML scraping
would have to guess at. robots.txt allows these paths ("Allow: /").

Raw API responses are cached one file per scheme, so a harvest interrupted
after 3000 schemes resumes at 3001 and re-rendering never re-fetches.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from ingestion.fetchers import ExtractedDocument

logger = logging.getLogger(__name__)

BASE = "https://www.myscheme.gov.in"
SEARCH_URL = f"{BASE}/api/apisetu/search/schemes"
DETAIL_URL = f"{BASE}/api/apisetu/schemes"
# The human-facing page is the canonical citation target, and its final path
# segment is the slug that eligibility_rules.json keys on.
PAGE_URL = BASE + "/schemes/{slug}"

DISCOVERY_METHOD = "myscheme_api"
PAGE_SIZE = 100


@dataclass
class HarvestResult:
    """Outcome of a harvest run."""

    listed: int = 0
    documents: list[ExtractedDocument] = field(default_factory=list)
    fetched: int = 0
    from_cache: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)


def _clean_md(text: str | None) -> str:
    """Normalise one of the API's *_md fields into plain markdown.

    The API's markdown keeps HTML entities and <br> tags from its rich-text
    editor; left in, they are tokenized as noise.

    Args:
        text: Raw markdown, possibly None.

    Returns:
        Cleaned markdown, stripped.
    """
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    return html.unescape(text).strip()


def _labels(value: Any) -> list[str]:
    """Flatten the API's {label, value} objects (single or list) into labels."""
    if not value:
        return []
    items = value if isinstance(value, list) else [value]
    return [str(i.get("label")) for i in items if isinstance(i, dict) and i.get("label")]


def scheme_facts(listing: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    """Collect the structured facts that make a scheme filterable.

    Args:
        listing: The scheme's `fields` from the search listing.
        detail: The `en` object from the detail endpoint.

    Returns:
        Ministry, level, states, categories and related facts. Missing values
        stay empty rather than guessed.
    """
    basic = detail.get("basicDetails") or {}
    return {
        "scheme_id": listing.get("slug") or "",
        "ministry": (_labels(basic.get("nodalMinistryName"))
                     or [listing.get("nodalMinistryName") or ""])[0],
        "department": (_labels(basic.get("nodalDepartmentName")) or [""])[0],
        "implementing_agency": basic.get("implementingAgency") or "",
        "level": listing.get("level") or (_labels(basic.get("level")) or [""])[0],
        "states": list(listing.get("beneficiaryState") or []),
        "categories": list(listing.get("schemeCategory") or _labels(basic.get("schemeCategory"))),
        "tags": list(listing.get("tags") or basic.get("tags") or []),
        "scheme_type": (_labels(basic.get("schemeType")) or [""])[0],
        "close_date": listing.get("schemeCloseDate") or basic.get("schemeCloseDate"),
    }


def render_markdown(
    listing: dict[str, Any], detail: dict[str, Any], documents: dict[str, Any], faqs: list[dict]
) -> str:
    """Render one scheme's API payloads as a sectioned markdown document.

    Headings deliberately match the section names scheme_extractor and the
    chunker already recognise, so downstream stages need no special casing.

    Args:
        listing: Search-listing fields.
        detail: Detail `en` object.
        documents: Documents `en` object.
        faqs: FAQ entries.

    Returns:
        Markdown text.
    """
    content = detail.get("schemeContent") or {}
    facts = scheme_facts(listing, detail)
    name = listing.get("schemeName") or (detail.get("basicDetails") or {}).get("schemeName", "")
    short = listing.get("schemeShortTitle")

    parts = [f"# {name}" + (f" ({short})" if short else "")]

    # Facts first: "which ministry runs this, and in which state" is a common
    # question, and it should be answerable from the chunk that names the scheme.
    fact_lines = [
        f"- {label}: {value}"
        for label, value in (
            ("Nodal ministry", facts["ministry"]),
            ("Nodal department", facts["department"]),
            ("Implementing agency", facts["implementing_agency"]),
            ("Level", facts["level"]),
            ("Applicable states", ", ".join(facts["states"])),
            ("Category", ", ".join(facts["categories"])),
            ("Scheme type", facts["scheme_type"]),
            ("Closing date", facts["close_date"]),
        )
        if value
    ]
    if fact_lines:
        parts += ["## Scheme Facts", "\n".join(fact_lines)]

    sections = (
        ("Details", _clean_md(content.get("detailedDescription_md"))
         or _clean_md(content.get("briefDescription"))),
        ("Benefits", _clean_md(content.get("benefits_md"))),
        ("Eligibility", _clean_md((detail.get("eligibilityCriteria") or {})
                                  .get("eligibilityDescription_md"))),
        ("Exclusions", _clean_md(content.get("exclusions_md"))),
    )
    for heading, body in sections:
        if body:
            parts += [f"## {heading}", body]

    process = [
        f"### {p.get('mode') or 'Process'}\n{_clean_md(p.get('process_md'))}"
        for p in detail.get("applicationProcess") or []
        if _clean_md(p.get("process_md"))
    ]
    if process:
        parts += ["## Application Process", "\n\n".join(process)]

    docs_md = _clean_md(documents.get("documentsRequired_md"))
    if docs_md:
        parts += ["## Documents Required", docs_md]

    faq_lines = [
        f"**Q: {_clean_md(f.get('question'))}**\n{_clean_md(f.get('answer_md'))}"
        for f in faqs
        if f.get("question")
    ]
    if faq_lines:
        parts += ["## Frequently Asked Questions", "\n\n".join(faq_lines)]

    refs = reference_urls(detail, with_titles=True)
    if refs:
        parts += ["## Sources and References", "\n".join(f"- {t}: {u}" for t, u in refs)]

    return "\n\n".join(parts) + "\n"


def reference_urls(detail: dict[str, Any], with_titles: bool = False) -> list[Any]:
    """Official documents a scheme cites — guidelines, manuals, notifications.

    These are discovery seeds: they point at the authoritative PDFs behind the
    scheme summary, which the document crawler can then fetch and parse.

    Args:
        detail: Detail `en` object.
        with_titles: Return (title, url) pairs instead of bare URLs.

    Returns:
        Stripped URLs (the API pads some with trailing spaces), in order.
    """
    out: list[Any] = []
    for ref in (detail.get("schemeContent") or {}).get("references") or []:
        url = (ref.get("url") or "").strip().replace(" ", "%20")
        if url.startswith("http"):
            out.append(((ref.get("title") or "Reference").strip(), url) if with_titles else url)
    return out


class MySchemeHarvester:
    """Fetch every scheme listed by the myScheme API into ExtractedDocuments."""

    def __init__(
        self,
        cache_dir: Path,
        concurrency: int = 3,
        delay_seconds: float = 0.3,
        timeout: float = 45.0,
        max_retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Configure the harvester.

        Args:
            cache_dir: One JSON file per scheme is kept here, for resumability.
            concurrency: Simultaneous detail fetches. Kept low: this is a
                public government service, not a CDN.
            delay_seconds: Pause after each request per worker.
            timeout: Per-request timeout.
            max_retries: Attempts per request before recording a failure.
            transport: Replaces the network, for offline tests.
        """
        self.cache_dir = cache_dir
        self.concurrency = concurrency
        self.delay_seconds = delay_seconds
        self.timeout = timeout
        self.max_retries = max_retries
        self.transport = transport

    async def _get_json(self, client: httpx.AsyncClient, url: str) -> dict[str, Any]:
        """GET a JSON endpoint with exponential backoff on transient errors."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await client.get(url)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}", request=response.request, response=response
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, json.JSONDecodeError):
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(min(2.0**attempt, 30.0))
        raise RuntimeError("unreachable")  # pragma: no cover

    async def list_all(self, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        """Page through the search API until every listed scheme is collected.

        Args:
            client: Shared HTTP client.

        Returns:
            Listing `fields` for every scheme, de-duplicated by slug.
        """
        listings: dict[str, dict[str, Any]] = {}
        start, total = 0, None
        while total is None or start < total:
            # A stable sort is required for offset paging. Under the default
            # relevance sort items shift between pages; measured 2026-10-05 it
            # silently skipped 76 of 5093 schemes. Name order skipped none.
            url = (f"{SEARCH_URL}?lang=en&q={quote('[]')}&keyword=&sort=schemename-asc"
                   f"&from={start}&size={PAGE_SIZE}")
            hits = (await self._get_json(client, url))["data"]["hits"]
            total = hits["page"]["total"]
            items = hits.get("items") or []
            if not items:
                break  # the API stopped short of its own total; do not spin
            for item in items:
                fields = item.get("fields") or {}
                if fields.get("slug"):
                    listings[fields["slug"]] = fields
            start += PAGE_SIZE
            await asyncio.sleep(self.delay_seconds)
        logger.info("myScheme lists %s schemes; collected %d", total, len(listings))
        return list(listings.values())

    async def _fetch_one(self, client: httpx.AsyncClient, slug: str) -> dict[str, Any]:
        """Fetch detail, documents and FAQs for one scheme."""
        detail = (await self._get_json(client, f"{DETAIL_URL}?slug={quote(slug)}&lang=en"))["data"]
        if not detail:
            # The portal lists some schemes it has no detail record for; its own
            # frontend sends those to /404. A portal-side gap, not a fetch error.
            raise LookupError("listed by the portal but its detail API returns no record")
        scheme_id = detail["_id"]
        docs = await self._get_json(client, f"{DETAIL_URL}/{scheme_id}/documents?lang=en")
        faqs = await self._get_json(client, f"{DETAIL_URL}/{scheme_id}/faqs?lang=en")
        return {
            "detail": detail.get("en") or {},
            "documents": ((docs.get("data") or {}).get("en")) or {},
            "faqs": (((faqs.get("data") or {}).get("en")) or {}).get("faqs") or [],
            "fetched_at": datetime.now(UTC).isoformat(),
        }

    async def harvest(
        self,
        limit: int | None = None,
        categories: list[str] | None = None,
        refresh: bool = False,
        on_document: Callable[[ExtractedDocument], Any] | None = None,
    ) -> HarvestResult:
        """Enumerate schemes and fetch (or load from cache) each one.

        Args:
            limit: Stop after this many schemes, for trial runs.
            categories: Only keep schemes in at least one of these categories.
            refresh: Re-fetch even when a cached copy exists.
            on_document: Called with each document as soon as it is built.
                When given, documents are handed off rather than collected in
                the result, so a full harvest never holds 5000+ documents at once.

        Returns:
            Documents (empty when on_document is given) plus counts and
            per-scheme failures.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        result = HarvestResult()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "From": "https://github.com/sushritapaul-beep/SETU",
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(headers=headers, timeout=self.timeout,
                                     follow_redirects=True, transport=self.transport) as client:
            listings = await self.list_all(client)
            result.listed = len(listings)
            if categories:
                wanted = {c.lower() for c in categories}
                listings = [x for x in listings
                            if wanted & {c.lower() for c in x.get("schemeCategory") or []}]
            listings = listings[:limit] if limit else listings

            gate = asyncio.Semaphore(self.concurrency)

            async def process(listing: dict[str, Any]) -> None:
                slug = listing["slug"]
                cache = self.cache_dir / f"{re.sub(r'[^A-Za-z0-9_-]', '_', slug)}.json"
                raw: dict[str, Any] | None = None
                if cache.exists() and not refresh:
                    raw = json.loads(cache.read_text(encoding="utf-8"))
                    result.from_cache += 1
                else:
                    async with gate:
                        try:
                            raw = await self._fetch_one(client, slug)
                        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                            error = f"{type(exc).__name__}: {exc}"
                            result.failures.append({"slug": slug, "error": error})
                            logger.warning("myScheme fetch failed for %s: %s", slug, exc)
                            return
                        finally:
                            await asyncio.sleep(self.delay_seconds)
                    raw["listing"] = listing
                    try:
                        cache.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
                    except OSError as exc:
                        # Disk full or similar: this scheme still enters the run's
                        # output, it just will not be resumable from cache.
                        logger.warning("Could not cache %s: %s", slug, exc)
                    result.fetched += 1
                    if result.fetched % 100 == 0:
                        logger.info("myScheme: fetched %d schemes", result.fetched)
                document = to_document(raw.get("listing") or listing, raw)
                if on_document is not None:
                    on_document(document)
                else:
                    result.documents.append(document)

            await asyncio.gather(*(process(x) for x in listings))

        logger.info("myScheme harvest: %d fetched, %d cached, %d failed",
                    result.fetched, result.from_cache, len(result.failures))
        return result


def to_document(listing: dict[str, Any], raw: dict[str, Any]) -> ExtractedDocument:
    """Build an ExtractedDocument from a cached raw scheme record.

    Args:
        listing: Search-listing fields.
        raw: {detail, documents, faqs, fetched_at}.

    Returns:
        The document, with structured facts in `extra` and the scheme's
        official references in `links` for the document crawler.
    """
    detail = raw.get("detail") or {}
    return ExtractedDocument(
        url=PAGE_URL.format(slug=listing["slug"]),
        text=render_markdown(listing, detail, raw.get("documents") or {}, raw.get("faqs") or []),
        title=listing.get("schemeName") or listing["slug"],
        doc_format="api",
        retrieved_at=raw.get("fetched_at") or datetime.now(UTC).isoformat(),
        content_type="application/json",
        links=reference_urls(detail),
        extra={"facts": scheme_facts(listing, detail), "discovery_method": DISCOVERY_METHOD},
    )
