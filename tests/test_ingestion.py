"""Offline tests for the ingestion layer — no network, no browser, no models."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ingestion import scraper as scraper_module
from ingestion.chunker import Chunk, SemanticParentChildChunker
from ingestion.parser import DoclingParser
from ingestion.scraper import STARTER_SCHEME_URLS, SchemeScraper

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _build_minimal_pdf(body_text: str) -> bytes:
    """Construct a tiny but structurally valid single-page PDF.

    Built in code rather than committed as a binary so the fixture stays
    reviewable in diffs and needs no PDF library to generate.

    Args:
        body_text: ASCII text to draw on the page.

    Returns:
        Complete PDF file bytes.
    """
    stream = f"BT /F1 24 Tf 72 700 Td ({body_text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        # The xref table stores each object's byte offset, so record it as we go.
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    return bytes(out)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Write a minimal PDF containing a recognisable marker string."""
    path = tmp_path / "sample_scheme.pdf"
    path.write_bytes(_build_minimal_pdf("SETU sample scheme eligibility document"))
    return path


SAMPLE_MARKDOWN = """## Overview

The scheme supports micro and small enterprises across India with credit
guarantees and subsidised interest rates for approved manufacturing units.

## Eligibility

Applicants must hold a valid Udyam registration certificate. The enterprise
must be classified as micro or small under the MSMED Act 2006. Annual turnover
must not exceed five crore rupees for micro enterprises.

## Benefits

Eligible units receive a credit guarantee covering up to eighty five percent of
the sanctioned loan amount, with no collateral required from the borrower.
"""


# --------------------------------------------------------------------------
# 1. Parser
# --------------------------------------------------------------------------

def test_parser_extracts_text_from_sample_pdf(
    sample_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real PDF is parsed to markdown via the fallback when Docling fails."""
    pymupdf4llm = pytest.importorskip(
        "pymupdf4llm", reason="pymupdf4llm not installed; run pip install -r requirements.txt"
    )
    assert pymupdf4llm is not None

    parser = DoclingParser()
    # Force the fallback path so this test never loads Docling's layout models.
    monkeypatch.setattr(
        DoclingParser,
        "_parse_with_docling",
        lambda self, path: (_ for _ in ()).throw(ImportError("docling disabled for test")),
    )

    markdown = parser.parse_pdf(sample_pdf)

    assert "SETU" in markdown
    assert "eligibility" in markdown.lower()


def test_parser_prefers_docling_when_it_succeeds(
    sample_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Docling output is used directly, without invoking the fallback."""
    monkeypatch.setattr(
        DoclingParser, "_parse_with_docling", lambda self, path: "# Parsed by Docling"
    )

    def _fail(self: DoclingParser, path: Path) -> str:
        raise AssertionError("fallback must not run when Docling succeeds")

    monkeypatch.setattr(DoclingParser, "_parse_with_pymupdf", _fail)

    assert DoclingParser().parse_pdf(sample_pdf) == "# Parsed by Docling"


def test_docling_can_be_disabled_by_env(
    sample_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SETU_DISABLE_DOCLING skips Docling entirely instead of failing slowly."""
    pytest.importorskip("pymupdf4llm")

    import ingestion.parser as parser_module

    monkeypatch.setattr(parser_module, "DISABLE_DOCLING", True)

    def _must_not_run(self: DoclingParser, path: Path) -> str:
        raise AssertionError("Docling must not be attempted when disabled")

    monkeypatch.setattr(DoclingParser, "_parse_with_docling", _must_not_run)

    assert "SETU" in DoclingParser().parse_pdf(sample_pdf)


def test_parser_raises_on_missing_file(tmp_path: Path) -> None:
    """A missing path fails immediately rather than deep inside a PDF library."""
    with pytest.raises(FileNotFoundError):
        DoclingParser().parse_pdf(tmp_path / "does_not_exist.pdf")


# --------------------------------------------------------------------------
# 2. Chunker
# --------------------------------------------------------------------------

def test_chunker_splits_markdown_into_linked_parents_and_children() -> None:
    """A known document yields parents, children, and correct parent linkage."""
    # Small budgets force multiple parents out of a short fixture document.
    chunker = SemanticParentChildChunker(parent_tokens=100, child_tokens=30)

    chunks = chunker.chunk_document(
        text=SAMPLE_MARKDOWN,
        source_url="https://www.myscheme.gov.in/schemes/mszcs",
        scheme_name="MSME Sustainable (ZED) Certification Scheme",
    )

    parents = [c for c in chunks if c.metadata.chunk_type == "parent"]
    children = [c for c in chunks if c.metadata.chunk_type == "child"]

    assert len(parents) >= 2, "short budget should produce several parents"
    assert len(children) > len(parents), "each parent should split into smaller children"

    parent_ids = {p.id for p in parents}
    assert all(p.parent_id is None for p in parents)
    # Every child must resolve to a real parent, or the retrieval phase cannot
    # expand a hit back to its context.
    assert all(c.parent_id in parent_ids for c in children)

    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(len(c.id) == 32 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks), "ids must be unique"

    sample = children[0]
    assert sample.metadata.source_url == "https://www.myscheme.gov.in/schemes/mszcs"
    assert sample.metadata.scheme_name == "MSME Sustainable (ZED) Certification Scheme"

    # Content must survive the split.
    assert any("Udyam" in c.text for c in children)


def test_chunker_ids_are_deterministic() -> None:
    """Re-chunking identical input reproduces identical ids, so upserts are idempotent."""
    chunker = SemanticParentChildChunker(parent_tokens=100, child_tokens=30)
    kwargs = {"source_url": "https://example.gov.in/x", "scheme_name": "Test Scheme"}

    first = chunker.chunk_document(text=SAMPLE_MARKDOWN, **kwargs)
    second = chunker.chunk_document(text=SAMPLE_MARKDOWN, **kwargs)

    assert [c.id for c in first] == [c.id for c in second]


def test_normalize_strips_noise_but_preserves_indic_joiners() -> None:
    """Zero-width noise goes; ZWNJ/ZWJ stay, because Indic scripts need them."""
    chunker = SemanticParentChildChunker()

    # U+FEFF and U+200B are scrape artefacts; U+200C/U+200D are meaningful.
    cleaned = chunker.normalize_text("Udyam﻿ registration​ required")
    assert "﻿" not in cleaned
    assert "​" not in cleaned
    assert "Udyam registration required" == cleaned

    hindi = "क‍ष हिन्दी‌ पाठ"
    assert chunker.normalize_text(hindi) == hindi, "Indic joiners must survive"

    assert chunker.normalize_text("a\n\n\n\n\nb") == "a\n\nb"


def test_chunker_output_is_free_of_zero_width_noise() -> None:
    """Noise present in the source never reaches a chunk."""
    dirty = SAMPLE_MARKDOWN.replace("Eligibility", "Eligibility﻿")
    chunks = SemanticParentChildChunker().chunk_document(dirty, "https://x.gov.in", "X")

    assert chunks, "fixture should produce chunks"
    assert all("﻿" not in c.text for c in chunks)


def test_strip_boilerplate_truncates_footer_and_removes_modals() -> None:
    """Site navigation and repeated dialog text are removed; content survives."""
    chunker = SemanticParentChildChunker()
    document = (
        "## Eligibility\n"
        "Must hold Udyam registration.\n"
        "Something went wrong. Please try again later. Ok\n"
        "Are you sure you want to sign out? CancelSign Out\n"
        "## Quick Links\n"
        "* [About Us](https://www.myscheme.gov.in/about)\n"
        "* [Contact Us](https://www.myscheme.gov.in/contact)\n"
    )

    cleaned = chunker.strip_boilerplate(document)

    assert "Udyam registration" in cleaned
    assert "Quick Links" not in cleaned
    assert "About Us" not in cleaned
    assert "Something went wrong" not in cleaned
    assert "sign out" not in cleaned


def test_strip_boilerplate_removes_ui_controls_and_nav_menu() -> None:
    """The header block of bare buttons and section bullets is removed."""
    chunker = SemanticParentChildChunker()
    document = (
        "Ok\nCancelSign Out\nBack\n"
        "  * Details\n  * Benefits\n  * Eligibility\n  * Documents Required\n"
        "CancelOk\n### Apply Now\n### Check Eligibility\n"
        "### Eligibility\n"
        "Should be registered in the Udyam portal to apply for the scheme.\n"
    )

    cleaned = chunker.strip_boilerplate(document)

    assert "CancelSign Out" not in cleaned
    assert "CancelOk" not in cleaned
    assert "Apply Now" not in cleaned
    # The bare menu bullet is gone...
    assert "* Details" not in cleaned
    # ...but the real heading of the same name and its content survive.
    assert "### Eligibility" in cleaned
    assert "Udyam portal" in cleaned


def test_strip_boilerplate_keeps_content_bullets_and_sentences() -> None:
    """Only exact UI matches are dropped; real bullets and prose are untouched."""
    chunker = SemanticParentChildChunker()
    document = (
        "### Documents Required\n"
        "  * Udyam Registration Number of the Enterprise.\n"
        "  * Valid Email ID for login credentials.\n"
        "You may cancel your application within 30 days.\n"
    )

    cleaned = chunker.strip_boilerplate(document)

    assert "Udyam Registration Number" in cleaned
    assert "Valid Email ID" in cleaned
    # "cancel" inside a sentence must not trigger the UI-line filter.
    assert "You may cancel your application" in cleaned


def test_is_low_quality_rejects_markup_but_keeps_prose() -> None:
    """Coordinate dumps and stubs are rejected; real scheme prose is not."""
    chunker = SemanticParentChildChunker()

    # Resembles the SVG path data that was polluting the index.
    svg = ".028 .008-.026 .008.112.112.188.055.163.341-.808.933-.785.933-.696" * 4
    assert chunker.is_low_quality(svg) is True
    assert chunker.is_low_quality("### Eligibility") is True, "too short to be useful"
    assert chunker.is_low_quality("   ") is True

    prose = (
        "Applicants must hold a valid Udyam registration certificate and the "
        "enterprise must be classified as micro or small under the MSMED Act."
    )
    assert chunker.is_low_quality(prose) is False


def test_is_low_quality_keeps_indic_prose() -> None:
    """Hindi and Tamil must not be filtered out — isalpha() covers both scripts."""
    chunker = SemanticParentChildChunker()

    hindi = (
        "आवेदक के पास वैध उद्यम पंजीकरण प्रमाणपत्र होना चाहिए और उद्यम को "
        "सूक्ष्म या लघु श्रेणी में वर्गीकृत होना चाहिए।"
    )
    tamil = (
        "விண்ணப்பதாரர் செல்லுபடியாகும் உத்யம் பதிவுச் சான்றிதழை வைத்திருக்க "
        "வேண்டும் மற்றும் நிறுவனம் குறு அல்லது சிறு வகையில் இருக்க வேண்டும்."
    )

    assert chunker.is_low_quality(hindi) is False
    assert chunker.is_low_quality(tamil) is False


def test_chunk_document_drops_boilerplate_chunks() -> None:
    """End to end, footer and markup never reach the chunk list."""
    document = SAMPLE_MARKDOWN + (
        "\n## Quick Links\n* [About Us](https://www.myscheme.gov.in/about)\n"
        + ".028 .996 .112 .188 .055 .163 .341 .808 .933 .785\n" * 20
    )

    chunks = SemanticParentChildChunker().chunk_document(
        document, "https://www.myscheme.gov.in/schemes/mszcs", "Test"
    )

    assert chunks, "real content must survive"
    assert all("Quick Links" not in c.text for c in chunks)
    assert all("About Us" not in c.text for c in chunks)
    assert any("Udyam" in c.text for c in chunks)


def test_chunker_returns_nothing_for_blank_input() -> None:
    """Blank documents produce no chunks instead of one empty chunk."""
    assert SemanticParentChildChunker().chunk_document("   \n  ", "https://x.gov.in", "X") == []


def test_chunk_to_dict_is_json_serialisable() -> None:
    """to_dict() flattens metadata for JSONL writing."""
    chunks = SemanticParentChildChunker().chunk_document(
        SAMPLE_MARKDOWN, "https://x.gov.in", "X Scheme"
    )
    record = chunks[0].to_dict()

    assert set(record) == {"id", "parent_id", "text", "metadata"}
    # Superset, not equality: provenance fields were added later and more may
    # follow. Pinning the exact set makes every schema addition a test failure.
    assert set(record["metadata"]) >= {"source_url", "scheme_name", "chunk_type"}
    assert set(record["metadata"]) >= {"doc_format", "retrieved_at", "doc_hash"}


# --------------------------------------------------------------------------
# 3. Scraper
# --------------------------------------------------------------------------

class _FakeMarkdown:
    """Stand-in for Crawl4AI's MarkdownGenerationResult object."""

    def __init__(self, raw: str) -> None:
        self.raw_markdown = raw


class _FakeResult:
    """Stand-in for a Crawl4AI CrawlResult."""

    def __init__(self, markdown: str, success: bool = True) -> None:
        self.success = success
        self.markdown = _FakeMarkdown(markdown)
        self.metadata = {"title": "MSME Sustainable (ZED) Certification Scheme"}
        self.error_message = "" if success else "simulated failure"


class _FakeCrawler:
    """Async context manager that returns canned pages instead of browsing."""

    def __init__(self, config: Any = None) -> None:
        self.config = config

    async def __aenter__(self) -> _FakeCrawler:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def arun(self, url: str, config: Any = None) -> _FakeResult:
        # One URL fails on purpose to prove the batch survives partial failure.
        if url.endswith("/broken"):
            return _FakeResult("", success=False)
        return _FakeResult(f"# Scheme page\n\nContent scraped from {url}")


def test_scraper_is_instantiable_and_scrapes_with_mocked_crawler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SchemeScraper constructs, and scrape_urls returns well-shaped records."""
    scraper = SchemeScraper(max_concurrency=2, timeout_seconds=5)
    assert scraper.max_concurrency == 2

    # Swap the browser for the fake and pretend the optional dep is present.
    monkeypatch.setattr(scraper_module, "AsyncWebCrawler", _FakeCrawler)
    monkeypatch.setattr(scraper_module, "CRAWL4AI_AVAILABLE", True)
    monkeypatch.setattr(scraper_module, "BrowserConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(scraper_module, "CrawlerRunConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(scraper_module, "CacheMode", type("CacheMode", (), {"BYPASS": "bypass"}))

    urls = ["https://www.myscheme.gov.in/schemes/mszcs", "https://www.myscheme.gov.in/broken"]
    records = asyncio.run(scraper.scrape_urls(urls))

    # The broken URL is dropped, not raised.
    assert len(records) == 1
    record = records[0]
    assert set(record) == {"url", "title", "markdown", "scraped_at"}
    assert record["url"] == urls[0]
    assert "Content scraped from" in record["markdown"]
    assert record["scraped_at"].endswith("+00:00"), "timestamps must be timezone-aware UTC"


def test_scraper_raises_clear_error_when_crawl4ai_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional dependency produces an actionable message."""
    monkeypatch.setattr(scraper_module, "CRAWL4AI_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="playwright install"):
        asyncio.run(SchemeScraper().scrape_urls(["https://example.gov.in"]))


def test_starter_urls_are_five_myscheme_scheme_pages() -> None:
    """The hardcoded starter corpus is the expected size and shape."""
    assert len(STARTER_SCHEME_URLS) == 5
    assert len(set(STARTER_SCHEME_URLS)) == 5
    assert all(u.startswith("https://www.myscheme.gov.in/schemes/") for u in STARTER_SCHEME_URLS)
