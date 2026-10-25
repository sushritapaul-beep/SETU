"""Offline tests for corpus expansion: myScheme API harvest, document store,
document-format coverage, and incremental Qdrant sync.

No network: the myScheme API is replaced with httpx.MockTransport, documents
are generated in memory, and Qdrant runs in-process.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from qdrant_client import QdrantClient

from ingestion.chunker import Chunk, ChunkMetadata
from ingestion.document_store import DocumentStore
from ingestion.fetchers import ExtractedDocument, _pdf_links, detect_format, extract
from ingestion.myscheme_api import MySchemeHarvester, reference_urls, render_markdown, to_document
from ingestion.sources import SOURCES_BY_KEY, Source, Tier, seed_urls
from retrieval.vectorstore import QdrantStore

# ------------------------------------------------------------- myScheme API --

LISTING = {
    "slug": "mszcs",
    "schemeName": "MSME Sustainable (ZED) Certification Scheme",
    "schemeShortTitle": "MSZCS",
    "level": "Central",
    "beneficiaryState": ["All"],
    "schemeCategory": ["Business & Entrepreneurship"],
    "nodalMinistryName": "Ministry Of Micro, Small and Medium Enterprises",
    "tags": ["Certification"],
}
DETAIL = {
    "basicDetails": {
        "implementingAgency": "Quality Council Of India (QCI)",
        "nodalMinistryName": {"label": "Ministry Of Micro, Small and Medium Enterprises"},
        "schemeType": {"label": "Central Sector Scheme"},
    },
    "schemeContent": {
        "detailedDescription_md": "Promotes Zero Defect Zero Effect &amp; quality.",
        "benefits_md": "Joining reward of ₹ 10,000<br>Subsidy up to 80%",
        "exclusions_md": "",
        "references": [
            {"title": "Guidelines", "url": "https://dcmsme.gov.in/Guidelines ZED.pdf "},
            {"title": "Broken", "url": ""},
        ],
    },
    "eligibilityCriteria": {"eligibilityDescription_md": "- Must be registered on Udyam."},
    "applicationProcess": [{"mode": "Online", "process_md": "**Step 01:** Visit the portal."}],
}
DOCS = {"documentsRequired_md": "1. Udyam Registration Number"}
FAQS = [{"question": "What is ZED?", "answer_md": "A certification drive."}]


def test_render_markdown_has_every_section_and_fact() -> None:
    """Sections use the headings the chunker and extractor already recognise."""
    text = render_markdown(LISTING, DETAIL, DOCS, FAQS)

    for heading in ("## Scheme Facts", "## Details", "## Benefits", "## Eligibility",
                    "## Application Process", "## Documents Required",
                    "## Frequently Asked Questions", "## Sources and References"):
        assert heading in text
    assert "## Exclusions" not in text, "empty sections must not be emitted"
    assert "Nodal ministry: Ministry Of Micro, Small and Medium Enterprises" in text
    assert "&amp;" not in text and "<br>" not in text, "editor markup must be cleaned"
    assert "₹ 10,000\nSubsidy" in text


def test_reference_urls_are_cleaned_and_empty_ones_dropped() -> None:
    """The API pads URLs with spaces; a raw space would break the fetch."""
    assert reference_urls(DETAIL) == ["https://dcmsme.gov.in/Guidelines%20ZED.pdf"]


def test_to_document_uses_the_canonical_page_and_carries_facts() -> None:
    """The page URL keeps the slug, which eligibility rules and citations key on."""
    raw = {"detail": DETAIL, "documents": DOCS, "faqs": FAQS,
           "fetched_at": "2026-10-05T00:00:00+00:00"}
    doc = to_document(LISTING, raw)

    assert doc.url == "https://www.myscheme.gov.in/schemes/mszcs"
    assert doc.extra["discovery_method"] == "myscheme_api"
    assert doc.extra["facts"]["level"] == "Central"
    assert doc.extra["facts"]["states"] == ["All"]
    assert doc.links == ["https://dcmsme.gov.in/Guidelines%20ZED.pdf"]
    assert doc.is_useful


def _fake_api(total: int, fail_slug: str | None = None,
              null_slug: str | None = None) -> httpx.MockTransport:
    """A myScheme API double that pages `total` schemes, 100 per page."""
    calls = {"detail": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/search/schemes" in url:
            start = int(request.url.params["from"])
            items = [{"fields": {**LISTING, "slug": f"s{i}"}}
                     for i in range(start, min(start + 100, total))]
            return httpx.Response(200, json={"data": {"hits": {
                "items": items, "page": {"total": total}}}})
        if "slug=" in url:
            calls["detail"] += 1
            slug = request.url.params["slug"]
            if slug == fail_slug:
                return httpx.Response(404)
            if slug == null_slug:
                return httpx.Response(200, json={"data": None, "statusCode": 200})
            return httpx.Response(200, json={"data": {"_id": f"id-{slug}", "en": DETAIL}})
        if url.endswith("/documents?lang=en"):
            return httpx.Response(200, json={"data": {"en": DOCS}})
        if url.endswith("/faqs?lang=en"):
            return httpx.Response(200, json={"data": {"en": {"faqs": FAQS}}})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def test_harvest_pages_through_the_whole_listing(tmp_path: Path) -> None:
    """Every listed scheme is enumerated across pages, not just the first page."""
    harvester = MySchemeHarvester(tmp_path, delay_seconds=0, transport=_fake_api(total=230))
    result = asyncio.run(harvester.harvest())

    assert result.listed == 230
    assert len(result.documents) == 230
    assert result.fetched == 230 and not result.failures


def test_harvest_resumes_from_cache_and_records_failures(tmp_path: Path) -> None:
    """A second run re-uses cached schemes; a failed scheme is reported, not dropped silently."""
    first = MySchemeHarvester(tmp_path, delay_seconds=0, max_retries=1,
                              transport=_fake_api(total=5, fail_slug="s3"))
    result = asyncio.run(first.harvest())
    assert result.fetched == 4
    assert [f["slug"] for f in result.failures] == ["s3"]

    transport = _fake_api(total=5)
    second = asyncio.run(
        MySchemeHarvester(tmp_path, delay_seconds=0, transport=transport).harvest()
    )
    assert second.from_cache == 4 and second.fetched == 1
    assert transport.calls["detail"] == 1, "only the previously failed scheme is re-fetched"


def test_portal_side_gaps_are_named_not_crashed(tmp_path: Path) -> None:
    """A listed scheme with no detail record is reported as a portal gap."""
    harvester = MySchemeHarvester(tmp_path, delay_seconds=0,
                                  transport=_fake_api(total=3, null_slug="s1"))
    result = asyncio.run(harvester.harvest())
    assert len(result.documents) == 2
    assert "detail API returns no record" in result.failures[0]["error"]


def test_harvest_category_filter(tmp_path: Path) -> None:
    """A category filter narrows the harvest without hiding the listing total."""
    harvester = MySchemeHarvester(tmp_path, delay_seconds=0, transport=_fake_api(total=3))
    result = asyncio.run(harvester.harvest(categories=["Health & Wellness"]))
    assert result.listed == 3 and result.documents == []


def test_harvest_hands_documents_off_instead_of_collecting(tmp_path: Path) -> None:
    """With on_document, a full harvest never accumulates every scheme in memory."""
    seen: list[ExtractedDocument] = []
    harvester = MySchemeHarvester(tmp_path, delay_seconds=0, transport=_fake_api(total=5))
    result = asyncio.run(harvester.harvest(on_document=seen.append))
    assert len(seen) == 5 and result.documents == []


# ----------------------------------------------------------- document store --

def _doc(url: str, text: str) -> ExtractedDocument:
    return ExtractedDocument(url=url, text=text, title="t", doc_format="html",
                             retrieved_at="2026-10-05T00:00:00+00:00")


def test_document_store_is_additive_and_keeps_history(tmp_path: Path) -> None:
    """Runs accumulate; a changed document archives its previous version."""
    store = DocumentStore(tmp_path)

    assert store.put(_doc("https://a.gov.in/x", "version one " * 20)) == "new"
    assert store.put(_doc("https://a.gov.in/y", "other page " * 20)) == "new"
    assert store.put(_doc("https://a.gov.in/x", "version one " * 20)) == "unchanged"
    assert store.put(_doc("https://a.gov.in/x/", "version two " * 20)) == "updated", \
        "a trailing slash is the same URL"

    docs = store.all()
    assert [d.url for d in docs] == ["https://a.gov.in/x/", "https://a.gov.in/y"]
    assert "version two" in docs[0].text
    assert store.history_count() == 1


def test_rebuild_streams_the_store_and_never_wipes_on_empty(tmp_path: Path) -> None:
    """Every stored document is chunked; an empty store leaves chunks.jsonl as it was."""
    from scripts.crawl import rebuild

    out, meta = tmp_path / "chunks.jsonl", tmp_path / "schemes.json"
    out.write_text("previous corpus\n", encoding="utf-8")
    assert rebuild(DocumentStore(tmp_path / "empty"), out, meta)["documents"] == 0
    assert out.read_text(encoding="utf-8") == "previous corpus\n"
    assert not meta.exists() and not (tmp_path / "chunks.jsonl.tmp").exists()

    store = DocumentStore(tmp_path / "store")
    store.put(_doc("https://a.gov.in/x", "## Eligibility\n" + "Must be an MSME. " * 40))
    store.put(_doc("https://a.gov.in/y", "## Benefits\n" + "Subsidy up to 25%. " * 40))
    stats = rebuild(store, out, meta)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert stats["documents"] == 2 and stats["chunks"] == len(lines) > 0
    assert {json.loads(x)["metadata"]["source_url"] for x in lines} == {
        "https://a.gov.in/x", "https://a.gov.in/y"}
    assert len(json.loads(meta.read_text(encoding="utf-8"))) == stats["schemes"]


# ---------------------------------------------------------- document formats --

def test_pdf_hyperlinks_become_discovery_links() -> None:
    """Links inside a guideline PDF reach the crawler's frontier."""
    pymupdf = pytest.importorskip("pymupdf")
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Eligibility guidelines for the scheme. " * 10)
    page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(72, 60, 300, 80),
                      "uri": "https://msme.gov.in/annexure-1.pdf"})
    payload = pdf.tobytes()

    # The link reader directly: running extract() would load Docling's layout
    # models, which is neither offline nor fast.
    assert _pdf_links(payload) == ["https://msme.gov.in/annexure-1.pdf"]
    assert _pdf_links(b"not a pdf") == [], "an unreadable PDF yields no links, not an error"


def test_pptx_is_extracted() -> None:
    """Scheme briefing decks are parsed instead of being skipped as unknown."""
    pptx = pytest.importorskip("pptx")
    deck = pptx.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[1])
    slide.shapes.title.text = "PMEGP briefing"
    slide.placeholders[1].text = "Margin money subsidy of 15 to 35 percent for new units."
    buffer = io.BytesIO()
    deck.save(buffer)

    doc = extract("https://kvic.gov.in/brief.pptx", buffer.getvalue(), "")
    assert doc.doc_format == "pptx"
    assert "Margin money subsidy" in doc.text


@pytest.mark.parametrize(("url", "ctype", "expected"), [
    ("https://x.gov.in/order.doc", "application/msword", "doc"),
    ("https://x.gov.in/annex.xls", "application/vnd.ms-excel", "xls"),
    ("https://x.gov.in/deck.ppt", "", "ppt"),
])
def test_legacy_office_formats_are_identified_not_misrouted(url: str, ctype: str,
                                                            expected: str) -> None:
    """Legacy binary Office files are reported as such, not as empty docx/xlsx."""
    payload = b"\xd0\xcf\x11\xe0 legacy compound file"
    assert detect_format(url, ctype, payload) == expected
    assert extract(url, payload, ctype).text == ""


def test_sitemaps_are_seeded() -> None:
    """A source's declared sitemaps join its seeds, so unlinked pages are reachable."""
    source = Source(key="t", name="t", domain="x.gov.in", tier=Tier.MINISTRY,
                    seeds=("https://x.gov.in/",), sitemaps=("https://x.gov.in/sitemap.xml",),
                    verified=True)
    SOURCES_BY_KEY["t"] = source
    try:
        assert [u for u, _ in seed_urls(["t"])] == ["https://x.gov.in/",
                                                     "https://x.gov.in/sitemap.xml"]
    finally:
        del SOURCES_BY_KEY["t"]


# ------------------------------------------------------- incremental indexing --

def _chunk(cid: str, kind: str) -> Chunk:
    return Chunk(id=cid, parent_id=None, text=f"Udyam registration {kind} {cid}",
                 metadata=ChunkMetadata("https://a.gov.in/x", "S", kind, states=("Gujarat",)))


def test_index_sync_skips_existing_and_prunes_stale() -> None:
    """Re-indexing embeds nothing twice, and removed chunks leave the index."""
    store = QdrantStore(client=QdrantClient(":memory:"), collection="sync", dim=4)
    store.ensure_collection(dim=4)
    old_child, parent = _chunk("a" * 32, "child"), _chunk("b" * 32, "parent")
    store.upsert_chunks([old_child], np.ones((1, 4), dtype=np.float32))
    store.upsert_sparse_only([parent])

    assert store.existing_chunk_ids(["a" * 32, "b" * 32, "c" * 32]) == {"a" * 32, "b" * 32}

    deleted = store.delete_points_not_in({"b" * 32})
    assert deleted == 1
    assert store.client.count("sync").count == 1

    hit = store.client.retrieve("sync", ids=[QdrantStore._to_point_id("b" * 32)],
                                with_payload=True)[0]
    assert hit.payload["states"] == ["Gujarat"], "scheme facts must reach the payload"


def test_chunks_jsonl_round_trips_new_metadata(tmp_path: Path) -> None:
    """New metadata fields survive the JSONL write/read used by ingest --index-only."""
    from scripts.ingest import load_chunks, write_chunks

    path = tmp_path / "chunks.jsonl"
    write_chunks([_chunk("d" * 32, "child")], path)
    loaded = load_chunks(path)
    assert list(loaded[0].metadata.states) == ["Gujarat"]
    row: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    assert row["metadata"]["discovery_method"] == ""
