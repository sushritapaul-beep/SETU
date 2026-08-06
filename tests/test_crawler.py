"""Offline tests for the crawler stack — no network, no browser, no downloads."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from ingestion import fetchers
from ingestion.crawl_state import CrawlState, content_hash, normalise_url
from ingestion.crawler import CrawlConfig, GovernmentCrawler
from ingestion.scheme_extractor import extract_scheme, find_deadlines, split_sections
from ingestion.sources import (
    OFFICIAL_EXCEPTIONS,
    SOURCES,
    SOURCES_BY_KEY,
    is_fetchable_url,
    is_official_url,
    seed_urls,
    source_for_url,
)

# --------------------------------------------------------------------------
# Official-source gate — the hardest requirement
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "https://www.myscheme.gov.in/schemes/mszcs",
        "https://msme.gov.in/all-schemes",
        "https://dcmsme.gov.in/schemes.htm",
        "https://nsap.nic.in/circular.do",
        "https://pmkisan.gov.in/Documents.aspx",
        "https://www.indiacode.nic.in/handle/123",
        "https://www.cgtmse.in/schemes",       # allowlisted statutory body
        "https://mudra.org.in/offerings",       # allowlisted PSU subsidiary
    ],
)
def test_official_urls_are_accepted(url: str) -> None:
    """Government domains and allowlisted statutory bodies pass the gate."""
    assert is_official_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://www.cleartax.in/s/msme-schemes",
        "https://bankbazaar.com/pmegp.html",
        "https://en.wikipedia.org/wiki/Udyam_Registration",
        "https://medium.com/@someone/indian-msme-schemes",
        "https://msme.gov.in.phishing-site.com/schemes",   # suffix-spoof attempt
        "https://notcgtmse.in/schemes",                     # substring of an exception
        "https://example.com/gov.in/schemes",               # gov.in only in the path
        "",
    ],
)
def test_unofficial_urls_are_rejected(url: str) -> None:
    """Aggregators, blogs and spoofed hosts never enter the corpus."""
    assert is_official_url(url) is False


def test_every_allowlist_exception_has_a_justification() -> None:
    """A domain may only bypass the TLD rule with a written reason."""
    for domain, reason in OFFICIAL_EXCEPTIONS.items():
        assert len(reason) > 20, f"{domain} lacks a justification"


def test_registered_sources_are_all_official() -> None:
    """No registered source may itself fail the officialness gate."""
    for source in SOURCES:
        for seed in source.seeds:
            assert is_official_url(seed), f"{source.key} seed is not official: {seed}"


def test_seed_urls_selects_by_key() -> None:
    """Seeds can be filtered to a subset of sources."""
    seeds = seed_urls(["msme_ministry"])
    assert seeds
    assert all(s.key == "msme_ministry" for _, s in seeds)
    with pytest.raises(KeyError):
        seed_urls(["not_a_real_source"])


def test_source_lookup_by_url() -> None:
    """A URL resolves back to the source that owns it."""
    source = source_for_url("https://www.myscheme.gov.in/schemes/ds")
    assert source is not None and source.key == "myscheme"


# --------------------------------------------------------------------------
# Format / path filtering
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://msme.gov.in/schemes", True),
        ("https://msme.gov.in/guide.pdf", True),
        ("https://msme.gov.in/data.xlsx", True),
        ("https://msme.gov.in/list.csv", True),
        ("https://msme.gov.in/api/schemes.json", True),
        ("https://msme.gov.in/sitemap.xml", True),
        ("https://msme.gov.in/bundle.zip", True),
        ("https://msme.gov.in/logo.png", False),
        ("https://msme.gov.in/video.mp4", False),
        ("https://msme.gov.in/login", False),
        ("https://msme.gov.in/user/dashboard", False),
    ],
)
def test_fetchable_url_filtering(url: str, expected: bool) -> None:
    """Only useful formats on non-transactional paths are queued."""
    assert is_fetchable_url(url) is expected


# --------------------------------------------------------------------------
# Crawl state: dedupe, retries, resumability
# --------------------------------------------------------------------------

@pytest.fixture
def state(tmp_path: Path) -> CrawlState:
    """A CrawlState backed by a temporary database."""
    with CrawlState(tmp_path / "crawl.db") as crawl_state:
        yield crawl_state


def test_url_normalisation_collapses_equivalents() -> None:
    """Trivially different URL spellings resolve to one frontier entry."""
    canonical = normalise_url("HTTPS://MSME.GOV.IN/Schemes/#section")
    assert canonical == "https://msme.gov.in/Schemes"
    assert normalise_url("https://msme.gov.in:443/a") == normalise_url("https://msme.gov.in/a")
    # Query strings survive: portals encode scheme ids there.
    assert "?id=7" in normalise_url("https://msme.gov.in/s?id=7")


def test_urls_are_added_once(state: CrawlState) -> None:
    """The same URL cannot be queued twice."""
    assert state.add_url("https://msme.gov.in/a", 0, "msme_ministry") is True
    assert state.add_url("https://msme.gov.in/a/", 0, "msme_ministry") is False
    assert len(state.next_batch(10)) == 1


def test_frontier_is_breadth_first(state: CrawlState) -> None:
    """Shallower URLs are handed out first, so coverage stays broad."""
    state.add_url("https://msme.gov.in/deep", 3, None)
    state.add_url("https://msme.gov.in/shallow", 0, None)

    batch = state.next_batch(2)
    assert batch[0].depth == 0


def test_identical_content_from_two_urls_deduplicates(state: CrawlState) -> None:
    """The same scheme on two portals is stored once."""
    text_hash = content_hash("Eligibility: must hold Udyam registration.")
    for url in ("https://msme.gov.in/a", "https://www.myscheme.gov.in/b"):
        state.add_url(url, 0, None)

    first = state.mark_done(
        "https://msme.gov.in/a", http_status=200, content_type="text/html",
        text_hash=text_hash, title="A", etag=None, last_modified=None, size=100,
    )
    second = state.mark_done(
        "https://www.myscheme.gov.in/b", http_status=200, content_type="text/html",
        text_hash=text_hash, title="B", etag=None, last_modified=None, size=100,
    )

    assert first is True, "first sighting is new content"
    assert second is False, "second sighting is a duplicate"
    assert state.stats()["unique_documents"] == 1


def test_content_hash_ignores_whitespace_reformatting() -> None:
    """Cosmetic reflowing must not defeat duplicate detection."""
    assert content_hash("a  b\n\nc") == content_hash("a b c")


def test_transient_failures_retry_then_become_terminal(state: CrawlState) -> None:
    """A URL is retried up to the limit, then dropped from the frontier."""
    state.add_url("https://msme.gov.in/flaky", 0, None)

    for _ in range(3):
        state.mark_failed("https://msme.gov.in/flaky", "timeout")

    assert state.next_batch(10) == []
    assert state.stats()["by_status"].get("failed") == 1


def test_permanent_failures_are_not_retried(state: CrawlState) -> None:
    """A 404 is terminal on the first attempt."""
    state.add_url("https://msme.gov.in/gone", 0, None)
    state.mark_failed("https://msme.gov.in/gone", "HTTP 404", permanent=True)

    assert state.next_batch(10) == []


def test_state_survives_reopen(tmp_path: Path) -> None:
    """Crawl progress persists, so an interrupted run resumes."""
    db = tmp_path / "crawl.db"
    with CrawlState(db) as first:
        first.add_url("https://msme.gov.in/a", 0, None)
    with CrawlState(db) as second:
        assert len(second.next_batch(10)) == 1


# --------------------------------------------------------------------------
# Format extraction
# --------------------------------------------------------------------------

def test_html_extraction_yields_text_and_links() -> None:
    """HTML gives readable text, a title and absolute links."""
    html = b"""<html><head><title>ZED Scheme</title></head><body>
    <script>var x=1;</script><nav>menu</nav>
    <h2>Eligibility</h2><p>Must hold Udyam registration.</p>
    <a href="/guide.pdf">Guide</a><a href="https://msme.gov.in/x">X</a>
    </body></html>"""

    doc = fetchers.extract_html("https://msme.gov.in/zed", html, "text/html")

    assert doc.title == "ZED Scheme"
    assert "Udyam registration" in doc.text
    assert "var x=1" not in doc.text, "scripts must be stripped"
    assert "https://msme.gov.in/guide.pdf" in doc.links


def test_csv_and_json_and_xml_extraction() -> None:
    """Tabular and structured formats flatten to searchable text."""
    csv_doc = fetchers.extract_csv(
        "https://data.gov.in/x.csv", b"scheme,limit\nPMEGP,2500000\n", "text/csv"
    )
    assert "PMEGP" in csv_doc.text

    json_doc = fetchers.extract_json(
        "https://api.gov.in/s.json", json.dumps({"scheme": "ZED"}).encode(), "application/json"
    )
    assert "ZED" in json_doc.text
    assert json_doc.extra["parsed"]["scheme"] == "ZED"

    sitemap = b"""<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://msme.gov.in/a</loc></url></urlset>"""
    xml_doc = fetchers.extract_xml("https://msme.gov.in/sitemap.xml", sitemap, "application/xml")
    assert "https://msme.gov.in/a" in xml_doc.links, "sitemap URLs feed the frontier"


def test_zip_extraction_reads_members() -> None:
    """Documents inside an archive are extracted, not just the container."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("guidelines.txt", "Eligibility: micro and small enterprises only.")
    doc = fetchers.extract_zip("https://msme.gov.in/b.zip", buffer.getvalue(), "application/zip")

    assert "micro and small enterprises" in doc.text


def test_format_detection_trusts_magic_bytes_over_headers() -> None:
    """A PDF mislabelled as HTML is still detected as a PDF."""
    assert fetchers.detect_format("https://x.gov.in/a", "text/html", b"%PDF-1.7 junk") == "pdf"
    assert fetchers.detect_format("https://x.gov.in/a.pdf", "", b"") == "pdf"
    assert fetchers.detect_format("https://x.gov.in/a", "text/csv", b"a,b") == "csv"


def test_unknown_format_degrades_quietly() -> None:
    """An unsupported payload yields empty text rather than raising."""
    doc = fetchers.extract("https://x.gov.in/a.bin", b"\x00\x01", "application/octet-stream")
    assert doc.text == ""
    assert doc.is_useful is False


# --------------------------------------------------------------------------
# Structured metadata
# --------------------------------------------------------------------------

SCHEME_TEXT = """# MSME Sustainable (ZED) Certification Scheme
Ministry of Micro, Small and Medium Enterprises

## Details
Promotes Zero Defect Zero Effect practices among MSMEs.

## Benefits
Certification Level 1: BRONZE: Rs 10,000/-
Joining reward of Rs 10,000/-

## Eligibility
* Should be a Micro, Small and Medium Enterprises (MSMEs) unit.
* Should be registered in Udyam portal.

## Application Process
Step 1: Visit the official website.

## Documents Required
1. Udyam Registration Number of the Enterprise.

## Frequently Asked Questions
Last date to apply is 31.03.2027 for the current cycle.
"""


def test_sections_split_into_canonical_keys() -> None:
    """Heading variants map onto stable field names."""
    sections = split_sections(SCHEME_TEXT)

    assert "Udyam portal" in sections["eligibility"]
    assert "BRONZE" in sections["benefits"]
    assert "Udyam Registration Number" in sections["documents_required"]
    assert "Visit the official website" in sections["application_process"]


def test_scheme_metadata_extraction_is_complete_and_provenanced() -> None:
    """A well-formed scheme page yields full metadata with provenance."""
    metadata = extract_scheme(
        SCHEME_TEXT,
        scheme_id="mszcs",
        name="MSME Sustainable (ZED) Certification Scheme",
        source_url="https://www.myscheme.gov.in/schemes/mszcs",
        retrieved_at="2026-08-06T12:00:00+00:00",
        doc_format="html",
        content_hash="abc123",
    )

    assert metadata.completeness == 1.0
    assert metadata.implementing_authority.startswith("Ministry of Micro")
    assert any("10,000" in a for a in metadata.amounts)
    assert metadata.retrieved_at == "2026-08-06T12:00:00+00:00"
    assert metadata.content_hash == "abc123"
    assert metadata.to_dict()["scheme_id"] == "mszcs"


def test_deadlines_need_deadline_context() -> None:
    """Only dates near deadline language are reported as deadlines."""
    assert find_deadlines("Last date to apply is 31.03.2027.")
    # A launch date is a date, but it is not a deadline.
    assert find_deadlines("The scheme was launched on 01.04.2022.") == []


def test_missing_sections_stay_none_rather_than_guessed() -> None:
    """Absent information is reported as missing, never invented."""
    metadata = extract_scheme(
        "## Details\nA scheme with no other sections.",
        scheme_id="x", name="X", source_url="https://msme.gov.in/x",
        retrieved_at="2026-08-06T00:00:00+00:00",
    )

    assert metadata.eligibility is None
    assert metadata.benefits is None
    assert metadata.deadlines == []
    assert metadata.completeness == 0.2


# --------------------------------------------------------------------------
# Crawler wiring
# --------------------------------------------------------------------------

COMPENDIUM_TEXT = """Ministry of Micro, Small and Medium Enterprises
List of Existing Schemes

1. Prime Minister's Employment Generation Programme (PMEGP)
Details
Credit-linked subsidy programme for generating self-employment.
Eligibility
Any individual above 18 years of age. Beneficiary must not have availed
assistance under any other similar scheme. VIRTUALLY all rural and urban
applicants may apply through the online portal maintained by KVIC.
Benefits
Subsidy of 15% to 35% of the project cost depending on category and area.
Documents Required
Aadhaar, project report and caste certificate where applicable.

2. Credit Guarantee Scheme for Micro and Small Enterprises (CGTMSE)
Details
Collateral-free credit guarantee for micro and small enterprises seeking
working capital and term loans from member lending institutions.
Eligibility
New and existing micro and small enterprises engaged in manufacturing or
service activity, excluding retail trade, educational institutions and
self-help groups as notified from time to time.
Benefits
Guarantee cover of up to 85% of the sanctioned credit facility.

3. MSME Sustainable ZED Certification Scheme
Details
Encourages Zero Defect Zero Effect manufacturing practices across the MSME
sector through structured assessment and certification.
Eligibility
Should be a Micro, Small or Medium Enterprise registered on the Udyam portal
and engaged in manufacturing activity as declared in the registration.
Benefits
Joining reward of Rs 10,000 and graded subsidy on certification cost.
"""


def test_compendium_splits_into_individual_schemes() -> None:
    """A ministry's multi-scheme PDF yields one record per scheme."""
    from ingestion.scheme_extractor import extract_compendium, split_compendium

    sections = split_compendium(COMPENDIUM_TEXT)
    assert len(sections) == 3

    titles = [t for t, _ in sections]
    assert any("PMEGP" in t or "Employment Generation" in t for t in titles)
    assert any("ZED" in t for t in titles)

    records = extract_compendium(
        COMPENDIUM_TEXT,
        source_url="https://champions.gov.in/Docs/MSME_Existing_Schemes.pdf",
        retrieved_at="2026-08-06T12:00:00+00:00",
        doc_format="pdf",
    )
    assert len(records) == 3
    # Each scheme now carries its own eligibility, which single-scheme
    # extraction could not produce from a compendium.
    assert all(r.eligibility for r in records)
    assert all(r.benefits for r in records)
    # Ids are unique and the source URL is anchored per scheme.
    assert len({r.scheme_id for r in records}) == 3
    assert all("#" in r.source_url for r in records)


def test_single_scheme_document_is_not_treated_as_a_compendium() -> None:
    """A normal scheme page falls through to single-scheme extraction."""
    from ingestion.scheme_extractor import extract_compendium

    assert extract_compendium(
        SCHEME_TEXT,
        source_url="https://www.myscheme.gov.in/schemes/mszcs",
        retrieved_at="2026-08-06T12:00:00+00:00",
    ) == []


def test_contents_page_entries_are_not_mistaken_for_schemes() -> None:
    """Short table-of-contents lines must not become scheme records."""
    from ingestion.scheme_extractor import split_compendium

    contents_only = "\n".join(
        f"{i}. Some Support Scheme For Enterprises Number {i}" for i in range(1, 8)
    )
    assert split_compendium(contents_only) == []


def _row(*cells: str) -> str:
    """Build one markdown table row, keeping the fixture source readable."""
    return "|" + "|".join(cells) + "|"


# Mirrors the real champions.gov.in PDF: a scheme table with <br> inside cells,
# headers repeated at page breaks, and continuation rows with empty lead cells.
_HEADER = _row(
    "**#**<br>**Name of**<br>**Scheme**", "**Objective**",
    "**Key Benefits**", "**Scheme**<br>**applicable for**",
)
_SEPARATOR = _row("---", "---", "---", "---")

TABLE_COMPENDIUM = "\n".join([
    "# MSME Schemes information for CHAMPIONS portal",
    "",
    _HEADER,
    _SEPARATOR,
    _row(
        "1.  Prime<br>Minister's<br>Employment<br>Generation<br>Programme<br>(PMEGP)",
        "Encourage<br>first<br>generation<br>entrepreneurs",
        "Subsidy of 15% to 35% of<br>the project cost",
        "New micro<br>enterprises<br>in rural and<br>urban areas",
    ),
    _row(
        "2.  Credit<br>Guarantee<br>Scheme for<br>Micro & Small<br>Enterprises<br>(CGTMSE)",
        "Credit<br>support<br>without<br>collateral",
        "Guarantee cover up to 85%<br>of the sanctioned credit",
        "Micro and<br>small<br>enterprises",
    ),
    # Page break: the header repeats mid-document.
    _HEADER,
    _SEPARATOR,
    # Continuation of CGTMSE — empty leading cells, no serial number.
    _row("", "", "Maximum subsidy would be<br>15% of the project cost", ""),
    _row(
        "3.  MSME<br>Sustainable<br>(ZED)<br>Certification<br>Scheme",
        "Promote<br>Zero Defect<br>Zero Effect",
        "Joining reward of Rs 10,000",
        "MSMEs<br>registered on<br>Udyam portal",
    ),
    "",
])


def test_tabular_compendium_yields_one_record_per_row() -> None:
    """A scheme table — the common government PDF shape — parses row by row."""
    from ingestion.scheme_extractor import extract_table_compendium

    records = extract_table_compendium(
        TABLE_COMPENDIUM,
        source_url="https://champions.gov.in/Docs/MSME_Existing_Schemes.pdf",
        retrieved_at="2026-08-06T12:00:00+00:00",
        doc_format="pdf",
    )

    assert len(records) == 3
    names = " | ".join(r.name for r in records)
    assert "PMEGP" in names and "CGTMSE" in names and "ZED" in names

    pmegp = records[0]
    assert "Encourage" in pmegp.details
    assert "15% to 35%" in pmegp.benefits
    assert "rural and urban" in pmegp.eligibility
    assert pmegp.scheme_id and "#" in pmegp.source_url
    # <br> markup must not survive into the extracted values.
    assert "<br>" not in pmegp.benefits


def test_repeated_table_headers_do_not_become_schemes() -> None:
    """Headers repeat at every PDF page break; they are not scheme rows."""
    from ingestion.scheme_extractor import extract_table_compendium

    records = extract_table_compendium(
        TABLE_COMPENDIUM,
        source_url="https://x.gov.in/a.pdf",
        retrieved_at="2026-08-06T12:00:00+00:00",
    )
    assert not any("Name of" in r.name for r in records)


def test_continuation_rows_attach_to_the_previous_scheme() -> None:
    """A row with an empty leading cell continues the scheme above it."""
    from ingestion.scheme_extractor import extract_table_compendium

    records = extract_table_compendium(
        TABLE_COMPENDIUM,
        source_url="https://x.gov.in/a.pdf",
        retrieved_at="2026-08-06T12:00:00+00:00",
    )
    cgtmse = next(r for r in records if "CGTMSE" in r.name)
    assert "Maximum subsidy" in cgtmse.benefits


def test_welfare_ministries_are_registered_and_verified() -> None:
    """The welfare half of the requirement is covered by verified sources."""
    verified = {s.key for s in SOURCES if s.verified}

    assert {"socialjustice", "minorityaffairs"} <= verified


def test_robots_disallowed_source_stays_unverified() -> None:
    """wcd.gov.in disallows all crawling; it must never be crawled by default.

    Kept in the registry rather than deleted so the reason survives — otherwise
    someone re-adds it, and the crawler silently skips it with no explanation.
    """
    wcd = SOURCES_BY_KEY["wcd"]

    assert wcd.verified is False
    assert "robots.txt" in wcd.notes.lower()
    assert wcd.key not in {s.key for _, s in seed_urls()}


def test_insecure_ssl_is_opt_in_and_justified() -> None:
    """Relaxed certificate checking is per-source and must state a reason."""
    relaxed = [s for s in SOURCES if s.insecure_ssl]

    assert len(relaxed) <= 2, "relaxing TLS verification must stay exceptional"
    for source in relaxed:
        assert "certificate" in source.notes.lower(), f"{source.key} lacks a reason"


def test_permanent_transport_errors_are_not_retried() -> None:
    """Certificate and DNS failures will not fix themselves between attempts."""
    from ingestion.crawler import _is_permanent_transport_error

    ssl_error = Exception(
        "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate"
    )
    dns_error = Exception("[Errno 11001] getaddrinfo failed")
    timeout = TimeoutError("read timed out")

    assert _is_permanent_transport_error(ssl_error) is True
    assert _is_permanent_transport_error(dns_error) is True
    # A timeout genuinely might succeed next time.
    assert _is_permanent_transport_error(timeout) is False


def test_crawler_refuses_unofficial_seed_urls(tmp_path: Path) -> None:
    """A non-official seed is rejected before any request is made."""
    import asyncio

    crawler = GovernmentCrawler(
        config=CrawlConfig(max_pages=1), state_path=tmp_path / "c.db"
    )
    with pytest.raises(ValueError, match="non-official"):
        asyncio.run(crawler.crawl(source_keys=[], extra_urls=["https://cleartax.in/x"]))


@pytest.mark.parametrize(
    "final_url,text,expected",
    [
        # Redirected to an error page while still returning HTTP 200.
        ("https://dcmsme.gov.in/ErrorPages/FileNotFound.htm", "x" * 300, True),
        ("https://x.gov.in/404.html", "x" * 300, True),
        # Short page whose text is an error message.
        ("https://x.gov.in/schemes", "File Not Found. Return to home.", True),
        ("https://x.gov.in/s", "The page you are looking for is unavailable.", True),
        # A long real page that merely mentions the phrase is not an error.
        ("https://x.gov.in/faq", "Eligibility details. " * 200 + " page not found", False),
        ("https://x.gov.in/schemes", "Eligibility: must hold Udyam registration. " * 20, False),
    ],
)
def test_soft_404_detection(final_url: str, text: str, expected: bool) -> None:
    """Error pages served with HTTP 200 are caught, real content is not."""
    from ingestion.crawler import looks_like_soft_404

    assert (looks_like_soft_404(final_url, text) is not None) is expected


def test_unverified_sources_are_excluded_from_bulk_crawls() -> None:
    """Sources whose live probe failed do not consume the page budget."""
    verified_only = {s.key for _, s in seed_urls()}
    everything = {s.key for _, s in seed_urls(include_unverified=True)}

    assert verified_only, "at least one source must be verified"
    assert verified_only < everything, "some sources are known-unreachable"
    # But naming one explicitly still works, so a fix can be tested.
    assert seed_urls(["kvic"])


def test_crawl_config_defaults_are_polite() -> None:
    """Default limits must not hammer government infrastructure."""
    config = CrawlConfig()

    assert config.respect_robots is True
    assert config.per_domain_concurrency <= 3
    assert config.same_domain_only is True
    assert config.max_retries >= 2
