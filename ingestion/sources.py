"""Registry of verified official government sources, and the gate that enforces them.

This module is the single point where "is this source official?" is decided. The
crawler refuses any URL that does not pass `is_official_url`, so unofficial
aggregators, blogs and coaching sites can never enter the corpus regardless of
how they were linked.

Two mechanisms, because neither alone is sufficient:

1. Government TLD suffixes (.gov.in, .nic.in) cover the overwhelming majority.
2. An explicit allowlist covers statutory bodies and public-sector undertakings
   that predate the .gov.in convention and legitimately sit on other domains
   (CGTMSE, SIDBI, MUDRA, NSIC). Each carries a justification comment; nothing
   goes in this list without one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Suffix match, so any subdomain of a government domain qualifies
# (dcmsme.gov.in, pmkisan.gov.in, tn.gov.in, ...).
OFFICIAL_SUFFIXES: tuple[str, ...] = (
    ".gov.in",
    ".nic.in",
    "india.gov.in",
)

# Statutory bodies and PSUs on non-.gov.in domains. Each entry names the body and
# why it counts as official; an entry without justification is a bug.
OFFICIAL_EXCEPTIONS: dict[str, str] = {
    "cgtmse.in":
        "Credit Guarantee Fund Trust for Micro and Small Enterprises — "
        "trust established by the Government of India and SIDBI",
    "sidbi.in":
        "Small Industries Development Bank of India — statutory body "
        "constituted under the SIDBI Act 1989",
    "mudra.org.in":
        "MUDRA Ltd — wholly owned subsidiary of SIDBI and the implementing "
        "agency for the PMMY scheme",
    "nsic.co.in":
        "National Small Industries Corporation — Government of India public "
        "sector undertaking under the Ministry of MSME",
    "kvic.org.in":
        "Khadi and Village Industries Commission — statutory body constituted "
        "under the KVIC Act 1956",
}

# Formats worth fetching. Anything else (images, video, archives of binaries) is
# discovered but skipped, so the frontier does not fill with noise.
FETCHABLE_EXTENSIONS: frozenset[str] = frozenset(
    {
        "",  # extensionless URLs are usually HTML routes
        ".html", ".htm", ".xhtml", ".aspx", ".php", ".jsp",
        ".pdf",
        ".doc", ".docx",
        ".xls", ".xlsx",
        ".ppt", ".pptx",
        ".csv",
        ".json",
        ".xml",
        ".zip",
        ".txt",
    }
)

# URL fragments that indicate application forms, logins and dashboards rather
# than scheme information. Crawling these wastes budget and can trip rate limits
# on transactional infrastructure.
EXCLUDED_PATH_HINTS: tuple[str, ...] = (
    "/login", "/signin", "/signup", "/register/", "/logout",
    "/captcha", "/payment", "/checkout",
    "/dashboard", "/admin",
    "/search?", "/print", "javascript:", "mailto:", "tel:",
)


class Tier(Enum):
    """How close a source is to being the authoritative text of a scheme."""

    AGGREGATOR = "aggregator"        # myScheme, india.gov.in — broad, structured
    MINISTRY = "ministry"            # the ministry that owns the scheme
    IMPLEMENTER = "implementer"      # agency running it (KVIC, NSIC, CGTMSE)
    STATE = "state"                  # state/UT government portals
    LEGAL = "legal"                  # Acts, gazette notifications


@dataclass(frozen=True)
class Source:
    """One official source the crawler is permitted to traverse."""

    key: str
    name: str
    domain: str
    tier: Tier
    seeds: tuple[str, ...]
    # myScheme and NSWS render scheme content client-side; a plain HTTP GET
    # returns an empty application shell, so these need a real browser.
    requires_js: bool = False
    # Politeness: government infrastructure is not a CDN.
    max_concurrency: int = 2
    delay_seconds: float = 1.0
    # False when a live probe could not reach the seed (dead path, DNS failure,
    # robots.txt refusal). Unverified sources are skipped unless explicitly
    # requested, so a crawl does not spend its page budget on known failures.
    verified: bool = False
    # Some government hosts serve a self-signed or incomplete certificate chain.
    # Disabling verification is a deliberate, per-source exception: the domain is
    # already on the official allowlist, and the content is public information.
    # It is never a global default — each use needs a documented reason.
    insecure_ssl: bool = False
    notes: str = ""
    sitemaps: tuple[str, ...] = field(default_factory=tuple)


# Every seed below was probed live on 2026-08-06. `verified` records what that
# probe returned, so nobody has to rediscover that a URL is dead. Sources that
# did not resolve are kept with verified=False and excluded from crawls by
# default — deleting them would lose the research, but crawling them wastes the
# budget on guaranteed failures.
SOURCES: tuple[Source, ...] = (
    Source(
        key="myscheme",
        name="myScheme — National Scheme Discovery Platform",
        domain="www.myscheme.gov.in",
        tier=Tier.AGGREGATOR,
        seeds=("https://www.myscheme.gov.in/search",),
        requires_js=True,
        verified=True,
        notes="MeitY/NeGD. 4700+ central and state schemes, MSME and welfare.",
    ),
    Source(
        key="india_gov",
        name="National Portal of India — Schemes",
        domain="www.india.gov.in",
        tier=Tier.AGGREGATOR,
        seeds=("https://www.india.gov.in/my-government/schemes",),
        verified=True,
        notes="Returns 403 to non-browser user agents; 200 with a browser UA.",
    ),
    Source(
        key="msme_ministry",
        name="Ministry of Micro, Small and Medium Enterprises",
        domain="msme.gov.in",
        tier=Tier.MINISTRY,
        seeds=("https://msme.gov.in/all-schemes",),
        requires_js=True,
        verified=True,
        notes="403 without a browser UA, and the page body is JS-rendered "
              "(HTTP fetch yields 0 characters), so it needs the browser path.",
    ),
    # ---------------------------------------------------------------- welfare --
    # These close the welfare half of the requirement. All three serve static
    # HTML with large link graphs, so recursive discovery works well on them.
    Source(
        key="socialjustice",
        name="Ministry of Social Justice and Empowerment",
        domain="socialjustice.gov.in",
        tier=Tier.MINISTRY,
        seeds=("https://socialjustice.gov.in/schemes", "https://socialjustice.gov.in/"),
        verified=True,
        notes="Welfare: SC/ST/OBC, disability, senior citizens, substance abuse. "
              "Static HTML, ~63-90 links per page.",
    ),
    Source(
        key="minorityaffairs",
        name="Ministry of Minority Affairs",
        domain="minorityaffairs.gov.in",
        tier=Tier.MINISTRY,
        seeds=("https://minorityaffairs.gov.in/",),
        verified=True,
        # Measured: the host presents a self-signed certificate in its chain, so
        # strict verification fails. The domain is government-owned and the
        # content is public, so verification is relaxed for this source only.
        insecure_ssl=True,
        notes="Welfare: minority scholarships, skill and livelihood schemes. "
              "Serves a self-signed certificate chain.",
    ),
    Source(
        key="kvic",
        name="Khadi and Village Industries Commission (PMEGP)",
        domain="kvic.gov.in",
        tier=Tier.IMPLEMENTER,
        seeds=("https://kvic.gov.in/",),
        # Measured 2026-10-05: robots.txt is "User-agent: * / Disallow: /", so
        # every page would be refused at fetch time anyway. PMEGP itself is on
        # myScheme and reached through the API harvest instead.
        verified=False,
        notes="EXCLUDED BY robots.txt — the site disallows all crawling. "
              "PMEGP content comes from the myScheme API harvest.",
    ),
    Source(
        key="champions",
        name="MSME CHAMPIONS — consolidated scheme document",
        domain="champions.gov.in",
        tier=Tier.IMPLEMENTER,
        seeds=("https://champions.gov.in/Docs/MSME_Existing_Schemes.pdf",),
        verified=True,
        notes="Verified 609KB application/pdf. Highest-value single MSME document.",
    ),
    Source(
        key="indiacode",
        name="India Code — MSMED Act 2006",
        domain="www.indiacode.nic.in",
        tier=Tier.LEGAL,
        seeds=("https://www.indiacode.nic.in/handle/123456789/2013",),
        verified=True,
        notes="Statutory basis for the MSME turnover and investment thresholds.",
    ),
    # --- probed and found unusable; kept for the record, excluded by default ---
    Source(
        key="dcmsme",
        name="Development Commissioner, MSME",
        domain="dcmsme.gov.in",
        tier=Tier.MINISTRY,
        seeds=("https://dcmsme.gov.in/",),
        requires_js=True,
        verified=False,
        notes="JS-rendered and /schemes.htm soft-404s. Needs a correct entry path.",
    ),
    Source(
        key="cgtmse",
        name="Credit Guarantee Fund Trust for Micro and Small Enterprises",
        domain="www.cgtmse.in",
        tier=Tier.IMPLEMENTER,
        seeds=("https://www.cgtmse.in/",),
        verified=True,
        notes="Re-probed 2026-10-05: homepage serves 8k chars of HTML with 42 "
              "links over plain HTTP; robots.txt allows crawling.",
    ),
    Source(
        key="pmkisan",
        name="PM-KISAN Samman Nidhi",
        domain="pmkisan.gov.in",
        tier=Tier.IMPLEMENTER,
        seeds=("https://pmkisan.gov.in/",),
        verified=False,
        notes="Welfare. Homepage resolves (2026-10-05) but serves a 21-char JS "
              "shell over HTTP; scheme text is on myScheme.",
    ),
    Source(
        key="udyam",
        name="Udyam Registration",
        domain="udyamregistration.gov.in",
        tier=Tier.IMPLEMENTER,
        seeds=("https://udyamregistration.gov.in/",),
        verified=False,
        notes="MSME classification thresholds. Homepage resolves (2026-10-05) but "
              "serves ~230 chars over HTTP; content is JS-rendered.",
    ),
    Source(
        key="tn_msme",
        name="Tamil Nadu MSME Department",
        domain="www.msmeonline.tn.gov.in",
        tier=Tier.STATE,
        seeds=("https://www.msmeonline.tn.gov.in/",),
        verified=False,
        notes="State coverage for the Tamil use case. Homepage resolves "
              "(2026-10-05) but serves ~70 chars over HTTP; JS-rendered. "
              "TN schemes are reached via the myScheme API instead.",
    ),
    Source(
        key="wcd",
        name="Ministry of Women and Child Development",
        domain="wcd.gov.in",
        tier=Tier.MINISTRY,
        seeds=("https://wcd.gov.in/",),
        # 2026-08-06 robots.txt disallowed all crawling; re-probed 2026-10-05 it
        # reads "Allow: /" and the homepage serves real HTML (113 links). The
        # crawler re-checks robots.txt on every run, so a reversal is honoured.
        verified=True,
        notes="Welfare: women and child schemes. robots.txt allows crawling "
              "as of 2026-10-05 (it disallowed all crawling in August).",
    ),
    Source(
        key="nsap",
        name="National Social Assistance Programme",
        domain="nsap.nic.in",
        tier=Tier.MINISTRY,
        seeds=("https://nsap.nic.in/",),
        verified=False,
        notes="Welfare: pensions. Neither nsap.nic.in nor nsap.gov.in resolved.",
    ),
    Source(
        key="rural",
        name="Ministry of Rural Development",
        domain="rural.gov.in",
        tier=Tier.MINISTRY,
        seeds=("https://rural.gov.in/",),
        verified=False,
        notes="Welfare: MGNREGA, NRLM, PMAY-G. rural.nic.in does not resolve; "
              "rural.gov.in resolves but returns a near-empty body.",
    ),
    Source(
        key="pmjay",
        name="Ayushman Bharat PM-JAY",
        domain="pmjay.gov.in",
        tier=Tier.IMPLEMENTER,
        seeds=("https://pmjay.gov.in/",),
        verified=False,
        notes="Welfare: health insurance. Connection timed out during probing.",
    ),
)

SOURCES_BY_KEY: dict[str, Source] = {s.key: s for s in SOURCES}


def is_official_url(url: str) -> bool:
    """Whether a URL belongs to a verified official government source.

    Args:
        url: Absolute URL to test.

    Returns:
        True if the host is a government domain or an allowlisted statutory body.
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False

    if any(host == s.lstrip(".") or host.endswith(s) for s in OFFICIAL_SUFFIXES):
        return True
    return any(host == d or host.endswith("." + d) for d in OFFICIAL_EXCEPTIONS)


def is_fetchable_url(url: str) -> bool:
    """Whether a URL points at a format worth downloading.

    Args:
        url: Absolute URL to test.

    Returns:
        True if the extension is fetchable and the path is not transactional.
    """
    lowered = url.lower()
    if any(hint in lowered for hint in EXCLUDED_PATH_HINTS):
        return False

    path = urlparse(lowered).path
    # Take the last dot segment only if it looks like a real extension; paths
    # such as /scheme.v2/details must not be read as ".v2/details".
    suffix = ""
    if "." in path.rsplit("/", 1)[-1]:
        suffix = "." + path.rsplit(".", 1)[-1]
        if len(suffix) > 6 or "/" in suffix:
            suffix = ""
    return suffix in FETCHABLE_EXTENSIONS


def source_for_url(url: str) -> Source | None:
    """Find the registered source that owns a URL, if any.

    Args:
        url: Absolute URL.

    Returns:
        The matching Source, or None when the URL is official but not from a
        registered seed domain (still crawlable, just unattributed).
    """
    host = (urlparse(url).hostname or "").lower()
    for source in SOURCES:
        domain = source.domain.lower()
        if host == domain or host.endswith("." + domain) or domain.endswith("." + host):
            return source
    return None


def seed_urls(
    keys: list[str] | None = None, include_unverified: bool = False
) -> list[tuple[str, Source]]:
    """Collect starting URLs for the selected sources.

    Args:
        keys: Source keys to include. None selects every registered source.
        include_unverified: Also seed sources whose live probe failed. Naming a
            source explicitly always includes it — the filter only applies when
            selecting them all.

    Returns:
        (url, source) pairs to seed the crawl frontier.

    Raises:
        KeyError: If a requested key is not registered.
    """
    if keys is not None:
        selected = tuple(SOURCES_BY_KEY[k] for k in keys)
    else:
        selected = tuple(s for s in SOURCES if s.verified or include_unverified)
    # Sitemaps are seeds too: extract_xml turns each <loc> into a link, so a
    # sitemap enumerates pages that navigation never links to.
    return [(url, source) for source in selected for url in (*source.seeds, *source.sitemaps)]
