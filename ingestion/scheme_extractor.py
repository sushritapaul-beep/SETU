"""Normalise a scraped document into structured scheme metadata.

Government scheme pages are remarkably consistent in their section headings —
Details, Benefits, Eligibility, Application Process, Documents Required — because
most are generated from the same templates. That regularity makes rule-based
section extraction reliable and, unlike an LLM pass, free, deterministic and
auditable.

Anything the rules cannot find is left as None rather than guessed. A missing
deadline is recoverable; an invented one is not.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Canonical field -> heading variants seen across portals. Matched
# case-insensitively against heading lines.
SECTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "details": ("details", "about the scheme", "overview", "introduction", "objective"),
    "benefits": ("benefits", "financial assistance", "quantum of assistance",
                 "incentives", "subsidy", "assistance provided"),
    "eligibility": ("eligibility", "eligibility criteria", "who can apply",
                    "target group", "beneficiaries"),
    "application_process": ("application process", "how to apply", "procedure",
                            "application procedure", "steps to apply"),
    "documents_required": ("documents required", "documents needed",
                           "required documents", "list of documents"),
    "exclusions": ("exclusions", "not eligible", "ineligible", "exceptions"),
    "faq": ("frequently asked questions", "faq", "faqs"),
    "references": ("sources and references", "references", "notification",
                   "guidelines", "circular"),
}

# Implementing authority, usually rendered as a breadcrumb or byline.
AUTHORITY_PATTERNS: tuple[str, ...] = (
    r"Ministry\s+[Oo]f\s+[A-Z][\w\s,&()-]{3,80}",
    r"Department\s+[Oo]f\s+[A-Z][\w\s,&()-]{3,80}",
    r"(?:Directorate|Commissionerate|Corporation|Board|Commission)\s+[Oo]f\s+[A-Z][\w\s,&()-]{3,80}",
)

# Deadlines and validity windows. Indian portals mix dd.mm.yyyy and dd/mm/yyyy.
DATE_PATTERNS: tuple[str, ...] = (
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{4}\b",
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
)

DEADLINE_HINTS: tuple[str, ...] = (
    "last date", "deadline", "apply before", "valid till", "valid up to",
    "closing date", "within one year", "on or before",
)

# Monetary amounts, in the Indian grouping convention.
AMOUNT_PATTERN = r"(?:₹|Rs\.?|INR)\s?[\d,]+(?:\.\d+)?(?:\s?(?:lakh|crore|lakhs|crores))?"


@dataclass
class SchemeMetadata:
    """Structured view of one scheme, extracted from an official document."""

    scheme_id: str
    name: str
    source_url: str
    retrieved_at: str
    doc_format: str = "html"

    details: str | None = None
    benefits: str | None = None
    eligibility: str | None = None
    application_process: str | None = None
    documents_required: str | None = None
    exclusions: str | None = None
    faq: str | None = None
    references: str | None = None

    implementing_authority: str | None = None
    deadlines: list[str] = field(default_factory=list)
    amounts: list[str] = field(default_factory=list)
    published_at: str | None = None
    modified_at: str | None = None
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable mapping.

        Returns:
            All fields as a plain dict.
        """
        return asdict(self)

    @property
    def completeness(self) -> float:
        """Fraction of the core sections that were found.

        Returns:
            0.0-1.0 across details, benefits, eligibility, application process
            and documents. Useful for triaging which pages extracted poorly.
        """
        core = (self.details, self.benefits, self.eligibility,
                self.application_process, self.documents_required)
        return sum(1 for value in core if value) / len(core)


def _heading_key(line: str) -> str | None:
    """Map a heading line to a canonical section key.

    Args:
        line: A single line of text.

    Returns:
        The canonical key, or None if the line is not a recognised heading.
    """
    cleaned = re.sub(r"^[#*\s>•\d.)-]+", "", line).strip().rstrip(":").lower()
    if not cleaned or len(cleaned) > 60:
        return None
    for key, variants in SECTION_PATTERNS.items():
        if any(cleaned == v or cleaned.startswith(v) for v in variants):
            return key
    return None


def split_sections(text: str) -> dict[str, str]:
    """Split document text into canonical scheme sections.

    Args:
        text: Full document text.

    Returns:
        Mapping of canonical section key to its body text. Sections not present
        in the document are simply absent.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in text.splitlines():
        key = _heading_key(line)
        if key:
            current = key
            sections.setdefault(current, [])
            continue
        if current and line.strip():
            sections[current].append(line.rstrip())

    return {k: "\n".join(v).strip() for k, v in sections.items() if any(v)}


def find_authority(text: str) -> str | None:
    """Identify the implementing ministry, department or agency.

    Args:
        text: Document text.

    Returns:
        The first plausible authority string, or None.
    """
    for pattern in AUTHORITY_PATTERNS:
        match = re.search(pattern, text)
        if match:
            # Collapse whitespace and trim trailing punctuation from the capture.
            return " ".join(match.group(0).split()).rstrip(" ,.-")
    return None


def find_deadlines(text: str) -> list[str]:
    """Collect date expressions that appear in a deadline context.

    Only dates near deadline language are returned; a scheme page is full of
    unrelated dates (launch dates, revision dates) that would be misleading if
    surfaced as deadlines.

    Args:
        text: Document text.

    Returns:
        Deduplicated snippets containing a deadline and its date.
    """
    found: list[str] = []
    for line in text.splitlines():
        lowered = line.lower()
        if not any(hint in lowered for hint in DEADLINE_HINTS):
            continue
        for pattern in DATE_PATTERNS:
            if re.search(pattern, line):
                snippet = " ".join(line.split())[:200]
                if snippet not in found:
                    found.append(snippet)
                break
    return found


def find_amounts(text: str, limit: int = 20) -> list[str]:
    """Collect monetary amounts mentioned in the document.

    Args:
        text: Document text.
        limit: Maximum distinct amounts to return.

    Returns:
        Deduplicated amount strings in document order.
    """
    seen: list[str] = []
    for match in re.finditer(AMOUNT_PATTERN, text):
        value = " ".join(match.group(0).split())
        if value not in seen:
            seen.append(value)
        if len(seen) >= limit:
            break
    return seen


# A compendium (a ministry's "all schemes" PDF) holds dozens of schemes in one
# file. Treating it as a single scheme yields nothing useful, because the section
# headings repeat once per scheme. These patterns find the per-scheme boundaries.
# Shape of a heading line, captured without reference to where the scheme
# keyword sits. Coupling the two was brittle: "Credit Guarantee Scheme for Micro
# and Small Enterprises (CGTMSE)" puts the keyword early and 41 characters of
# title after it, which a fixed trailing allowance silently rejected.
SCHEME_TITLE_PATTERNS: tuple[str, ...] = (
    r"^\s*\d{1,2}[.)]\s+([A-Z][^\n]{10,150})\s*$",   # "1. Some Scheme (ABBR)"
    r"^#{1,4}\s+([A-Z][^\n]{10,150})\s*$",            # markdown heading
    r"^\s*([A-Z][A-Z\s,&()\-'/]{15,150})\s*$",        # ALL-CAPS title line
)

# A candidate heading is only a scheme if it names one. Checked against the
# captured title rather than woven into the pattern.
SCHEME_KEYWORDS: tuple[str, ...] = (
    "scheme", "programme", "program", "yojana", "mission", "abhiyan",
    "fund", "nidhi", "subsidy", "grant", "pension", "scholarship", "awas",
)

# Below this, a "scheme" is a table-of-contents entry rather than a real section.
MIN_SCHEME_SECTION_CHARS = 400
# A document with fewer detected schemes than this is treated as a single scheme.
MIN_SCHEMES_FOR_COMPENDIUM = 3


def find_scheme_boundaries(text: str) -> list[tuple[int, str]]:
    """Locate where each scheme starts in a multi-scheme document.

    Args:
        text: Full document text.

    Returns:
        (line index, scheme title) pairs in document order.
    """
    boundaries: list[tuple[int, str]] = []
    seen: set[str] = set()

    for index, line in enumerate(text.splitlines()):
        for pattern in SCHEME_TITLE_PATTERNS:
            match = re.match(pattern, line)
            if not match:
                continue
            title = " ".join(match.group(1).split()).strip(" .:-")
            key = title.lower()
            # Shape alone is not enough — the heading must actually name a scheme.
            if not any(word in key for word in SCHEME_KEYWORDS):
                break
            # A compendium lists each scheme twice: once in the contents, once
            # as the section. Keep the first occurrence and let the length
            # filter discard it if it turns out to be the contents entry.
            if key not in seen:
                seen.add(key)
                boundaries.append((index, title))
            break

    return boundaries


def split_compendium(text: str) -> list[tuple[str, str]]:
    """Split a multi-scheme document into one section per scheme.

    Args:
        text: Full document text.

    Returns:
        (scheme title, section text) pairs. Empty when the document does not
        look like a compendium, so the caller falls back to single-scheme
        handling.
    """
    boundaries = find_scheme_boundaries(text)
    if len(boundaries) < MIN_SCHEMES_FOR_COMPENDIUM:
        return []

    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    for position, (start, title) in enumerate(boundaries):
        end = boundaries[position + 1][0] if position + 1 < len(boundaries) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        # Short blocks are contents-page entries, not the scheme itself.
        if len(body) >= MIN_SCHEME_SECTION_CHARS:
            sections.append((title, body))

    return sections


# Government "all schemes" documents are very often a table, one row per scheme,
# rather than prose with headings. PDF-to-markdown conversion renders these as
# pipe tables with <br> inside cells, so they need a separate parser.
TABLE_COLUMN_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("name", ("name of scheme", "name of the scheme", "scheme name", "scheme")),
    ("details", ("objective", "objectives", "description", "purpose", "about")),
    ("benefits", ("key benefits", "benefits", "assistance", "incentive",
                  "quantum of assistance", "financial assistance")),
    ("eligibility", ("applicable for", "eligibility", "eligible", "target",
                     "beneficiary", "who can apply")),
    ("application_process", ("how to apply", "application", "apply", "procedure")),
    ("references", ("link", "website", "url", "reference", "guidelines")),
)

# Enough columns to be a real table rather than a stray pipe-delimited line.
MIN_TABLE_COLUMNS = 3


def _split_table_row(line: str) -> list[str]:
    """Split one markdown table row into cleaned cell values.

    Args:
        line: A line beginning and ending with a pipe.

    Returns:
        Cell strings with <br> collapsed to spaces and emphasis stripped.
    """
    # Strip exactly one delimiter from each end. str.strip("|") would remove
    # every consecutive pipe, collapsing "|||continuation||" to a single cell —
    # and the empty leading cells are precisely what marks a continuation row.
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]

    cells = body.split("|")
    cleaned: list[str] = []
    for cell in cells:
        value = re.sub(r"<br\s*/?>", " ", cell, flags=re.I)
        value = re.sub(r"<sup>.*?</sup>", "", value, flags=re.I | re.S)
        value = value.replace("**", "").replace("*", "")
        cleaned.append(" ".join(value.split()).strip())
    return cleaned


def _map_table_columns(header_cells: list[str]) -> dict[int, str]:
    """Map table column positions onto canonical scheme fields.

    Args:
        header_cells: Cleaned header row cells.

    Returns:
        Column index -> canonical field name, for columns that matched.
    """
    mapping: dict[int, str] = {}
    for index, cell in enumerate(header_cells):
        lowered = cell.lower()
        for field_name, variants in TABLE_COLUMN_MAP:
            if field_name in mapping.values():
                continue
            if any(variant in lowered for variant in variants):
                mapping[index] = field_name
                break
    return mapping


def extract_table_compendium(
    text: str,
    *,
    source_url: str,
    retrieved_at: str,
    doc_format: str = "pdf",
    published_at: str | None = None,
    modified_at: str | None = None,
    content_hash: str = "",
) -> list[SchemeMetadata]:
    """Extract one record per row from a tabular scheme compendium.

    Handles the quirks of PDF-derived tables: headers repeat at every page
    break, and a scheme's text often continues on rows whose leading cell is
    empty.

    Args:
        text: Full document text, as markdown.
        source_url: Official URL the document came from.
        retrieved_at: ISO retrieval timestamp.
        doc_format: Source format.
        published_at: Publication date, when known.
        modified_at: Last-modified date, when known.
        content_hash: Hash of the whole document.

    Returns:
        One record per scheme row. Empty when no scheme table was found.
    """
    mapping: dict[int, str] = {}
    rows: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        # Separator row: |---|---|
        if set(stripped.replace("|", "").replace(" ", "")) <= {"-", ":"}:
            continue

        cells = _split_table_row(stripped)
        if len(cells) < MIN_TABLE_COLUMNS:
            continue

        candidate = _map_table_columns(cells)
        # A header repeats at each page break; re-reading it is how the parser
        # stays aligned when column counts shift mid-document.
        if "name" in candidate.values() and len(candidate) >= 2:
            mapping = candidate
            continue
        if not mapping:
            continue

        name_index = next((i for i, f in mapping.items() if f == "name"), 0)
        name_cell = cells[name_index] if name_index < len(cells) else ""
        # A leading serial number marks a new scheme; without one, the row
        # continues the previous scheme's text.
        starts_new = bool(re.match(r"^\d{1,2}[.)]\s*\S", name_cell))

        if starts_new:
            if current:
                rows.append(current)
            current = {"name": re.sub(r"^\d{1,2}[.)]\s*", "", name_cell).strip()}
            for index, field_name in mapping.items():
                if field_name == "name" or index >= len(cells):
                    continue
                if cells[index]:
                    current[field_name] = cells[index]
        elif current:
            for index, field_name in mapping.items():
                if index >= len(cells) or not cells[index]:
                    continue
                key = "details" if field_name == "name" else field_name
                current[key] = (current.get(key, "") + " " + cells[index]).strip()

    if current:
        rows.append(current)

    records: list[SchemeMetadata] = []
    for row in rows:
        name = row.get("name", "").strip()
        # Require a plausible name and some substance, so header fragments and
        # empty rows do not become schemes.
        if len(name) < 8 or sum(len(v) for v in row.values()) < 60:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60]
        records.append(
            SchemeMetadata(
                scheme_id=slug,
                name=name,
                source_url=f"{source_url}#{slug}",
                retrieved_at=retrieved_at,
                doc_format=doc_format,
                published_at=published_at,
                modified_at=modified_at,
                content_hash=content_hash,
                details=row.get("details"),
                benefits=row.get("benefits"),
                eligibility=row.get("eligibility"),
                application_process=row.get("application_process"),
                references=row.get("references"),
                implementing_authority=find_authority(text),
                amounts=find_amounts(" ".join(row.values())),
                deadlines=find_deadlines(" ".join(row.values())),
            )
        )

    if records:
        logger.info("Parsed %d schemes from the table at %s", len(records), source_url)
    return records


def extract_compendium(
    text: str,
    *,
    source_url: str,
    retrieved_at: str,
    doc_format: str = "pdf",
    published_at: str | None = None,
    modified_at: str | None = None,
    content_hash: str = "",
) -> list[SchemeMetadata]:
    """Extract one metadata record per scheme from a multi-scheme document.

    Args:
        text: Full document text.
        source_url: Official URL the document came from.
        retrieved_at: ISO retrieval timestamp.
        doc_format: Source format.
        published_at: Publication date, when known.
        modified_at: Last-modified date, when known.
        content_hash: Hash of the whole document.

    Returns:
        One record per detected scheme. Empty if the document is not a
        compendium, so the caller can fall back to single-scheme extraction.
    """
    # Tabular compendiums are more common than prose ones in government PDFs,
    # so try that shape first.
    tabular = extract_table_compendium(
        text,
        source_url=source_url,
        retrieved_at=retrieved_at,
        doc_format=doc_format,
        published_at=published_at,
        modified_at=modified_at,
        content_hash=content_hash,
    )
    if tabular:
        return tabular

    sections = split_compendium(text)
    if not sections:
        return []

    records: list[SchemeMetadata] = []
    for title, body in sections:
        # Slugify the title into a stable id, then disambiguate with the source
        # so two ministries' "Design Scheme" entries do not collide.
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
        records.append(
            extract_scheme(
                body,
                scheme_id=slug,
                name=title,
                source_url=f"{source_url}#{slug}",
                retrieved_at=retrieved_at,
                doc_format=doc_format,
                published_at=published_at,
                modified_at=modified_at,
                content_hash=content_hash,
            )
        )

    logger.info("Split compendium at %s into %d schemes", source_url, len(records))
    return records


def extract_scheme(
    text: str,
    *,
    scheme_id: str,
    name: str,
    source_url: str,
    retrieved_at: str,
    doc_format: str = "html",
    published_at: str | None = None,
    modified_at: str | None = None,
    content_hash: str = "",
) -> SchemeMetadata:
    """Build structured scheme metadata from document text.

    Args:
        text: Full extracted document text.
        scheme_id: Stable identifier, usually the URL slug.
        name: Human-readable scheme name.
        source_url: Official URL the text came from.
        retrieved_at: ISO timestamp of retrieval.
        doc_format: Source format (html, pdf, docx...).
        published_at: Publication date, when the source exposed one.
        modified_at: Last-modified date, when the source exposed one.
        content_hash: Hash of the text, for provenance.

    Returns:
        Populated metadata; fields the rules could not find stay None.
    """
    sections = split_sections(text)
    metadata = SchemeMetadata(
        scheme_id=scheme_id,
        name=name,
        source_url=source_url,
        retrieved_at=retrieved_at,
        doc_format=doc_format,
        published_at=published_at,
        modified_at=modified_at,
        content_hash=content_hash,
        implementing_authority=find_authority(text),
        deadlines=find_deadlines(text),
        amounts=find_amounts(text),
        **{key: sections.get(key) for key in SECTION_PATTERNS},
    )

    logger.debug(
        "Extracted %s: completeness=%.0f%%, authority=%s, %d deadlines, %d amounts",
        scheme_id, metadata.completeness * 100, metadata.implementing_authority,
        len(metadata.deadlines), len(metadata.amounts),
    )
    return metadata
