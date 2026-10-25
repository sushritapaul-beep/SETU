"""Format detection and text extraction for every document type the crawl meets.

Government portals publish the same scheme as an HTML page, a PDF circular, a
DOCX order and an XLSX annexure, sometimes all four. Each needs different
handling, but the pipeline downstream only wants text plus provenance — so every
extractor returns the same `ExtractedDocument`.

Extractors never raise for bad content. A corrupt PDF in a 4000-URL crawl must
degrade to a logged failure, not abort the run.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

# Content types mapped to the extractor family that handles them.
CONTENT_TYPE_MAP: dict[str, str] = {
    "text/html": "html",
    "application/xhtml+xml": "html",
    "application/pdf": "pdf",
    # Legacy binary Office formats get their own family: python-docx and
    # openpyxl cannot read them, and routing them there produced an empty
    # document indistinguishable from a page with no content.
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "text/csv": "csv",
    "application/json": "json",
    "application/xml": "xml",
    "text/xml": "xml",
    "application/zip": "zip",
    "text/plain": "text",
}

EXTENSION_MAP: dict[str, str] = {
    ".html": "html", ".htm": "html", ".xhtml": "html",
    ".aspx": "html", ".php": "html", ".jsp": "html",
    ".pdf": "pdf",
    ".doc": "doc", ".docx": "docx",
    ".xls": "xls", ".xlsx": "xlsx",
    ".ppt": "ppt", ".pptx": "pptx",
    ".csv": "csv",
    ".json": "json",
    ".xml": "xml",
    ".zip": "zip",
    ".txt": "text",
}

# Magic bytes, checked last but trusted most: government servers frequently
# serve PDFs as application/octet-stream or even text/html.
MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "pdf"),
    (b"PK\x03\x04", "zip_family"),  # docx/xlsx/zip all share this
    (b"<?xml", "xml"),
    (b"<!DOCTYPE html", "html"),
    (b"<html", "html"),
)

# Reused across PDFs, keyed by the OCR flag; see extract_pdf.
_PDF_PARSERS: dict[bool, Any] = {}

# A document shorter than this is almost certainly an error page or a stub.
MIN_USEFUL_CHARS = 120


@dataclass
class ExtractedDocument:
    """Text plus provenance for one fetched resource."""

    url: str
    text: str
    title: str
    doc_format: str
    retrieved_at: str
    content_type: str = ""
    published_at: str | None = None
    modified_at: str | None = None
    links: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_useful(self) -> bool:
        """Whether the extraction produced enough text to be worth indexing."""
        return len(self.text.strip()) >= MIN_USEFUL_CHARS


def detect_format(url: str, content_type: str, payload: bytes) -> str:
    """Decide which extractor a payload needs.

    Checks magic bytes last because they are the most reliable signal — many
    government servers mislabel content types, and a PDF served as text/html
    would otherwise be parsed as a web page and yield nothing.

    Args:
        url: Source URL, used for its extension.
        content_type: Response Content-Type header.
        payload: First bytes of the body.

    Returns:
        One of: html, pdf, docx, xlsx, csv, json, xml, zip, text, unknown.
    """
    head = payload[:512]
    for signature, family in MAGIC_SIGNATURES:
        if head.startswith(signature):
            if family != "zip_family":
                return family
            # Zip container: look inside to tell docx from xlsx from a plain zip.
            try:
                with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                    names = set(zf.namelist())
                    if any(n.startswith("word/") for n in names):
                        return "docx"
                    if any(n.startswith("xl/") for n in names):
                        return "xlsx"
                    if any(n.startswith("ppt/") for n in names):
                        return "pptx"
            except (zipfile.BadZipFile, OSError):
                pass
            return "zip"

    base = (content_type or "").split(";")[0].strip().lower()
    if base in CONTENT_TYPE_MAP:
        return CONTENT_TYPE_MAP[base]

    suffix = Path(url.split("?")[0]).suffix.lower()
    return EXTENSION_MAP.get(suffix, "unknown")


# ----------------------------------------------------------------- extractors --

def extract_html(url: str, payload: bytes, content_type: str) -> ExtractedDocument:
    """Extract readable text, title, dates and links from an HTML page.

    Args:
        url: Source URL.
        payload: Raw response body.
        content_type: Response content type.

    Returns:
        The extracted document, with discovered links for the frontier.
    """
    from bs4 import BeautifulSoup

    # lxml is lenient with the malformed markup common on older portals.
    soup = BeautifulSoup(payload, "lxml")

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form"]):
        tag.decompose()

    title = (soup.title.string or "").strip() if soup.title else ""

    published = modified = None
    for meta in soup.find_all("meta"):
        name = (meta.get("property") or meta.get("name") or "").lower()
        content = meta.get("content")
        if not content:
            continue
        if name in {"article:published_time", "datepublished", "dc.date"}:
            published = content
        elif name in {"article:modified_time", "datemodified", "last-modified"}:
            modified = content

    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    from urllib.parse import urljoin

    links = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
            links.append(urljoin(url, href))

    return ExtractedDocument(
        url=url, text=text, title=title, doc_format="html",
        retrieved_at=datetime.now(UTC).isoformat(), content_type=content_type,
        published_at=published, modified_at=modified, links=links,
    )


def extract_pdf(
    url: str, payload: bytes, content_type: str, enable_ocr: bool = False
) -> ExtractedDocument:
    """Extract markdown from a PDF, falling back across parsers.

    Reuses the existing DoclingParser so PDF handling stays in one place — it
    already tries Docling first (which preserves the tables that hold
    eligibility criteria) and falls back to pymupdf4llm.

    Args:
        url: Source URL.
        payload: Raw PDF bytes.
        content_type: Response content type.
        enable_ocr: Run OCR for scanned documents.

    Returns:
        The extracted document; empty text when every parser failed.
    """
    import tempfile

    from ingestion.parser import DoclingParser

    text = ""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    # One parser per OCR setting for the whole process: DoclingParser loads its
    # layout and OCR models on first use, and a fresh instance per PDF reloaded
    # them every time until the crawl ran out of memory and crashed.
    if enable_ocr not in _PDF_PARSERS:
        _PDF_PARSERS[enable_ocr] = DoclingParser(enable_ocr=enable_ocr)
    parser = _PDF_PARSERS[enable_ocr]
    try:
        text = parser.parse_pdf(tmp_path)
    except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop a crawl
        logger.warning("PDF extraction failed for %s: %s", url, exc)
    finally:
        tmp_path.unlink(missing_ok=True)

    title = Path(url.split("?")[0]).stem.replace("_", " ").replace("-", " ").strip()
    return ExtractedDocument(
        url=url, text=text, title=title, doc_format="pdf",
        retrieved_at=datetime.now(UTC).isoformat(), content_type=content_type,
        # Guidelines cite their annexures, amendments and forms by hyperlink;
        # without these the crawler can never reach a document that is only
        # linked from inside another document.
        links=_pdf_links(payload),
    )


def _pdf_links(payload: bytes) -> list[str]:
    """Collect the external hyperlinks embedded in a PDF.

    Args:
        payload: Raw PDF bytes.

    Returns:
        http(s) link targets, de-duplicated, in page order. Empty when the PDF
        cannot be opened — link discovery must never fail an extraction.
    """
    try:
        import pymupdf

        with pymupdf.open(stream=payload, filetype="pdf") as pdf:
            uris = [link.get("uri", "") for page in pdf for link in page.get_links()]
    except Exception as exc:  # noqa: BLE001 - links are a bonus, not the extraction
        logger.debug("Could not read PDF links: %s", exc)
        return []
    return list(dict.fromkeys(u.strip() for u in uris if u.strip().startswith("http")))


def extract_docx(url: str, payload: bytes, content_type: str) -> ExtractedDocument:
    """Extract paragraphs and tables from a Word document.

    Args:
        url: Source URL.
        payload: Raw DOCX bytes.
        content_type: Response content type.

    Returns:
        The extracted document.
    """
    import docx

    parts: list[str] = []
    title = ""
    try:
        document = docx.Document(io.BytesIO(payload))
        core = document.core_properties
        title = (core.title or "").strip()
        parts.extend(p.text.strip() for p in document.paragraphs if p.text.strip())
        # Tables carry eligibility criteria far more often than prose does.
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
    except Exception as exc:  # noqa: BLE001
        logger.warning("DOCX extraction failed for %s: %s", url, exc)

    if not title:
        title = Path(url.split("?")[0]).stem.replace("_", " ").strip()
    return ExtractedDocument(
        url=url, text="\n".join(parts), title=title, doc_format="docx",
        retrieved_at=datetime.now(UTC).isoformat(), content_type=content_type,
    )


def extract_xlsx(
    url: str, payload: bytes, content_type: str, max_rows: int = 2000
) -> ExtractedDocument:
    """Flatten spreadsheet cells into pipe-delimited rows.

    Args:
        url: Source URL.
        payload: Raw XLSX bytes.
        content_type: Response content type.
        max_rows: Row cap per sheet, so a large dataset cannot swamp the corpus.

    Returns:
        The extracted document, one section per sheet.
    """
    import openpyxl

    parts: list[str] = []
    try:
        # read_only + data_only keeps memory bounded and resolves formulas to
        # their cached values, which is what a reader would see.
        book = openpyxl.load_workbook(io.BytesIO(payload), read_only=True, data_only=True)
        for sheet in book.worksheets:
            parts.append(f"## Sheet: {sheet.title}")
            for index, row in enumerate(sheet.iter_rows(values_only=True)):
                if index >= max_rows:
                    parts.append(f"[truncated after {max_rows} rows]")
                    break
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    parts.append(" | ".join(cells))
        book.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("XLSX extraction failed for %s: %s", url, exc)

    return ExtractedDocument(
        url=url, text="\n".join(parts),
        title=Path(url.split("?")[0]).stem.replace("_", " ").strip(),
        doc_format="xlsx", retrieved_at=datetime.now(UTC).isoformat(),
        content_type=content_type,
    )


def extract_csv(
    url: str, payload: bytes, content_type: str, max_rows: int = 2000
) -> ExtractedDocument:
    """Flatten CSV rows into pipe-delimited lines.

    Args:
        url: Source URL.
        payload: Raw CSV bytes.
        content_type: Response content type.
        max_rows: Row cap.

    Returns:
        The extracted document.
    """
    import chardet

    parts: list[str] = []
    try:
        # Government CSVs are often cp1252 or an Indic encoding, not UTF-8.
        guess = chardet.detect(payload[:20000]) or {}
        text = payload.decode(guess.get("encoding") or "utf-8", errors="replace")
        for index, row in enumerate(csv.reader(io.StringIO(text))):
            if index >= max_rows:
                parts.append(f"[truncated after {max_rows} rows]")
                break
            cells = [c.strip() for c in row if c.strip()]
            if cells:
                parts.append(" | ".join(cells))
    except Exception as exc:  # noqa: BLE001
        logger.warning("CSV extraction failed for %s: %s", url, exc)

    return ExtractedDocument(
        url=url, text="\n".join(parts),
        title=Path(url.split("?")[0]).stem.replace("_", " ").strip(),
        doc_format="csv", retrieved_at=datetime.now(UTC).isoformat(),
        content_type=content_type,
    )


def extract_json(url: str, payload: bytes, content_type: str) -> ExtractedDocument:
    """Render a JSON document as indented text.

    Args:
        url: Source URL.
        payload: Raw JSON bytes.
        content_type: Response content type.

    Returns:
        The extracted document, with the parsed object kept in `extra` so a
        structured extractor can use it without re-parsing.
    """
    text, parsed = "", None
    try:
        parsed = json.loads(payload.decode("utf-8", errors="replace"))
        text = json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("JSON extraction failed for %s: %s", url, exc)

    return ExtractedDocument(
        url=url, text=text,
        title=Path(url.split("?")[0]).stem.replace("_", " ").strip(),
        doc_format="json", retrieved_at=datetime.now(UTC).isoformat(),
        content_type=content_type, extra={"parsed": parsed} if parsed is not None else {},
    )


def extract_xml(url: str, payload: bytes, content_type: str) -> ExtractedDocument:
    """Extract text nodes from XML, and URLs when it is a sitemap.

    Args:
        url: Source URL.
        payload: Raw XML bytes.
        content_type: Response content type.

    Returns:
        The extracted document. Sitemap <loc> entries land in `links`, which is
        how the crawler discovers pages without following navigation.
    """
    parts: list[str] = []
    links: list[str] = []
    try:
        root = ElementTree.fromstring(payload)
        for element in root.iter():
            # Sitemaps namespace their tags; match on the local name.
            tag = element.tag.rsplit("}", 1)[-1].lower()
            value = (element.text or "").strip()
            if not value:
                continue
            if tag == "loc":
                links.append(value)
            else:
                parts.append(value)
    except ElementTree.ParseError as exc:
        logger.warning("XML parse failed for %s: %s", url, exc)

    return ExtractedDocument(
        url=url, text="\n".join(parts),
        title=Path(url.split("?")[0]).stem.replace("_", " ").strip(),
        doc_format="xml", retrieved_at=datetime.now(UTC).isoformat(),
        content_type=content_type, links=links,
    )


def extract_zip(
    url: str, payload: bytes, content_type: str, max_members: int = 25
) -> ExtractedDocument:
    """Extract text from documents inside a ZIP archive.

    Ministries publish scheme guideline bundles as ZIPs, so the contents matter
    even though the container itself has no text.

    Args:
        url: Source URL.
        payload: Raw ZIP bytes.
        content_type: Response content type.
        max_members: Cap on members processed, guarding against zip bombs.

    Returns:
        The extracted document, concatenating each member's text.
    """
    parts: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for info in archive.infolist()[:max_members]:
                if info.is_dir() or info.file_size > 50 * 1024 * 1024:
                    continue
                member_format = EXTENSION_MAP.get(Path(info.filename).suffix.lower())
                if member_format not in {"pdf", "docx", "xlsx", "pptx", "csv", "text", "html"}:
                    continue
                data = archive.read(info)
                inner = extract(f"{url}#{info.filename}", data, "", member_format)
                if inner.text.strip():
                    parts.append(f"## {info.filename}\n{inner.text}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ZIP extraction failed for %s: %s", url, exc)

    return ExtractedDocument(
        url=url, text="\n\n".join(parts),
        title=Path(url.split("?")[0]).stem.replace("_", " ").strip(),
        doc_format="zip", retrieved_at=datetime.now(UTC).isoformat(),
        content_type=content_type,
    )


def extract_pptx(url: str, payload: bytes, content_type: str) -> ExtractedDocument:
    """Extract slide text and tables from a PowerPoint deck.

    Ministries publish scheme briefings and training decks as PPTX.

    Args:
        url: Source URL.
        payload: Raw PPTX bytes.
        content_type: Response content type.

    Returns:
        The extracted document, one section per slide.
    """
    from pptx import Presentation

    parts: list[str] = []
    try:
        deck = Presentation(io.BytesIO(payload))
        for number, slide in enumerate(deck.slides, start=1):
            lines: list[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame and shape.text_frame.text.strip():
                    lines.append(shape.text_frame.text.strip())
                if getattr(shape, "has_table", False) and shape.has_table:
                    for row in shape.table.rows:
                        cells = [c.text.strip() for c in row.cells if c.text.strip()]
                        if cells:
                            lines.append(" | ".join(cells))
            if lines:
                parts.append(f"## Slide {number}\n" + "\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        logger.warning("PPTX extraction failed for %s: %s", url, exc)

    return ExtractedDocument(
        url=url, text="\n\n".join(parts),
        title=Path(url.split("?")[0]).stem.replace("_", " ").strip(),
        doc_format="pptx", retrieved_at=datetime.now(UTC).isoformat(),
        content_type=content_type,
    )


def extract_text(url: str, payload: bytes, content_type: str) -> ExtractedDocument:
    """Decode a plain text document.

    Args:
        url: Source URL.
        payload: Raw bytes.
        content_type: Response content type.

    Returns:
        The extracted document.
    """
    import chardet

    guess = chardet.detect(payload[:20000]) or {}
    return ExtractedDocument(
        url=url,
        text=payload.decode(guess.get("encoding") or "utf-8", errors="replace"),
        title=Path(url.split("?")[0]).stem.replace("_", " ").strip(),
        doc_format="text", retrieved_at=datetime.now(UTC).isoformat(),
        content_type=content_type,
    )


def extract(
    url: str,
    payload: bytes,
    content_type: str,
    doc_format: str | None = None,
    enable_ocr: bool = False,
) -> ExtractedDocument:
    """Dispatch to the right extractor for a payload.

    Args:
        url: Source URL.
        payload: Raw response body.
        content_type: Response Content-Type.
        doc_format: Force a format instead of detecting one.
        enable_ocr: Enable OCR for scanned PDFs.

    Returns:
        The extracted document. An unknown format yields empty text rather than
        raising, so the crawl records a skip and continues.
    """
    resolved = doc_format or detect_format(url, content_type, payload)

    if resolved == "html":
        return extract_html(url, payload, content_type)
    if resolved == "pdf":
        return extract_pdf(url, payload, content_type, enable_ocr=enable_ocr)
    if resolved == "docx":
        return extract_docx(url, payload, content_type)
    if resolved == "xlsx":
        return extract_xlsx(url, payload, content_type)
    if resolved == "csv":
        return extract_csv(url, payload, content_type)
    if resolved == "json":
        return extract_json(url, payload, content_type)
    if resolved == "xml":
        return extract_xml(url, payload, content_type)
    if resolved == "zip":
        return extract_zip(url, payload, content_type)
    if resolved == "text":
        return extract_text(url, payload, content_type)
    if resolved == "pptx":
        return extract_pptx(url, payload, content_type)

    # doc/xls/ppt (legacy binary Office) and anything unrecognised: reported by
    # format, so the crawl records "unsupported format" rather than "no text".
    logger.info("No extractor for format %r at %s", resolved, url)
    return ExtractedDocument(
        url=url, text="", title="", doc_format=resolved or "unknown",
        retrieved_at=datetime.now(UTC).isoformat(), content_type=content_type,
    )
