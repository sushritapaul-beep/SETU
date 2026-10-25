"""Convert scheme PDFs to markdown, preferring Docling and falling back to PyMuPDF.

Docling reconstructs reading order, tables and headings, which matters because
scheme eligibility criteria are usually laid out in tables. PyMuPDF4LLM is far
faster and dependency-light but flattens structure, so it is used only when
Docling fails.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Set SETU_DISABLE_DOCLING=1 to skip Docling entirely and parse with pymupdf4llm.
#
# Docling is worth trying because it reconstructs the tables that hold
# eligibility criteria, but it is heavy and environment-sensitive. On this
# Windows machine it has failed two different ways — a symlink-privilege error
# while caching models, then torch inductor attempting a JIT compile with an
# MSVC toolchain that is not installed. Both failures cost minutes before
# raising, which is far worse than not trying at all, and pymupdf4llm handled
# the same documents successfully in every case.
DISABLE_DOCLING = os.environ.get("SETU_DISABLE_DOCLING", "").lower() in {"1", "true", "yes"}

# Docling pulls in torch, which may try to JIT-compile kernels. Without a C++
# compiler that attempt fails slowly. Disabling dynamo makes it fail fast, or
# not attempt at all.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


class DoclingParser:
    """Parse PDFs into markdown with an automatic lightweight fallback."""

    def __init__(self, enable_ocr: bool = False) -> None:
        """Configure the parser.

        Args:
            enable_ocr: Run OCR over scanned pages. Off by default because it is
                slow and most portal PDFs already contain a text layer.
        """
        self.enable_ocr = enable_ocr
        # Docling's DocumentConverter loads layout models on construction, so it
        # is built once on first use and reused rather than per document.
        self._converter = None

    def parse_pdf(self, path: Path) -> str:
        """Convert a PDF to markdown.

        Args:
            path: Path to an existing PDF file.

        Returns:
            The document as markdown text.

        Raises:
            FileNotFoundError: If the path does not exist.
            RuntimeError: If both Docling and the PyMuPDF fallback fail.
        """
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        if DISABLE_DOCLING:
            logger.debug("Docling disabled by SETU_DISABLE_DOCLING; using pymupdf4llm")
            markdown = self._parse_with_pymupdf(path)
            if not markdown.strip():
                raise RuntimeError(
                    f"No extractable text in {path} "
                    "(is it a scanned PDF? try DoclingParser(enable_ocr=True))"
                )
            logger.info("Parsed %s with pymupdf4llm (%d chars)", path.name, len(markdown))
            return markdown

        try:
            markdown = self._parse_with_docling(path)
            if markdown.strip():
                logger.info("Parsed %s with Docling (%d chars)", path.name, len(markdown))
                return markdown
            # An empty result is a failure in disguise; fall through rather than
            # writing a blank document into the corpus.
            logger.warning("Docling produced no text for %s; falling back", path.name)
        except Exception as exc:  # noqa: BLE001 - any Docling failure should degrade, not crash
            logger.warning(
                "Docling failed for %s (%s: %s); falling back to pymupdf4llm",
                path.name,
                type(exc).__name__,
                exc,
            )

        try:
            markdown = self._parse_with_pymupdf(path)
        except Exception as exc:  # noqa: BLE001 - convert to one clear error below
            raise RuntimeError(f"Both Docling and pymupdf4llm failed for {path}: {exc}") from exc

        if not markdown.strip():
            raise RuntimeError(
                f"No extractable text in {path} "
                "(is it a scanned PDF? try DoclingParser(enable_ocr=True))"
            )

        logger.info("Parsed %s with pymupdf4llm fallback (%d chars)", path.name, len(markdown))
        return markdown

    def _parse_with_docling(self, path: Path) -> str:
        """Run Docling's structured conversion.

        Args:
            path: Path to the PDF.

        Returns:
            Markdown produced by Docling.
        """
        if self._converter is None:
            # Imported lazily: Docling pulls in torch and layout models, which
            # is far too heavy to pay for at module import time.
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter, PdfFormatOption

            # Docling's default docling-parse backend commits ~6 GB of private
            # memory per page in flight (measured: 22 GB on an 18-page PDF), so
            # pages fail with std::bad_alloc and the process can segfault once
            # the system commit limit is hit. The pdfium backend peaked at
            # 2.3 GB on the same PDF and kept every page.
            self._converter = DocumentConverter(format_options={
                InputFormat.PDF: PdfFormatOption(backend=PyPdfiumDocumentBackend),
            })

        result = self._converter.convert(str(path))
        return result.document.export_to_markdown()

    def _parse_with_pymupdf(self, path: Path) -> str:
        """Run the lightweight PyMuPDF extraction path.

        Args:
            path: Path to the PDF.

        Returns:
            Markdown produced by pymupdf4llm.
        """
        import pymupdf4llm

        return pymupdf4llm.to_markdown(str(path))
