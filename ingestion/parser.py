"""Convert scheme PDFs to markdown, preferring Docling and falling back to PyMuPDF.

Docling reconstructs reading order, tables and headings, which matters because
scheme eligibility criteria are usually laid out in tables. PyMuPDF4LLM is far
faster and dependency-light but flattens structure, so it is used only when
Docling fails.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()

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
