"""Split scheme documents into linked parent/child chunks for retrieval.

This implements the parent-document pattern: small child chunks are what get
embedded and searched (precise matching), while the larger parent chunk is what
gets handed to the LLM (enough surrounding context to actually answer). Each
child therefore carries the id of the parent it came from.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Splitters count characters, but chunk budgets are quoted in tokens. Four
# characters per token is the usual English approximation; Devanagari and Tamil
# tokenize less efficiently, so real token counts will run somewhat higher.
CHARS_PER_TOKEN = 4

# Markdown-aware split points, tried in order. Headings come first so a chunk
# boundary lands between sections rather than mid-sentence.
MARKDOWN_SEPARATORS = ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""]

# Invisible characters that portal HTML leaves scattered through the text. They
# carry no meaning but are still tokenized, so they add noise to every embedding
# and appear as mojibake inside quoted citations.
#
# NOTE: U+200C (ZWNJ) and U+200D (ZWJ) are deliberately NOT stripped — they are
# semantically load-bearing in Devanagari and Tamil, and removing them would
# corrupt the Hindi and Tamil content this system exists to serve.
ZERO_WIDTH_NOISE = {
    "﻿": "",  # BOM / zero-width no-break space
    "​": "",  # zero-width space
    " ": " ",  # non-breaking space -> ordinary space
}


# Headings that begin myScheme's site-wide footer. Everything from the first
# marker onward is navigation, not scheme content, so the document is truncated
# there rather than filtered line by line.
FOOTER_MARKERS = (
    "## Quick Links",
    "## Useful Links",
    "## Important Links",
)

# Modal, interstitial and sign-in text repeated on every scheme page. Matched
# per line so surrounding real content survives.
BOILERPLATE_PHRASES = (
    "You're being redirected to an external website",
    "Your mobile number will be shared with Jan Samarth",
    "Something went wrong. Please try again later",
    "Are you sure you want to sign out",
    "You need to sign in before applying for schemes",
    "You have already submitted an application for this scheme",
    "It seems you have already initiated your application",
    "Enter scheme name to search",
)

# Standalone UI controls that the scraper renders as their own lines. Matched on
# the whole stripped line (case-insensitively) rather than as substrings, so a
# sentence that merely contains "cancel" is never touched.
UI_NOISE_LINES = frozenset(
    {
        "ok",
        "cancel",
        "cancelok",
        "cancelsign in",
        "cancelsign out",
        "sign in",
        "sign out",
        "back",
        "###",
        "### apply now",
        "### check eligibility",
        "apply now",
        "check eligibility",
    }
)

# The in-page section menu, emitted as a bare bullet list above the real content.
# Only dropped when a line is exactly one of these bullets — the same words as a
# "### Eligibility" heading or inside a sentence are kept.
NAV_LABELS = frozenset(
    {
        "details",
        "benefits",
        "eligibility",
        "application process",
        "documents required",
        "frequently asked questions",
        "sources and references",
        "feedback",
        "exclusions",
    }
)

# Quality floor for a chunk to be worth embedding.
MIN_CHUNK_CHARS = 80

# Below this share of letters a chunk is coordinate data or markup, not prose.
# Uses str.isalpha(), which is true for Devanagari and Tamil as well as Latin,
# so Hindi and Tamil content is never penalised by this check.
MIN_ALPHA_RATIO = 0.55


@dataclass(frozen=True)
class ChunkMetadata:
    """Provenance attached to every chunk, used later for citation and filtering.

    The provenance fields default to empty so chunk files written before they
    existed still load. Without them a chunk cannot answer "how stale is this?",
    which matters when the underlying source is a government rule that changes.
    """

    source_url: str
    scheme_name: str
    chunk_type: str  # "parent" or "child"
    # --- provenance ---
    doc_format: str = "html"
    retrieved_at: str = ""
    published_at: str | None = None
    modified_at: str | None = None
    source_title: str = ""
    # Hash of the source document, linking a chunk back to the exact text and to
    # the crawl-state record that captured it.
    doc_hash: str = ""


@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit of text plus its provenance."""

    id: str
    parent_id: str | None
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict[str, Any]:
        """Render as a plain dict suitable for JSONL serialisation.

        Returns:
            A JSON-safe mapping of every field.
        """
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "text": self.text,
            "metadata": asdict(self.metadata),
        }


@dataclass
class SemanticParentChildChunker:
    """Produce large parent chunks and the smaller child chunks nested inside them.

    Children are split by recursive character splitting, which keeps this stage
    runnable with no model downloads.
    """

    parent_tokens: int = 2048
    child_tokens: int = 512
    child_overlap_tokens: int = 64
    _parent_splitter: RecursiveCharacterTextSplitter = field(init=False, repr=False)
    _child_splitter: RecursiveCharacterTextSplitter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the character splitters used for parents and oversized children."""
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_tokens * CHARS_PER_TOKEN,
            # Parents are context containers, not match targets, so overlap here
            # would mostly duplicate text handed to the LLM.
            chunk_overlap=0,
            separators=MARKDOWN_SEPARATORS,
        )
        child_chars = self.child_tokens * CHARS_PER_TOKEN
        # Overlap must stay below chunk size or the splitter refuses to build.
        # Cap it at a quarter of the chunk so small child budgets stay valid;
        # at the default 512/64 this is a no-op.
        overlap_chars = min(self.child_overlap_tokens * CHARS_PER_TOKEN, child_chars // 4)
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chars,
            chunk_overlap=overlap_chars,
            separators=MARKDOWN_SEPARATORS,
        )
        # A quality floor near the chunk budget would reject almost everything,
        # so cap it at half the child size. At the 512-token default this is a
        # no-op; it only matters for deliberately small configurations.
        self._min_chunk_chars = min(MIN_CHUNK_CHARS, child_chars // 2)

    def chunk_document(
        self,
        text: str,
        source_url: str,
        scheme_name: str,
        provenance: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split one document into parent chunks and their child chunks.

        Args:
            text: The document body, as markdown.
            source_url: Where the document came from; used for citation.
            scheme_name: Human-readable scheme title.
            provenance: Optional retrieval metadata (doc_format, retrieved_at,
                published_at, modified_at, source_title, doc_hash) carried onto
                every chunk, so a chunk can report how current it is.

        Returns:
            Parents and children interleaved, each parent immediately followed by
            its own children. Empty if the document had no usable text.
        """
        if not text or not text.strip():
            logger.warning("Empty document for %s; nothing to chunk", source_url)
            return []

        # Normalise before splitting so chunk ids hash the cleaned text and stay
        # stable regardless of invisible characters upstream.
        text = self.normalize_text(text)
        text = self.strip_boilerplate(text)
        if not text.strip():
            logger.warning("Document for %s was entirely boilerplate", source_url)
            return []

        chunks: list[Chunk] = []
        # Drop junk parents outright: if a parent is navigation or markup, every
        # child split out of it is too.
        all_parents = self._parent_splitter.split_text(text)
        parent_texts = [p for p in all_parents if not self.is_low_quality(p)]
        # Counts parents and children alike, so the log line reflects everything
        # the quality filter removed.
        dropped_low_quality = len(all_parents) - len(parent_texts)

        for parent_index, parent_text in enumerate(parent_texts):
            parent_id = self._make_id(source_url, "parent", parent_index, parent_text)
            chunks.append(
                Chunk(
                    id=parent_id,
                    parent_id=None,
                    text=parent_text,
                    metadata=ChunkMetadata(
                        source_url, scheme_name, "parent", **(provenance or {})
                    ),
                )
            )

            for child_index, child_text in enumerate(self._split_children(parent_text)):
                # A good parent can still yield a junk child (a stray SVG block
                # or a heading-only fragment), so children are checked too.
                if self.is_low_quality(child_text):
                    dropped_low_quality += 1
                    continue
                chunks.append(
                    Chunk(
                        # Child ids are seeded with the parent id so two identical
                        # sentences under different parents never collide.
                        id=self._make_id(source_url, f"child-{parent_id}", child_index, child_text),
                        parent_id=parent_id,
                        text=child_text,
                        metadata=ChunkMetadata(
                            source_url, scheme_name, "child", **(provenance or {})
                        ),
                    )
                )

        logger.info(
            "Chunked %s into %d parents and %d children (%d low-quality chunks dropped)",
            scheme_name,
            len(parent_texts),
            len(chunks) - len(parent_texts),
            dropped_low_quality,
        )
        return chunks

    def _split_children(self, parent_text: str) -> list[str]:
        """Split one parent's text into child chunks.

        Args:
            parent_text: Text of a single parent chunk.

        Returns:
            Child chunk texts, each within the configured token budget.
        """
        return self._child_splitter.split_text(parent_text)

    @classmethod
    def strip_boilerplate(cls, text: str) -> str:
        """Remove site navigation and modal text from a scraped document.

        Portal pages wrap real content in a footer and a set of hidden dialogs
        that appear on every scheme. Left in, they are embedded, compete in
        retrieval, and can end up quoted in a citation.

        Args:
            text: Normalised document text.

        Returns:
            Text truncated at the footer, with boilerplate lines removed.
        """
        # Truncate at whichever footer heading appears first.
        cut = len(text)
        for marker in FOOTER_MARKERS:
            position = text.find(marker)
            if position != -1:
                cut = min(cut, position)
        text = text[:cut]

        kept = [
            line
            for line in text.splitlines()
            if not any(phrase in line for phrase in BOILERPLATE_PHRASES)
            and not cls._is_ui_line(line)
        ]
        return "\n".join(kept)

    @staticmethod
    def _is_ui_line(line: str) -> bool:
        """Whether a single line is a bare UI control or section-menu bullet.

        Args:
            line: One line of document text.

        Returns:
            True if the line is chrome rather than content.
        """
        stripped = line.strip().lower()
        if stripped in UI_NOISE_LINES:
            return True
        # A bare menu bullet such as "* Documents Required" — but not a heading
        # ("### Documents Required") or a bullet with real text after it.
        if stripped.startswith(("* ", "- ")):
            return stripped[2:].strip() in NAV_LABELS
        return False

    def is_low_quality(self, text: str) -> bool:
        """Decide whether a chunk is too poor to be worth embedding.

        Args:
            text: Candidate chunk text.

        Returns:
            True if the chunk is too short or contains too few letters to be
            prose (SVG path data, coordinate dumps, stray punctuation).
        """
        stripped = text.strip()
        if len(stripped) < self._min_chunk_chars:
            return True
        # Spaces are excluded so that whitespace cannot mask a low letter count.
        dense = [c for c in stripped if not c.isspace()]
        if not dense:
            return True
        alpha_ratio = sum(c.isalpha() for c in dense) / len(dense)
        return alpha_ratio < MIN_ALPHA_RATIO

    @staticmethod
    def normalize_text(text: str) -> str:
        """Strip invisible scrape artefacts while preserving Indic joiners.

        Args:
            text: Raw document text from a scrape or PDF parse.

        Returns:
            Text with zero-width noise removed and runs of blank lines collapsed.
        """
        for noise, replacement in ZERO_WIDTH_NOISE.items():
            text = text.replace(noise, replacement)
        # Collapse 3+ newlines to a paragraph break so blank space does not eat
        # into a chunk's character budget.
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")
        return text

    @staticmethod
    def _make_id(source_url: str, kind: str, index: int, text: str) -> str:
        """Derive a stable chunk id from its content and position.

        Deterministic rather than random so that re-ingesting an unchanged
        document produces identical ids, making vector upserts idempotent.

        Args:
            source_url: Document origin.
            kind: Discriminator such as "parent" or "child-<parent_id>".
            index: Position within its group.
            text: Chunk body.

        Returns:
            A 32-character hex digest.
        """
        seed = f"{source_url}|{kind}|{index}|{text}".encode()
        return hashlib.sha256(seed).hexdigest()[:32]
