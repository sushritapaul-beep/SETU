"""One-shot ingestion: scrape, parse, chunk, embed and index scheme documents.

Runs the full pipeline end to end: pages and PDFs become chunks in
data/processed/chunks.jsonl, which are then embedded via Jina's API and
upserted into Qdrant Cloud.

Usage:
    python scripts/ingest.py                      # the five starter URLs
    python scripts/ingest.py --no-scrape          # local PDFs only
    python scripts/ingest.py --urls URL1 URL2     # explicit URLs
    python scripts/ingest.py --no-index           # stop after chunks.jsonl
    python scripts/ingest.py --index-only         # re-index existing chunks.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Allow `python scripts/ingest.py` from the repo root without installing the
# project as a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from ingestion.chunker import Chunk, ChunkMetadata, SemanticParentChildChunker  # noqa: E402
from ingestion.parser import DoclingParser  # noqa: E402
from ingestion.scraper import STARTER_SCHEME_URLS, SchemeScraper  # noqa: E402

logger = logging.getLogger("setu.ingest")


def parse_args() -> argparse.Namespace:
    """Define and read command-line options.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Scrape, parse and chunk scheme documents.")
    parser.add_argument(
        "--urls",
        nargs="*",
        default=None,
        help="URLs to scrape. Defaults to the five starter myScheme URLs.",
    )
    parser.add_argument(
        "--no-scrape",
        action="store_true",
        help="Skip web scraping and process only local PDFs.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory of PDFs to parse. Defaults to data/raw.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to data/processed/chunks.jsonl.",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Stop after writing chunks.jsonl; skip embedding and Qdrant upsert.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Skip scraping and parsing; embed and index an existing chunks.jsonl.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


async def collect_scraped_documents(urls: list[str]) -> list[tuple[str, str, str]]:
    """Scrape URLs into (text, source_url, scheme_name) triples.

    Args:
        urls: Pages to fetch.

    Returns:
        One triple per successfully scraped page.
    """
    scraper = SchemeScraper()
    try:
        records = await scraper.scrape_urls(urls)
    except RuntimeError as exc:
        # Missing browser/deps should not destroy a run that also has PDFs.
        logger.error("Scraping unavailable: %s", exc)
        return []
    return [(r["markdown"], r["url"], r["title"]) for r in records]


def collect_pdf_documents(pdf_dir: Path) -> list[tuple[str, str, str]]:
    """Parse every PDF in a directory into (text, source_url, scheme_name) triples.

    Args:
        pdf_dir: Directory to scan (non-recursively) for .pdf files.

    Returns:
        One triple per successfully parsed PDF.
    """
    if not pdf_dir.is_dir():
        logger.warning("PDF directory %s does not exist; skipping", pdf_dir)
        return []

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.info("No PDFs found in %s", pdf_dir)
        return []

    parser = DoclingParser()
    documents: list[tuple[str, str, str]] = []
    for pdf in pdfs:
        try:
            markdown = parser.parse_pdf(pdf)
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the run
            logger.error("Failed to parse %s: %s", pdf.name, exc)
            continue
        # Local files have no URL, so record a file:// origin for citation.
        documents.append((markdown, pdf.resolve().as_uri(), pdf.stem))
    return documents


def write_chunks(chunks: list[Chunk], out_path: Path) -> None:
    """Write chunks to disk as JSON Lines.

    JSONL rather than a single JSON array so later phases can stream the file
    without holding the whole corpus in memory.

    Args:
        chunks: Chunks to persist.
        out_path: Destination file; parent directories are created.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            # ensure_ascii=False keeps Devanagari and Tamil readable in the file.
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    """Read chunks back from a JSONL file.

    Args:
        path: Path to a chunks.jsonl produced by write_chunks.

    Returns:
        The deserialised chunks.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"No chunk file at {path}. Run without --index-only first.")

    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            chunks.append(
                Chunk(
                    id=record["id"],
                    parent_id=record["parent_id"],
                    text=record["text"],
                    metadata=ChunkMetadata(**record["metadata"]),
                )
            )
    return chunks


def index_chunks(chunks: list[Chunk]) -> int:
    """Embed chunks via Jina's API and upsert them into Qdrant Cloud.

    Args:
        chunks: Chunks to index.

    Returns:
        Number of points written, or 0 if indexing could not run.
    """
    # Imported here so that --no-index runs never open a Qdrant connection
    # or need JINA_API_KEY set at all.
    from retrieval.embeddings import JinaEmbedder
    from retrieval.vectorstore import QdrantStore

    if not chunks:
        logger.warning("No chunks to index")
        return 0

    embedder = JinaEmbedder()
    store = QdrantStore(dim=embedder.dim)

    try:
        store.ensure_collection(dim=embedder.dim)
    except Exception as exc:  # noqa: BLE001 - a down Qdrant must not lose the chunks
        logger.error(
            "Could not reach Qdrant at %s (%s). chunks.jsonl is still written; "
            "check QDRANT_URL/QDRANT_API_KEY in .env and re-run with --index-only.",
            get_settings().QDRANT_URL,
            exc,
        )
        return 0

    logger.info("Embedding %d chunks with %s", len(chunks), embedder.model_name)
    vectors = embedder.embed_batch([c.text for c in chunks])
    return store.upsert_chunks(chunks, vectors)


async def main() -> int:
    """Run the scrape -> parse -> chunk pipeline.

    Returns:
        Process exit code: 0 on success, 1 if nothing could be ingested.
    """
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    settings = get_settings()
    pdf_dir = args.pdf_dir or settings.raw_data_dir
    out_path = args.out or (settings.processed_data_dir / "chunks.jsonl")

    if args.index_only:
        chunks = load_chunks(out_path)
        logger.info("Loaded %d chunks from %s", len(chunks), out_path)
        indexed = index_chunks(chunks)
        logger.info("Indexed %d points", indexed)
        return 0 if indexed else 1

    documents: list[tuple[str, str, str]] = []
    if not args.no_scrape:
        urls = args.urls if args.urls else STARTER_SCHEME_URLS
        logger.info("Scraping %d URL(s)", len(urls))
        documents.extend(await collect_scraped_documents(urls))
    documents.extend(collect_pdf_documents(pdf_dir))

    if not documents:
        logger.error(
            "No documents ingested. Check network access, run "
            "`python -m playwright install chromium`, or drop PDFs into %s",
            pdf_dir,
        )
        return 1

    # No embeddings passed: child chunks are split by character here, so this
    # stage stays model-free. index_chunks() below is what calls Jina.
    chunker = SemanticParentChildChunker()
    all_chunks: list[Chunk] = []
    for text, source_url, scheme_name in documents:
        all_chunks.extend(chunker.chunk_document(text, source_url, scheme_name))

    write_chunks(all_chunks, out_path)

    parents = sum(1 for c in all_chunks if c.metadata.chunk_type == "parent")
    logger.info(
        "Wrote %d chunks (%d parents, %d children) from %d document(s) to %s",
        len(all_chunks),
        parents,
        len(all_chunks) - parents,
        len(documents),
        out_path,
    )

    if args.no_index:
        logger.info("--no-index set; skipping embedding and Qdrant upsert")
        return 0

    indexed = index_chunks(all_chunks)
    logger.info("Indexed %d points into collection '%s'", indexed, get_settings().QDRANT_COLLECTION)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
