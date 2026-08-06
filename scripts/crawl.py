"""Crawl official government sources and write chunks with full provenance.

Runs the discovery pipeline end to end: crawl -> extract -> structured metadata
-> chunk -> JSONL. Indexing stays in scripts/ingest.py, so a crawl can be
inspected before anything reaches the vector store.

Usage:
    python scripts/crawl.py --list-sources
    python scripts/crawl.py --sources msme_ministry dcmsme --max-pages 50
    python scripts/crawl.py --sources myscheme --max-depth 2 --dry-run
    python scripts/crawl.py --all --max-pages 2000 --enable-ocr
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from ingestion.chunker import Chunk, SemanticParentChildChunker  # noqa: E402
from ingestion.crawl_state import content_hash  # noqa: E402
from ingestion.crawler import CrawlConfig, GovernmentCrawler  # noqa: E402
from ingestion.fetchers import ExtractedDocument  # noqa: E402
from ingestion.scheme_extractor import extract_compendium, extract_scheme  # noqa: E402
from ingestion.sources import SOURCES, is_official_url  # noqa: E402

logger = logging.getLogger("setu.crawl")


def parse_args() -> argparse.Namespace:
    """Define and read command-line options.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Crawl official government scheme sources.")
    parser.add_argument("--sources", nargs="*", help="Source keys to crawl.")
    parser.add_argument("--all", action="store_true", help="Crawl every registered source.")
    parser.add_argument("--urls", nargs="*", help="Extra official URLs to seed.")
    parser.add_argument("--list-sources", action="store_true", help="List sources and exit.")
    parser.add_argument("--max-depth", type=int, default=2, help="Link depth from each seed.")
    parser.add_argument("--max-pages", type=int, default=200, help="Page budget for this run.")
    parser.add_argument("--enable-ocr", action="store_true", help="OCR scanned PDFs (slow).")
    parser.add_argument("--ignore-robots", action="store_true",
                        help="Do not honour robots.txt. Use only with permission.")
    parser.add_argument("--allow-cross-domain", action="store_true",
                        help="Follow links to other official domains.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Crawl and report, but write nothing. Uses a throwaway "
                             "state database so it does not consume the real frontier.")
    parser.add_argument("--reset", action="store_true",
                        help="Delete crawl state first, forcing a full re-crawl.")
    parser.add_argument("--out", type=Path, default=None, help="Chunk JSONL output path.")
    parser.add_argument("--metadata-out", type=Path, default=None,
                        help="Structured scheme metadata JSON output path.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def list_sources() -> None:
    """Print the registered official sources."""
    print(f"\n{len(SOURCES)} registered official sources:\n")
    for source in SOURCES:
        flags = []
        if source.requires_js:
            flags.append("JS")
        marker = f"  [{','.join(flags)}]" if flags else ""
        print(f"  {source.key:16} {source.tier.value:12} {source.domain}{marker}")
        if source.notes:
            print(f"                   {source.notes}")
    print()


def scheme_id_from_url(url: str) -> str:
    """Derive a stable scheme identifier from a URL.

    Args:
        url: Source URL.

    Returns:
        The final path segment without extension, or the host if the path is
        empty.
    """
    path = url.split("?")[0].rstrip("/")
    tail = path.rsplit("/", 1)[-1]
    stem = tail.rsplit(".", 1)[0] if "." in tail else tail
    return stem or (url.split("//")[-1].split("/")[0])


def to_chunks(
    documents: list[ExtractedDocument], chunker: SemanticParentChildChunker
) -> tuple[list[Chunk], list[dict[str, Any]]]:
    """Convert crawled documents into chunks and structured metadata.

    Args:
        documents: Extracted documents from the crawl.
        chunker: Configured chunker.

    Returns:
        (chunks, scheme metadata records).
    """
    chunks: list[Chunk] = []
    metadata_records: list[dict[str, Any]] = []

    for document in documents:
        doc_hash = content_hash(document.text)
        scheme_id = scheme_id_from_url(document.url)
        name = document.title or scheme_id.replace("-", " ").replace("_", " ").title()

        # A ministry's "all schemes" PDF holds dozens of schemes; extracting it
        # as one scheme finds nothing, because every section heading repeats.
        compendium = extract_compendium(
            document.text,
            source_url=document.url,
            retrieved_at=document.retrieved_at,
            doc_format=document.doc_format,
            published_at=document.published_at,
            modified_at=document.modified_at,
            content_hash=doc_hash,
        )
        if compendium:
            metadata_records.extend(record.to_dict() for record in compendium)
        else:
            metadata_records.append(
                extract_scheme(
                    document.text,
                    scheme_id=scheme_id,
                    name=name,
                    source_url=document.url,
                    retrieved_at=document.retrieved_at,
                    doc_format=document.doc_format,
                    published_at=document.published_at,
                    modified_at=document.modified_at,
                    content_hash=doc_hash,
                ).to_dict()
            )

        # Provenance travels with every chunk, so a citation can state both what
        # was said and when it was retrieved.
        provenance = {
            "doc_format": document.doc_format,
            "retrieved_at": document.retrieved_at,
            "published_at": document.published_at,
            "modified_at": document.modified_at,
            "source_title": document.title,
            "doc_hash": doc_hash,
        }
        chunks.extend(
            chunker.chunk_document(document.text, document.url, name, provenance=provenance)
        )

    return chunks, metadata_records


async def main() -> int:
    """Run the crawl and write results.

    Returns:
        Process exit code: 0 on success, 1 if nothing was ingested.
    """
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    if args.list_sources:
        list_sources()
        return 0

    if not args.sources and not args.all and not args.urls:
        logger.error("Choose --sources, --all or --urls. Use --list-sources to see options.")
        return 1

    for url in args.urls or []:
        if not is_official_url(url):
            logger.error("Refusing non-official URL: %s", url)
            return 1

    settings = get_settings()
    config = CrawlConfig(
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        enable_ocr=args.enable_ocr,
        respect_robots=not args.ignore_robots,
        same_domain_only=not args.allow_cross_domain,
    )
    if args.ignore_robots:
        logger.warning("robots.txt is being ignored — ensure you have permission to do so")

    default_state = Path(__file__).resolve().parent.parent / "data" / "crawl" / "crawl_state.db"
    if args.reset and default_state.exists():
        default_state.unlink()
        logger.info("Cleared crawl state at %s", default_state)

    # A dry run must not mark URLs as fetched in the durable frontier, or the
    # real run that follows finds nothing left to do.
    state_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.dry_run:
        state_dir = tempfile.TemporaryDirectory(prefix="setu-dryrun-")
        state_path = Path(state_dir.name) / "crawl_state.db"
    else:
        state_path = default_state

    try:
        crawler = GovernmentCrawler(config=config, state_path=state_path)
        result = await crawler.crawl(
            source_keys=None if args.all else args.sources,
            extra_urls=args.urls,
        )
    finally:
        if state_dir is not None:
            state_dir.cleanup()

    print("\n" + "=" * 70)
    print(f"  documents extracted : {len(result.documents)}")
    for key, value in result.stats.items():
        print(f"  {key:20}: {value}")
    if result.failures:
        print(f"\n  failures ({len(result.failures)}):")
        for failure in result.failures[:10]:
            print(f"    {failure['url'][:80]} -> {failure['error'][:60]}")
    print("=" * 70)

    if not result.documents:
        logger.error("No documents extracted")
        return 1

    chunker = SemanticParentChildChunker()
    chunks, metadata_records = to_chunks(result.documents, chunker)

    by_format: dict[str, int] = {}
    for document in result.documents:
        by_format[document.doc_format] = by_format.get(document.doc_format, 0) + 1
    complete = sum(1 for m in metadata_records
                   if m["eligibility"] and m["benefits"])
    print(f"\n  formats            : {by_format}")
    print(f"  chunks             : {len(chunks)}")
    print(f"  with eligibility+benefits: {complete}/{len(metadata_records)}")

    if args.dry_run:
        print("\n  --dry-run: nothing written\n")
        return 0

    out_path = args.out or (settings.processed_data_dir / "chunks.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    metadata_path = args.metadata_out or (settings.processed_data_dir / "schemes.json")
    metadata_path.write_text(
        json.dumps(metadata_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n  chunks   -> {out_path}")
    print(f"  metadata -> {metadata_path}")
    print("  next: python scripts/ingest.py --index-only\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
