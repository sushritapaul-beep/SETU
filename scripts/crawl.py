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
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from ingestion.chunker import Chunk, SemanticParentChildChunker  # noqa: E402
from ingestion.crawl_state import CrawlState, content_hash  # noqa: E402
from ingestion.crawler import CrawlConfig, GovernmentCrawler  # noqa: E402
from ingestion.document_store import DocumentStore  # noqa: E402
from ingestion.fetchers import ExtractedDocument  # noqa: E402
from ingestion.myscheme_api import MySchemeHarvester  # noqa: E402
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
    parser.add_argument("--myscheme", action="store_true",
                        help="Harvest every scheme via myScheme's public API.")
    parser.add_argument("--myscheme-limit", type=int, default=None,
                        help="Harvest at most this many myScheme schemes (trial runs).")
    parser.add_argument("--myscheme-categories", nargs="*", default=None,
                        help="Only harvest schemes in these myScheme categories.")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch myScheme records even when cached.")
    parser.add_argument("--follow-references", type=int, default=0, metavar="N",
                        help="Fetch up to N official documents (guidelines, notifications) "
                             "that stored documents link to. 0 disables.")
    parser.add_argument("--rebuild-only", action="store_true",
                        help="Skip fetching; rebuild chunks.jsonl from the document store.")
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
        facts = document.extra.get("facts") or {}

        # A ministry's "all schemes" PDF holds dozens of schemes; extracting it
        # as one scheme finds nothing, because every section heading repeats.
        # An API record is one scheme by construction, so it skips the check.
        compendium = [] if facts else extract_compendium(
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
            "scheme_id": facts.get("scheme_id") or scheme_id,
            "ministry": facts.get("ministry") or "",
            "level": facts.get("level") or "",
            "states": tuple(facts.get("states") or ()),
            "categories": tuple(facts.get("categories") or ()),
            "discovery_method": document.extra.get("discovery_method") or "crawl",
        }
        chunks.extend(
            chunker.chunk_document(document.text, document.url, name, provenance=provenance)
        )

    return chunks, metadata_records


def rebuild(store: DocumentStore, out_path: Path, metadata_path: Path | None) -> dict[str, Any]:
    """Rebuild chunks.jsonl and schemes.json from the whole document store.

    Chunks are rebuilt from the whole store, not this run's output: an
    incremental crawl returns only what changed, and writing just that would
    silently shrink the corpus on every repeat run.

    Streams one document at a time and writes its chunks immediately, so peak
    memory tracks the largest document rather than the corpus. Chunks go to a
    temporary file that replaces the old one only on success, so a crash
    mid-rebuild leaves the previous chunks.jsonl intact.

    Args:
        store: The document store.
        out_path: Destination for chunks (os.devnull for a dry run).
        metadata_path: Destination for scheme metadata; None writes none.

    Returns:
        Counts: documents, chunks, schemes, complete, formats.
    """
    chunker = SemanticParentChildChunker()
    metadata_records: list[dict[str, Any]] = []
    formats: dict[str, int] = {}
    documents = chunks = 0
    real = out_path != Path(os.devnull)
    tmp = out_path.with_name(out_path.name + ".tmp") if real else out_path
    if real:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as handle:
        for document in store.iter():
            doc_chunks, records = to_chunks([document], chunker)
            for chunk in doc_chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            metadata_records += records
            formats[document.doc_format] = formats.get(document.doc_format, 0) + 1
            documents += 1
            chunks += len(doc_chunks)
    if real:
        if documents:
            os.replace(tmp, out_path)
        else:
            tmp.unlink()  # an empty store must not wipe an existing chunks.jsonl
    if metadata_path is not None and documents:
        # json.dump writes incrementally; json.dumps would build the whole
        # file as one string first (measured: the run's peak allocation).
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata_records, handle, indent=2, ensure_ascii=False)
    return {
        "documents": documents,
        "chunks": chunks,
        "schemes": len(metadata_records),
        "complete": sum(1 for m in metadata_records if m["eligibility"] and m["benefits"]),
        "formats": formats,
    }

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

    fetching = args.sources or args.all or args.urls or args.myscheme or args.follow_references
    if not fetching and not args.rebuild_only:
        logger.error("Choose --sources, --all, --urls, --myscheme, --follow-references or "
                     "--rebuild-only. Use --list-sources to see options.")
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

    crawl_dir = Path(__file__).resolve().parent.parent / "data" / "crawl"
    default_state = crawl_dir / "crawl_state.db"
    if args.reset and default_state.exists():
        default_state.unlink()
        logger.info("Cleared crawl state at %s", default_state)

    # A dry run must not mark URLs as fetched in the durable frontier, or the
    # real run that follows finds nothing left to do — nor touch the store.
    state_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.dry_run:
        state_dir = tempfile.TemporaryDirectory(prefix="setu-dryrun-")
        state_path = Path(state_dir.name) / "crawl_state.db"
        store = DocumentStore(Path(state_dir.name) / "documents")
    else:
        state_path = default_state
        store = DocumentStore(crawl_dir / "documents")

    # Every extracted document goes straight to the store and is not kept in
    # memory: the store is the only copy, and later passes stream from it.
    outcomes: dict[str, int] = {}

    def persist(document: ExtractedDocument) -> None:
        # Written the moment a document exists, so a crash mid-crawl loses only
        # what was in flight rather than everything fetched so far.
        status = store.put(document)
        outcomes[status] = outcomes.get(status, 0) + 1

    if args.myscheme:
        harvest = await MySchemeHarvester(crawl_dir / "myscheme").harvest(
            limit=args.myscheme_limit, categories=args.myscheme_categories,
            refresh=args.refresh, on_document=persist,
        )
        print(f"\n  myScheme listed {harvest.listed}: {harvest.fetched} fetched, "
              f"{harvest.from_cache} cached, {len(harvest.failures)} failed")
        for failure in harvest.failures[:10]:
            print(f"    {failure['slug']} -> {failure['error'][:80]}")

    if args.sources or args.all or args.urls:
        # Self-healing: a URL the frontier calls "done" whose text never reached
        # the store (a crash between the two) is fetched again, not lost.
        with CrawlState(state_path) as frontier:
            missing = [u for u in frontier.done_urls() if not store.has(u)]
            if missing:
                logger.info("Requeued %d fetched-but-unstored URLs", frontier.requeue(missing))
        result = await GovernmentCrawler(config=config, state_path=state_path,
                                         on_document=persist).crawl(
            source_keys=None if args.all else args.sources, extra_urls=args.urls,
        )
        print(f"\n  crawl: {len(result.documents)} new documents, stats={result.stats}")
        for failure in result.failures[:10]:
            print(f"    {failure['url'][:80]} -> {failure['error'][:60]}")

    if args.follow_references:
        # Guidelines and notifications cited by schemes already in the store
        # (or harvested just now). Fetched as documents only (depth 0): the
        # citing page vouches for the document, not for its whole site.
        # Everything this run fetched is already in the store, so one streamed
        # pass over it sees every known URL and every citation.
        known: set[str] = set()
        cited: set[str] = set()
        for d in store.iter():
            known.add(d.url)
            if d.extra.get("discovery_method"):
                cited.update(link for link in d.links if is_official_url(link))
        refs = sorted(cited - known)[: args.follow_references]
        logger.info("Following %d official references", len(refs))
        if refs:
            ref_config = CrawlConfig(max_depth=0, max_pages=len(refs),
                                     enable_ocr=args.enable_ocr,
                                     respect_robots=not args.ignore_robots)
            def persist_reference(document: ExtractedDocument) -> None:
                document.extra["discovery_method"] = "scheme_reference"
                persist(document)

            ref_result = await GovernmentCrawler(config=ref_config, state_path=state_path,
                                                 on_document=persist_reference
                                                 ).crawl(source_keys=[], extra_urls=refs)
            print(f"\n  references: {len(refs)} attempted, "
                  f"{len(ref_result.documents)} new documents")
            for failure in ref_result.failures[:10]:
                print(f"    {failure['url'][:80]} -> {failure['error'][:60]}")

    if args.dry_run:
        stats = rebuild(store, Path(os.devnull), None)
    else:
        out_path = args.out or (settings.processed_data_dir / "chunks.jsonl")
        metadata_path = args.metadata_out or (settings.processed_data_dir / "schemes.json")
        stats = rebuild(store, out_path, metadata_path)

    print(f"\n  this run: {outcomes}  |  store: {stats['documents']} current documents, "
          f"{store.history_count()} superseded versions kept")
    if not stats["documents"]:
        logger.error("Document store is empty")
        return 1
    print(f"  formats            : {stats['formats']}")
    print(f"  chunks             : {stats['chunks']}")
    print(f"  with eligibility+benefits: {stats['complete']}/{stats['schemes']}")

    if args.dry_run:
        if state_dir is not None:
            state_dir.cleanup()
        print("\n  --dry-run: nothing written\n")
        return 0

    print(f"\n  chunks   -> {out_path}")
    print(f"  metadata -> {metadata_path}")
    print("  next: python scripts/ingest.py --index-only\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
