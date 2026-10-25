"""Report corpus coverage stage by stage, from discovery to the vector index.

Each stage is measured against the stage before it, so a loss shows up where it
happens: discovered but never fetched, fetched but rejected, stored but never
chunked, chunked but never indexed. Where the true denominator is unknowable
(how many official documents exist in total), the report says so instead of
printing a percentage.

Usage:
    python scripts/coverage.py                 # local state only
    python scripts/coverage.py --online        # also ask myScheme for its total
    python scripts/coverage.py --qdrant        # also count indexed points
    python scripts/coverage.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings  # noqa: E402
from ingestion.document_store import DocumentStore  # noqa: E402
from ingestion.sources import SOURCES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CRAWL = ROOT / "data" / "crawl"


def _ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)" if denominator \
        else f"{numerator}/0"


def crawl_stage(db: Path) -> dict[str, Any]:
    """Discovery, retrieval and rejection counts from the crawl frontier."""
    if not db.exists():
        return {}
    conn = sqlite3.connect(db)
    by_status = dict(conn.execute("SELECT status, COUNT(*) FROM urls GROUP BY status"))
    # Reasons are grouped by their leading phrase so "HTTP 404" across 300 URLs
    # is one line, not 300.
    reasons = Counter(
        (row[0] or "").split(" (")[0].split(":")[0][:60]
        for row in conn.execute("SELECT error FROM urls WHERE status IN ('failed','skipped')")
    )
    per_source = {
        key: dict(Counter(s for k, s in rows if k == key))
        for rows in [conn.execute("SELECT source_key, status FROM urls").fetchall()]
        for key in {k for k, _ in rows}
    }
    conn.close()
    return {"by_status": by_status, "loss_reasons": dict(reasons.most_common(15)),
            "per_source": per_source}


def myscheme_total() -> int | None:
    """How many schemes myScheme itself lists — the denominator for its coverage."""
    import httpx

    from ingestion.myscheme_api import SEARCH_URL
    try:
        response = httpx.get(f"{SEARCH_URL}?lang=en&q=%5B%5D&keyword=&sort=&from=0&size=1",
                             timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        return int(response.json()["data"]["hits"]["page"]["total"])
    except Exception:  # noqa: BLE001 - an offline report is still useful
        return None


def main() -> int:
    """Print (and optionally save) the coverage report.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Corpus coverage report.")
    parser.add_argument("--online", action="store_true", help="Query myScheme's live total.")
    parser.add_argument("--qdrant", action="store_true", help="Count points in Qdrant.")
    parser.add_argument("--json", type=Path, default=None, help="Also write the report here.")
    args = parser.parse_args()

    report: dict[str, Any] = {}

    verified = [s for s in SOURCES if s.verified]
    crawl = crawl_stage(CRAWL / "crawl_state.db")
    crawled = {k for k, statuses in crawl.get("per_source", {}).items()
               if k and statuses.get("done")}
    report["sources"] = {"registered": len(SOURCES), "verified": len(verified),
                         "with_fetched_documents": sorted(crawled)}
    report["crawl"] = crawl

    cached = len(list((CRAWL / "myscheme").glob("*.json"))) if (CRAWL / "myscheme").exists() else 0
    total = myscheme_total() if args.online else None
    report["myscheme"] = {"listed_by_portal": total, "fetched_and_cached": cached}

    store = DocumentStore(CRAWL / "documents")
    docs = store.all()
    report["documents"] = {
        "current": len(docs),
        "superseded_versions": store.history_count(),
        "by_discovery_method": dict(Counter(d.extra.get("discovery_method", "crawl")
                                            for d in docs)),
        "by_format": dict(Counter(d.doc_format for d in docs)),
        "below_quality_floor": sum(1 for d in docs if not d.is_useful),
    }

    chunks_path = get_settings().processed_data_dir / "chunks.jsonl"
    rows = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if chunks_path.exists() else []
    chunked_urls = {r["metadata"]["source_url"] for r in rows}
    report["chunks"] = {
        "total": len(rows),
        "by_type": dict(Counter(r["metadata"]["chunk_type"] for r in rows)),
        "documents_represented": len(chunked_urls),
        "documents_with_no_chunks": sorted({d.url for d in docs} - chunked_urls)[:20],
    }

    if args.qdrant:
        from retrieval.vectorstore import QdrantStore
        store_q = QdrantStore()
        report["index"] = {"points": store_q.client.count(store_q.collection, exact=True).count}

    # ---- printed summary: each stage against its predecessor ----
    st = crawl.get("by_status", {})
    print("\nSOURCE COVERAGE  (denominator = registry; the true official universe is larger)")
    print(f"  verified sources with fetched documents: "
          f"{_ratio(len(crawled & {s.key for s in verified}), len(verified))}")
    print("\nCRAWL FUNNEL")
    attempted = sum(st.get(k, 0) for k in ("done", "duplicate", "failed", "skipped"))
    print(f"  discovered URLs           : {sum(st.values())}")
    print(f"  attempted                 : {_ratio(attempted, sum(st.values()))}")
    print(f"  kept (done)               : {_ratio(st.get('done', 0), attempted)}")
    print(f"  never attempted (pending) : {st.get('pending', 0)}")
    for reason, count in crawl.get("loss_reasons", {}).items():
        print(f"    lost: {count:5}  {reason}")
    print("\nMYSCHEME API")
    print(f"  fetched : {_ratio(cached, total) if total else f'{cached} (run --online for total)'}")
    print("\nDOCUMENTS")
    for key, value in report["documents"].items():
        print(f"  {key:22}: {value}")
    print("\nCHUNKS / INDEX")
    print(f"  documents chunked : {_ratio(len(chunked_urls), len(docs))}")
    print(f"  chunks            : {report['chunks']['total']} {report['chunks']['by_type']}")
    if "index" in report:
        print(f"  indexed points    : {_ratio(report['index']['points'], len(rows))}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
