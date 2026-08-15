"""Run RAGAS over the gold set and record the scores.

Uses Groq Llama-3.3 as the judge rather than the OpenAI default, so evaluation
costs nothing beyond the free tier and needs no second provider.

Usage:
    python eval/ragas_eval.py
    python eval/ragas_eval.py --limit 5          # quick check
    python eval/ragas_eval.py --lang hi          # one language only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _install_vertexai_shim() -> None:
    """Satisfy a stale import inside ragas 0.4.3.

    ragas.llms.base does `from langchain_community.chat_models.vertexai import
    ChatVertexAI`, but langchain-community 0.4.x removed that module. Pinning
    langchain-community back to 0.3.x would drag langchain-core to 0.3.x and
    break the pinned langchain 1.3.11, so a stub is registered instead. SETU
    never uses Vertex AI, so nothing can call into it.
    """
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    module = types.ModuleType(name)

    class ChatVertexAI:  # noqa: D106 - placeholder for an unused provider
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Vertex AI is not used by SETU")

    module.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules[name] = module


_install_vertexai_shim()

from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from agents.graph import build_graph, initial_state  # noqa: E402
from config.settings import get_settings  # noqa: E402
from observability.langfuse_setup import build_config, flush  # noqa: E402

logger = logging.getLogger("setu.eval")

# Metric key -> factory. strictness=1 on ResponseRelevancy is required against
# Groq, which rejects the multi-generation sampling RAGAS requests by default.
AVAILABLE_METRICS: dict[str, Any] = {
    "faithfulness": Faithfulness,
    "answer_relevancy": lambda: ResponseRelevancy(strictness=1),
    "context_precision": LLMContextPrecisionWithReference,
    "context_recall": LLMContextRecall,
}

GOLD_SET = Path(__file__).resolve().parent / "gold_set.jsonl"
HISTORY_DIR = Path(__file__).resolve().parent / "history"
# Agent outputs are cached here before judging. Generating them costs ~45 min and
# most of a day's free-tier tokens; losing that because the judge ran out of
# quota is not acceptable, so the two phases are separable.
RUNS_DIR = HISTORY_DIR / "_runs"

# A cheaper judge by default. Scoring 25 answers across 4 metrics is ~100 calls,
# and the larger answer-generation model exhausts a 100k/day free tier before
# it finishes. The 8B model is a weaker judge, so record which one produced a
# score.
DEFAULT_JUDGE = "llama-3.1-8b-instant"

# answer_relevancy's own similarity scoring needs a real local embedding
# model — a HuggingFace Hub id, not the runtime pipeline's Jina API model.
RAGAS_EMBEDDING_MODEL = "BAAI/bge-m3"


def parse_args() -> argparse.Namespace:
    """Read command-line options.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Score SETU against the gold set with RAGAS.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N queries.")
    parser.add_argument("--lang", choices=["hi", "ta", "hinglish"], help="Filter by language.")
    parser.add_argument("--out", type=Path, default=None, help="Override the output path.")
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE,
        help=f"Groq model used for scoring (default: {DEFAULT_JUDGE}).",
    )
    parser.add_argument(
        "--from-cache",
        type=Path,
        default=None,
        help="Score a previously cached run instead of re-running the agent.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Run the agent and cache the answers, but do not score them.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=sorted(AVAILABLE_METRICS),
        default=None,
        help=(
            "Metrics to score. Each one costs a judge call per sample, and the "
            "judge model is throttled to 6000 tokens/minute, so halving the "
            "metric count roughly halves the wall time. Defaults to all four."
        ),
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser.parse_args()


def build_metrics(names: list[str] | None) -> list[Any]:
    """Instantiate the requested RAGAS metrics.

    Args:
        names: Metric keys to build. None builds all of them.

    Returns:
        Metric instances ready to pass to evaluate().
    """
    selected = names or sorted(AVAILABLE_METRICS)
    return [AVAILABLE_METRICS[name]() for name in selected]


def load_gold_set(path: Path, lang: str | None, limit: int | None) -> list[dict[str, Any]]:
    """Read gold-set rows, optionally filtered and stratified.

    When a limit is applied it samples proportionally across languages. The file
    is ordered hi-*, ta-*, cm-*, so a naive head-slice would return only Hindi
    and produce scores that say nothing about a multilingual system.

    Args:
        path: Path to gold_set.jsonl.
        lang: Language filter, or None for all.
        limit: Maximum rows, or None for all.

    Returns:
        The selected rows, grouped by language in the original order.

    Raises:
        FileNotFoundError: If the gold set is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Gold set not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if lang:
        rows = [r for r in rows if r["lang"] == lang]
    if not limit or limit >= len(rows):
        return rows

    by_lang: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_lang.setdefault(row["lang"], []).append(row)

    # Proportional share per language, with at least one row each so no language
    # silently drops out of the report.
    selected: list[dict[str, Any]] = []
    for code, group in by_lang.items():
        share = max(1, round(limit * len(group) / len(rows)))
        selected.extend(group[:share])
        logger.info("stratified: %s -> %d of %d", code, min(share, len(group)), len(group))
    return selected[:limit]


def run_agent(graph: Any, row: dict[str, Any]) -> dict[str, Any]:
    """Run one gold-set query through the agent.

    Args:
        graph: The compiled agent graph.
        row: A gold-set row.

    Returns:
        {answer, contexts} for RAGAS. Empty answer on failure, which RAGAS
        scores as unfaithful rather than crashing the run.
    """
    config, _ = build_config(f"eval-{row['id']}", row["query"], extra_tags=["eval:ragas"])
    try:
        state = graph.invoke(initial_state(row["query"]), config=config)
    except Exception as exc:  # noqa: BLE001 - one bad query must not end the run
        logger.error("Agent failed on %s: %s", row["id"], exc)
        return {"answer": "", "contexts": []}

    # Scored in English: the gold answers are English, and judging a translation
    # would measure the translation API rather than the RAG pipeline.
    answer = state.get("answer_en") or state.get("final_answer") or ""
    contexts = [c.get("text", "") for c in (state.get("reranked_chunks") or [])]
    return {"answer": answer, "contexts": contexts}


def build_judge(model: str) -> tuple[Any, Any]:
    """Construct the LLM and embeddings RAGAS uses to score.

    Args:
        model: Groq model id to judge with.

    Returns:
        (llm, embeddings) already wrapped for RAGAS.

    Raises:
        RuntimeError: If the configured LLM has no credentials.
    """
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    problems = settings.validate_llm_credentials()
    if problems:
        raise RuntimeError("; ".join(problems))

    judge = ChatGroq(
        model=model,
        # Deterministic judging: a sampled judge makes runs incomparable.
        temperature=0.0,
        api_key=settings.GROQ_API_KEY.get_secret_value() if settings.GROQ_API_KEY else None,
    )
    # answer_relevancy needs a real embedding model for its own similarity
    # scoring. This used to reuse settings.EMBEDDING_MODEL, back when that was
    # BAAI/bge-m3 (a HuggingFace Hub id HuggingFaceEmbeddings can load). It's
    # jina-embeddings-v3 now — a Jina API model id, not loadable here — so
    # RAGAS's own embedding choice is deliberately decoupled from the
    # pipeline's and hardcoded to a real local model instead.
    embeddings = HuggingFaceEmbeddings(model_name=RAGAS_EMBEDDING_MODEL)
    return LangchainLLMWrapper(judge), LangchainEmbeddingsWrapper(embeddings)


def main() -> int:
    """Evaluate the agent and write a dated report.

    Returns:
        Process exit code: 0 on success, 1 if nothing could be evaluated.
    """
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)-7s | %(name)s | %(message)s")

    if args.from_cache:
        cached = json.loads(args.from_cache.read_text(encoding="utf-8"))
        rows, samples = cached["rows"], cached["samples"]
        logger.info("Loaded %d cached answers from %s", len(samples), args.from_cache)
    else:
        rows = load_gold_set(GOLD_SET, args.lang, args.limit)
        if not rows:
            logger.error("No gold-set rows selected")
            return 1
        logger.info("Evaluating %d queries", len(rows))

        graph = build_graph()
        samples = []
        for index, row in enumerate(rows, start=1):
            logger.info("[%d/%d] %s (%s)", index, len(rows), row["id"], row["lang"])
            result = run_agent(graph, row)
            samples.append(
                {
                    "user_input": row["query"],
                    "response": result["answer"],
                    "retrieved_contexts": result["contexts"] or [""],
                    "reference": row["expected_answer"],
                }
            )

        # Cache BEFORE judging. Generation is the expensive half; if the judge
        # runs out of quota, re-run with --from-cache instead of starting over.
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = RUNS_DIR / f"{datetime.now(UTC):%Y-%m-%d-%H%M%S}.json"
        cache_path.write_text(
            json.dumps({"rows": rows, "samples": samples}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Cached answers to %s", cache_path)

        empty = sum(1 for s in samples if not s["response"].strip())
        if empty:
            logger.warning(
                "%d/%d answers are empty (likely rate-limited); scores will be depressed",
                empty,
                len(samples),
            )

        if args.generate_only:
            print(f"\nCached {len(samples)} answers to {cache_path}")
            print(f"Score them later with:\n  python eval/ragas_eval.py --from-cache {cache_path}")
            return 0

    llm, embeddings = build_judge(args.judge_model)
    dataset = EvaluationDataset.from_list(samples)

    # RAGAS parallelises metrics by default. llama-3.1-8b-instant allows only
    # 6000 tokens/minute and each judging call carries the full context (~2500
    # tokens), so even two workers spends most of its time in 429 backoff. One
    # worker is slower per step but finishes sooner overall.
    from ragas.run_config import RunConfig

    run_config = RunConfig(max_workers=1, max_retries=10, max_wait=120, timeout=600)

    metrics = build_metrics(args.metrics)
    logger.info(
        "Scoring %d samples on %d metrics (%d judge calls)",
        len(samples), len(metrics), len(samples) * len(metrics),
    )

    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    summary = {
        key: (round(float(value), 4) if isinstance(value, int | float) else value)
        for key, value in scores._repr_dict.items()
    }
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "queries": len(rows),
        "lang_filter": args.lang,
        # Recorded because a weaker judge produces different numbers; comparing
        # runs scored by different models would be meaningless.
        "judge_model": args.judge_model,
        "answer_model": get_settings().GROQ_MODEL,
        # Recorded so a partial run is never mistaken for a full one.
        "metrics_scored": args.metrics or sorted(AVAILABLE_METRICS),
        "empty_answers": sum(1 for s in samples if not s["response"].strip()),
        "scores": summary,
        "per_query": [
            {"id": row["id"], "lang": row["lang"], "scheme_id": row["scheme_id"]}
            for row in rows
        ],
    }

    out_path = args.out or (HISTORY_DIR / f"{datetime.now(UTC):%Y-%m-%d}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 60)
    for name, value in summary.items():
        print(f"  {name:32} {value}")
    print("=" * 60)
    print(f"written: {out_path}")

    flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
