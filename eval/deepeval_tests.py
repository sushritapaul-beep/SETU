"""DeepEval quality gate: a fast subset check suitable for CI.

Separate from tests/ because these call a real LLM and a live Qdrant, so they
are excluded from the normal offline suite. Run them explicitly:

    pytest eval/deepeval_tests.py -m deepeval -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.graph import build_graph, initial_state  # noqa: E402
from config.settings import get_settings  # noqa: E402
from observability.langfuse_setup import build_config, flush  # noqa: E402

GOLD_SET = Path(__file__).resolve().parent / "gold_set.jsonl"

# Thresholds the pipeline must clear for the build to pass.
MIN_FAITHFULNESS = 0.7
MIN_ANSWER_RELEVANCY = 0.7

# Kept small so the whole gate finishes inside ~2 minutes: 10 English-answer
# checks, skewed to the schemes with the richest scraped content.
SUBSET_IDS = [
    "hi-01", "hi-02", "hi-03", "hi-07", "hi-08",
    "hi-11", "hi-12", "ta-03", "cm-01", "cm-04",
]


def _load_subset() -> list[dict[str, Any]]:
    """Read the gold-set rows used by this gate.

    Returns:
        The selected rows, in SUBSET_IDS order.
    """
    if not GOLD_SET.exists():
        return []
    rows = {
        json.loads(line)["id"]: json.loads(line)
        for line in GOLD_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return [rows[i] for i in SUBSET_IDS if i in rows]


SUBSET = _load_subset()


def _requirements_met() -> tuple[bool, str]:
    """Check that a live LLM and a populated collection are available.

    Returns:
        (ready, reason). reason explains the skip when not ready.
    """
    settings = get_settings()
    problems = settings.validate_llm_credentials()
    if problems:
        return False, "; ".join(problems)
    try:
        from retrieval.vectorstore import QdrantStore

        store = QdrantStore()
        if not store.client.collection_exists(store.collection):
            return False, f"collection '{store.collection}' does not exist"
        if store.client.count(store.collection, exact=True).count == 0:
            return False, f"collection '{store.collection}' is empty"
    except Exception as exc:  # noqa: BLE001 - a skip beats a confusing failure
        return False, f"Qdrant unreachable: {exc}"
    return True, ""


READY, SKIP_REASON = _requirements_met()

pytestmark = [
    pytest.mark.deepeval,
    pytest.mark.skipif(not READY, reason=f"live dependencies unavailable: {SKIP_REASON}"),
    pytest.mark.skipif(not SUBSET, reason="gold_set.jsonl is missing or empty"),
]


@pytest.fixture(scope="module", autouse=True)
def _flush_traces() -> Any:
    """Send buffered spans before the module's process exits.

    Langfuse batches in a background thread, so without this the traces for a
    whole eval run are discarded at exit — the run that most needs a trace is
    the one that produced no trace at all.
    """
    yield
    flush()


@pytest.fixture(scope="module")
def graph() -> Any:
    """Compile the agent once for the whole module."""
    return build_graph()


@pytest.fixture(scope="module")
def metrics() -> tuple[Any, Any]:
    """Build the DeepEval metrics, judged by the configured Groq model.

    Returns:
        (faithfulness, answer_relevancy) metric instances.
    """
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.models import GPTModel

    settings = get_settings()
    # DeepEval routes through LiteLLM, which addresses Groq as "groq/<model>".
    judge = GPTModel(
        model=f"groq/{settings.GROQ_MODEL}",
        api_key=settings.GROQ_API_KEY.get_secret_value() if settings.GROQ_API_KEY else None,
        temperature=0.0,
    )
    return (
        FaithfulnessMetric(threshold=MIN_FAITHFULNESS, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=MIN_ANSWER_RELEVANCY, model=judge, include_reason=True),
    )


@pytest.mark.parametrize("row", SUBSET, ids=[r["id"] for r in SUBSET])
def test_answer_is_faithful_and_relevant(
    row: dict[str, Any], graph: Any, metrics: tuple[Any, Any]
) -> None:
    """Each subset query must produce a grounded, on-topic answer.

    Args:
        row: A gold-set row.
        graph: The compiled agent.
        metrics: (faithfulness, answer_relevancy).
    """
    from deepeval.test_case import LLMTestCase

    faithfulness, relevancy = metrics

    config, _ = build_config(f"deepeval-{row['id']}", row["query"], extra_tags=["eval:deepeval"])
    state = graph.invoke(initial_state(row["query"]), config=config)

    # Judged in English: the gold answers are English, so scoring the translated
    # output would measure the translation API rather than the RAG pipeline.
    answer = state.get("answer_en") or state.get("final_answer") or ""
    contexts = [c.get("text", "") for c in (state.get("reranked_chunks") or [])]

    assert answer.strip(), f"{row['id']} produced no answer"
    assert contexts, f"{row['id']} retrieved no context"

    case = LLMTestCase(
        input=row["query"],
        actual_output=answer,
        expected_output=row["expected_answer"],
        retrieval_context=contexts,
    )

    faithfulness.measure(case)
    relevancy.measure(case)

    assert faithfulness.score >= MIN_FAITHFULNESS, (
        f"{row['id']} faithfulness {faithfulness.score:.2f} < {MIN_FAITHFULNESS}: "
        f"{faithfulness.reason}"
    )
    assert relevancy.score >= MIN_ANSWER_RELEVANCY, (
        f"{row['id']} answer_relevancy {relevancy.score:.2f} < {MIN_ANSWER_RELEVANCY}: "
        f"{relevancy.reason}"
    )
