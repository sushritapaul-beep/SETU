"""Smoke test: run one Hindi, one Tamil and one Hinglish query through the agent.

Shows the full multilingual path — detected language, English translation,
retrieved chunks, English answer, and the answer translated back.

Requires: a populated Qdrant collection, a working LLM, and access to the GATED
IndicTrans2 repos (accept the terms on HuggingFace, then set HUGGINGFACE_TOKEN).

Usage:
    python scripts/smoke_indic.py
    python scripts/smoke_indic.py --detect-only   # no translation models needed
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.graph import build_graph, initial_state  # noqa: E402
from config.settings import get_settings  # noqa: E402
from indic.lang_detect import IndicLanguageDetector  # noqa: E402

logger = logging.getLogger("setu.smoke_indic")

# One query per script path: native Devanagari, native Tamil, romanised mix.
QUERIES: list[tuple[str, str]] = [
    ("Hindi", "ZED प्रमाणन के लिए कौन से दस्तावेज़ आवश्यक हैं?"),
    ("Tamil", "இந்த திட்டத்திற்கு விண்ணப்பிக்க என்ன ஆவணங்கள் தேவை?"),
    ("Hinglish", "ZED certification ke liye kya documents chahiye"),
]


def parse_args() -> argparse.Namespace:
    """Read command-line options.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Smoke-test the Indic language layer.")
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only run language detection; skips the gated translation models.",
    )
    parser.add_argument(
        "--only",
        choices=[label.lower() for label, _ in QUERIES],
        help=(
            "Run a single query. The full pipeline holds ~6GB of models "
            "(BGE-M3 + reranker + two IndicTrans2 models); on a 16GB machine, "
            "running all three back to back can exhaust RAM."
        ),
    )
    return parser.parse_args()


def print_detection(label: str, query: str, detection: dict[str, Any]) -> None:
    """Render a language-detection result.

    Args:
        label: Human-readable name for the query's language.
        query: The query text.
        detection: Result from IndicLanguageDetector.detect.
    """
    print("\n" + "=" * 78)
    print(f"{label}: {query}")
    print("=" * 78)
    print(f"  detected   : {detection['lang_code']} / {detection['script']} "
          f"(confidence {detection['confidence']:.4f}, model {detection.get('model')})")
    print(f"  codes      : {detection.get('codes')}")


def print_run(state: dict[str, Any]) -> None:
    """Render a full agent run.

    Args:
        state: Final graph state.
    """
    print(f"  intent     : {state.get('intent')}")
    print(f"  query_en   : {state.get('query_en')}")
    print(f"  retrieved  : {len(state.get('retrieved_chunks') or [])} -> "
          f"kept {len(state.get('reranked_chunks') or [])}")
    for citation in (state.get("citations") or [])[:3]:
        print(f"      [{citation['label']}] {citation['scheme_name'][:52]}")
    print(f"  faithfulness: {state.get('faithfulness_score')}")

    print("\n  ENGLISH ANSWER:")
    print(textwrap.fill(state.get("answer_en") or "(none)", width=74,
                        initial_indent="    ", subsequent_indent="    "))
    print("\n  TRANSLATED ANSWER:")
    print(textwrap.fill(state.get("final_answer") or "(none)", width=74,
                        initial_indent="    ", subsequent_indent="    "))


def main() -> int:
    """Run each query and print the multilingual pipeline's output.

    Returns:
        Process exit code: 0 on success, 1 if prerequisites are missing.
    """
    args = parse_args()
    logging.basicConfig(level="WARNING", format="%(levelname)-7s | %(name)s | %(message)s")
    settings = get_settings()

    detector = IndicLanguageDetector()
    selected = [(label, q) for label, q in QUERIES if not args.only or label.lower() == args.only]

    if args.detect_only:
        for label, query in selected:
            print_detection(label, query, detector.detect(query))
        print("\n" + "=" * 78)
        return 0

    problems = settings.validate_llm_credentials()
    if problems:
        logger.error("LLM not configured: %s", "; ".join(problems))
        return 1
    if settings.HUGGINGFACE_TOKEN is None:
        print("\nHUGGINGFACE_TOKEN is not set. The IndicTrans2 repos are gated:")
        print("  1. Accept terms at https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M")
        print("  2. Accept terms at https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M")
        print("  3. Put a read token in HUGGINGFACE_TOKEN in .env")
        print("\nRe-run with --detect-only to exercise language detection alone.")
        return 1

    graph = build_graph()
    for index, (label, query) in enumerate(selected, start=1):
        print_detection(label, query, detector.detect(query))
        state = graph.invoke(
            initial_state(query),
            config={"configurable": {"thread_id": f"indic-smoke-{index}"}},
        )
        print_run(state)

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
