"""Smoke test: run the compiled agent graph on three English queries.

Requires a working LLM (GROQ_API_KEY in .env, or LLM_PROVIDER=ollama with a
local daemon) and a populated Qdrant collection.

Usage:
    python scripts/smoke_agent.py
    python scripts/smoke_agent.py --query "Who can apply for the Design Scheme?"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.graph import build_graph, initial_state  # noqa: E402
from agents.prompts import prompt_versions  # noqa: E402
from config.settings import get_settings  # noqa: E402

logger = logging.getLogger("setu.smoke_agent")

# One query per intent the classifier knows about, so a run exercises the
# eligibility branch and the non-eligibility branch.
DEFAULT_QUERIES = [
    "What documents are required for ZED certification?",
    "I run a small manufacturing unit in Gujarat with 2 crore turnover and Udyam "
    "registration. Am I eligible for technology acquisition assistance?",
    "How much financial assistance does the Design Scheme provide?",
]


def parse_args() -> argparse.Namespace:
    """Read command-line options.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Smoke-test the SETU agent graph.")
    parser.add_argument(
        "--query", action="append", help="Query to run; repeatable. Defaults to three built-ins."
    )
    parser.add_argument("--verbose", action="store_true", help="Show node-level logging.")
    return parser.parse_args()


def print_state(index: int, query: str, state: dict[str, Any]) -> None:
    """Render one run's final state.

    Args:
        index: 1-based query number.
        query: The question asked.
        state: Final graph state.
    """
    print("\n" + "=" * 78)
    print(f"QUERY {index}: {query}")
    print("=" * 78)

    print(f"  lang               : {state.get('lang')}")
    print(f"  intent             : {state.get('intent')}")
    print(f"  profile            : {json.dumps(state.get('profile') or {}, ensure_ascii=False)}")
    print(f"  retrieved / kept   : {len(state.get('retrieved_chunks') or [])}"
          f" / {len(state.get('reranked_chunks') or [])}")
    print(f"  faithfulness       : {state.get('faithfulness_score')}")
    print(f"  retries            : {state.get('retries')}")

    verdict = state.get("eligibility_result")
    if verdict:
        print(f"  ELIGIBILITY        : {verdict['eligible']} ({verdict['scheme_name']})")
        print(f"    reason           : {verdict['reason']}")
        if verdict["missing_fields"]:
            print(f"    missing          : {', '.join(verdict['missing_fields'])}")
    else:
        print("  ELIGIBILITY        : not applicable for this intent")

    print("\n  ANSWER:")
    answer = state.get("final_answer") or "(none)"
    print(textwrap.fill(answer, width=74, initial_indent="    ", subsequent_indent="    "))

    citations = state.get("citations") or []
    if citations:
        print("\n  CITATIONS:")
        for citation in citations:
            print(f"    [{citation['label']}] {citation['scheme_name']}")
            print(f"               {citation['source_url']}")


def main() -> int:
    """Run every query through the graph and print results.

    Returns:
        Process exit code: 0 on success, 1 if the LLM is not configured.
    """
    args = parse_args()
    logging.basicConfig(
        level="INFO" if args.verbose else "WARNING",
        format="%(levelname)-7s | %(name)s | %(message)s",
    )
    settings = get_settings()

    problems = settings.validate_llm_credentials()
    if problems:
        logger.error("LLM not configured: %s", "; ".join(problems))
        print("\nFix: copy .env.example to .env and set GROQ_API_KEY, "
              "or set LLM_PROVIDER=ollama with a local Ollama daemon running.")
        return 1

    queries = args.query or DEFAULT_QUERIES
    print(f"provider : {settings.LLM_PROVIDER} ({settings.GROQ_MODEL})")
    print(f"prompts  : {prompt_versions()}")

    graph = build_graph()
    for index, query in enumerate(queries, start=1):
        # A distinct thread_id per query keeps the checkpointer from replaying
        # one question's state into the next.
        state = graph.invoke(
            initial_state(query),
            config={"configurable": {"thread_id": f"smoke-{index}"}},
        )
        print_state(index, query, state)

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
