"""The SETU agent: a LangGraph state machine with a self-critique retry loop.

Flow:

    START
      -> detect_language          (IndicLID: hi / ta / code_mixed / en)
      -> translate_query_to_en    (no-op when the query is already English)
      -> classify_intent
      -> extract_profile
      -> retrieve_hybrid  <-------------+
      -> rerank                         |
      -> check_eligibility              | retry with a rewritten query
      -> generate_answer                | (faithfulness < 0.7, retries < 2)
      -> self_critique -----------------+
      -> translate_answer_from_en (no-op when the query was English)
      -> END

Retrieval and generation always happen in English: the corpus is English, so
translating at the edges is far cheaper and more accurate than maintaining a
separate index per language.

The loop is what makes this an agent rather than a pipeline: the graph judges
its own output and can decide to go back and retrieve differently.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    check_eligibility,
    classify_intent,
    detect_language,
    extract_profile,
    generate_answer,
    rerank,
    retrieve_hybrid,
    self_critique,
    translate_answer_from_en,
    translate_query_to_en,
)

logger = logging.getLogger(__name__)


class SetuState(TypedDict, total=False):
    """State carried through the graph.

    total=False so nodes can return partial updates and early nodes need not
    populate keys that later nodes own.
    """

    user_query: str
    lang: str
    intent: str
    # Indic layer (Phase 5): script and confidence come from IndicLID;
    # lang_codes carries every language above the code-mixed threshold.
    script: str
    lang_confidence: float
    lang_codes: list[str]
    # The English form of the query, used for retrieval and generation.
    query_en: str
    # The answer before translation, kept so evaluation scores the model's own
    # output rather than a translation of it.
    answer_en: str
    profile: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    eligibility_result: dict[str, Any] | None
    draft_answer: str
    faithfulness_score: float
    retries: int
    final_answer: str
    citations: list[dict[str, Any]]
    # Not in the original field list: holds the critic's rewritten query so the
    # retry loop can search differently without overwriting the user's words.
    search_query: str


def route_after_critique(state: SetuState) -> Literal["retry", "finalize"]:
    """Decide whether to loop back to retrieval or end the run.

    The retry decision itself is made in self_critique, which sets final_answer
    only when finalizing; this router just reads that, so the loop-termination
    logic exists in exactly one place and cannot disagree with itself.

    Args:
        state: Current graph state.

    Returns:
        "retry" to search again, "finalize" to end.
    """
    return "finalize" if state.get("final_answer") else "retry"


def build_graph(checkpointer: Any | None = None) -> Any:
    """Assemble and compile the SETU agent graph.

    Args:
        checkpointer: Checkpoint saver. Defaults to an in-memory saver, which
            keeps per-thread state for multi-turn conversations but does not
            survive a restart.

    Returns:
        The compiled graph, ready to .invoke() or .stream().
    """
    builder = StateGraph(SetuState)

    builder.add_node("detect_language", detect_language)
    builder.add_node("translate_query_to_en", translate_query_to_en)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("extract_profile", extract_profile)
    builder.add_node("retrieve_hybrid", retrieve_hybrid)
    builder.add_node("rerank", rerank)
    builder.add_node("check_eligibility", check_eligibility)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("self_critique", self_critique)
    builder.add_node("translate_answer_from_en", translate_answer_from_en)

    builder.add_edge(START, "detect_language")
    builder.add_edge("detect_language", "translate_query_to_en")
    builder.add_edge("translate_query_to_en", "classify_intent")
    builder.add_edge("classify_intent", "extract_profile")
    builder.add_edge("extract_profile", "retrieve_hybrid")
    builder.add_edge("retrieve_hybrid", "rerank")
    builder.add_edge("rerank", "check_eligibility")
    builder.add_edge("check_eligibility", "generate_answer")
    builder.add_edge("generate_answer", "self_critique")

    builder.add_conditional_edges(
        "self_critique",
        route_after_critique,
        {"retry": "retrieve_hybrid", "finalize": "translate_answer_from_en"},
    )
    builder.add_edge("translate_answer_from_en", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def initial_state(user_query: str) -> SetuState:
    """Build a clean starting state for a query.

    Args:
        user_query: The user's question, in any supported language.

    Returns:
        A state dict with counters initialised, so nodes never see None.
    """
    return SetuState(
        user_query=user_query,
        profile={},
        retrieved_chunks=[],
        reranked_chunks=[],
        retries=0,
        citations=[],
    )


# Module-level compiled graph, as the API and UI layers will import this.
# Built at import time so graph construction errors surface immediately rather
# than on the first request.
graph = build_graph()
