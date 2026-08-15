"""Individual LangGraph node implementations.

Every node takes the full SetuState and returns only the keys it changed, which
is what lets LangGraph merge updates and checkpoint each step independently.

All external dependencies (LLM, vector store, reranker, rule checker) are reached
through module-level getter functions so tests can monkeypatch them without a
network, a model download, or a running Qdrant.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from agents.prompts import GROUNDED_ANSWERER, PROFILE_EXTRACTOR, QUERY_CLASSIFIER, SELF_CRITIC
from agents.tools import EligibilityChecker
from config.settings import get_settings
from observability.langfuse_setup import observation, score_trace, set_trace_io

if TYPE_CHECKING:  # avoids a circular import: graph.py imports this module
    from agents.graph import SetuState

logger = logging.getLogger(__name__)

# Retry policy for the self-critique loop.
FAITHFULNESS_THRESHOLD = 0.7
MAX_RETRIES = 2

# Evaluation scores answer_en, so translating the answer back is wasted work
# there: it loads a second ~1GB model and runs the slow quadratic decode path.
# Set SETU_SKIP_ANSWER_TRANSLATION=1 to skip it. Never set this for serving —
# users would receive English answers to Hindi questions.
SKIP_ANSWER_TRANSLATION = os.environ.get("SETU_SKIP_ANSWER_TRANSLATION", "").lower() in {
    "1",
    "true",
    "yes",
}

VALID_INTENTS = {"eligibility", "procedure", "list", "benefit_amount"}

# Citation labels use a short id prefix: full 32-char digests make the prompt
# noisy and the model more likely to mangle them.
CITATION_LEN = 8


# ---------------------------------------------------------------- providers --

@lru_cache(maxsize=1)
def get_llm() -> Any:
    """Build the configured chat model.

    Returns:
        A LangChain chat model for the provider named in settings.

    Raises:
        RuntimeError: If the selected provider is missing credentials.
    """
    settings = get_settings()
    problems = settings.validate_llm_credentials()
    if problems:
        raise RuntimeError("; ".join(problems))

    if settings.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.GROQ_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_retries=settings.LLM_MAX_RETRIES,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            api_key=settings.GROQ_API_KEY.get_secret_value() if settings.GROQ_API_KEY else None,
            # qwen3.6-27b is a reasoning model; SETU parses JSON straight out of
            # the response, so thinking output must not leak into it. Ignored
            # by non-reasoning models if GROQ_MODEL is overridden.
            reasoning_effort="none",
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
    )


@lru_cache(maxsize=1)
def get_store() -> Any:
    """Build the Qdrant store wired to an embedder.

    Returns:
        A QdrantStore ready for hybrid search.
    """
    from retrieval.embeddings import JinaEmbedder
    from retrieval.vectorstore import QdrantStore

    embedder = JinaEmbedder()
    return QdrantStore(embedder=embedder)


@lru_cache(maxsize=1)
def get_reranker() -> Any:
    """Build the cross-encoder reranker.

    Returns:
        A JinaReranker instance.
    """
    from retrieval.reranker import JinaReranker

    return JinaReranker()


@lru_cache(maxsize=1)
def get_checker() -> EligibilityChecker:
    """Build the deterministic eligibility checker.

    Returns:
        An EligibilityChecker over the configured rules file.
    """
    return EligibilityChecker()


@lru_cache(maxsize=1)
def get_detector() -> Any:
    """Build the IndicLID language detector.

    Returns:
        An IndicLanguageDetector instance.
    """
    from indic.lang_detect import IndicLanguageDetector

    return IndicLanguageDetector()


@lru_cache(maxsize=1)
def get_translator() -> Any:
    """Build the Sarvam-backed translator.

    Returns:
        An IndicTranslator instance.
    """
    from indic.translator import IndicTranslator

    return IndicTranslator()


# ------------------------------------------------------------------ helpers --

def _invoke_llm(prompt: str, run_name: str) -> str:
    """Call the chat model and return its text, or an empty string on failure.

    Wrapped so that a rate limit or a dead API degrades the answer instead of
    crashing the whole graph run.

    Args:
        prompt: Fully rendered prompt text.
        run_name: Name for this call in traces. Without it all four LLM calls in
            a run are recorded as "ChatGroq" and cannot be told apart or
            compared against each other.

    Returns:
        The model's text response, or "" if the call failed.
    """
    try:
        # Only run_name is passed: LangChain merges this with the ambient config,
        # so the inherited Langfuse callbacks stay attached.
        response = get_llm().invoke(prompt, config={"run_name": run_name})
    except Exception as exc:  # noqa: BLE001 - any provider error must degrade
        logger.error("LLM call failed (%s): %s", type(exc).__name__, exc)
        return ""
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


def _parse_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response.

    Models wrap JSON in prose or markdown fences even when told not to, so the
    object is located rather than assumed to be the whole string.

    Args:
        text: Raw model output.

    Returns:
        The parsed object, or an empty dict if none could be read.
    """
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.S)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return {}
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse JSON from model output: %s", exc)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks as numbered, citable passages.

    Args:
        chunks: Hits with "chunk_id" and "text".

    Returns:
        A newline-separated context block.
    """
    lines = []
    for chunk in chunks:
        label = (chunk.get("chunk_id") or "")[:CITATION_LEN]
        text = " ".join(chunk.get("text", "").split())
        lines.append(f"[{label}] ({chunk.get('scheme_name', 'unknown scheme')}) {text}")
    return "\n\n".join(lines)


def _chunk_digest(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarise chunks for a trace: what was found, without repeating the text.

    The full passage text is already on the enclosing node span, so recording it
    again on the retrieval observation would double the trace payload for no
    extra insight. What is needed here is which passages ranked where.

    Args:
        chunks: Retrieval hits.

    Returns:
        One compact record per chunk, in rank order.
    """
    return [
        {
            "chunk_id": chunk.get("chunk_id"),
            "scheme_name": chunk.get("scheme_name"),
            "score": chunk.get("score") or chunk.get("rerank_score"),
        }
        for chunk in chunks
    ]


def _scheme_id_from_url(source_url: str) -> str:
    """Derive a scheme_id from a myScheme URL.

    The final path segment of a myScheme URL is the same slug used as scheme_id
    in eligibility_rules.json, so no separate lookup table is needed.

    Args:
        source_url: A scheme page URL.

    Returns:
        The slug, or "" if the URL is empty.
    """
    return source_url.rstrip("/").rsplit("/", 1)[-1] if source_url else ""


# -------------------------------------------------------------------- nodes --

def _effective_query(state: SetuState) -> str:
    """The query text downstream nodes should search and reason over.

    Prefers the critic's rewritten query, then the English translation, then the
    user's original words — so retrieval always runs against English text when a
    translation exists.

    Args:
        state: Current graph state.

    Returns:
        The query to use.
    """
    return state.get("search_query") or state.get("query_en") or state["user_query"]


def detect_language(state: SetuState) -> dict[str, Any]:
    """Identify the query's language and script with IndicLID.

    Args:
        state: Current graph state.

    Returns:
        State update with "lang", "script" and "lang_confidence".
    """
    try:
        result = get_detector().detect(state["user_query"])
    except Exception as exc:  # noqa: BLE001 - never fail a query on detection
        logger.error("Language detection unavailable (%s): %s", type(exc).__name__, exc)
        return {"lang": "en", "script": "Latn", "lang_confidence": 0.0}

    logger.info(
        "detect_language: %s (%s, %.2f) via %s",
        result["lang_code"],
        result["script"],
        result["confidence"],
        result.get("model", "?"),
    )
    return {
        "lang": result["lang_code"],
        "script": result["script"],
        "lang_confidence": result["confidence"],
        "lang_codes": result.get("codes", []),
    }


def translate_query_to_en(state: SetuState) -> dict[str, Any]:
    """Translate a non-English query into English for retrieval.

    The corpus is English, so matching happens in English rather than
    maintaining a separate index per language.

    Args:
        state: Current graph state.

    Returns:
        State update with "query_en".
    """
    lang = state.get("lang", "en")
    query = state["user_query"]

    # Guard inside the node rather than with a conditional edge: the graph stays
    # linear and readable, and an English query simply passes straight through.
    if lang == "en":
        return {"query_en": query}

    source = _source_language_tag(state)
    try:
        english = get_translator().to_english(query, source)
    except Exception as exc:  # noqa: BLE001 - fall back to the original text
        logger.error("Query translation failed (%s): %s", type(exc).__name__, exc)
        return {"query_en": query}

    logger.info("translate_query_to_en: %r -> %r", query[:60], english[:60])
    return {"query_en": english}


def translate_answer_from_en(state: SetuState) -> dict[str, Any]:
    """Translate the finished English answer back into the user's language.

    Args:
        state: Current graph state.

    Returns:
        State update with "answer_en" preserved and "final_answer" localised.
    """
    lang = state.get("lang", "en")
    answer = state.get("final_answer") or ""

    if lang == "en" or not answer.strip():
        return {"answer_en": answer}

    if SKIP_ANSWER_TRANSLATION:
        logger.info("translate_answer_from_en: skipped (SETU_SKIP_ANSWER_TRANSLATION)")
        return {"answer_en": answer}

    target = _source_language_tag(state)
    try:
        localised = get_translator().from_english(answer, target)
    except Exception as exc:  # noqa: BLE001 - an English answer beats no answer
        logger.error("Answer translation failed (%s): %s", type(exc).__name__, exc)
        return {"answer_en": answer}

    # The query's own script is Latin — either genuinely code-mixed (two
    # languages clear IndicLID's threshold) or a single language IndicLID is
    # confident is romanised (e.g. "hin_Latn" alone). Either way the user
    # typed in Latin script, so the answer should come back in it too, rather
    # than Devanagari/Tamil they may not read fluently.
    if state.get("script") == "Latn":
        localised = get_translator().to_romanized(localised, target)

    logger.info("translate_answer_from_en: -> %s", target)
    # The English text is kept so evaluation can score the model's actual output
    # rather than a translation of it.
    return {"answer_en": answer, "final_answer": localised}


def _source_language_tag(state: SetuState) -> str:
    """Work out which language to translate from or into.

    Args:
        state: Current graph state.

    Returns:
        A language tag IndicTranslator can resolve to a FLORES-200 code.
    """
    lang = state.get("lang", "en")
    if lang == "code_mixed":
        # Code-mixed text has no single source language. Prefer the first
        # detected non-English code, since that is the part English retrieval
        # would otherwise miss.
        for code in state.get("lang_codes") or []:
            if not code.startswith("eng"):
                return code
        return "hin_Deva"
    script = state.get("script") or ""
    return f"{lang}_{script}" if script else lang


def classify_intent(state: SetuState) -> dict[str, Any]:
    """Classify what kind of answer the user is asking for.

    Args:
        state: Current graph state.

    Returns:
        State update with "intent".
    """
    raw = _invoke_llm(QUERY_CLASSIFIER.format(query=state["user_query"]), "classify-intent")
    parsed = _parse_json(raw)
    intent = parsed.get("intent")
    if intent not in VALID_INTENTS:
        # An unusable classification should not stop the run; "procedure" is the
        # most general intent and never triggers an eligibility verdict.
        logger.warning("Unrecognised intent %r; defaulting to 'procedure'", intent)
        intent = "procedure"
    return {"intent": intent}


def extract_profile(state: SetuState) -> dict[str, Any]:
    """Pull a structured applicant profile out of the user's message.

    Args:
        state: Current graph state.

    Returns:
        State update with "profile".
    """
    # Only eligibility questions consume the profile — check_eligibility discards
    # it for every other intent. Extracting it regardless burnt one LLM call per
    # query for nothing, which is both slower and, on a metered tier, wasteful.
    if state.get("intent") != "eligibility":
        logger.debug("extract_profile: skipped for intent=%s", state.get("intent"))
        return {"profile": {}}

    raw = _invoke_llm(PROFILE_EXTRACTOR.format(query=_effective_query(state)), "extract-profile")
    profile = _parse_json(raw)
    # Drop nulls so the checker reports them as missing rather than comparing
    # against None.
    profile = {k: v for k, v in profile.items() if v is not None}
    logger.debug("extract_profile: %s", profile)
    return {"profile": profile}


def retrieve_hybrid(state: SetuState) -> dict[str, Any]:
    """Run hybrid dense + BM25 + RRF retrieval.

    Uses the rewritten search query when the self-critique loop supplied one.

    Args:
        state: Current graph state.

    Returns:
        State update with "retrieved_chunks".
    """
    from retrieval.vectorstore import QdrantStore

    settings = get_settings()
    query = _effective_query(state)

    # Typed as a retriever rather than a plain span: this is the step whose
    # recall determines whether the answer can be grounded at all, and Langfuse
    # only breaks retrieval out in its own analytics when it is typed as one.
    with observation("retrieve-hybrid", as_type="retriever", input=query) as span:
        try:
            hits = get_store().hybrid_search(
                query_text=query,
                k_dense=settings.RETRIEVAL_TOP_K,
                k_sparse=settings.RETRIEVAL_TOP_K,
                top_k=settings.RETRIEVAL_TOP_K,
                filters=QdrantStore.child_only_filter(),
            )
        except Exception as exc:  # noqa: BLE001 - a down Qdrant degrades to no context
            logger.error("Retrieval failed (%s): %s", type(exc).__name__, exc)
            hits = []

        if span is not None:
            span.update(
                output=_chunk_digest(hits),
                # Records that the critic rewrote the query, which is otherwise
                # invisible: a retry looks identical to a first attempt.
                metadata={"is_retry": bool(state.get("search_query")), "candidates": len(hits)},
            )

    logger.info("retrieve_hybrid: %d candidates for %r", len(hits), query)
    return {"retrieved_chunks": hits}


def rerank(state: SetuState) -> dict[str, Any]:
    """Reorder candidates with the cross-encoder and keep the best few.

    Args:
        state: Current graph state.

    Returns:
        State update with "reranked_chunks".
    """
    candidates = state.get("retrieved_chunks") or []
    if not candidates:
        return {"reranked_chunks": []}

    query = _effective_query(state)
    with observation("rerank-cross-encoder", as_type="retriever", input=query) as span:
        try:
            top = get_reranker().rerank(query, candidates, top_k=get_settings().RERANK_TOP_N)
        except Exception as exc:  # noqa: BLE001 - fall back to fusion order
            logger.error("Rerank failed (%s): %s; using fusion order", type(exc).__name__, exc)
            top = candidates[: get_settings().RERANK_TOP_N]

        if span is not None:
            span.update(
                output=_chunk_digest(top),
                metadata={"candidates_in": len(candidates), "kept": len(top)},
            )

    logger.info("rerank: kept %d of %d", len(top), len(candidates))
    return {"reranked_chunks": top}


def check_eligibility(state: SetuState) -> dict[str, Any]:
    """Run the deterministic eligibility check when the question calls for one.

    Skipped for non-eligibility intents so that a procedural question never
    produces a qualification verdict as a side effect.

    Args:
        state: Current graph state.

    Returns:
        State update with "eligibility_result" (None when not applicable).
    """
    if state.get("intent") != "eligibility":
        return {"eligibility_result": None}

    chunks = state.get("reranked_chunks") or []
    if not chunks:
        return {"eligibility_result": None}

    # The best-ranked passage determines which scheme the user is asking about.
    scheme_id = _scheme_id_from_url(chunks[0].get("source_url", ""))
    profile = state.get("profile") or {}

    try:
        checker = get_checker()
    except Exception as exc:  # noqa: BLE001 - missing rules file must not crash
        logger.error("Eligibility checker unavailable (%s): %s", type(exc).__name__, exc)
        return {"eligibility_result": None}

    # Rules are hand-authored per scheme and the corpus is broader than the rule
    # set. Emitting "Unknown scheme" as a verdict would read to a user as a
    # rejection; returning no verdict lets the answer stay informational, which
    # is the honest outcome when eligibility genuinely cannot be computed.
    if not checker.has_scheme(scheme_id):
        logger.info("check_eligibility: no rules for %r; answering informationally", scheme_id)
        return {"eligibility_result": None}

    try:
        result = checker.check(scheme_id, profile)
    except Exception as exc:  # noqa: BLE001 - one bad rule must not crash the graph
        logger.error("Eligibility check failed (%s): %s", type(exc).__name__, exc)
        return {"eligibility_result": None}

    logger.info("check_eligibility: %s -> eligible=%s", scheme_id, result["eligible"])
    return {"eligibility_result": result}


def generate_answer(state: SetuState) -> dict[str, Any]:
    """Write a cited answer grounded strictly in the retrieved passages.

    Args:
        state: Current graph state.

    Returns:
        State update with "draft_answer" and "citations".
    """
    chunks = state.get("reranked_chunks") or []
    if not chunks:
        return {
            "draft_answer": (
                "I could not find information about this in the indexed scheme "
                "documents, so I cannot answer reliably."
            ),
            "citations": [],
        }

    eligibility = state.get("eligibility_result")
    if eligibility:
        eligibility_block = (
            "Deterministic eligibility verdict (computed by rule checker, state it exactly):\n"
            f"- Scheme: {eligibility['scheme_name']}\n"
            f"- Eligible: {eligibility['eligible']}\n"
            f"- Reason: {eligibility['reason']}"
        )
    else:
        eligibility_block = ""

    draft = _invoke_llm(
        GROUNDED_ANSWERER.format(
            context=_format_context(chunks),
            eligibility_block=eligibility_block,
            # The English form: context passages are English, so mixing scripts
            # in one prompt would only make grounding harder.
            query=_effective_query(state),
        ),
        "generate-answer",
    )
    if not draft:
        draft = "The language model was unavailable, so I cannot answer right now."

    citations = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "label": (chunk.get("chunk_id") or "")[:CITATION_LEN],
            "source_url": chunk.get("source_url"),
            "scheme_name": chunk.get("scheme_name"),
        }
        for chunk in chunks
    ]
    return {"draft_answer": draft, "citations": citations}


def self_critique(state: SetuState) -> dict[str, Any]:
    """Score the draft's faithfulness and decide whether to retry or finalize.

    Sets "final_answer" only when finalizing; the router keys off that, so the
    retry decision lives in exactly one place.

    Args:
        state: Current graph state.

    Returns:
        State update with "faithfulness_score", possibly "retries",
        "search_query" and "final_answer".
    """
    draft = state.get("draft_answer") or ""
    chunks = state.get("reranked_chunks") or []
    retries = state.get("retries") or 0

    raw = _invoke_llm(
        SELF_CRITIC.format(
            context=_format_context(chunks),
            query=_effective_query(state),
            draft_answer=draft,
        ),
        "self-critique",
    )
    parsed = _parse_json(raw)

    try:
        score = float(parsed.get("faithfulness", 1.0))
    except (TypeError, ValueError):
        score = 1.0
    score = max(0.0, min(1.0, score))

    # A failed critic must not trap the run in a loop, so an unparseable
    # response finalizes rather than retrying.
    wants_retry = parsed.get("decision") == "retry"
    will_retry = wants_retry and score < FAITHFULNESS_THRESHOLD and retries < MAX_RETRIES

    logger.info(
        "self_critique: faithfulness=%.2f retries=%d -> %s",
        score,
        retries,
        "retry" if will_retry else "finalize",
    )

    # The agent already judges its own output; recording that judgement is what
    # turns the trace list into something filterable by quality. Sent on every
    # pass, including retried ones, so a run that had to retry is visible as
    # such rather than only by its final score.
    score_trace(
        "faithfulness",
        score,
        comment=f"retry {retries}" if will_retry else f"finalized after {retries} retries",
    )

    if will_retry:
        better = parsed.get("better_query") or state["user_query"]
        return {
            "faithfulness_score": score,
            "retries": retries + 1,
            "search_query": better,
        }

    # Promote the question and answer to the trace's own input/output. The
    # LangGraph handler would otherwise show the whole state dict — every
    # retrieved passage included — where the trace list needs a readable pair.
    set_trace_io(input=state["user_query"], output=draft)

    return {"faithfulness_score": score, "final_answer": draft}
