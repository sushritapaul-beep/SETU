"""FastAPI service exposing the SETU agent.

Two endpoints answer questions: /query returns the finished answer, and
/query/stream emits server-sent events as each graph node completes, so the UI
can show progress instead of a spinner for the ~10s a full run takes.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agents.graph import build_graph, initial_state
from api.schemas import (
    Citation,
    EligibilityResult,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    StreamEvent,
)
from config.settings import get_settings
from observability.langfuse_setup import build_config, flush

logger = logging.getLogger(__name__)

# Human-readable progress lines, so the UI does not leak node names to users.
NODE_LABELS = {
    "detect_language": "Detecting language",
    "translate_query_to_en": "Translating your question",
    "classify_intent": "Understanding the question",
    "extract_profile": "Reading your details",
    "retrieve_hybrid": "Searching scheme documents",
    "rerank": "Ranking the best passages",
    "check_eligibility": "Checking eligibility rules",
    "generate_answer": "Writing the answer",
    "self_critique": "Checking the answer against sources",
    "translate_answer_from_en": "Translating the answer back",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the graph once at startup and flush traces on shutdown.

    Args:
        app: The FastAPI application.

    Yields:
        Control back to the server while it runs.
    """
    logging.basicConfig(
        level=get_settings().LOG_LEVEL,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    # Compiled once: rebuilding per request would discard the checkpointer and
    # reload models on every call.
    app.state.graph = build_graph()
    logger.info("SETU agent graph compiled")
    yield
    flush()


app = FastAPI(
    title="SETU",
    description="Multilingual agentic RAG over Indian government MSME schemes.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # The Gradio UI runs on 7860; both hostnames are listed because browsers
    # treat them as distinct origins.
    allow_origins=[
        "http://localhost:7860",
        "http://127.0.0.1:7860",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_response(state: dict[str, Any], trace_id: str) -> QueryResponse:
    """Convert a final graph state into the API response model.

    Args:
        state: Final LangGraph state.
        trace_id: Correlation id for this run.

    Returns:
        The response payload.
    """
    verdict = state.get("eligibility_result")
    return QueryResponse(
        answer=state.get("final_answer") or "",
        citations=[Citation(**c) for c in (state.get("citations") or [])],
        lang=state.get("lang") or "en",
        eligibility=EligibilityResult(**verdict) if verdict else None,
        trace_id=trace_id,
        answer_en=state.get("answer_en"),
        intent=state.get("intent"),
        faithfulness_score=state.get("faithfulness_score"),
        retries=state.get("retries") or 0,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report whether the service and its dependencies can answer questions.

    Returns:
        Status plus per-dependency detail. Reports "degraded" rather than
        failing, so a load balancer can distinguish "up but unusable" from down.
    """
    settings = get_settings()
    qdrant_ok = False
    points = 0

    try:
        from retrieval.vectorstore import QdrantStore

        store = QdrantStore()
        if store.client.collection_exists(store.collection):
            points = store.client.count(store.collection, exact=True).count
            qdrant_ok = points > 0
    except Exception as exc:  # noqa: BLE001 - health checks never raise
        logger.warning("Health check could not reach Qdrant: %s", exc)

    llm_ok = not settings.validate_llm_credentials()
    return HealthResponse(
        status="ok" if (qdrant_ok and llm_ok) else "degraded",
        qdrant=qdrant_ok,
        collection=settings.QDRANT_COLLECTION,
        points=points,
        llm_provider=settings.LLM_PROVIDER,
        llm_configured=llm_ok,
        langfuse_enabled=settings.is_langfuse_configured(),
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a question and return the complete result.

    Args:
        request: The user's question and optional session id.

    Returns:
        A cited answer, with an eligibility verdict when applicable.

    Raises:
        HTTPException: 503 if the agent could not run.
    """
    session_id = request.session_id or str(uuid.uuid4())
    config, trace_id = build_config(session_id, request.query, extra_tags=["endpoint:query"])

    state = initial_state(request.query)
    if request.lang:
        # An explicit choice overrides detection; detect_language will not
        # overwrite it because the node only runs before this is read.
        state["lang"] = request.lang

    try:
        # The graph is synchronous and CPU-bound; ainvoke lets the event loop
        # serve other requests while it runs.
        final = await app.state.graph.ainvoke(state, config=config)
    except Exception as exc:  # noqa: BLE001 - surface as a clean 503
        logger.exception("Agent run failed")
        raise HTTPException(status_code=503, detail=f"Agent unavailable: {exc}") from exc

    return _to_response(final, trace_id)


async def _event_stream(request: QueryRequest) -> AsyncIterator[str]:
    """Yield SSE frames as each graph node completes.

    Args:
        request: The user's question and optional session id.

    Yields:
        Server-sent event frames, terminated by a "final" or "error" event.
    """
    session_id = request.session_id or str(uuid.uuid4())
    config, trace_id = build_config(session_id, request.query, extra_tags=["endpoint:stream"])

    state = initial_state(request.query)
    if request.lang:
        state["lang"] = request.lang

    def frame(event: StreamEvent) -> str:
        """Encode one event in SSE wire format."""
        return f"data: {event.model_dump_json()}\n\n"

    merged: dict[str, Any] = {}
    try:
        # stream_mode="updates" yields {node_name: state_delta} per completed
        # node, which is exactly the granularity the UI wants.
        async for chunk in app.state.graph.astream(state, config=config, stream_mode="updates"):
            for node_name, delta in chunk.items():
                if isinstance(delta, dict):
                    merged.update(delta)
                yield frame(
                    StreamEvent(
                        event="node",
                        node=node_name,
                        message=NODE_LABELS.get(node_name, node_name),
                    )
                )

        response = _to_response(merged, trace_id)
        yield frame(StreamEvent(event="final", data=response.model_dump()))
    except Exception as exc:  # noqa: BLE001 - report in-band; headers are already sent
        logger.exception("Streaming run failed")
        yield frame(StreamEvent(event="error", message=str(exc)))
    finally:
        flush()


@app.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """Answer a question, streaming progress as server-sent events.

    Args:
        request: The user's question and optional session id.

    Returns:
        A text/event-stream response.
    """
    return StreamingResponse(
        _event_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Stops nginx and similar proxies from buffering the stream, which
            # would defeat the point of streaming.
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def root() -> dict[str, Any]:
    """Point callers at the docs and the health endpoint.

    Returns:
        A short service description.
    """
    return {
        "service": "SETU",
        "docs": "/docs",
        "health": "/health",
        "endpoints": ["POST /query", "POST /query/stream"],
    }


# Kept so `python api/main.py` works as well as `uvicorn api.main:app`.
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
