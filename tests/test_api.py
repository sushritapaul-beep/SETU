"""Offline tests for the API and observability layers — the graph is stubbed."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from api.schemas import QueryRequest, QueryResponse
from observability import langfuse_setup

FINAL_STATE: dict[str, Any] = {
    "final_answer": "You need your Udyam Registration Number [39f4b029].",
    "answer_en": "You need your Udyam Registration Number [39f4b029].",
    "lang": "hin",
    "intent": "procedure",
    "faithfulness_score": 0.95,
    "retries": 0,
    "citations": [
        {
            "chunk_id": "39f4b029" + "0" * 24,
            "label": "39f4b029",
            "source_url": "https://www.myscheme.gov.in/schemes/mszcs",
            "scheme_name": "MSME Sustainable (ZED) Certification Scheme",
        }
    ],
    "eligibility_result": None,
    "reranked_chunks": [{"text": "Udyam Registration Number.", "chunk_id": "a" * 32}],
}


class _FakeGraph:
    """Stands in for the compiled LangGraph."""

    def __init__(self, state: dict[str, Any] | None = None, fail: bool = False) -> None:
        self.state = state or FINAL_STATE
        self.fail = fail
        self.configs: list[dict[str, Any]] = []

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.configs.append(config)
        if self.fail:
            raise RuntimeError("agent exploded")
        return self.state

    async def astream(self, state: dict[str, Any], config: dict[str, Any], stream_mode: str):
        self.configs.append(config)
        if self.fail:
            raise RuntimeError("agent exploded")
        # LangGraph's "updates" mode yields {node_name: delta} per node.
        yield {"detect_language": {"lang": "hin"}}
        yield {"retrieve_hybrid": {"retrieved_chunks": []}}
        yield {"self_critique": self.state}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient whose app has a stubbed graph and no Langfuse."""
    monkeypatch.setattr(api_main, "build_graph", lambda: _FakeGraph())
    # Tracing off: a real handler would try to reach a Langfuse server.
    monkeypatch.setattr(langfuse_setup, "get_callback_handler", lambda: None)
    langfuse_setup.get_langfuse_client.cache_clear()
    with TestClient(api_main.app) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

def test_query_request_rejects_empty_query() -> None:
    """An empty question is a client error, not something the agent must handle."""
    with pytest.raises(ValueError):
        QueryRequest(query="")


def test_query_response_requires_trace_id() -> None:
    """Every answer is correlatable, so trace_id is mandatory."""
    with pytest.raises(ValueError):
        QueryResponse(answer="hi")


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

def test_health_reports_dependency_detail(client: TestClient) -> None:
    """/health always answers, describing which dependencies are usable."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert set(body) >= {"qdrant", "collection", "llm_provider", "langfuse_enabled"}


def test_query_returns_answer_and_citations(client: TestClient) -> None:
    """A successful run maps graph state onto the response contract."""
    response = client.post("/query", json={"query": "ZED documents?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("You need your Udyam")
    assert body["lang"] == "hin"
    assert body["trace_id"]
    assert len(body["citations"]) == 1
    assert body["citations"][0]["label"] == "39f4b029"
    assert body["faithfulness_score"] == 0.95


def test_query_reuses_session_id_as_thread_id(client: TestClient) -> None:
    """session_id becomes the checkpointer thread_id, so turns share state."""
    client.post("/query", json={"query": "first", "session_id": "abc-123"})

    config = client.app.state.graph.configs[-1]
    assert config["configurable"]["thread_id"] == "abc-123"


def test_query_surfaces_agent_failure_as_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """A crashed agent returns a clean 503 rather than a stack trace."""
    monkeypatch.setattr(api_main, "build_graph", lambda: _FakeGraph(fail=True))
    monkeypatch.setattr(langfuse_setup, "get_callback_handler", lambda: None)

    with TestClient(api_main.app) as test_client:
        response = test_client.post("/query", json={"query": "anything"})

    assert response.status_code == 503
    assert "Agent unavailable" in response.json()["detail"]


def test_stream_emits_node_events_then_final(client: TestClient) -> None:
    """The SSE stream reports progress and ends with the complete answer."""
    with client.stream("POST", "/query/stream", json={"query": "ZED documents?"}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line[len("data: ") :])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    kinds = [e["event"] for e in events]
    assert kinds[-1] == "final", "stream must terminate with a final event"
    assert kinds.count("node") == 3

    # Node events carry a human-readable label, not the raw node name.
    assert events[0]["message"] == "Detecting language"

    payload = events[-1]["data"]
    assert payload["answer"].startswith("You need your Udyam")
    assert payload["citations"][0]["label"] == "39f4b029"


def test_cors_allows_the_gradio_origin(client: TestClient) -> None:
    """The UI on :7860 is a different origin and must be permitted."""
    response = client.options(
        "/query",
        headers={
            "Origin": "http://localhost:7860",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code in {200, 204}
    assert response.headers["access-control-allow-origin"] == "http://localhost:7860"


# --------------------------------------------------------------------------
# Observability
# --------------------------------------------------------------------------

def test_build_config_carries_thread_id_and_prompt_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every run records which prompt versions produced it."""
    monkeypatch.setattr(langfuse_setup, "get_callback_handler", lambda: None)

    config, trace_id = langfuse_setup.build_config("sess-1", "a question")

    assert config["configurable"]["thread_id"] == "sess-1"
    assert trace_id
    assert "query_classifier" in config["metadata"]["prompt_versions"]
    assert "setu" in config["tags"]
    # No handler means no callbacks key, so LangGraph runs untraced.
    assert "callbacks" not in config


def test_tracing_disabled_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Langfuse keys disable tracing instead of raising."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    langfuse_setup.get_langfuse_client.cache_clear()

    from config.settings import Settings

    monkeypatch.setattr(
        langfuse_setup, "get_settings", lambda: Settings(_env_file=None)
    )

    assert langfuse_setup.get_langfuse_client() is None
    assert langfuse_setup.get_callback_handler() is None
    # Flushing with no client is a no-op, not an error.
    langfuse_setup.flush()

    langfuse_setup.get_langfuse_client.cache_clear()
