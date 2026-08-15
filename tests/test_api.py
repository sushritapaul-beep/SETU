"""Offline tests for the API and observability layers — the graph is stubbed."""

from __future__ import annotations

import json
import time
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


def test_cors_allows_the_configured_ui_origin(client: TestClient) -> None:
    """The Chainlit UI runs as a separate process/deployment and must be permitted."""
    response = client.options(
        "/query",
        headers={
            "Origin": "http://localhost:8001",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code in {200, 204}
    assert response.headers["access-control-allow-origin"] == "http://localhost:8001"


# --------------------------------------------------------------------------
# Security: API key and rate limiting
# --------------------------------------------------------------------------

def test_no_api_key_configured_leaves_service_open(client: TestClient) -> None:
    """Local development runs without auth; that is the documented default."""
    assert client.post("/query", json={"query": "x"}).status_code == 200


def test_configured_api_key_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a key set, answering endpoints reject callers that lack it."""
    from pydantic import SecretStr

    from config.settings import Settings

    settings = Settings(_env_file=None)
    settings.API_KEY = SecretStr("secret-key")
    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(api_main, "build_graph", lambda: _FakeGraph())
    monkeypatch.setattr(langfuse_setup, "get_callback_handler", lambda: None)

    with TestClient(api_main.app) as test_client:
        assert test_client.post("/query", json={"query": "x"}).status_code == 401
        wrong = test_client.post(
            "/query", json={"query": "x"}, headers={"X-API-Key": "wrong"}
        )
        assert wrong.status_code == 401
        ok = test_client.post(
            "/query", json={"query": "x"}, headers={"X-API-Key": "secret-key"}
        )
        assert ok.status_code == 200

    # /health must stay reachable without a key, or orchestrators cannot probe it.
    with TestClient(api_main.app) as test_client:
        assert test_client.get("/health").status_code == 200


def test_rate_limit_prefers_client_id_header_over_source_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proxying caller (the UI) can identify each visitor individually.

    Without this, every visitor's request reaches the API from the same
    source (the UI's own connection), so IP-based limiting would treat every
    visitor of a deployed demo as one shared caller.
    """
    limiter = api_main.RateLimiter(limit=1, window_seconds=60)
    monkeypatch.setattr(api_main, "rate_limiter", limiter)

    class _FakeClient:
        host = "10.0.0.1"  # same source IP for both calls below

    class _FakeRequest:
        client = _FakeClient()

    # Two distinct X-Client-Id values from the same source IP are two
    # distinct callers, not one exhausting a shared limit.
    api_main.enforce_rate_limit(_FakeRequest(), x_client_id="visitor-a")
    api_main.enforce_rate_limit(_FakeRequest(), x_client_id="visitor-b")


def test_rate_limiter_rejects_bursts_then_recovers() -> None:
    """A caller over its limit gets 429 with Retry-After; others are unaffected."""
    from fastapi import HTTPException

    limiter = api_main.RateLimiter(limit=3, window_seconds=60)

    for _ in range(3):
        limiter.check("1.2.3.4")

    with pytest.raises(HTTPException) as excinfo:
        limiter.check("1.2.3.4")
    assert excinfo.value.status_code == 429
    assert "Retry-After" in excinfo.value.headers

    # The limit is per client, so one abuser cannot lock everyone out.
    limiter.check("5.6.7.8")


def test_rate_limit_window_expires() -> None:
    """Old requests fall out of the window rather than counting forever."""
    limiter = api_main.RateLimiter(limit=1, window_seconds=60)
    limiter.check("1.2.3.4")

    # Simulate the first hit ageing past the window.
    limiter._hits["1.2.3.4"] = [time.monotonic() - 61]
    limiter.check("1.2.3.4")


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


def test_build_config_sets_langfuse_trace_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session, tags and name use the keys Langfuse actually reads.

    A plain "session_id" metadata key is inert: it leaves the Sessions view
    empty while looking correct in the config dict.
    """
    monkeypatch.setattr(langfuse_setup, "get_callback_handler", lambda: None)

    config, _ = langfuse_setup.build_config(
        "sess-9", "a question", extra_tags=["endpoint:query"], user_id="user-7"
    )

    metadata = config["metadata"]
    assert metadata["langfuse_session_id"] == "sess-9"
    assert metadata["langfuse_user_id"] == "user-7"
    assert set(metadata["langfuse_tags"]) == {"setu", "endpoint:query"}
    # A name that identifies the operation, not the run.
    assert config["run_name"] == "answer-scheme-question"


def test_build_config_omits_user_id_when_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """An anonymous run must not report an empty string as a user."""
    monkeypatch.setattr(langfuse_setup, "get_callback_handler", lambda: None)

    config, _ = langfuse_setup.build_config("sess-1", "q")

    assert "langfuse_user_id" not in config["metadata"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("my aadhaar is 1234 5678 9012", "my aadhaar is [AADHAAR]"),
        ("aadhaar 123456789012 ok", "aadhaar [AADHAAR] ok"),
        ("pan ABCDE1234F please", "pan [PAN] please"),
        ("call 9876543210", "call [PHONE]"),
        ("call +91 9876543210", "call [PHONE]"),
        ("mail me at ram.kumar@example.co.in", "mail me at [EMAIL]"),
    ],
)
def test_mask_pii_redacts_direct_identifiers(raw: str, expected: str) -> None:
    """Identifiers are redacted before anything reaches the trace store."""
    assert langfuse_setup.mask_pii(data=raw) == expected


def test_mask_pii_keeps_eligibility_attributes() -> None:
    """Income, category and turnover survive: they make a verdict debuggable.

    Masking these would remove exactly the fields needed to explain why the
    rule checker returned the verdict it did, and none identifies a person.
    """
    profile = {"annual_income": 250000, "category": "OBC", "turnover": 5000000}

    assert langfuse_setup.mask_pii(data=profile) == profile


@pytest.mark.parametrize(
    "identifier",
    [
        # A UUID's middle segments are often all digits and match 4-4-4 exactly.
        "self_critique:88a7a294-4029-4771-8299-939abaabe47c",
        "run-1234-5678-9012-abc",
        "chunk_0536b788",
    ],
)
def test_mask_pii_leaves_internal_identifiers_intact(identifier: str) -> None:
    """Correlation ids must survive masking.

    Redacting part of a checkpoint namespace or run id protects nothing and
    breaks the ids used to tie observations in a trace back together.
    """
    assert langfuse_setup.mask_pii(data=identifier) == identifier


def test_mask_pii_recurses_into_nested_structures() -> None:
    """Nested payloads are masked too, not just top-level strings."""
    data = {"messages": [{"content": "pan ABCDE1234F"}]}

    assert langfuse_setup.mask_pii(data=data) == {"messages": [{"content": "pan [PAN]"}]}


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
