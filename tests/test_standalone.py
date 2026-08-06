"""Offline tests for the Spaces standalone pipeline — no models, no network."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ui import standalone


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a tiny snapshot and point the module at it."""
    records = [
        {
            "id": f"{1:032x}",
            "parent_id": None,
            "text": "Should be registered in the Udyam portal to apply.",
            "metadata": {
                "source_url": "https://www.myscheme.gov.in/schemes/mszcs",
                "scheme_name": "MSME Sustainable (ZED) Certification Scheme",
                "chunk_type": "child",
            },
        },
        {
            "id": f"{2:032x}",
            "parent_id": None,
            "text": "Tamil Nadu handloom weaver subsidy for textile units.",
            "metadata": {
                "source_url": "https://www.myscheme.gov.in/schemes/msme-stes",
                "scheme_name": "Msme Skill Training And Employment Scheme",
                "chunk_type": "child",
            },
        },
    ]
    (tmp_path / "chunks.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    # Deterministic unit vectors so cosine ordering is predictable.
    vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    np.save(tmp_path / "vectors.npy", vectors)

    monkeypatch.setattr(standalone, "SNAPSHOT_DIR", tmp_path)
    standalone._load_index.cache_clear()
    return tmp_path


class _FakeEmbedder:
    """Returns a vector aligned with the first snapshot chunk."""

    def embed_query(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class _FakeLLM:
    """Echoes a fixed cited answer."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> Any:
        self.prompts.append(prompt)

        class _R:
            content = "You must register on the Udyam portal [00000000]."

        return _R()


def test_snapshot_available_reports_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing snapshot is detected rather than crashing at query time."""
    monkeypatch.setattr(standalone, "SNAPSHOT_DIR", tmp_path / "nope")
    assert standalone.snapshot_available() is False


def test_snapshot_available_detects_a_real_one(snapshot: Path) -> None:
    """Both files present means the snapshot is usable."""
    assert standalone.snapshot_available() is True


def test_answer_retrieves_cites_and_generates(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The standalone path returns a generated answer with real citations."""
    llm = _FakeLLM()
    monkeypatch.setattr(standalone, "_get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(standalone, "_get_llm", lambda: llm)
    # No reranker: the Space degrades to fusion order when memory is tight.
    monkeypatch.setattr(standalone, "_get_reranker", lambda: None)

    result = standalone.answer("How do I register for ZED?", top_k=2)

    assert "Udyam portal" in result["answer"]
    assert result["citations"], "an answer must carry citations"
    assert result["citations"][0]["label"] == "00000000"
    assert result["citations"][0]["source_url"].startswith("https://www.myscheme.gov.in/")
    # The prompt must actually contain the retrieved passages.
    assert "Udyam portal" in llm.prompts[0]


def test_answer_degrades_when_generation_fails(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead LLM still returns the retrieved sources instead of nothing."""

    class _DeadLLM:
        def invoke(self, prompt: str) -> Any:
            raise RuntimeError("no credits")

    monkeypatch.setattr(standalone, "_get_embedder", lambda: _FakeEmbedder())
    monkeypatch.setattr(standalone, "_get_llm", lambda: _DeadLLM())
    monkeypatch.setattr(standalone, "_get_reranker", lambda: None)

    result = standalone.answer("anything", top_k=2)

    assert "Could not generate" in result["answer"]
    assert result["citations"], "sources survive a generation failure"


def test_answer_reports_missing_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Without a snapshot the user gets an explanation, not a traceback."""
    monkeypatch.setattr(standalone, "SNAPSHOT_DIR", tmp_path / "absent")
    standalone._load_index.cache_clear()

    result = standalone.answer("anything")

    assert "Retrieval unavailable" in result["answer"]
    assert result["citations"] == []
