"""Offline tests for the agent layer — no LLM, no Qdrant, no model downloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents import nodes as nodes_module
from agents.graph import SetuState, build_graph, initial_state, route_after_critique
from agents.prompts import ALL_PROMPTS, prompt_versions
from agents.tools import EligibilityChecker

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

RULES = {
    "schema_version": 1,
    "schemes": [
        {
            "scheme_id": "testscheme",
            "name": "Test MSME Scheme",
            "source_url": "https://www.myscheme.gov.in/schemes/testscheme",
            "rules": [
                {
                    "field": "entity_type",
                    "operator": "in",
                    "value": ["micro", "small"],
                    "description": "Must be a micro or small enterprise.",
                },
                {
                    "field": "turnover_lakhs",
                    "operator": "<=",
                    "value": 500,
                    "description": "Turnover must not exceed Rs 5 crore.",
                },
                {
                    "field": "state",
                    "operator": "==",
                    "value": "Tamil Nadu",
                    "description": "Unit must be in Tamil Nadu.",
                },
                {
                    "field": "udyam_registered",
                    "operator": "==",
                    "value": True,
                    "description": "Must hold Udyam registration.",
                },
            ],
        }
    ],
}


@pytest.fixture
def checker(tmp_path: Path) -> EligibilityChecker:
    """An EligibilityChecker over a small hand-crafted rule set."""
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(RULES), encoding="utf-8")
    return EligibilityChecker(rules_path=path)


CHUNKS = [
    {
        "chunk_id": "a" * 32,
        "text": "Should be registered in the Udyam portal to apply for this scheme.",
        "source_url": "https://www.myscheme.gov.in/schemes/testscheme",
        "scheme_name": "Test MSME Scheme",
        "score": 0.9,
    },
    {
        "chunk_id": "b" * 32,
        "text": "The scheme provides a credit guarantee of eighty five percent.",
        "source_url": "https://www.myscheme.gov.in/schemes/testscheme",
        "scheme_name": "Test MSME Scheme",
        "score": 0.7,
    },
]


class _FakeLLM:
    """Chat model stub that returns queued responses in order."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else ""

        class _Response:
            content = text

        return _Response()


class _FakeStore:
    """Vector store stub returning a fixed candidate list."""

    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.hits = CHUNKS if hits is None else hits
        self.queries: list[str] = []

    def hybrid_search(self, query_text: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.queries.append(query_text)
        return self.hits


class _FakeReranker:
    """Reranker stub that preserves input order."""

    def rerank(
        self, query: str, hits: list[dict[str, Any]], top_k: int = 5
    ) -> list[dict[str, Any]]:
        return hits[:top_k]


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch, checker: EligibilityChecker) -> dict[str, Any]:
    """Replace every external dependency of the nodes module with a stub."""
    store = _FakeStore()
    monkeypatch.setattr(nodes_module, "get_store", lambda: store)
    monkeypatch.setattr(nodes_module, "get_reranker", lambda: _FakeReranker())
    monkeypatch.setattr(nodes_module, "get_checker", lambda: checker)
    return {"store": store}


def _set_llm(monkeypatch: pytest.MonkeyPatch, responses: list[str]) -> _FakeLLM:
    """Install a queued fake LLM into the nodes module."""
    llm = _FakeLLM(responses)
    monkeypatch.setattr(nodes_module, "get_llm", lambda: llm)
    return llm


# --------------------------------------------------------------------------
# 1. Graph compiles
# --------------------------------------------------------------------------

def test_graph_compiles_without_error() -> None:
    """The state machine assembles and exposes every declared node."""
    compiled = build_graph()

    assert compiled is not None
    node_names = set(compiled.get_graph().nodes)
    for expected in (
        "detect_language",
        "classify_intent",
        "extract_profile",
        "retrieve_hybrid",
        "rerank",
        "check_eligibility",
        "generate_answer",
        "self_critique",
    ):
        assert expected in node_names


def test_module_level_graph_is_importable() -> None:
    """The exported graph object exists, so API and UI layers can import it."""
    from agents.graph import graph

    assert graph is not None


def test_prompts_are_versioned() -> None:
    """Every prompt carries a version, for traceable eval runs."""
    versions = prompt_versions()

    assert set(versions) == {
        "query_classifier",
        "profile_extractor",
        "grounded_answerer",
        "self_critic",
    }
    assert all(v for v in versions.values())
    assert all(p.template.strip() for p in ALL_PROMPTS.values())


# --------------------------------------------------------------------------
# 2. Eligibility checker
# --------------------------------------------------------------------------

def test_eligibility_passes_for_fully_qualifying_profile(checker: EligibilityChecker) -> None:
    """A profile meeting every rule is eligible with no failures or gaps."""
    result = checker.check(
        "testscheme",
        {
            "entity_type": "small",
            "turnover_lakhs": 200,
            "state": "Tamil Nadu",
            "udyam_registered": True,
        },
    )

    assert result["eligible"] is True
    assert len(result["matched_rules"]) == 4
    assert result["failed_rules"] == []
    assert result["missing_fields"] == []


def test_eligibility_fails_and_reports_which_rule_broke(checker: EligibilityChecker) -> None:
    """A disqualifying value is reported with the actual value that caused it."""
    result = checker.check(
        "testscheme",
        {
            "entity_type": "medium",  # not in [micro, small]
            "turnover_lakhs": 9000,  # over the 500 ceiling
            "state": "Tamil Nadu",
            "udyam_registered": True,
        },
    )

    assert result["eligible"] is False
    assert len(result["failed_rules"]) == 2
    failed_fields = {r["field"] for r in result["failed_rules"]}
    assert failed_fields == {"entity_type", "turnover_lakhs"}
    assert any(r["actual"] == 9000 for r in result["failed_rules"])


def test_missing_information_never_counts_as_eligible(checker: EligibilityChecker) -> None:
    """Unknown fields block a positive verdict instead of being assumed to pass."""
    result = checker.check("testscheme", {"entity_type": "micro"})

    assert result["eligible"] is False
    assert set(result["missing_fields"]) == {"turnover_lakhs", "state", "udyam_registered"}
    assert result["failed_rules"] == []
    assert "need more information" in result["reason"]


def test_unknown_scheme_is_reported_not_raised(checker: EligibilityChecker) -> None:
    """An unrecognised scheme id yields a clear negative rather than an exception."""
    result = checker.check("nope", {"entity_type": "micro"})

    assert result["eligible"] is False
    assert "Unknown scheme" in result["reason"]


def test_type_mismatch_is_a_failure_not_a_crash(checker: EligibilityChecker) -> None:
    """Comparing a string turnover against a number fails gracefully."""
    result = checker.check(
        "testscheme",
        {
            "entity_type": "micro",
            "turnover_lakhs": "two hundred",
            "state": "Tamil Nadu",
            "udyam_registered": True,
        },
    )

    assert result["eligible"] is False
    assert any("error" in r for r in result["failed_rules"])


def test_real_rules_file_loads_and_covers_five_schemes() -> None:
    """The shipped rules file parses and matches the five indexed schemes."""
    real = EligibilityChecker()

    assert set(real.schemes) == {"mszcs", "ds", "msme-stes", "samsmeata", "ipsasmerctse"}
    for scheme in real.schemes.values():
        assert scheme.rules, f"{scheme.scheme_id} has no rules"
        for rule in scheme.rules:
            assert rule.operator in {"==", "<=", ">=", "in"}
            assert rule.description


# --------------------------------------------------------------------------
# 3. Self-critique loop terminates
# --------------------------------------------------------------------------

def test_self_critique_loop_terminates_within_max_retries(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """A critic that always demands a retry still ends after MAX_RETRIES."""
    # Every critic response is a low score demanding another attempt.
    always_retry = json.dumps(
        {
            "faithfulness": 0.1,
            "decision": "retry",
            "reason": "unsupported",
            "better_query": "rewritten query",
        }
    )
    # Enough responses for classify/extract/answer/critic across all passes.
    _set_llm(monkeypatch, [always_retry] * 40)

    compiled = build_graph()
    result = compiled.invoke(
        initial_state("Am I eligible?"),
        config={"configurable": {"thread_id": "t1"}, "recursion_limit": 50},
    )

    assert result["retries"] == nodes_module.MAX_RETRIES
    assert result["final_answer"], "loop must finalize rather than spin forever"
    # Retrieval ran once plus one per retry.
    assert len(wired["store"].queries) == nodes_module.MAX_RETRIES + 1


def test_high_faithfulness_finalizes_immediately(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """A confident critic ends the run on the first pass."""
    _set_llm(
        monkeypatch,
        [
            json.dumps({"intent": "procedure"}),
            "Apply on the Udyam portal [aaaaaaaa].",
            json.dumps({"faithfulness": 0.95, "decision": "finalize", "reason": "ok"}),
        ],
    )

    result = build_graph().invoke(
        initial_state("How do I apply?"),
        config={"configurable": {"thread_id": "t2"}},
    )

    assert result["retries"] == 0
    assert result["faithfulness_score"] == 0.95
    assert result["final_answer"] == "Apply on the Udyam portal [aaaaaaaa]."
    assert len(wired["store"].queries) == 1


def test_router_reads_final_answer() -> None:
    """The router's decision follows solely from whether an answer was finalized."""
    assert route_after_critique(SetuState(final_answer="done")) == "finalize"
    assert route_after_critique(SetuState()) == "retry"


# --------------------------------------------------------------------------
# 4. Node behaviour
# --------------------------------------------------------------------------

def test_eligibility_intent_produces_a_deterministic_verdict(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """An eligibility question runs the rule checker on the extracted profile."""
    _set_llm(
        monkeypatch,
        [
            json.dumps({"intent": "eligibility"}),
            json.dumps(
                {
                    "entity_type": "small",
                    "turnover_lakhs": 200,
                    "state": "Tamil Nadu",
                    "udyam_registered": True,
                }
            ),
            "Yes, you qualify [aaaaaaaa].",
            json.dumps({"faithfulness": 0.9, "decision": "finalize", "reason": "ok"}),
        ],
    )

    result = build_graph().invoke(
        initial_state("I run a small unit in Tamil Nadu, am I eligible?"),
        config={"configurable": {"thread_id": "t3"}},
    )

    verdict = result["eligibility_result"]
    assert verdict is not None
    assert verdict["eligible"] is True
    assert verdict["scheme_id"] == "testscheme"


def test_profile_extraction_is_skipped_for_non_eligibility_intents(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """No LLM call is spent extracting a profile that will never be used."""
    llm = _set_llm(
        monkeypatch,
        [
            json.dumps({"intent": "benefit_amount"}),
            "The scheme provides 25% [aaaaaaaa].",
            json.dumps({"faithfulness": 0.9, "decision": "finalize", "reason": "ok"}),
        ],
    )

    result = build_graph().invoke(
        initial_state("How much does the scheme provide?"),
        config={"configurable": {"thread_id": "skip-1"}},
    )

    assert result["profile"] == {}
    # classify + generate + critique only — the profile-extractor prompt never ran.
    assert len(llm.prompts) == 3
    assert not any("Extract a structured applicant profile" in p for p in llm.prompts)


def test_profile_extraction_still_runs_for_eligibility(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """Eligibility questions must still pay for profile extraction."""
    llm = _set_llm(
        monkeypatch,
        [
            json.dumps({"intent": "eligibility"}),
            json.dumps({"entity_type": "small"}),
            "Answer [aaaaaaaa].",
            json.dumps({"faithfulness": 0.9, "decision": "finalize", "reason": "ok"}),
        ],
    )

    result = build_graph().invoke(
        initial_state("Am I eligible?"), config={"configurable": {"thread_id": "skip-2"}}
    )

    assert result["profile"] == {"entity_type": "small"}
    assert any("Extract a structured applicant profile" in p for p in llm.prompts)


def test_non_eligibility_intent_skips_the_verdict(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """A procedural question must not emit a qualification verdict."""
    _set_llm(
        monkeypatch,
        [
            json.dumps({"intent": "procedure"}),
            "Follow these steps [aaaaaaaa].",
            json.dumps({"faithfulness": 0.9, "decision": "finalize", "reason": "ok"}),
        ],
    )

    result = build_graph().invoke(
        initial_state("How do I apply?"), config={"configurable": {"thread_id": "t4"}}
    )

    assert result["eligibility_result"] is None


def test_llm_failure_degrades_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """A dead LLM yields a graceful answer, not an exception."""

    class _DeadLLM:
        def invoke(self, prompt: str) -> Any:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(nodes_module, "get_llm", lambda: _DeadLLM())

    result = build_graph().invoke(
        initial_state("Anything?"), config={"configurable": {"thread_id": "t5"}}
    )

    assert result["intent"] == "procedure"  # safe default
    assert "unavailable" in result["final_answer"].lower()


def test_no_retrieval_results_produces_an_honest_answer(
    monkeypatch: pytest.MonkeyPatch, checker: EligibilityChecker
) -> None:
    """With no context the agent says so rather than inventing one."""
    monkeypatch.setattr(nodes_module, "get_store", lambda: _FakeStore(hits=[]))
    monkeypatch.setattr(nodes_module, "get_reranker", lambda: _FakeReranker())
    monkeypatch.setattr(nodes_module, "get_checker", lambda: checker)
    _set_llm(monkeypatch, [json.dumps({"intent": "list"}), "", ""])

    result = build_graph().invoke(
        initial_state("Tell me about a scheme we never indexed"),
        config={"configurable": {"thread_id": "t6"}},
    )

    assert result["citations"] == []
    assert "could not find" in result["final_answer"].lower()


# --------------------------------------------------------------------------
# 5. Indic layer (mocked)
# --------------------------------------------------------------------------

class _FakeDetector:
    """Language detector stub returning a fixed verdict."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.seen: list[str] = []

    def detect(self, text: str) -> dict[str, Any]:
        self.seen.append(text)
        return self.result


class _FakeTranslator:
    """Translator stub that tags text instead of running a model."""

    def __init__(self) -> None:
        self.to_en_calls: list[tuple[str, str]] = []
        self.from_en_calls: list[tuple[str, str]] = []

    def to_english(self, text: str, src_lang: str) -> str:
        self.to_en_calls.append((text, src_lang))
        return "ENGLISH: what documents are required"

    def from_english(self, text: str, tgt_lang: str) -> str:
        self.from_en_calls.append((text, tgt_lang))
        return f"[{tgt_lang}] {text}"


def _wire_indic(
    monkeypatch: pytest.MonkeyPatch, detection: dict[str, Any]
) -> tuple[_FakeDetector, _FakeTranslator]:
    """Install fake detector and translator into the nodes module."""
    detector = _FakeDetector(detection)
    translator = _FakeTranslator()
    monkeypatch.setattr(nodes_module, "get_detector", lambda: detector)
    monkeypatch.setattr(nodes_module, "get_translator", lambda: translator)
    return detector, translator


HINDI_DETECTION = {
    "lang_code": "hin",
    "script": "Deva",
    "confidence": 0.99,
    "codes": ["hin_Deva"],
    "model": "FTN",
}


def test_hindi_query_is_translated_in_and_out(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """A Hindi query is translated to English for retrieval and back for display."""
    _, translator = _wire_indic(monkeypatch, HINDI_DETECTION)
    _set_llm(
        monkeypatch,
        [
            json.dumps({"intent": "procedure"}),
            "You need your Udyam Registration Number [aaaaaaaa].",
            json.dumps({"faithfulness": 0.9, "decision": "finalize", "reason": "ok"}),
        ],
    )

    result = build_graph().invoke(
        initial_state("ZED प्रमाणन के लिए कौन से दस्तावेज़ आवश्यक हैं?"),
        config={"configurable": {"thread_id": "indic-1"}},
    )

    assert result["lang"] == "hin"
    assert result["script"] == "Deva"
    # Retrieval ran against the English translation, not the Devanagari original.
    assert result["query_en"] == "ENGLISH: what documents are required"
    assert wired["store"].queries == ["ENGLISH: what documents are required"]
    # The English answer is preserved for evaluation...
    assert result["answer_en"] == "You need your Udyam Registration Number [aaaaaaaa]."
    # ...and the user-facing answer is localised.
    assert result["final_answer"].startswith("[hin_Deva]")
    assert translator.from_en_calls[0][1] == "hin_Deva"


def test_english_query_skips_translation_entirely(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """An English query must not pay for two translation round trips."""
    _, translator = _wire_indic(
        monkeypatch,
        {"lang_code": "en", "script": "Latn", "confidence": 1.0, "codes": ["eng_Latn"]},
    )
    _set_llm(
        monkeypatch,
        [
            json.dumps({"intent": "procedure"}),
            "Apply on the Udyam portal [aaaaaaaa].",
            json.dumps({"faithfulness": 0.9, "decision": "finalize", "reason": "ok"}),
        ],
    )

    result = build_graph().invoke(
        initial_state("What documents are required?"),
        config={"configurable": {"thread_id": "indic-2"}},
    )

    assert translator.to_en_calls == []
    assert translator.from_en_calls == []
    assert result["final_answer"] == "Apply on the Udyam portal [aaaaaaaa]."


def test_code_mixed_input_picks_the_non_english_language(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """Hinglish translates via its Indic half, since English needs no translating."""
    _, translator = _wire_indic(
        monkeypatch,
        {
            "lang_code": "code_mixed",
            "script": "Latn",
            "confidence": 0.55,
            "codes": ["eng_Latn", "hin_Latn"],
            "model": "FTR",
        },
    )
    _set_llm(
        monkeypatch,
        [
            json.dumps({"intent": "procedure"}),
            "Answer [aaaaaaaa].",
            json.dumps({"faithfulness": 0.9, "decision": "finalize", "reason": "ok"}),
        ],
    )

    result = build_graph().invoke(
        initial_state("ZED certification ke liye documents kya chahiye"),
        config={"configurable": {"thread_id": "indic-3"}},
    )

    assert result["lang"] == "code_mixed"
    # eng_Latn is skipped in favour of the Indic code.
    assert translator.to_en_calls[0][1] == "hin_Latn"


def test_translation_failure_falls_back_to_original_text(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """A dead translator degrades to the untranslated query, never to nothing."""

    class _DeadTranslator:
        def to_english(self, text: str, src_lang: str) -> str:
            raise RuntimeError("model repo is gated")

        def from_english(self, text: str, tgt_lang: str) -> str:
            raise RuntimeError("model repo is gated")

    monkeypatch.setattr(nodes_module, "get_detector", lambda: _FakeDetector(HINDI_DETECTION))
    monkeypatch.setattr(nodes_module, "get_translator", lambda: _DeadTranslator())
    _set_llm(
        monkeypatch,
        [
            json.dumps({"intent": "procedure"}),
            "English answer [aaaaaaaa].",
            json.dumps({"faithfulness": 0.9, "decision": "finalize", "reason": "ok"}),
        ],
    )

    original = "ZED प्रमाणन के लिए कौन से दस्तावेज़ आवश्यक हैं?"
    result = build_graph().invoke(
        initial_state(original), config={"configurable": {"thread_id": "indic-4"}}
    )

    assert result["query_en"] == original, "must fall back to the user's words"
    # The English answer still reaches the user rather than being lost.
    assert result["final_answer"] == "English answer [aaaaaaaa]."


def test_detector_failure_defaults_to_english(
    monkeypatch: pytest.MonkeyPatch, wired: dict[str, Any]
) -> None:
    """If IndicLID cannot load, the graph still answers in English."""

    class _DeadDetector:
        def detect(self, text: str) -> dict[str, Any]:
            raise RuntimeError("fasttext unavailable")

    monkeypatch.setattr(nodes_module, "get_detector", lambda: _DeadDetector())
    _set_llm(
        monkeypatch,
        [
            json.dumps({"intent": "procedure"}),
            "Answer [aaaaaaaa].",
            json.dumps({"faithfulness": 0.9, "decision": "finalize", "reason": "ok"}),
        ],
    )

    result = build_graph().invoke(
        initial_state("anything"), config={"configurable": {"thread_id": "indic-5"}}
    )

    assert result["lang"] == "en"
    assert result["final_answer"] == "Answer [aaaaaaaa]."


def test_graph_includes_the_translation_nodes() -> None:
    """Both Indic nodes are wired into the compiled graph."""
    names = set(build_graph().get_graph().nodes)

    assert "translate_query_to_en" in names
    assert "translate_answer_from_en" in names


def test_parse_json_survives_markdown_fences() -> None:
    """Models wrap JSON in fences; the parser copes."""
    assert nodes_module._parse_json('```json\n{"intent": "list"}\n```') == {"intent": "list"}
    prose = 'Sure! {"intent": "list"} hope that helps'
    assert nodes_module._parse_json(prose) == {"intent": "list"}
    assert nodes_module._parse_json("not json at all") == {}
    assert nodes_module._parse_json("") == {}
