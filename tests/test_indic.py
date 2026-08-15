"""Offline tests for the Indic layer — fastText and the Azure Translator API call are mocked."""

from __future__ import annotations

from typing import Any

import pytest

from indic.lang_detect import IndicLanguageDetector
from indic.translator import ENGLISH_TAG, IndicTranslator, to_azure_lang

# --------------------------------------------------------------------------
# Language detection
# --------------------------------------------------------------------------

class _FakeFastText:
    """Stand-in for a loaded fastText model.

    Mirrors the real object's shape: predictions come from ``model.f.predict``
    and are (probability, label) tuples.
    """

    def __init__(self, predictions: list[tuple[float, str]]) -> None:
        self.f = self  # the real model exposes the C++ binding as .f
        self.predictions = predictions
        self.seen: list[str] = []

    def predict(
        self, text: str, k: int, threshold: float, on_error: str
    ) -> list[tuple[float, str]]:
        self.seen.append(text)
        return self.predictions[:k]


@pytest.fixture(autouse=True)
def _clear_detector_cache() -> None:
    """Keep one test's fake model out of the next test."""
    IndicLanguageDetector.clear_cache()


def _detector_with(
    monkeypatch: pytest.MonkeyPatch, predictions: list[tuple[float, str]]
) -> IndicLanguageDetector:
    """Build a detector whose model loading is replaced by a fake."""
    monkeypatch.setattr(
        IndicLanguageDetector, "_load", classmethod(lambda cls, repo: _FakeFastText(predictions))
    )
    return IndicLanguageDetector()


def test_detects_native_hindi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Devanagari input is routed to the native model and reported as Hindi."""
    detector = _detector_with(
        monkeypatch, [(0.99, "__label__hin_Deva"), (0.01, "__label__mai_Deva")]
    )

    result = detector.detect("उद्यम पंजीकरण के लिए कौन पात्र है?")

    assert result["lang_code"] == "hin"
    assert result["script"] == "Deva"
    assert result["confidence"] == pytest.approx(0.99)
    assert result["model"] == "FTN", "Devanagari must use the native-script model"


def test_detects_tamil_and_normalises_script_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """IndicLID's 'Tamil' is normalised to the ISO 15924 code the pipeline wants."""
    detector = _detector_with(monkeypatch, [(0.99, "__label__tam_Tamil")])

    result = detector.detect("இந்த திட்டத்திற்கு என்ன ஆவணங்கள் தேவை?")

    assert result["lang_code"] == "tam"
    assert result["script"] == "Taml", "IndicLID says 'Tamil'; FLORES wants 'Taml'"


def test_english_is_reported_as_en(monkeypatch: pytest.MonkeyPatch) -> None:
    """Latin-script English routes to the roman model and normalises to 'en'."""
    detector = _detector_with(monkeypatch, [(1.0, "__label__eng_Latn")])

    result = detector.detect("What documents are required for ZED certification?")

    assert result["lang_code"] == "en"
    assert result["model"] == "FTR", "Latin script must use the roman model"


def test_two_confident_languages_are_code_mixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both languages clearing the threshold marks the input as code-mixed."""
    detector = _detector_with(
        monkeypatch, [(0.55, "__label__hin_Latn"), (0.40, "__label__eng_Latn")]
    )

    result = detector.detect("ZED certification ke liye documents kya chahiye")

    assert result["lang_code"] == "code_mixed"
    assert result["codes"] == ["hin_Latn", "eng_Latn"]


def test_one_dominant_language_is_not_code_mixed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A confident single prediction stays a single language."""
    detector = _detector_with(
        monkeypatch, [(0.90, "__label__hin_Latn"), (0.05, "__label__eng_Latn")]
    )

    assert detector.detect("mujhe loan chahiye")["lang_code"] == "hin"


def test_blank_and_failing_input_degrade_to_english(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty text or a broken model yields English rather than an exception."""
    assert IndicLanguageDetector().detect("")["lang_code"] == "en"

    def _boom(cls: Any, repo: str) -> Any:
        raise RuntimeError("fasttext missing")

    monkeypatch.setattr(IndicLanguageDetector, "_load", classmethod(_boom))
    result = IndicLanguageDetector().detect("कुछ भी")

    assert result["lang_code"] == "en"
    assert result["confidence"] == 0.0


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------

def test_to_azure_lang_maps_iso_codes() -> None:
    """Short and three-letter codes both resolve to Azure's language codes."""
    assert to_azure_lang("hi") == "hi"
    assert to_azure_lang("hin") == "hi"
    assert to_azure_lang("ta") == "ta"
    assert to_azure_lang("en") == ENGLISH_TAG
    assert to_azure_lang("hin_Deva") == "hi"


def test_to_azure_lang_ignores_script_suffix() -> None:
    """Azure has one code per language, so a romanised tag maps the same way."""
    assert to_azure_lang("hin_Latn") == "hi"
    assert to_azure_lang("tam_Latn") == "ta"
    assert to_azure_lang("eng_Latn") == ENGLISH_TAG


def test_to_azure_lang_falls_back_to_english_for_unknown() -> None:
    """An unrecognised code degrades to English rather than raising."""
    assert to_azure_lang("klingon") == ENGLISH_TAG


def test_translate_is_a_noop_when_languages_match() -> None:
    """Identical source and target skip the API call entirely."""
    translator = IndicTranslator()

    assert translator.from_english("already english", "en") == "already english"
    assert translator.to_english("", "hi") == ""


def test_segment_isolates_citations_from_prose() -> None:
    """Citation markers are held out of translation so they survive verbatim."""
    pieces = IndicTranslator._segment(
        "Documents are required [39f4b029]. Trainees must be 18 to 25 [57fd5bb0]."
    )

    verbatim = [text for translate, text in pieces if not translate]
    assert verbatim == ["[39f4b029]", "[57fd5bb0]"]
    # Nothing is lost: the pieces reconstruct the original exactly.
    assert "".join(text for _, text in pieces) == (
        "Documents are required [39f4b029]. Trainees must be 18 to 25 [57fd5bb0]."
    )


def test_segment_splits_long_prose_into_sentences() -> None:
    """Long text is split so no single segment exceeds the configured ceiling."""
    long_text = ("This is a sentence about the scheme eligibility criteria. " * 40).strip()

    pieces = IndicTranslator._segment(long_text)

    assert len(pieces) > 1, "long prose must be broken up"
    assert all(len(text) <= 2000 for _, text in pieces)


def test_segment_leaves_short_text_alone() -> None:
    """Text with no citations and no length problem passes through as one piece."""
    assert IndicTranslator._segment("Short answer.") == [(True, "Short answer.")]


def test_translation_failure_returns_input_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed API call must not lose the user's text."""
    translator = IndicTranslator()

    def _boom(text: str, src_lang: str, tgt_lang: str) -> Any:
        raise RuntimeError("Azure Translator unreachable")

    monkeypatch.setattr(translator, "_request", _boom)

    original = "उद्यम पंजीकरण"
    assert translator.to_english(original, "hi") == original


def test_to_romanized_converts_native_script_to_latin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hindi Devanagari is romanised via Azure's /transliterate endpoint."""
    translator = IndicTranslator()
    calls: list[tuple[str, str, str, str]] = []

    def _fake(text: str, language: str, from_script: str, to_script: str) -> str:
        calls.append((text, language, from_script, to_script))
        return "udyam panjikaran"

    monkeypatch.setattr(translator, "_transliterate_request", _fake)

    assert translator.to_romanized("उद्यम पंजीकरण", "hi") == "udyam panjikaran"
    assert calls == [("उद्यम पंजीकरण", "hi", "Deva", "Latn")]


def test_to_romanized_skips_languages_with_no_native_script_mapping() -> None:
    """A language outside NATIVE_SCRIPT passes through unchanged rather than erroring."""
    translator = IndicTranslator()

    assert translator.to_romanized("Bonjour", "fr") == "Bonjour"


def test_to_romanized_failure_returns_input_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed transliteration call must not lose the answer."""
    translator = IndicTranslator()

    def _boom(text: str, language: str, from_script: str, to_script: str) -> Any:
        raise RuntimeError("Azure Translator unreachable")

    monkeypatch.setattr(translator, "_transliterate_request", _boom)

    original = "उद्यम पंजीकरण"
    assert translator.to_romanized(original, "hi") == original
