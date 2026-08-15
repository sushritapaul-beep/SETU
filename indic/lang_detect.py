"""Language identification for Indic and code-mixed input, using AI4Bharat IndicLID.

IndicLID ships two fastText classifiers rather than one: a native-script model
(Devanagari, Tamil, Bengali, ...) and a roman-script model for transliterated
text like Hinglish. Which one to consult is decided from the input's Unicode
blocks, because a romanised Hindi sentence carries no Devanagari at all.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

# HuggingFace repos holding the two fastText models. Both are public.
FTN_REPO = "ai4bharat/IndicLID-FTN"  # native scripts
FTR_REPO = "ai4bharat/IndicLID-FTR"  # roman / transliterated
MODEL_FILE = "model_baseline_roman.bin"  # both repos use this filename

# Unicode ranges covering the Indic scripts IndicLID supports. Used only to pick
# which model to ask, not to classify the language itself.
INDIC_RANGES = (
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali / Assamese
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0600, 0x06FF),  # Arabic (Urdu)
    (0x1C50, 0x1C7F),  # Ol Chiki (Santali)
)

# IndicLID spells some scripts differently from the standard ISO 15924 /
# FLORES-200 codes the rest of the pipeline uses, so labels are normalised
# here rather than at every call site.
SCRIPT_ALIASES = {
    "Tamil": "Taml",
    "Telugu": "Telu",
    "Olch": "Olck",
}

DEFAULT_CODE_MIXED_THRESHOLD = 0.3


class IndicLanguageDetector:
    """Detect the language and script of a query, including code-mixed input."""

    # fastText models are ~100MB each; held on the class so repeated
    # construction never reloads them.
    _model_cache: ClassVar[dict[str, Any]] = {}

    def __init__(self, code_mixed_threshold: float = DEFAULT_CODE_MIXED_THRESHOLD) -> None:
        """Configure the detector without loading any model.

        Args:
            code_mixed_threshold: Minimum probability for a second language to
                count as genuinely present rather than classifier noise.
        """
        self.code_mixed_threshold = code_mixed_threshold

    # ------------------------------------------------------------- loading --

    @classmethod
    def _load(cls, repo: str) -> Any:
        """Load and cache one fastText model.

        Args:
            repo: HuggingFace repo id holding the model.

        Returns:
            A loaded fastText model.
        """
        if repo not in cls._model_cache:
            # Imported lazily so importing this module does not require fasttext.
            import fasttext
            from huggingface_hub import hf_hub_download

            logger.info("Loading IndicLID model %s", repo)
            path = hf_hub_download(repo, MODEL_FILE)
            cls._model_cache[repo] = fasttext.load_model(path)
        return cls._model_cache[repo]

    @staticmethod
    def _predict(model: Any, text: str, k: int = 5) -> list[tuple[float, str]]:
        """Run fastText prediction, bypassing its NumPy 2 incompatibility.

        fasttext 0.9.2's ``predict()`` wraps results in ``np.array(..., copy=False)``,
        which NumPy 2 rejects. The underlying C++ binding returns plain tuples and
        is called directly instead.

        Args:
            model: A loaded fastText model.
            text: Text to classify.
            k: Number of candidate labels to return.

        Returns:
            (probability, label) pairs, most probable first.
        """
        # fastText treats a newline as a document separator and will raise on it.
        cleaned = " ".join(text.split())
        return model.f.predict(cleaned, k, 0.0, "strict")

    # ----------------------------------------------------------- detection --

    @staticmethod
    def _has_indic_script(text: str) -> bool:
        """Whether the text contains any supported Indic script character.

        Args:
            text: Input text.

        Returns:
            True if at least one character falls in an Indic Unicode block.
        """
        return any(
            any(low <= ord(char) <= high for low, high in INDIC_RANGES) for char in text
        )

    @staticmethod
    def _split_label(label: str) -> tuple[str, str]:
        """Split a fastText label into language and script codes.

        Args:
            label: A label such as "__label__hin_Deva" or "__label__other".

        Returns:
            (language_code, script_code); script is "Zyyy" when unknown.
        """
        bare = label.removeprefix("__label__")
        if "_" not in bare:
            return bare, "Zyyy"  # ISO 15924 code for "undetermined"
        lang, script = bare.split("_", 1)
        return lang, SCRIPT_ALIASES.get(script, script)

    def detect(self, text: str) -> dict[str, Any]:
        """Identify the language and script of a piece of text.

        Args:
            text: The user's query, in any supported language or script.

        Returns:
            {
              "lang_code": ISO-639-3 code, "en", or "code_mixed",
              "script": ISO 15924 script code,
              "confidence": probability of the top label,
              "codes": all languages above the code-mixed threshold, as
                       "<lang>_<script>" tags,
              "model": which IndicLID model answered ("FTN" or "FTR"),
            }
            An empty or unclassifiable input reports English at zero confidence,
            so the agent degrades to its default path instead of failing.
        """
        if not text or not text.strip():
            return self._unknown()

        use_native = self._has_indic_script(text)
        repo = FTN_REPO if use_native else FTR_REPO

        try:
            model = self._load(repo)
            predictions = self._predict(model, text)
        except Exception as exc:  # noqa: BLE001 - detection must never break a query
            logger.error("Language detection failed (%s): %s", type(exc).__name__, exc)
            return self._unknown()

        if not predictions:
            return self._unknown()

        # Every language the classifier considers genuinely present.
        above = [(prob, label) for prob, label in predictions if prob > self.code_mixed_threshold]
        top_prob, top_label = predictions[0]
        lang, script = self._split_label(top_label)

        codes = [
            "{}_{}".format(*self._split_label(label)) for _, label in above
        ] or [f"{lang}_{script}"]

        result: dict[str, Any] = {
            "lang_code": "en" if lang == "eng" else lang,
            "script": script,
            "confidence": float(top_prob),
            "codes": codes,
            "model": "FTN" if use_native else "FTR",
        }

        # Two languages clearing the threshold means the classifier could not
        # commit to one — the signature of genuinely code-mixed text.
        if len(above) >= 2:
            result["lang_code"] = "code_mixed"
            logger.info("Detected code-mixed input: %s", result["codes"])

        return result

    @staticmethod
    def _unknown() -> dict[str, Any]:
        """Fallback result used when detection is impossible.

        Returns:
            An English result at zero confidence.
        """
        return {
            "lang_code": "en",
            "script": "Latn",
            "confidence": 0.0,
            "codes": ["eng_Latn"],
            "model": "none",
        }

    @classmethod
    def clear_cache(cls) -> None:
        """Drop cached models so tests and long-running processes can free memory."""
        cls._model_cache.clear()

    @classmethod
    def warm(cls) -> None:
        """Eagerly load both fastText classifiers.

        Loading is otherwise lazy on first ``detect()`` call, which means the
        first real user query pays the download-and-load cost (and its
        transient memory spike) inline. Called once at process startup so
        that cost lands before the service reports healthy instead of on a
        user's request.
        """
        cls._load(FTN_REPO)
        cls._load(FTR_REPO)
