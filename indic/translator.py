"""Indic <-> English translation using Azure AI Translator.

Azure replaces a locally-loaded IndicTrans2 (previously ~2GB across two
checkpoints, CPU-bound quadratic decoding) with an API call. Chosen over
Sarvam AI's translate API for this project because Azure's free tier renews
monthly and Sarvam's is a one-time, non-renewing signup credit.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Citation markers emitted by the answer generator, e.g. "[39f4b029]". These must
# survive translation byte-for-byte, so they are isolated here before any text
# reaches the translator.
CITATION_RE = re.compile(r"\[[0-9a-fA-F]{6,}\]")

# Azure's limit is 50,000 characters per request; this segment size is a
# readability/citation-safety margin, not a workaround for that limit.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")
MAX_CHARS_PER_SEGMENT = 2000

AZURE_TRANSLATE_URL = "https://api.cognitive.microsofttranslator.com/translate"
AZURE_TRANSLITERATE_URL = "https://api.cognitive.microsofttranslator.com/transliterate"
API_VERSION = "3.0"

ENGLISH_TAG = "en"

# Native script per Azure language code, for the languages this project
# actually surfaces in the UI. Azure's /transliterate supports many more
# language/script pairs; this only covers what to_romanized ever needs to
# convert to Latin.
NATIVE_SCRIPT = {"hi": "Deva", "ta": "Taml"}
LATIN_SCRIPT = "Latn"

# Azure uses plain ISO 639-1 codes with no script/region suffix. Detected
# languages are mapped here so a caller can pass "hi", "hin", or a
# script-tagged form like "hin_Deva" from IndicLID.
LANG_TO_AZURE = {
    "hi": "hi", "hin": "hi",
    "ta": "ta", "tam": "ta",
    "bn": "bn", "ben": "bn",
    "mr": "mr", "mar": "mr",
    "te": "te", "tel": "te",
    "kn": "kn", "kan": "kn",
    "ml": "ml", "mal": "ml",
    "gu": "gu", "guj": "gu",
    "pa": "pa", "pan": "pa",
    "or": "or", "ory": "or",
    "ur": "ur", "urd": "ur",
    "as": "as", "asm": "as",
    "en": ENGLISH_TAG, "eng": ENGLISH_TAG,
}


def to_azure_lang(lang: str) -> str:
    """Normalise a language identifier to an Azure Translator language code.

    Args:
        lang: An ISO code ("hi"), an ISO-639-3 code ("hin"), or a script-tagged
            form ("hin_Deva", "hin_Latn") — the script suffix is ignored, since
            Azure has one code per language.

    Returns:
        An Azure language code such as "hi".
    """
    base = lang.split("_", 1)[0].lower()
    return LANG_TO_AZURE.get(base, ENGLISH_TAG)


class IndicTranslator:
    """Translate between Indic languages and English via Azure AI Translator."""

    def __init__(self) -> None:
        """Configure the translator."""
        settings = get_settings()
        self._api_key = settings.AZURE_TRANSLATOR_KEY
        self._region = settings.AZURE_TRANSLATOR_REGION

    # --------------------------------------------------------- translation --

    @staticmethod
    def _segment(text: str) -> list[tuple[bool, str]]:
        """Break text into translatable and verbatim pieces.

        Citation markers are isolated so they can be reattached untouched, and
        very long runs of prose are split on sentence boundaries.

        Args:
            text: Text about to be translated.

        Returns:
            (translate?, piece) pairs that concatenate back to the input.
        """
        pieces: list[tuple[bool, str]] = []
        cursor = 0
        for match in CITATION_RE.finditer(text):
            if match.start() > cursor:
                pieces.append((True, text[cursor : match.start()]))
            pieces.append((False, match.group(0)))  # verbatim
            cursor = match.end()
        if cursor < len(text):
            pieces.append((True, text[cursor:]))

        expanded: list[tuple[bool, str]] = []
        for translate, piece in pieces:
            if not translate or len(piece) <= MAX_CHARS_PER_SEGMENT:
                expanded.append((translate, piece))
                continue
            # Keep the whitespace that separated sentences so the rejoined text
            # does not run together.
            for index, sentence in enumerate(SENTENCE_SPLIT_RE.split(piece)):
                if sentence.strip():
                    expanded.append((True, (" " if index else "") + sentence))
        return expanded

    def _segment_and_convert(self, text: str, convert: Callable[[str], str]) -> str:
        """Split text into citation-safe pieces, convert each, and reassemble.

        Shared by translation and transliteration: both need the same
        citation-marker isolation, sentence splitting, and rejoining.

        Args:
            text: Text to process.
            convert: Called with one translatable piece; returns its converted
                form. Raising propagates to the caller unconverted.

        Returns:
            The reassembled, converted text. The input unchanged if it holds
            nothing translatable.
        """
        pieces = self._segment(text)
        to_convert = [piece for translate, piece in pieces if translate and piece.strip()]
        if not to_convert:
            return text

        converted = [convert(piece) for piece in to_convert]

        # Reassemble in original order, restoring citations byte-for-byte.
        output: list[str] = []
        index = 0
        for translate, piece in pieces:
            if translate and piece.strip():
                output.append(converted[index])
                index += 1
            else:
                output.append(piece)
        joined = " ".join(part.strip() for part in output if part.strip())
        # Segmenting can leave a space before trailing punctuation (including
        # the Devanagari danda), which reads as a typo in the final answer.
        return re.sub(r"\s+([.।,;:!?])", r"\1", joined).strip()

    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Run one translation in either direction.

        Args:
            text: Text to translate.
            src_lang: Azure source language code.
            tgt_lang: Azure target language code.

        Returns:
            The translated text, or the input unchanged if translation failed.
        """
        if not text or not text.strip():
            return text
        if src_lang == tgt_lang:
            return text

        try:
            return self._segment_and_convert(
                text, lambda piece: self._request(piece, src_lang, tgt_lang)
            )
        except Exception as exc:  # noqa: BLE001 - never lose the user's query
            logger.error(
                "Translation %s->%s failed (%s): %s; returning input unchanged",
                src_lang,
                tgt_lang,
                type(exc).__name__,
                exc,
            )
            return text

    def _request(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Call Azure Translator's /translate for one segment."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Ocp-Apim-Subscription-Key"] = self._api_key.get_secret_value()
        if self._region:
            headers["Ocp-Apim-Subscription-Region"] = self._region

        response = httpx.post(
            AZURE_TRANSLATE_URL,
            params={"api-version": API_VERSION, "from": src_lang, "to": tgt_lang},
            headers=headers,
            json=[{"Text": text}],
            timeout=30,
        )
        response.raise_for_status()
        return response.json()[0]["translations"][0]["text"]

    def _transliterate_request(
        self, text: str, language: str, from_script: str, to_script: str
    ) -> str:
        """Call Azure Translator's /transliterate for one segment."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Ocp-Apim-Subscription-Key"] = self._api_key.get_secret_value()
        if self._region:
            headers["Ocp-Apim-Subscription-Region"] = self._region

        response = httpx.post(
            AZURE_TRANSLITERATE_URL,
            params={
                "api-version": API_VERSION,
                "language": language,
                "fromScript": from_script,
                "toScript": to_script,
            },
            headers=headers,
            json=[{"Text": text}],
            timeout=30,
        )
        response.raise_for_status()
        return response.json()[0]["text"]

    def to_romanized(self, text: str, lang: str) -> str:
        """Convert native-script text into Latin-script (romanised) form.

        For Hinglish input: the user wrote their query in Latin script, so the
        answer should come back the same way rather than in Devanagari/Tamil
        they may not read fluently.

        Args:
            text: Native-script text, already translated into `lang`.
            lang: The text's language, as an ISO code or tag.

        Returns:
            The romanised text. Unchanged if transliteration failed, or if
            `lang` has no supported native-script mapping (NATIVE_SCRIPT).
        """
        if not text or not text.strip():
            return text
        azure_lang = to_azure_lang(lang)
        script = NATIVE_SCRIPT.get(azure_lang)
        if script is None:
            return text

        try:
            return self._segment_and_convert(
                text,
                lambda piece: self._transliterate_request(piece, azure_lang, script, LATIN_SCRIPT),
            )
        except Exception as exc:  # noqa: BLE001 - a native-script answer beats no answer
            logger.error(
                "Transliteration %s->Latn failed (%s): %s; returning input unchanged",
                azure_lang,
                type(exc).__name__,
                exc,
            )
            return text

    def to_english(self, text: str, src_lang: str) -> str:
        """Translate Indic text into English for retrieval.

        Args:
            text: Text in an Indic language.
            src_lang: Source language as an ISO code or tag.

        Returns:
            The English translation, or the input unchanged on failure.
        """
        return self._translate(text, to_azure_lang(src_lang), ENGLISH_TAG)

    def from_english(self, text: str, tgt_lang: str) -> str:
        """Translate English text into an Indic language for display.

        Args:
            text: English text.
            tgt_lang: Target language as an ISO code or tag.

        Returns:
            The translated text, or the input unchanged on failure.
        """
        return self._translate(text, ENGLISH_TAG, to_azure_lang(tgt_lang))
