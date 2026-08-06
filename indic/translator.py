"""Indic <-> English translation using AI4Bharat IndicTrans2 (distilled 200M).

The distilled 200M checkpoints are used rather than the 1B ones because this
runs on CPU in the demo path, where the 1B model takes tens of seconds per
query.

IMPORTANT: the IndicTrans2 model repos on HuggingFace are GATED. You must accept
the terms at https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M and
put a token in HUGGINGFACE_TOKEN before this module can load anything.
"""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Citation markers emitted by the answer generator, e.g. "[39f4b029]". These must
# survive translation byte-for-byte: IndicTrans2 otherwise transliterates them
# into the target script ("[57எஃப்டி5பிபி0]"), which destroys the citation.
CITATION_RE = re.compile(r"\[[0-9a-fA-F]{6,}\]")

# Long inputs get truncated at max_length, silently losing the end of an answer.
# Anything longer than this is split on sentence boundaries and translated piece
# by piece instead.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")
MAX_CHARS_PER_SEGMENT = 300

INDIC_TO_EN_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"
EN_TO_INDIC_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"

ENGLISH_TAG = "eng_Latn"

# IndicTrans2 uses FLORES-200 tags. Detected languages are mapped here so a
# caller can pass a plain ISO code like "hi" or "hin".
LANG_TO_FLORES = {
    "hi": "hin_Deva",
    "hin": "hin_Deva",
    "ta": "tam_Taml",
    "tam": "tam_Taml",
    "bn": "ben_Beng",
    "ben": "ben_Beng",
    "mr": "mar_Deva",
    "mar": "mar_Deva",
    "te": "tel_Telu",
    "tel": "tel_Telu",
    "kn": "kan_Knda",
    "kan": "kan_Knda",
    "ml": "mal_Mlym",
    "mal": "mal_Mlym",
    "gu": "guj_Gujr",
    "guj": "guj_Gujr",
    "pa": "pan_Guru",
    "pan": "pan_Guru",
    "or": "ory_Orya",
    "ory": "ory_Orya",
    "ur": "urd_Arab",
    "urd": "urd_Arab",
    "as": "asm_Beng",
    "asm": "asm_Beng",
    "en": ENGLISH_TAG,
    "eng": ENGLISH_TAG,
}


def to_flores(lang: str) -> str:
    """Normalise a language identifier to a FLORES-200 tag.

    Romanised input (for example "hin_Latn" from IndicLID) is mapped to the
    native-script tag, because IndicTrans2 has no romanised input mode.

    Args:
        lang: An ISO code ("hi"), an ISO-639-3 code ("hin"), or a full tag
            ("hin_Deva", "hin_Latn").

    Returns:
        A FLORES-200 tag such as "hin_Deva".
    """
    if "_" in lang:
        base, script = lang.split("_", 1)
        if script == "Latn" and base not in {"eng"}:
            # Transliterated Indic text: fall back to the native-script tag.
            return LANG_TO_FLORES.get(base, ENGLISH_TAG)
        return lang
    return LANG_TO_FLORES.get(lang.lower(), ENGLISH_TAG)


class IndicTranslator:
    """Translate between Indic languages and English with IndicTrans2."""

    # Models are ~1GB each; cached on the class so a request never reloads them.
    _model_cache: ClassVar[dict[str, tuple[Any, Any]]] = {}

    def __init__(
        self,
        device: str | None = None,
        max_length: int = 256,
        num_beams: int = 5,
        batch_size: int = 4,
    ) -> None:
        """Configure the translator without loading any model.

        Args:
            device: Torch device. None auto-selects CUDA when available.
            max_length: Maximum generated tokens per sentence.
            num_beams: Beam width. 5 is the IndicTrans2 reference setting.
            batch_size: Segments generated at once. Peak memory scales with
                batch_size * num_beams, and a long answer can split into dozens
                of segments — translating them all in one batch is enough to
                exhaust RAM on a 16GB machine that is already holding BGE-M3 and
                the reranker.
        """
        self.max_length = max_length
        self.num_beams = num_beams
        self.batch_size = batch_size
        self._device = device
        self._processor: Any | None = None

    @property
    def device(self) -> str:
        """The torch device translation runs on.

        Returns:
            "cuda" when a GPU is present and none was forced, else "cpu".
        """
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("IndicTranslator using device: %s", self._device)
        return self._device

    # ------------------------------------------------------------- loading --

    def _load(self, model_id: str) -> tuple[Any, Any]:
        """Load and cache a tokenizer/model pair.

        Args:
            model_id: HuggingFace model id.

        Returns:
            (tokenizer, model), with the model already moved to the device.

        Raises:
            RuntimeError: If the gated repo cannot be accessed.
        """
        if model_id in self._model_cache:
            return self._model_cache[model_id]

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        settings = get_settings()
        token = settings.HUGGINGFACE_TOKEN
        auth = token.get_secret_value() if token else None

        logger.info("Loading %s on %s (first use downloads ~1GB)", model_id, self.device)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, token=auth
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, trust_remote_code=True, token=auth
            )
        except Exception as exc:  # noqa: BLE001 - turn a load failure into actionable guidance
            detail = str(exc)
            # Only blame gating when the error actually is gating; a memory or
            # disk failure wearing an auth message sends you debugging the
            # wrong thing entirely.
            if "gated" in detail.lower() or "401" in detail or "restricted" in detail.lower():
                hint = (
                    f"IndicTrans2 repos are gated: accept the terms at "
                    f"https://huggingface.co/{model_id} and set HUGGINGFACE_TOKEN in .env."
                )
            elif "paging file" in detail.lower() or "1455" in detail:
                hint = (
                    "Windows ran out of virtual memory (os error 1455). The pipeline holds "
                    "~6GB of models; increase the page file size, or run fewer models per "
                    "process."
                )
            else:
                hint = "Check network access and available memory."
            raise RuntimeError(
                f"Could not load {model_id}. {hint} Underlying error: {type(exc).__name__}: {exc}"
            ) from exc

        model = model.to(self.device)
        model.eval()
        self._model_cache[model_id] = (tokenizer, model)
        return tokenizer, model

    @property
    def processor(self) -> Any:
        """The IndicTransToolkit preprocessor, loaded on first use.

        Returns:
            An IndicProcessor instance.
        """
        if self._processor is None:
            from IndicTransToolkit.processor import IndicProcessor

            self._processor = IndicProcessor(inference=True)
        return self._processor

    # --------------------------------------------------------- translation --

    @staticmethod
    def _segment(text: str) -> list[tuple[bool, str]]:
        """Break text into translatable and verbatim pieces.

        Citation markers are isolated so they can be reattached untouched, and
        long runs of prose are split on sentence boundaries so nothing is lost
        to the generator's length limit.

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

    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Run one translation in either direction.

        Args:
            text: Text to translate.
            src_lang: FLORES-200 source tag.
            tgt_lang: FLORES-200 target tag.

        Returns:
            The translated text, or the input unchanged if translation failed.
        """
        if not text or not text.strip():
            return text
        if src_lang == tgt_lang:
            return text

        model_id = EN_TO_INDIC_MODEL if src_lang == ENGLISH_TAG else INDIC_TO_EN_MODEL
        pieces = self._segment(text)
        to_translate = [piece for translate, piece in pieces if translate and piece.strip()]
        if not to_translate:
            return text

        try:
            import torch

            tokenizer, model = self._load(model_id)
            prepared = self.processor.preprocess_batch(
                to_translate, src_lang=src_lang, tgt_lang=tgt_lang
            )

            translated: list[str] = []
            for start in range(0, len(prepared), self.batch_size):
                window = prepared[start : start + self.batch_size]
                encoded = tokenizer(
                    window, truncation=True, padding=True, return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        num_beams=self.num_beams,
                        max_length=self.max_length,
                        min_length=0,
                        num_return_sequences=1,
                        # REQUIRED with transformers >= 4.5x. IndicTrans2's remote
                        # modelling code reads past_key_values[0][0].shape, i.e.
                        # the legacy tuple cache. Newer transformers hands it an
                        # EncoderDecoderCache whose first entry is None, raising
                        # AttributeError. Disabling the cache avoids that path;
                        # segments are short, so the cost is small.
                        use_cache=False,
                    )

                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                translated.extend(self.processor.postprocess_batch(decoded, lang=tgt_lang))
                # Release the graph's activations before the next window.
                del encoded, generated

            # Reassemble in original order, restoring citations byte-for-byte.
            output: list[str] = []
            index = 0
            for translate, piece in pieces:
                if translate and piece.strip():
                    output.append(translated[index])
                    index += 1
                else:
                    output.append(piece)
            joined = " ".join(part.strip() for part in output if part.strip())
            # Segmenting can leave a space before trailing punctuation (including
            # the Devanagari danda), which reads as a typo in the final answer.
            return re.sub(r"\s+([.।,;:!?])", r"\1", joined).strip()
        except Exception as exc:  # noqa: BLE001 - never lose the user's query
            logger.error(
                "Translation %s->%s failed (%s): %s; returning input unchanged",
                src_lang,
                tgt_lang,
                type(exc).__name__,
                exc,
            )
            return text

    def to_english(self, text: str, src_lang: str) -> str:
        """Translate Indic text into English for retrieval.

        Args:
            text: Text in an Indic language.
            src_lang: Source language as an ISO code or FLORES tag.

        Returns:
            The English translation, or the input unchanged on failure.
        """
        return self._translate(text, to_flores(src_lang), ENGLISH_TAG)

    def from_english(self, text: str, tgt_lang: str) -> str:
        """Translate English text into an Indic language for display.

        Args:
            text: English text.
            tgt_lang: Target language as an ISO code or FLORES tag.

        Returns:
            The translated text, or the input unchanged on failure.
        """
        return self._translate(text, ENGLISH_TAG, to_flores(tgt_lang))

    @classmethod
    def clear_cache(cls) -> None:
        """Drop cached models so tests and long-running processes can free memory."""
        cls._model_cache.clear()
