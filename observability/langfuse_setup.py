"""Langfuse tracing setup for the SETU agent.

Tracing is optional by design: if the keys are absent the factory returns None
and every caller runs untraced. A missing observability backend should never be
the reason a citizen cannot get an answer.

Everything here degrades to a no-op rather than raising. The same rule applies
to the helpers used from inside graph nodes (`observation`, `score_trace`,
`set_trace_io`): an observability failure must never change an answer.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Any

from config.settings import get_settings

logger = logging.getLogger(__name__)

# Direct identifiers are redacted before any span leaves the process. Scheme
# questions routinely carry them ("my Aadhaar is 1234 5678 9012", "call me on
# ..."), and a trace store is the wrong place to accumulate them.
#
# Eligibility attributes — income, social category, turnover, gender — are
# deliberately NOT masked: they are exactly what makes a wrong eligibility
# verdict debuggable, and none of them identifies a person on its own.
_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Aadhaar: 12 digits, conventionally written in 4-4-4 groups. Matched before
    # the phone pattern so a 12-digit id is not partly consumed as a 10-digit one.
    #
    # The [-\w] guards matter more than they look: a UUID's middle segments are
    # frequently all-digit ("...-4029-4771-8299-..."), which matches 4-4-4
    # exactly. Without the guards every LangGraph checkpoint namespace and run id
    # in the trace gets partly redacted, destroying the ids used to correlate
    # observations while protecting nothing.
    (re.compile(r"(?<![-\w])\d{4}[ -]?\d{4}[ -]?\d{4}(?![-\w])"), "[AADHAAR]"),
    # PAN: five letters, four digits, one letter.
    (re.compile(r"(?<![-\w])[A-Z]{5}\d{4}[A-Z](?![-\w])"), "[PAN]"),
    # Indian mobile: 10 digits starting 6-9, optionally +91 prefixed. Bounded by
    # lookarounds rather than \b, which cannot match before a leading "+" and
    # would leave it stranded outside the redaction.
    (re.compile(r"(?<![-\w])(?:\+?91[ -]?)?[6-9]\d{9}(?![-\w])"), "[PHONE]"),
    (re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
)


def mask_pii(*, data: Any, **_: Any) -> Any:
    """Redact direct identifiers from a value on its way to Langfuse.

    Passed to the Langfuse client as its ``mask`` hook, so it runs over every
    input, output and metadata value the SDK records, including those the
    LangChain callback handler produces.

    Args:
        data: Any value the SDK is about to record.

    Returns:
        The value with Aadhaar/PAN/phone/email spans replaced by placeholders.
        Non-string leaves are returned unchanged.
    """
    if isinstance(data, str):
        for pattern, placeholder in _PII_PATTERNS:
            data = pattern.sub(placeholder, data)
        return data
    if isinstance(data, dict):
        return {key: mask_pii(data=value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask_pii(data=item) for item in data]
    return data


@lru_cache(maxsize=1)
def get_langfuse_client() -> Any | None:
    """Build the Langfuse client, or None when tracing is not configured.

    Returns:
        A Langfuse client, or None if keys are missing or the SDK fails to load.
    """
    settings = get_settings()
    if not settings.is_langfuse_configured():
        logger.info("Langfuse keys not set; running untraced")
        return None

    try:
        from langfuse import Langfuse

        public = settings.LANGFUSE_PUBLIC_KEY
        secret = settings.LANGFUSE_SECRET_KEY
        client = Langfuse(
            public_key=public.get_secret_value() if public else None,
            secret_key=secret.get_secret_value() if secret else None,
            host=settings.LANGFUSE_HOST,
            mask=mask_pii,
        )
        logger.info("Langfuse client ready (host=%s)", settings.LANGFUSE_HOST)
        return client
    except Exception as exc:  # noqa: BLE001 - observability must never break the app
        logger.error("Could not initialise Langfuse (%s): %s", type(exc).__name__, exc)
        return None


def get_callback_handler() -> Any | None:
    """Build a LangChain callback handler that reports spans to Langfuse.

    Returns:
        A callback handler to pass in a LangGraph config's "callbacks" list, or
        None when tracing is disabled.
    """
    if get_langfuse_client() is None:
        return None

    try:
        # Langfuse v3+ moved the handler here; v2 had it at langfuse.callback.
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as exc:  # noqa: BLE001 - degrade to untraced
        logger.error("Could not build Langfuse handler (%s): %s", type(exc).__name__, exc)
        return None


def build_config(
    session_id: str,
    user_query: str,
    extra_tags: list[str] | None = None,
    user_id: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Assemble a LangGraph invocation config, with tracing when available.

    Args:
        session_id: Conversation id; also becomes the checkpointer thread_id.
        user_query: The question, recorded as trace metadata.
        extra_tags: Additional Langfuse tags.
        user_id: Caller identity, when the deployment knows one.

    Returns:
        (config, trace_id). trace_id is a local uuid when tracing is disabled, so
        callers always have something to correlate logs by.
    """
    from agents.prompts import prompt_versions

    trace_id = str(uuid.uuid4())
    tags = ["setu", *(extra_tags or [])]
    config: dict[str, Any] = {
        "configurable": {"thread_id": session_id},
        # A fixed, descriptive name: the trace list is filterable by operation
        # only if the name identifies the operation rather than the run. Without
        # it every SETU trace is called "LangGraph".
        "run_name": "answer-scheme-question",
        "metadata": {
            # The langfuse_-prefixed keys are the documented way to promote these
            # to real trace attributes. Plain "session_id" stays inert metadata,
            # which leaves the Sessions view empty.
            "langfuse_session_id": session_id,
            "langfuse_tags": tags,
            "trace_id": trace_id,
            # Recorded so a score regression can be traced to exact prompt wording.
            "prompt_versions": prompt_versions(),
            "query_preview": user_query[:200],
        },
        "tags": tags,
    }
    if user_id:
        config["metadata"]["langfuse_user_id"] = user_id

    handler = get_callback_handler()
    if handler is not None:
        config["callbacks"] = [handler]

    return config, trace_id


@contextmanager
def observation(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """Open a typed Langfuse observation, or do nothing when tracing is off.

    Used to give steps that are not LangChain components their correct
    observation type — a retrieval reported as a generic span is invisible to
    Langfuse's retrieval analytics and to the agent graph.

    Only the setup is guarded: an exception raised inside the body propagates
    normally, so the span records the failure instead of hiding it.

    Args:
        name: Observation name, naming the operation not the run.
        as_type: A Langfuse observation type, e.g. "retriever" or "tool".
        **kwargs: Passed through to the SDK, e.g. input=.

    Yields:
        The live observation, or None when untraced.
    """
    client = get_langfuse_client()
    ctx: Any = nullcontext(None)
    if client is not None:
        try:
            ctx = client.start_as_current_observation(name=name, as_type=as_type, **kwargs)
        except Exception as exc:  # noqa: BLE001 - degrade to untraced
            logger.warning("Could not open Langfuse observation %r: %s", name, exc)
            ctx = nullcontext(None)
    with ctx as span:
        yield span


def _client_with_active_trace() -> Any | None:
    """The client, but only when a trace is actually being recorded.

    The "current trace" helpers are no-ops outside a trace, and the SDK warns
    each time one is called without one. That happens on every untraced run —
    tests, the smoke scripts — so the check is done here instead, keeping real
    warnings meaningful.

    Returns:
        The Langfuse client, or None when untraced or outside a span.
    """
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        return client if client.get_current_trace_id() else None
    except Exception:  # noqa: BLE001 - treat an unreadable context as untraced
        return None


def score_trace(name: str, value: float | str, comment: str | None = None) -> None:
    """Attach a score to the trace currently being recorded.

    Scores are what make traces filterable by quality: without them a bad run
    looks exactly like a good one in the trace list.

    Args:
        name: Score name, e.g. "faithfulness".
        value: Numeric or categorical score value.
        comment: Optional free-text note.
    """
    client = _client_with_active_trace()
    if client is None:
        return
    try:
        client.score_current_trace(name=name, value=value, comment=comment)
    except Exception as exc:  # noqa: BLE001 - scoring must never break a run
        logger.warning("Could not record Langfuse score %r: %s", name, exc)


def set_trace_io(input: Any = None, output: Any = None) -> None:  # noqa: A002
    """Set the current trace's top-level input and output.

    Without this the trace list shows the raw LangGraph state — every retrieved
    chunk and every intermediate key — where the question and the answer should
    be.

    The SDK marks this deprecated: in v4 a trace's input/output is meant to come
    from its root observation, which here is the span the LangChain handler
    creates and this code does not own. Overriding is still the only way to get
    a readable trace list without wrapping every graph invocation in a hand-made
    root span. Safe while requirements pin langfuse<5; when that pin moves, the
    replacement is to own the root span at the call sites in api/main.py.

    Args:
        input: Value to show as the trace input.
        output: Value to show as the trace output.
    """
    client = _client_with_active_trace()
    if client is None:
        return
    try:
        client.set_current_trace_io(input=input, output=output)
    except Exception as exc:  # noqa: BLE001 - never break a run over presentation
        logger.warning("Could not set Langfuse trace io: %s", exc)


def flush() -> None:
    """Push any buffered spans to Langfuse.

    Langfuse batches in the background, so a short-lived process (an eval run,
    a CLI script) can exit before its traces are sent.
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:  # noqa: BLE001 - flushing must never raise
        logger.warning("Langfuse flush failed: %s", exc)
