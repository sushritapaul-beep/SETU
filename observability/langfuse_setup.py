"""Langfuse tracing setup for the SETU agent.

Tracing is optional by design: if the keys are absent the factory returns None
and every caller runs untraced. A missing observability backend should never be
the reason a citizen cannot get an answer.
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any

from config.settings import get_settings

logger = logging.getLogger(__name__)


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
) -> tuple[dict[str, Any], str]:
    """Assemble a LangGraph invocation config, with tracing when available.

    Args:
        session_id: Conversation id; also becomes the checkpointer thread_id.
        user_query: The question, recorded as trace metadata.
        extra_tags: Additional Langfuse tags.

    Returns:
        (config, trace_id). trace_id is a local uuid when tracing is disabled, so
        callers always have something to correlate logs by.
    """
    from agents.prompts import prompt_versions

    trace_id = str(uuid.uuid4())
    config: dict[str, Any] = {
        "configurable": {"thread_id": session_id},
        "metadata": {
            "session_id": session_id,
            "trace_id": trace_id,
            # Recorded so a score regression can be traced to exact prompt wording.
            "prompt_versions": prompt_versions(),
            "query_preview": user_query[:200],
        },
        "tags": ["setu", *(extra_tags or [])],
    }

    handler = get_callback_handler()
    if handler is not None:
        config["callbacks"] = [handler]

    return config, trace_id


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
