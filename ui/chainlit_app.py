"""Chainlit chat UI for SETU — production entrypoint, deployed to Render.

Talks to the FastAPI service over /query/stream rather than importing the
agent — a thin client, so the heavy work happens in one process and this UI
can restart without reloading anything.

Usage:
    chainlit run ui/chainlit_app.py -w
    SETU_API_URL=https://setu-api.onrender.com chainlit run ui/chainlit_app.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import chainlit as cl
import httpx
from chainlit.data.base import BaseDataLayer
from chainlit.input_widget import Select
from chainlit.types import Feedback, PageInfo, PaginatedResponse, Pagination, ThreadDict, ThreadFilter

logger = logging.getLogger("setu.ui")

API_URL = os.environ.get("SETU_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 300.0  # a full run with translation + retries can take minutes

# Citation markers like "[39f4b029]" are internal chunk ids the generator
# embeds for grounding/attribution — meaningful for the Sources lookup, not
# for a person reading the answer, so they're stripped before display.
CITATION_RE = re.compile(r"\s?\[[0-9a-fA-F]{6,}\]")
# Removing a marker can leave a stray space before punctuation behind.
DANGLING_SPACE_RE = re.compile(r"\s+([.।,;:!?])")


def _strip_citations(text: str) -> str:
    """Remove internal chunk-id markers and the whitespace they leave behind."""
    return DANGLING_SPACE_RE.sub(r"\1", CITATION_RE.sub("", text))

# label -> code. "Auto-detect" leaves detection to IndicLID, which is the
# point of the Indic layer; the explicit choices exist for when it gets it
# wrong. The Select widget works in labels, hence label -> code rather than
# the other way round.
LANGUAGES: dict[str, str] = {
    "Auto-detect": "auto",
    "English": "en",
    "हिन्दी (Hindi)": "hi",
    "தமிழ் (Tamil)": "ta",
}
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "code_mixed": "Hinglish",
}

# Graph nodes -> what a user should actually see. Several internal stages
# (intent classification, profile extraction, the self-critique retry check)
# are backend/QA mechanics with nothing to show a person, so they map to
# None and are skipped entirely. Related stages collapse onto the same
# label so they render as one status, not a near-duplicate right after it.
NODE_LABELS: dict[str, str | None] = {
    "detect_language": "Understanding your question",
    "translate_query_to_en": "Understanding your question",
    "classify_intent": None,
    "extract_profile": None,
    "retrieve_hybrid": "Searching official sources",
    "rerank": "Searching official sources",
    "check_eligibility": "Checking eligibility",
    "generate_answer": "Writing your answer",
    "self_critique": None,
    "translate_answer_from_en": "Finalising your answer",
}


# ---------------------------------------------------------------- feedback --
# Chainlit's thumbs-up/down UI only appears once a data layer is registered,
# and BaseDataLayer's contract is full thread/step/element persistence (17
# methods) even though the only thing worth persisting here is feedback
# itself — Langfuse already traces every run, and there's no conversation-
# history/resume feature in this app. Everything except the two feedback
# methods is a deliberate, safe no-op.
FEEDBACK_DB = Path(__file__).resolve().parent.parent / "data" / "feedback.sqlite3"


def _feedback_db() -> sqlite3.Connection:
    FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            for_id TEXT NOT NULL,
            thread_id TEXT,
            value INTEGER NOT NULL,
            comment TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    return conn


class SetuDataLayer(BaseDataLayer):
    """Persists user feedback on answers; everything else is a no-op."""

    async def upsert_feedback(self, feedback: Feedback) -> str:
        feedback_id = feedback.id or str(uuid.uuid4())
        conn = _feedback_db()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO feedback (id, for_id, thread_id, value, comment) "
                "VALUES (?, ?, ?, ?, ?)",
                (feedback_id, feedback.forId, feedback.threadId, feedback.value, feedback.comment),
            )
            conn.commit()
        finally:
            conn.close()
        return feedback_id

    async def delete_feedback(self, feedback_id: str) -> bool:
        conn = _feedback_db()
        try:
            conn.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
            conn.commit()
        finally:
            conn.close()
        return True

    # No auth and no thread-history UI in this app, so nothing below needs
    # a real implementation — each is called by Chainlit's own lifecycle
    # regardless, and must return a harmless, correctly-shaped value.
    async def get_user(self, identifier: str) -> Any | None:
        return None

    async def create_user(self, user: Any) -> Any | None:
        return None

    async def create_element(self, element: Any) -> None:
        return None

    async def get_element(self, thread_id: str, element_id: str) -> Any | None:
        return None

    async def delete_element(self, element_id: str, thread_id: str | None = None) -> None:
        return None

    async def create_step(self, step_dict: Any) -> None:
        return None

    async def update_step(self, step_dict: Any) -> None:
        return None

    async def delete_step(self, step_id: str) -> None:
        return None

    async def get_thread_author(self, thread_id: str) -> str:
        return ""

    async def delete_thread(self, thread_id: str) -> None:
        return None

    async def list_threads(self, pagination: Pagination, filters: ThreadFilter) -> PaginatedResponse:
        return PaginatedResponse(
            pageInfo=PageInfo(hasNextPage=False, startCursor=None, endCursor=None), data=[]
        )

    async def get_thread(self, thread_id: str) -> ThreadDict | None:
        return None

    async def update_thread(
        self,
        thread_id: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> None:
        return None

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        return None

    async def get_favorite_steps(self, user_id: str) -> list:
        return []


@cl.data_layer
def get_data_layer() -> BaseDataLayer:
    return SetuDataLayer()


# ----------------------------------------------------------------- starters --
# A blank chat box is the biggest drop-off point for a first-time visitor who
# doesn't know what's askable. These double as a mini-catalogue of what SETU
# covers, in the language the answer will actually come back in.
@cl.set_starters
async def starters() -> list[cl.Starter]:
    return [
        cl.Starter(
            label="ZED certification documents",
            message="What documents are required for ZED certification?",
        ),
        cl.Starter(
            label="ZED प्रमाणन के दस्तावेज़",
            message="ZED प्रमाणन के लिए कौन से दस्तावेज़ आवश्यक हैं?",
        ),
        cl.Starter(
            label="MSME training age limit",
            message="MSME skill training scheme mein trainee ki age limit kya hai?",
        ),
        cl.Starter(
            label="தகுதி விவரம்",
            message="இந்த திட்டத்திற்கு யார் தகுதியானவர்?",
        ),
    ]


@cl.on_chat_start
async def start() -> None:
    """Assign a session id and offer the language selector.

    One session id per browser session keeps each visitor's graph state
    separate on the API side.
    """
    cl.user_session.set("session_id", str(uuid.uuid4()))
    cl.user_session.set("language", "auto")
    await cl.ChatSettings(
        [
            Select(
                id="language",
                label="Language",
                values=list(LANGUAGES.keys()),
                initial_index=0,
            ),
        ]
    ).send()


@cl.on_settings_update
async def update_settings(settings: dict[str, Any]) -> None:
    """Store the user's language choice for subsequent turns."""
    label = settings.get("language", "Auto-detect")
    cl.user_session.set("language", LANGUAGES.get(label, "auto"))


def _eligibility_markdown(eligibility: dict[str, Any]) -> str:
    """Render an eligibility verdict as an inline-element-ready markdown block.

    Args:
        eligibility: The API's EligibilityResult payload.

    Returns:
        Markdown summarising the verdict, failed rules, and missing fields.
    """
    verdict = "✅ Eligible" if eligibility.get("eligible") else "❌ Not confirmed"
    lines = [f"**{verdict}** — {eligibility.get('scheme_name', '')}", "", eligibility.get("reason", "")]
    if eligibility.get("failed_rules"):
        lines += ["", "**Failed criteria**"]
        lines += [f"- {rule['description']}" for rule in eligibility["failed_rules"]]
    if eligibility.get("missing_fields"):
        lines += ["", "**Still needed:** " + ", ".join(eligibility["missing_fields"])]
    # An authoritative-sounding wrong eligibility verdict is a worse failure
    # than an uncertain-sounding right one — this is a deterministic engine,
    # not legal advice, and rules can change after the corpus was indexed.
    lines += ["", "*Rules can change — confirm with your local DIC or Udyam office before applying.*"]
    return "\n".join(lines)


def _sources_markdown(citations: list[dict[str, Any]]) -> str:
    """Render citations as a deduplicated, inline-element-ready markdown list.

    Args:
        citations: Citation records from the API.

    Returns:
        Markdown listing one link per source scheme. Internal chunk ids
        aren't included — they identify a retrieval unit, not something a
        person reading the answer has any use for.
    """
    seen: dict[str, str] = {}
    for citation in citations:
        url = citation.get("source_url") or ""
        if url and url not in seen:
            seen[url] = citation.get("scheme_name") or url
    # A citation is only worth trusting if it's visibly an official .gov.in
    # domain, not just a blue link — that glanceable signal matters more for
    # a civic-tech answer than for a generic chatbot citation.
    lines = []
    for url, name in seen.items():
        domain = urlparse(url).netloc
        lines.append(f"- [{name}]({url}) — official source: `{domain}`" if domain else f"- [{name}]({url})")
    return "\n".join(lines)


@cl.on_message
async def main(message: cl.Message) -> None:
    """Stream one answer from the SETU API behind a single evolving status line.

    One message is created up front and its content is rewritten in place as
    each stage completes — "Understanding your question..." becomes
    "Searching official sources..." becomes the final answer — the same
    single-line-that-morphs-into-the-answer pattern ChatGPT/Claude/Gemini use,
    rather than a stack of separate step rows.
    """
    session_id = cl.user_session.get("session_id")
    language = cl.user_session.get("language") or "auto"

    payload: dict[str, Any] = {"query": message.content, "session_id": session_id}
    if language != "auto":
        payload["lang"] = language

    status = cl.Message(content="Thinking...")
    await status.send()

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            async with client.stream("POST", f"{API_URL}/query/stream", json=payload) as response:
                if response.status_code != 200:
                    status.content = f"⚠️ The SETU API returned {response.status_code}. Is it running?"
                    await status.update()
                    return

                last_label: str | None = None

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    event = json.loads(line[len("data: ") :])

                    if event["event"] == "node":
                        label = NODE_LABELS.get(event.get("node") or "")
                        # None means "internal, don't show"; repeating the
                        # same label back-to-back would just be visual noise.
                        if label is None or label == last_label:
                            continue
                        last_label = label
                        status.content = f"{label}..."
                        await status.update()

                    elif event["event"] == "error":
                        status.content = f"⚠️ {event.get('message', 'Something went wrong.')}"
                        await status.update()
                        return

                    elif event["event"] == "final":
                        data = event.get("data") or {}
                        elements = []
                        if data.get("eligibility"):
                            elements.append(
                                cl.Text(
                                    name="Eligibility",
                                    content=_eligibility_markdown(data["eligibility"]),
                                    display="inline",
                                )
                            )
                        if data.get("citations"):
                            elements.append(
                                cl.Text(
                                    name="Sources",
                                    content=_sources_markdown(data["citations"]),
                                    display="inline",
                                )
                            )

                        answer = _strip_citations(data.get("answer") or "(no answer)")
                        lang_name = LANGUAGE_NAMES.get(data.get("lang", ""), "")
                        footer = f"\n\n*Answered in {lang_name}.*" if lang_name else ""

                        status.content = answer + footer
                        status.elements = elements
                        await status.update()
                        return
    except httpx.ConnectError:
        status.content = (
            f"⚠️ Cannot reach the SETU API at {API_URL}.\n\n"
            "Start it with: `uvicorn api.main:app --reload`"
        )
        await status.update()
    except Exception as exc:  # noqa: BLE001 - show the user something actionable
        logger.exception("UI request failed")
        status.content = f"⚠️ Request failed: {exc}"
        await status.update()
