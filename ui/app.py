"""Gradio chat UI for SETU.

Talks to the FastAPI service over /query/stream rather than importing the agent,
so the UI stays a thin client: the heavy models live in one process and the UI
can be restarted without reloading them.

Usage:
    python ui/app.py
    SETU_API_URL=http://localhost:8000 python ui/app.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("setu.ui")

API_URL = os.environ.get("SETU_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 300.0  # a full run with translation can take minutes on CPU

# Standalone mode answers from a bundled snapshot instead of calling the API.
# Hugging Face Spaces sets SPACE_ID, so a Space switches over automatically.
STANDALONE = (
    os.environ.get("SETU_STANDALONE", "").lower() in {"1", "true", "yes"}
    or os.environ.get("SPACE_ID") is not None
)

# Value -> label. "auto" leaves detection to IndicLID, which is the point of the
# Indic layer; the explicit choices exist for when detection gets it wrong.
LANGUAGES: dict[str, str] = {
    "auto": "Auto-detect",
    "en": "English",
    "hi": "हिन्दी (Hindi)",
    "ta": "தமிழ் (Tamil)",
}

EXAMPLES = [
    "What documents are required for ZED certification?",
    "ZED प्रमाणन के लिए कौन से दस्तावेज़ आवश्यक हैं?",
    "MSME skill training scheme mein trainee ki age limit kya hai?",
    "இந்த திட்டத்திற்கு யார் தகுதியானவர்?",
]


def format_sources(citations: list[dict[str, Any]], eligibility: dict[str, Any] | None) -> str:
    """Render citations and any eligibility verdict as collapsible markdown.

    Args:
        citations: Citation records from the API.
        eligibility: Eligibility verdict, or None.

    Returns:
        Markdown with <details> blocks, or an empty string if there is nothing
        to show.
    """
    blocks: list[str] = []

    if eligibility:
        verdict = "✅ Eligible" if eligibility.get("eligible") else "❌ Not confirmed"
        lines = [
            f"<details><summary><b>{verdict}</b> — {eligibility.get('scheme_name', '')}</summary>",
            "",
            f"{eligibility.get('reason', '')}",
            "",
        ]
        if eligibility.get("failed_rules"):
            lines.append("**Failed criteria**")
            lines += [f"- {r['description']} (you: `{r.get('actual')}`)"
                      for r in eligibility["failed_rules"]]
            lines.append("")
        if eligibility.get("missing_fields"):
            lines.append("**Still needed:** " + ", ".join(eligibility["missing_fields"]))
            lines.append("")
        lines.append("</details>")
        blocks.append("\n".join(lines))

    if citations:
        # Several passages usually come from one scheme; dedupe so the panel
        # lists sources, not chunks.
        seen: dict[str, str] = {}
        for citation in citations:
            url = citation.get("source_url") or ""
            if url and url not in seen:
                seen[url] = citation.get("scheme_name") or url
        lines = [f"<details><summary>📄 Sources ({len(seen)})</summary>", ""]
        lines += [f"- [{name}]({url})" for url, name in seen.items()]
        labels = ", ".join(f"`{c['label']}`" for c in citations if c.get("label"))
        if labels:
            lines += ["", f"Passage ids: {labels}"]
        lines.append("</details>")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _answer_standalone(message: str) -> Iterator[str]:
    """Answer from the bundled snapshot, without the API or the full agent.

    Args:
        message: The user's question.

    Yields:
        A progress line, then the cited answer.
    """
    from ui.standalone import answer as standalone_answer

    yield "⏳ Searching scheme documents…"
    result = standalone_answer(message)
    panel = format_sources(result["citations"], None)
    note = (
        "\n\n<sub>Demo mode: retrieval + reranking + cited generation from a "
        "bundled snapshot. Translation and the self-critique loop run in the "
        "full local version.</sub>"
    )
    yield result["answer"] + ("\n\n" + panel if panel else "") + note


def stream_answer(
    message: str, history: list[Any], language: str, session_id: str
) -> Iterator[str]:
    """Stream one answer from the API, yielding progress then the final text.

    Args:
        message: The user's question.
        history: Gradio chat history (unused; the API keeps state per session).
        language: Selected language code, or "auto".
        session_id: Conversation id shared across turns.

    Yields:
        Partial markdown to display, progress first and the answer last.
    """
    if STANDALONE:
        yield from _answer_standalone(message)
        return

    payload: dict[str, Any] = {"query": message, "session_id": session_id}
    if language and language != "auto":
        payload["lang"] = language

    steps: list[str] = []
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            with client.stream("POST", f"{API_URL}/query/stream", json=payload) as response:
                if response.status_code != 200:
                    yield f"⚠️ API returned {response.status_code}. Is the server running?"
                    return

                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    event = json.loads(line[len("data: ") :])

                    if event["event"] == "node":
                        steps.append(event.get("message") or event.get("node") or "")
                        # Show the trail so the user sees what it is doing.
                        yield "\n".join(f"⏳ {s}…" for s in steps[-4:])

                    elif event["event"] == "error":
                        yield f"⚠️ {event.get('message', 'Unknown error')}"
                        return

                    elif event["event"] == "final":
                        data = event.get("data") or {}
                        answer = data.get("answer") or "(no answer)"
                        panel = format_sources(data.get("citations") or [], data.get("eligibility"))
                        detected = data.get("lang", "")
                        footer = f"\n\n<sub>language: `{detected}`"
                        if data.get("faithfulness_score") is not None:
                            footer += f" · faithfulness: `{data['faithfulness_score']}`"
                        footer += "</sub>"
                        yield answer + ("\n\n" + panel if panel else "") + footer
                        return
    except httpx.ConnectError:
        yield (
            f"⚠️ Cannot reach the SETU API at {API_URL}.\n\n"
            "Start it with: `uvicorn api.main:app --reload`"
        )
    except Exception as exc:  # noqa: BLE001 - show the user something actionable
        logger.exception("UI request failed")
        yield f"⚠️ Request failed: {exc}"


def build_ui() -> gr.Blocks:
    """Assemble the Gradio interface.

    Returns:
        The Blocks app, ready to launch.
    """
    with gr.Blocks(title="SETU", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# SETU\n"
            "**Scheme Eligibility & Translation Understanding for Bharat**\n\n"
            "Ask about Indian government MSME schemes in English, Hindi, Tamil or "
            "Hinglish. Answers are grounded in official scheme pages and cited; "
            "eligibility is decided by deterministic rules, never generated."
        )

        with gr.Row():
            language = gr.Dropdown(
                choices=[(label, value) for value, label in LANGUAGES.items()],
                value="auto",
                label="Language",
                scale=1,
            )
            gr.Markdown(f"<sub>API: `{API_URL}`</sub>")

        # One id per browser session keeps each visitor's graph state separate.
        session_id = gr.State(lambda: str(uuid.uuid4()))

        gr.ChatInterface(
            fn=stream_answer,
            additional_inputs=[language, session_id],
            type="messages",
            examples=[[e] for e in EXAMPLES],
            cache_examples=False,
            title=None,
            description=None,
        )

    return demo


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)-7s | %(name)s | %(message)s")
    build_ui().launch(server_name="127.0.0.1", server_port=7860, show_api=False)
