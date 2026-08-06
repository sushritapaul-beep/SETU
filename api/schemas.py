"""Pydantic request/response models for the SETU API.

Defined separately from the app so the UI, tests and any future client can
import the contract without pulling in FastAPI or the agent graph.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """A user question, optionally tied to a conversation."""

    query: str = Field(..., min_length=1, max_length=2000, description="The user's question.")
    session_id: str | None = Field(
        default=None,
        description=(
            "Conversation id. Maps to the LangGraph checkpointer's thread_id, so "
            "reusing it continues the same conversation."
        ),
    )
    lang: str | None = Field(
        default=None,
        description="Force a language ('hi', 'ta', 'en'). None auto-detects with IndicLID.",
    )


class Citation(BaseModel):
    """One retrieved passage backing part of an answer."""

    chunk_id: str | None = None
    label: str | None = Field(default=None, description="Short id as it appears in the answer.")
    source_url: str | None = None
    scheme_name: str | None = None


class EligibilityRule(BaseModel):
    """A single rule evaluated during an eligibility check."""

    field: str
    operator: str
    value: Any
    description: str
    actual: Any | None = None


class EligibilityResult(BaseModel):
    """The deterministic eligibility verdict for one scheme."""

    eligible: bool
    scheme_id: str
    scheme_name: str
    source_url: str = ""
    matched_rules: list[EligibilityRule] = Field(default_factory=list)
    failed_rules: list[EligibilityRule] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    reason: str = ""


class QueryResponse(BaseModel):
    """A complete, cited answer."""

    answer: str = Field(..., description="Final answer, in the user's language.")
    citations: list[Citation] = Field(default_factory=list)
    lang: str = Field(default="en", description="Detected language code, or 'code_mixed'.")
    eligibility: EligibilityResult | None = Field(
        default=None,
        description="Present only for eligibility questions; computed by rules, never generated.",
    )
    trace_id: str = Field(..., description="Langfuse trace id, or a local uuid when untraced.")
    # Diagnostics that make a bad answer explainable without opening Langfuse.
    answer_en: str | None = None
    intent: str | None = None
    faithfulness_score: float | None = None
    retries: int = 0


class StreamEvent(BaseModel):
    """One server-sent event emitted while the graph runs."""

    event: Literal["node", "final", "error"]
    node: str | None = Field(default=None, description="Graph node that just finished.")
    message: str | None = Field(default=None, description="Human-readable progress line.")
    data: dict[str, Any] | None = Field(
        default=None, description="Payload; the final event carries the answer."
    )


class HealthResponse(BaseModel):
    """Service and dependency status."""

    status: Literal["ok", "degraded"]
    qdrant: bool = Field(..., description="Whether the collection is reachable and non-empty.")
    collection: str
    points: int = 0
    llm_provider: str
    llm_configured: bool
    langfuse_enabled: bool
