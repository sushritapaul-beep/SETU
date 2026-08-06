"""Type-safe application configuration for SETU, loaded once from the environment.

Every tunable value in the system lives here so that behaviour can be changed via
.env without touching code, and so tests can construct an isolated Settings
object instead of depending on the developer's local environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor all paths to the repo root derived from this file, so they resolve
# identically whether the process starts from the CLI, pytest, uvicorn, or Docker.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

LLMProvider = Literal["groq", "ollama"]


class Settings(BaseSettings):
    """Validated runtime configuration assembled from .env and process environment.

    Attributes are grouped by subsystem. Secrets use ``SecretStr`` so they are
    masked in logs, tracebacks and ``repr()`` output rather than leaking.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Ignore unrelated vars (CI runners and Docker inject many) instead of
        # failing to start.
        extra="ignore",
    )

    # ---------- LLM provider ----------
    LLM_PROVIDER: LLMProvider = "groq"

    GROQ_API_KEY: SecretStr | None = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:8b"

    # Deterministic by default: this is a factual, citation-bearing RAG system,
    # and eval scores are only comparable across runs if sampling is disabled.
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    LLM_MAX_RETRIES: int = Field(default=2, ge=0)
    LLM_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0)

    # ---------- Embeddings & reranking ----------
    HUGGINGFACE_TOKEN: SecretStr | None = None
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    # BGE-M3 dense width. Must equal the Qdrant collection's vector size, or
    # every upsert fails — kept here so both sides read one number.
    EMBEDDING_DIM: int = Field(default=1024, gt=0)

    # ---------- Vector store ----------
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: SecretStr | None = None
    QDRANT_COLLECTION: str = "setu_schemes"

    # ---------- Retrieval knobs ----------
    # Fetch wide, then let the cross-encoder narrow: recall is cheap at this
    # stage, and the reranker is what actually determines answer quality.
    RETRIEVAL_TOP_K: int = Field(default=20, gt=0)
    RERANK_TOP_N: int = Field(default=5, gt=0)

    # ---------- Observability ----------
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_SECRET_KEY: SecretStr | None = None
    LANGFUSE_HOST: str = "http://localhost:3000"

    # ---------- API server ----------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = Field(default=8000, gt=0, le=65535)
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator(
        "GROQ_API_KEY",
        "HUGGINGFACE_TOKEN",
        "QDRANT_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        mode="before",
    )
    @classmethod
    def _blank_secret_is_unset(cls, value: object) -> object:
        """Treat an empty or whitespace-only secret as absent.

        A commented-out .env line like `LANGFUSE_PUBLIC_KEY=` yields "", which
        would otherwise become SecretStr("") — not None — and make every
        "is it configured?" check wrongly answer yes.

        Args:
            value: The raw environment value.

        Returns:
            None when the value is blank, otherwise the value unchanged.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    # ---------- Derived paths ----------
    # Exposed as properties rather than fields so they are never overridable from
    # the environment — the on-disk layout is a fact about the repo, not config.

    @property
    def data_dir(self) -> Path:
        """Root of all project data."""
        return PROJECT_ROOT / "data"

    @property
    def raw_data_dir(self) -> Path:
        """Directory holding scraped/uploaded scheme PDFs and HTML."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Directory holding parsed markdown chunks."""
        return self.data_dir / "processed"

    @property
    def eligibility_rules_path(self) -> Path:
        """Path to the hand-authored deterministic eligibility rules."""
        return self.data_dir / "eligibility_rules.json"

    # ---------- Helpers ----------

    def validate_llm_credentials(self) -> list[str]:
        """Report configuration problems that would stop the selected LLM working.

        Returns a list instead of raising so that importing config never crashes
        an offline test run; callers that actually need an LLM check this and
        fail loudly at that point.

        Returns:
            Human-readable problem descriptions. Empty list means usable config.
        """
        problems: list[str] = []
        if self.LLM_PROVIDER == "groq" and self.GROQ_API_KEY is None:
            problems.append(
                "LLM_PROVIDER=groq but GROQ_API_KEY is unset. "
                "Add it to .env or set LLM_PROVIDER=ollama."
            )
        if self.LLM_PROVIDER == "ollama" and not self.OLLAMA_BASE_URL:
            problems.append("LLM_PROVIDER=ollama but OLLAMA_BASE_URL is empty.")
        return problems

    def is_langfuse_configured(self) -> bool:
        """Whether both Langfuse keys are present, i.e. tracing can be enabled.

        Returns:
            True when tracing should be wired up; False to run untraced.
        """
        return self.LANGFUSE_PUBLIC_KEY is not None and self.LANGFUSE_SECRET_KEY is not None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton.

    Cached because reading and validating .env on every access would be wasteful
    in request handlers; tests call ``get_settings.cache_clear()`` to reset.

    Returns:
        The shared, validated Settings instance.
    """
    return Settings()
