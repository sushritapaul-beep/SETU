"""Offline tests for config.settings — no network, no .env file, no services."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.settings import PROJECT_ROOT, Settings, get_settings

# Every test constructs Settings with _env_file=None so the developer's real
# .env can never leak in and make results machine-dependent.
NO_ENV_FILE = {"_env_file": None}


def test_defaults_apply_when_environment_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty environment yields the documented defaults, not an error."""
    for var in ("LLM_PROVIDER", "GROQ_API_KEY", "QDRANT_URL", "LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(**NO_ENV_FILE)

    assert settings.LLM_PROVIDER == "groq"
    assert settings.GROQ_API_KEY is None
    assert settings.QDRANT_URL == "http://localhost:6333"
    assert settings.QDRANT_COLLECTION == "setu_schemes"
    assert settings.EMBEDDING_MODEL == "BAAI/bge-m3"
    assert settings.EMBEDDING_DIM == 1024
    assert settings.LLM_TEMPERATURE == 0.0
    assert settings.LOG_LEVEL == "INFO"


def test_dummy_environment_variables_are_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Values set in the environment override defaults and are typed correctly."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_dummy_key_for_tests")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_dummy_token_for_tests")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-dummy")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-dummy")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.internal:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-dummy-key")
    monkeypatch.setenv("API_PORT", "9001")

    settings = Settings(**NO_ENV_FILE)

    assert settings.LLM_PROVIDER == "ollama"
    assert settings.QDRANT_URL == "http://qdrant.internal:6333"
    # Env vars arrive as strings; pydantic coerces to the annotated type.
    assert settings.API_PORT == 9001
    assert isinstance(settings.API_PORT, int)
    assert settings.GROQ_API_KEY is not None
    assert settings.GROQ_API_KEY.get_secret_value() == "gsk_dummy_key_for_tests"


def test_secrets_are_masked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Secret values must not appear in logs or tracebacks."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_super_secret_value")

    settings = Settings(**NO_ENV_FILE)

    assert "gsk_super_secret_value" not in repr(settings)
    assert "gsk_super_secret_value" not in str(settings.GROQ_API_KEY)


def test_invalid_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo'd provider fails at startup rather than at first LLM call."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")

    with pytest.raises(ValidationError):
        Settings(**NO_ENV_FILE)


def test_out_of_range_values_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Field constraints catch nonsensical numeric configuration."""
    monkeypatch.setenv("LLM_TEMPERATURE", "5.0")

    with pytest.raises(ValidationError):
        Settings(**NO_ENV_FILE)


def test_validate_llm_credentials_flags_missing_groq_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting Groq without a key is reported, not raised."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    problems = Settings(**NO_ENV_FILE).validate_llm_credentials()

    assert len(problems) == 1
    assert "GROQ_API_KEY" in problems[0]


def test_validate_llm_credentials_passes_for_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local fallback needs no API key, so a missing Groq key is fine."""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert Settings(**NO_ENV_FILE).validate_llm_credentials() == []


def test_blank_secrets_are_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty .env line must not read as a configured credential."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "   ")
    monkeypatch.setenv("GROQ_API_KEY", "")

    settings = Settings(**NO_ENV_FILE)

    assert settings.LANGFUSE_PUBLIC_KEY is None
    assert settings.LANGFUSE_SECRET_KEY is None
    assert settings.GROQ_API_KEY is None
    # The whole point: blank keys must not claim tracing is on.
    assert settings.is_langfuse_configured() is False
    assert settings.validate_llm_credentials(), "blank Groq key must be reported missing"


def test_is_langfuse_configured_requires_both_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracing turns on only when the key pair is complete."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert Settings(**NO_ENV_FILE).is_langfuse_configured() is False

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-dummy")
    assert Settings(**NO_ENV_FILE).is_langfuse_configured() is False

    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-dummy")
    assert Settings(**NO_ENV_FILE).is_langfuse_configured() is True


def test_derived_paths_point_inside_the_repo() -> None:
    """Path properties resolve relative to the repo, independent of cwd."""
    settings = Settings(**NO_ENV_FILE)

    assert settings.raw_data_dir == PROJECT_ROOT / "data" / "raw"
    assert settings.processed_data_dir == PROJECT_ROOT / "data" / "processed"
    assert settings.eligibility_rules_path.name == "eligibility_rules.json"
    # These directories are committed with .gitkeep, so they must exist already.
    assert settings.raw_data_dir.is_dir()
    assert settings.processed_data_dir.is_dir()


def test_get_settings_is_cached() -> None:
    """Repeated calls reuse one instance so .env is parsed once per process."""
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        # Leave no cached instance behind for other tests in the session.
        get_settings.cache_clear()
