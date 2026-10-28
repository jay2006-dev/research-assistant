"""Milestone 1 acceptance tests — configuration and startup."""

import hashlib
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from asterline.config import PROJECT_ROOT, Settings
from asterline.main import app


# ── Configuration ────────────────────────────────────────────────


def test_default_config_loads():
    """Settings instantiates with correct defaults; no API key required."""
    settings = Settings(_env_file=None)
    assert settings.app_name == "Asterline Research Assistant"
    assert settings.debug is False
    assert settings.api_key == ""


def test_env_override(monkeypatch):
    """Environment variables override default settings."""
    monkeypatch.setenv("APP_NAME", "Test App")
    settings = Settings(_env_file=None)
    assert settings.app_name == "Test App"


def test_relevance_distance_setting(monkeypatch):
    """Settings includes configurable max_relevance_distance defaulting to 0.70."""
    settings = Settings(_env_file=None)
    assert settings.max_relevance_distance == 0.70

    monkeypatch.setenv("MAX_RELEVANCE_DISTANCE", "0.65")
    custom = Settings(_env_file=None)
    assert custom.max_relevance_distance == 0.65


def test_corpus_path_resolves():
    """corpus_abs_path is absolute and points to an existing file.

    If the corpus is missing, the test fails with an actionable message
    telling the developer exactly where the file is expected.
    """
    settings = Settings(_env_file=None)
    corpus = settings.corpus_abs_path
    assert corpus.is_absolute(), f"Expected absolute path, got {corpus}"
    assert corpus.exists(), (
        f"Corpus file not found at: {corpus}\n"
        f"  Expected location: {PROJECT_ROOT / 'data' / 'corpus.txt'}\n"
        f"  Ensure the assignment dataset is placed at data/corpus.txt "
        f"relative to the repository root."
    )


def test_api_key_not_in_repr():
    """Secret fields must not leak into repr or str output."""
    settings = Settings(_env_file=None, api_key="super-secret-key-12345")
    text = repr(settings)
    assert "super-secret-key-12345" not in text, (
        "api_key value was exposed in repr(settings)"
    )


def test_api_key_env_var_aliases(monkeypatch):
    """Settings accepts both API_KEY and GEMINI_API_KEY environment variables."""
    # Ensure clean environment
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # 1. Neither variable configured -> defaults to empty string
    empty_settings = Settings(_env_file=None)
    assert empty_settings.api_key == ""

    # 2. GEMINI_API_KEY populates api_key
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-secret-key-1")
    gemini_settings = Settings(_env_file=None)
    assert gemini_settings.api_key == "gemini-test-secret-key-1"
    assert "gemini-test-secret-key-1" not in repr(gemini_settings)

    # 3. API_KEY populates api_key
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("API_KEY", "api-test-secret-key-2")
    api_settings = Settings(_env_file=None)
    assert api_settings.api_key == "api-test-secret-key-2"
    assert "api-test-secret-key-2" not in repr(api_settings)


def test_langsmith_settings_defaults_and_env_propagation(monkeypatch):
    """LangSmith configuration defaults and environment variable propagation."""
    import os

    # Clean environment
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)

    # Defaults
    settings = Settings(_env_file=None)
    assert settings.langsmith_project == "asterline-research-assistant"
    assert settings.langsmith_tracing is False
    assert settings.langsmith_api_key == ""
    assert "fake-ls-key" not in repr(Settings(_env_file=None, langsmith_api_key="fake-ls-key"))

    # When tracing enabled, environment variables are propagated
    active = Settings(
        _env_file=None,
        langsmith_tracing=True,
        langsmith_api_key="fake-ls-key",
        langsmith_project="asterline-research-assistant",
    )
    assert os.environ.get("LANGSMITH_TRACING") == "true"
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGSMITH_PROJECT") == "asterline-research-assistant"
    assert os.environ.get("LANGSMITH_API_KEY") == "fake-ls-key"


def test_timeout_settings_validation(monkeypatch):
    """request_timeout and workflow_timeout defaults are valid and enforce positive values."""
    from pydantic import ValidationError

    # Defaults
    settings = Settings(_env_file=None)
    assert settings.request_timeout == 30.0
    assert settings.workflow_timeout == 120.0

    # Non-positive values fail validation
    with pytest.raises(ValidationError):
        Settings(_env_file=None, request_timeout=0.0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, request_timeout=-5.0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, workflow_timeout=0.0)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, workflow_timeout=-10.0)

    # Positive overrides succeed
    custom = Settings(_env_file=None, request_timeout=15.0, workflow_timeout=60.0)
    assert custom.request_timeout == 15.0
    assert custom.workflow_timeout == 60.0


def test_config_does_not_modify_corpus():
    """Loading configuration must never alter the corpus file.

    Verifies both content integrity (SHA-256) and filesystem metadata
    (mtime_ns).  If the corpus is missing, the test fails early with an
    actionable message rather than an opaque FileNotFoundError.
    """
    corpus = PROJECT_ROOT / "data" / "corpus.txt"
    if not corpus.exists():
        pytest.fail(
            f"Cannot run corpus-integrity check: file not found at {corpus}.\n"
            f"Place the assignment dataset at data/corpus.txt before running tests."
        )

    stat_before = corpus.stat()
    hash_before = hashlib.sha256(corpus.read_bytes()).hexdigest()

    # Load settings twice to be thorough
    Settings(_env_file=None)
    Settings(_env_file=None)

    stat_after = corpus.stat()
    hash_after = hashlib.sha256(corpus.read_bytes()).hexdigest()

    assert hash_before == hash_after, "Corpus content was modified during config loading"
    assert stat_before.st_mtime_ns == stat_after.st_mtime_ns, (
        "Corpus mtime changed during config loading"
    )


# ── FastAPI Health Endpoint ──────────────────────────────────────


def test_health_endpoint():
    """GET /health returns 200 with status 'ok' (synchronous, deterministic)."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "Asterline Research Assistant"
