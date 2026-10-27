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
