"""Centralised, validated application configuration.

Path resolution
---------------
All relative paths (``corpus_path``, ``chroma_persist_dir``, etc.) are resolved
against ``PROJECT_ROOT``, which is derived from the filesystem location of this
module — **not** from the process's current working directory.

This means commands like ``uvicorn``, ``pytest``, or ``streamlit`` produce
consistent results regardless of which directory they are launched from.

``.env`` loading
----------------
The ``.env`` file is likewise loaded from ``PROJECT_ROOT / ".env"``.  If that
file does not exist, pydantic-settings silently falls back to environment
variables and built-in defaults.  No error is raised for a missing ``.env``.

Secret fields
-------------
``api_key`` and ``langsmith_api_key`` use ``Field(repr=False)`` so they are
never included in ``repr(settings)`` or casual log output.  Both default to an
empty string and are **not** required during Milestone 1.
"""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Anchor to the repository root ────────────────────────────────
# This file lives at  src/asterline/config.py
# PROJECT_ROOT   =  …/asterline-research-assistant/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── General ──────────────────────────────────────────────────
    app_name: str = "Asterline Research Assistant"
    debug: bool = False

    # ── Paths (relative to PROJECT_ROOT unless absolute) ─────────
    corpus_path: str = "data/corpus.txt"
    chroma_persist_dir: str = "data/chroma"
    sqlite_url: str = "sqlite+aiosqlite:///data/asterline.db"
    memory_db_path: str = "data/asterline_user_memory.db"
    active_pointer_file: str = "data/chroma_active_pointer.json"
    build_lock_file: str = "data/index_build.lock"

    # ── Vector Index & Retrieval (Milestone 2) ───────────────────
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_collection_name: str = "asterline_knowledge_base"
    retrieval_top_k: int = 5
    max_relevance_distance: float = 0.70

    # ── Model Provider ───────────────────────────────────────────
    model_provider: str = Field(
        default="gemini",
        validation_alias=AliasChoices("model_provider", "llm_provider"),
        description="Active model provider: 'gemini' (default) or 'groq'.",
    )
    model_name: str = "gemini-2.5-flash"
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("api_key", "gemini_api_key"),
        repr=False,
    )
    groq_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("groq_api_key"),
        repr=False,
    )
    groq_model_name: str = "openai/gpt-oss-120b"
    request_timeout: float = Field(
        default=30.0,
        gt=0,
        description="Per-request HTTP timeout in seconds (must be positive).",
    )
    workflow_timeout: float = Field(
        default=120.0,
        gt=0,
        description="Overall research workflow execution deadline in seconds (must be positive).",
    )

    # ── LangSmith (optional) ─────────────────────────────────────
    langsmith_api_key: str = Field(default="", repr=False)
    langsmith_project: str = "asterline-research-assistant"
    langsmith_tracing: bool = False

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        import os

        if self.langsmith_tracing:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            if self.langsmith_project:
                os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
                os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
            if self.langsmith_api_key:
                os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
                os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
        else:
            if os.environ.get("LANGSMITH_TRACING") == "false" or "LANGSMITH_TRACING" not in os.environ:
                os.environ["LANGSMITH_TRACING"] = "false"
                os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # ── Path helpers ─────────────────────────────────────────────

    def resolve_path(self, relative: str) -> Path:
        """Resolve a path relative to ``PROJECT_ROOT``.

        If *relative* is already an absolute path, it is returned unchanged.
        """
        p = Path(relative)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def corpus_abs_path(self) -> Path:
        """Absolute path to the corpus file."""
        return self.resolve_path(self.corpus_path)

    @property
    def chroma_abs_dir(self) -> Path:
        """Absolute path to the Chroma persist directory."""
        return self.resolve_path(self.chroma_persist_dir)

    @property
    def memory_db_abs_path(self) -> Path:
        """Absolute path to the SQLite user memory database."""
        return self.resolve_path(self.memory_db_path)

    @property
    def active_pointer_abs_path(self) -> Path:
        """Absolute path to the active collection pointer JSON."""
        return self.resolve_path(self.active_pointer_file)

    @property
    def build_lock_abs_path(self) -> Path:
        """Absolute path to the index build lock file."""
        return self.resolve_path(self.build_lock_file)
