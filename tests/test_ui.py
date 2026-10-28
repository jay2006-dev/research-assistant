"""Tests for Asterline Streamlit research assistant UI and execution helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from asterline.app import (
    execute_research,
    get_memory_store,
    get_settings,
    get_vector_store,
    render_response,
)
from asterline.config import Settings
from asterline.memory import UserMemoryStore
from asterline.providers import FakeLLMProvider
from asterline.state import WorkflowOutcome
from asterline.store import IndexNotInitializedError, VectorStore


@pytest.fixture
def temp_memory_store(tmp_path: Path) -> UserMemoryStore:
    """Fixture providing an isolated SQLite memory store."""
    db_file = tmp_path / "test_ui_memory.db"
    return UserMemoryStore(db_file)


def test_ui_helpers_initialization() -> None:
    """Verify core UI factory helpers instantiate valid components."""
    settings = get_settings()
    assert isinstance(settings, Settings)

    mem_store = get_memory_store(settings)
    assert isinstance(mem_store, UserMemoryStore)

    vs = get_vector_store(settings)
    assert isinstance(vs, VectorStore)


def test_ui_execute_research_uninitialized_index_error() -> None:
    """Verify execute_research safely handles and reports uninitialized vector store."""
    mock_vs = MagicMock(spec=VectorStore)
    mock_vs.get_active_collection.side_effect = IndexNotInitializedError("No active index found.")

    res = execute_research(
        query="What is the pricing?",
        user_id="test_user",
        store=mock_vs,
    )

    assert res["outcome"] == "error"
    assert res["error_type"] == "IndexNotInitializedError"
    assert "Please run 'uv run python -m asterline.cli index'" in res["error_message"]


def test_ui_execute_research_success_workflow(temp_memory_store: UserMemoryStore) -> None:
    """Verify execute_research maps successful graph outcomes and citations cleanly."""
    mock_vs = MagicMock(spec=VectorStore)
    mock_vs.get_active_collection.return_value = MagicMock()

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "outcome": WorkflowOutcome.SUCCESS,
        "memory_status": "loaded",
        "research_cycle_count": 1,
        "evidence": [
            {"chunk_id": "pricing:0", "evidence_id": "pricing:0", "title": "Plans and billing"}
        ],
        "response": {
            "outcome": "success",
            "answer": "Launch costs $59 per month.",
            "claims": [
                {
                    "claim_text": "Launch costs $59 per month.",
                    "citations": [
                        {"chunk_id": "pricing:0", "snippet": "Launch costs $59 per month."}
                    ],
                }
            ],
            "citations": [{"chunk_id": "pricing:0", "snippet": "Launch costs $59 per month."}],
            "conflicts": [],
            "cycles_used": 1,
        },
    }

    with patch("asterline.app.create_research_graph", return_value=mock_graph):
        res = execute_research(
            query="How much does Launch cost?",
            user_id="eval_alice",
            provider=FakeLLMProvider(),
            store=mock_vs,
            memory_store=temp_memory_store,
        )

    assert res["outcome"] == "success"
    assert res["answer"] == "Launch costs $59 per month."
    assert len(res["claims"]) == 1
    assert res["claims"][0]["claim_text"] == "Launch costs $59 per month."
    assert res["claims"][0]["citations"][0]["chunk_id"] == "pricing:0"
    assert res["memory_status"] == "loaded"
    assert res["cycles_used"] == 1

    # Invariant check: Ordinary research query execution MUST NOT write to the user memory store
    memories_after = temp_memory_store.recall(user_id="eval_alice")
    assert len(memories_after) == 0, "Research query execution must not automatically write memories"


def test_ui_execute_research_insufficient_evidence(temp_memory_store: UserMemoryStore) -> None:
    """Verify execute_research correctly surfaces insufficient evidence without hallucination."""
    mock_vs = MagicMock(spec=VectorStore)
    mock_vs.get_active_collection.return_value = MagicMock()

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "outcome": WorkflowOutcome.INSUFFICIENT_EVIDENCE,
        "memory_status": "empty",
        "research_cycle_count": 2,
        "evidence": [],
        "response": {
            "outcome": "insufficient_evidence",
            "answer": "The corporate knowledge base does not contain evidence about quantum computing.",
            "claims": [],
            "citations": [],
            "conflicts": [],
            "cycles_used": 2,
        },
    }

    with patch("asterline.app.create_research_graph", return_value=mock_graph):
        res = execute_research(
            query="What are quantum computing integration specifications?",
            user_id="eval_user",
            provider=FakeLLMProvider(),
            store=mock_vs,
            memory_store=temp_memory_store,
        )

    assert res["outcome"] == "insufficient_evidence"
    assert "quantum computing" in res["answer"]
    assert len(res["claims"]) == 0
    assert len(res["citations"]) == 0


def test_ui_user_memory_isolation_and_operations(temp_memory_store: UserMemoryStore) -> None:
    """Verify user memory operations in UI context strictly enforce user namespace isolation."""
    # 1. Add context for Alice
    m_alice = temp_memory_store.remember(
        user_id="alice",
        category="preference",
        key="target_tier",
        content="Interested in Launch plan.",
    )
    assert m_alice.status == "active"

    # 2. Verify Bob sees zero active memories (isolation invariant)
    bob_memories = temp_memory_store.recall(user_id="bob")
    assert len(bob_memories) == 0

    # 3. Correct context for Alice
    m_corrected = temp_memory_store.correct(
        user_id="alice",
        key="target_tier",
        new_content="Updated to Team plan.",
    )
    assert m_corrected.content == "Updated to Team plan."

    # 4. Forget context for Alice
    forgotten_count = temp_memory_store.forget(user_id="alice", key="target_tier")
    assert forgotten_count == 2  # Active + superseded record wiped

    assert len(temp_memory_store.recall(user_id="alice")) == 0


def test_ui_execute_research_provider_error(temp_memory_store: UserMemoryStore) -> None:
    """Verify execute_research cleanly surfaces provider errors without crashes or secret leakage."""
    mock_vs = MagicMock(spec=VectorStore)
    mock_vs.get_active_collection.return_value = MagicMock()

    mock_provider = MagicMock(spec=FakeLLMProvider)
    mock_provider.invoke.side_effect = RuntimeError("Provider timeout with secret-api-key-12345")

    settings = Settings(api_key="secret-api-key-12345", _env_file=None)

    with patch("asterline.app.create_research_graph", side_effect=RuntimeError("Provider connection failed with secret-api-key-12345")):
        res = execute_research(
            query="Pricing info?",
            user_id="user_1",
            settings=settings,
            provider=mock_provider,
            store=mock_vs,
            memory_store=temp_memory_store,
        )

    assert res["outcome"] == "error"
    assert "secret-api-key-12345" not in res["error_message"]
    assert "[REDACTED]" in res["error_message"] or "failed" in res["error_message"]


def test_ui_render_response_branches() -> None:
    """Verify render_response runs across all outcome branches without exceptions."""
    with patch("streamlit.success"), patch("streamlit.warning"), patch("streamlit.error"), patch("streamlit.info"), patch("streamlit.markdown"), patch("streamlit.metric"), patch("streamlit.caption"), patch("streamlit.code"), patch("streamlit.expander"):
        # 1. Success with claims and citations
        render_response({
            "outcome": "success",
            "query": "How much does Launch cost?",
            "answer": "Launch costs $59 per month.",
            "claims": [
                {
                    "claim_text": "Launch costs $59 per month.",
                    "citations": [{"chunk_id": "pricing:0", "snippet": "Launch costs $59 per month."}],
                }
            ],
            "cycles_used": 1,
            "memory_status": "loaded",
            "relevant_chunks": ["pricing:0"],
        })

        # 2. Insufficient evidence
        render_response({
            "outcome": "insufficient_evidence",
            "query": "Quantum specs?",
            "answer": "Insufficient evidence in corpus.",
            "claims": [],
            "cycles_used": 2,
            "memory_status": "empty",
            "relevant_chunks": [],
        })

        # 3. Conflict detected
        render_response({
            "outcome": "conflict_detected",
            "query": "Retention conflict?",
            "answer": "Conflicting evidence detected.",
            "conflicts": ["pricing:1 says 21 days", "retention:0 says 20 days"],
            "cycles_used": 2,
            "memory_status": "loaded",
            "relevant_chunks": ["pricing:1", "retention:0"],
        })

        # 4. Error
        render_response({
            "outcome": "error",
            "error_type": "IndexNotInitializedError",
            "error_message": "Index missing.",
        })


def test_streamlit_app_startup_and_rendering() -> None:
    """Verify Streamlit app renders successfully via AppTest without crashes or syntax errors."""
    app_file = str(Path(__file__).resolve().parent.parent / "src" / "asterline" / "app.py")
    at = AppTest.from_file(app_file)
    at.run()

    assert not at.exception, f"AppTest raised an exception during startup: {at.exception}"
    assert len(at.title) > 0
    assert "Asterline Enterprise Research Assistant" in at.title[0].value
    # Check that clear button and submit button exist
    buttons = [b.label for b in at.button]
    assert "🔄 New Question / Clear View" in buttons
