"""Unit and regression tests for researcher node and relevance filtering."""

from collections import namedtuple
from unittest.mock import MagicMock

from asterline.config import Settings
from asterline.providers import FakeLLMProvider
from asterline.researcher import (
    _apply_distance_filter,
    _apply_semantic_relevance,
    _is_explicitly_negative,
    researcher_node,
)
from asterline.state import EvidenceItem, ResearchState

# Mock SearchResult matching VectorStore.search output contract
SearchResult = namedtuple(
    "SearchResult",
    [
        "chunk_id",
        "doc_id",
        "title",
        "content",
        "category",
        "source_url",
        "published",
        "version",
        "distance",
    ],
)


def _make_evidence(eid: str, dist: float, content: str = "Test content") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=eid,
        chunk_id=eid,
        doc_id="DOC-01",
        title="Doc Title",
        content=content,
        distance=dist,
        is_relevant=False,
    )


def test_researcher_runs_with_default_settings():
    """researcher_node successfully accesses max_relevance_distance on default settings."""
    cfg = Settings(_env_file=None)
    mock_store = MagicMock()
    mock_store.search.return_value = []

    state: ResearchState = {"query": "life support systems"}
    result = researcher_node(state, store=mock_store, settings=cfg)

    assert result["evidence"] == []
    assert result["relevant_evidence"] == []
    assert "No evidence found" in result["error_message"]


def test_semantic_relevance_label_parsing_cases():
    """Verify explicit negative phrases vs ambiguous and positive phrases."""
    # Positive / ambiguous phrases must NOT be classified as negative
    assert _is_explicitly_negative("known to be relevant") is False
    assert _is_explicitly_negative("innovative and relevant") is False
    assert _is_explicitly_negative("no issue found; relevant") is False
    assert _is_explicitly_negative("not unrelated") is False

    # Explicit negatives MUST be classified as negative
    assert _is_explicitly_negative("not relevant") is True
    assert _is_explicitly_negative("unrelated") is True
    assert _is_explicitly_negative("irrelevant") is True
    assert _is_explicitly_negative("0: irrelevant") is True
    assert _is_explicitly_negative("no") is True


def test_semantic_relevance_demotes_only_explicit_negatives():
    """_apply_semantic_relevance demotes only explicit negatives, preserving ambiguous phrases."""
    items = [
        _make_evidence("CH-000", 0.1, "Known system details"),
        _make_evidence("CH-001", 0.2, "Innovative design doc"),
        _make_evidence("CH-002", 0.3, "Normal operation specs"),
        _make_evidence("CH-003", 0.4, "Related telemetry"),
        _make_evidence("CH-004", 0.5, "Unrelated cafeteria menu"),
        _make_evidence("CH-005", 0.6, "Off-topic notes"),
    ]
    for item in items:
        item.is_relevant = True

    response_text = (
        "0: known to be relevant\n"
        "1: innovative and relevant\n"
        "2: no issue found; relevant\n"
        "3: not unrelated\n"
        "4: not relevant\n"
        "5: unrelated\n"
    )
    fake_provider = FakeLLMProvider(default_text=response_text)

    filtered = _apply_semantic_relevance(
        query="Atmospheric scrubbers",
        evidence=items,
        provider=fake_provider,
        secrets=[],
    )

    # 0, 1, 2, 3 must remain True
    assert filtered[0].is_relevant is True, "'known to be relevant' was falsely demoted"
    assert filtered[1].is_relevant is True, "'innovative and relevant' was falsely demoted"
    assert filtered[2].is_relevant is True, "'no issue found; relevant' was falsely demoted"
    assert filtered[3].is_relevant is True, "'not unrelated' was falsely demoted"

    # 4 and 5 must be demoted to False
    assert filtered[4].is_relevant is False, "'not relevant' failed to demote"
    assert filtered[5].is_relevant is False, "'unrelated' failed to demote"


def test_researcher_node_filters_distance_and_semantics_offline():
    """Deterministic offline test of researcher_node distance partitioning and semantic demotion."""
    cfg = Settings(_env_file=None, max_relevance_distance=0.70)
    mock_store = MagicMock()

    # Result 0: distance 0.20 (below 0.70, LLM confirms relevant) -> relevant
    # Result 1: distance 0.40 (below 0.70, LLM demotes as irrelevant) -> not relevant
    # Result 2: distance 0.85 (above 0.70 threshold) -> not relevant
    mock_store.search.return_value = [
        SearchResult("CH-001", "DOC-01", "Title 1", "Content 1", "cat", "", "2026", 1, 0.20),
        SearchResult("CH-002", "DOC-02", "Title 2", "Content 2", "cat", "", "2026", 1, 0.40),
        SearchResult("CH-003", "DOC-03", "Title 3", "Content 3", "cat", "", "2026", 1, 0.85),
    ]

    fake_provider = FakeLLMProvider(default_text="0: relevant\n1: irrelevant")

    state: ResearchState = {"query": "test query"}
    result = researcher_node(state, store=mock_store, settings=cfg, provider=fake_provider)

    assert len(result["evidence"]) == 3, "All retrieved items should be recorded in evidence"
    assert len(result["relevant_evidence"]) == 1, "Only CH-001 should pass distance + semantic filters"
    assert result["relevant_evidence"][0]["evidence_id"] == "CH-001"
