"""Unit and regression tests for Milestone 3 state contracts."""

from asterline.state import (
    AnswerClaim,
    Citation,
    EvidenceItem,
    ResearchState,
    SourceConflict,
    VerificationStatus,
)


def test_answer_claim_computed_field_evidence_ids():
    """AnswerClaim.evidence_ids is serialized into model_dump output."""
    claim = AnswerClaim(
        claim_text="Asterline uses a dual-redundant life support loop.",
        citations=[
            Citation(evidence_id="CH-001", chunk_id="CH-001", snippet="dual-redundant life support"),
            Citation(evidence_id="CH-002", chunk_id="CH-002", snippet="life support loop"),
        ],
        verification_status=VerificationStatus.SUPPORTED,
    )

    dumped = claim.model_dump()
    assert "evidence_ids" in dumped, "evidence_ids was omitted from serialized dictionary"
    assert dumped["evidence_ids"] == ["CH-001", "CH-002"]


def test_research_state_fields():
    """ResearchState TypedDict accepts max_iterations and iteration_count."""
    state: ResearchState = {
        "query": "Test query",
        "sub_queries": ["Test query"],
        "research_cycle_count": 1,
        "max_research_cycles": 2,
        "iteration_count": 1,
        "max_iterations": 2,
        "evidence_store": {},
        "relevant_chunk_ids": [],
        "new_chunk_ids": [],
        "new_chunk_count": 0,
        "evidence": [],
        "relevant_evidence": [],
        "claims": [],
        "route_decision": "verify",
    }
    assert state["research_cycle_count"] == 1
    assert state["max_research_cycles"] == 2
    assert state["iteration_count"] == 1
    assert state["max_iterations"] == 2
    assert state["evidence_store"] == {}


def test_research_response_cycles_used():
    """ResearchResponse includes cycles_used and iterations_used."""
    from asterline.state import ResearchResponse, WorkflowOutcome

    resp = ResearchResponse(
        answer="Grounded answer",
        outcome=WorkflowOutcome.SUCCESS,
        cycles_used=2,
        iterations_used=2,
    )
    assert resp.cycles_used == 2
    assert resp.iterations_used == 2
