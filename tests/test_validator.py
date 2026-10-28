"""Tests for Deterministic Validator: independent citation integrity and response assembly."""

from asterline.state import ResearchState, VerificationStatus, WorkflowOutcome
from asterline.validator import validator_node


def _make_store():
    return {
        "chunk-1": {
            "chunk_id": "chunk-1",
            "title": "Corporate Code of Conduct",
            "content": "All employees must complete annual security awareness training by December 31.",
            "category": "compliance",
        },
        "chunk-memory": {
            "chunk_id": "chunk-memory",
            "title": "Personal Notes",
            "content": "The CEO likes coffee at 9 AM.",
            "category": "personal_memory",
        },
    }


def test_validator_accepts_valid_supported_claims():
    """Validator accepts claim that has both an explicit SUPPORTED status and matching citations."""
    store = _make_store()
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Employees must complete security training annually.",
                "citations": [
                    {
                        "evidence_id": "chunk-1",
                        "snippet": "annual security awareness training",
                    }
                ],
            }
        ],
        "evidence_store": store,
        "verification_report": {
            "claim_results": [
                {
                    "claim_index": 0,
                    "claim_text": "Employees must complete security training annually.",
                    "status": VerificationStatus.SUPPORTED.value,
                }
            ],
            "all_claims_supported": True,
            "all_citations_valid": True,
        },
        "research_cycle_count": 1,
    }

    result = validator_node(state)
    resp = result["response"]

    assert result["outcome"] == WorkflowOutcome.SUCCESS.value
    assert resp["outcome"] == WorkflowOutcome.SUCCESS.value
    assert "Employees must complete security training annually." in resp["answer"]
    assert len(resp["claims"]) == 1
    assert len(resp["citations"]) == 1
    assert resp["citations"][0]["evidence_id"] == "chunk-1"


def test_validator_rejects_unsupported_claim_despite_matching_snippet():
    """Validator rejects claim if verifier status is UNSUPPORTED, even if snippet matches verbatim."""
    store = _make_store()
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Employees must complete security training bi-monthly.",
                "citations": [
                    {
                        "evidence_id": "chunk-1",
                        "snippet": "annual security awareness training",  # Real snippet, but claim is distorted
                    }
                ],
            }
        ],
        "evidence_store": store,
        "verification_report": {
            "claim_results": [
                {
                    "claim_index": 0,
                    "claim_text": "Employees must complete security training bi-monthly.",
                    "status": VerificationStatus.UNSUPPORTED.value,  # Verifier identified failure
                }
            ],
            "all_claims_supported": False,
            "all_citations_valid": True,
        },
        "research_cycle_count": 1,
    }

    result = validator_node(state)
    resp = result["response"]

    # When all claims are rejected, validator must emit INSUFFICIENT_EVIDENCE
    assert result["outcome"] == WorkflowOutcome.INSUFFICIENT_EVIDENCE.value
    assert resp["outcome"] == WorkflowOutcome.INSUFFICIENT_EVIDENCE.value
    assert resp["claims"] == []
    assert resp["citations"] == []
    assert "Insufficient evidence" in resp["answer"]


def test_validator_rejects_fabricated_evidence_id():
    """Validator rejects citations that reference fabricated or non-existent chunk IDs."""
    store = _make_store()
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Security training is required.",
                "citations": [
                    {
                        "evidence_id": "fabricated-id-404",
                        "snippet": "annual security awareness training",
                    }
                ],
            }
        ],
        "evidence_store": store,
        "verification_report": {
            "claim_results": [
                {
                    "claim_index": 0,
                    "claim_text": "Security training is required.",
                    "status": VerificationStatus.SUPPORTED.value,
                }
            ],
        },
        "research_cycle_count": 1,
    }

    result = validator_node(state)
    assert result["outcome"] == WorkflowOutcome.INSUFFICIENT_EVIDENCE.value
    assert result["response"]["claims"] == []


def test_validator_rejects_mismatched_snippet():
    """Validator rejects citations whose snippets are absent from corpus text."""
    store = _make_store()
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Security training is required.",
                "citations": [
                    {
                        "evidence_id": "chunk-1",
                        "snippet": "hallucinated snippet not found anywhere in document",
                    }
                ],
            }
        ],
        "evidence_store": store,
        "verification_report": {
            "claim_results": [
                {
                    "claim_index": 0,
                    "claim_text": "Security training is required.",
                    "status": VerificationStatus.SUPPORTED.value,
                }
            ],
        },
        "research_cycle_count": 1,
    }

    result = validator_node(state)
    assert result["outcome"] == WorkflowOutcome.INSUFFICIENT_EVIDENCE.value
    assert result["response"]["claims"] == []


def test_validator_rejects_personal_memory_citations():
    """Validator rejects citations referencing personal memory chunks."""
    store = _make_store()
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "The executive enjoys morning beverages.",
                "citations": [
                    {
                        "evidence_id": "chunk-memory",
                        "snippet": "likes coffee at 9 AM",
                    }
                ],
            }
        ],
        "evidence_store": store,
        "verification_report": {
            "claim_results": [
                {
                    "claim_index": 0,
                    "claim_text": "The executive enjoys morning beverages.",
                    "status": VerificationStatus.SUPPORTED.value,
                }
            ],
        },
        "research_cycle_count": 1,
    }

    result = validator_node(state)
    assert result["outcome"] == WorkflowOutcome.INSUFFICIENT_EVIDENCE.value
    assert result["response"]["claims"] == []


def test_validator_preserves_neutral_conflicts():
    """Validator attaches detected source conflicts to terminal response."""
    store = _make_store()
    conflict_dict = {
        "description": "Discrepancy in training deadlines between manuals.",
        "evidence_ids": ["chunk-1"],
        "resolution": "unresolved",
    }
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Employees must complete security training annually.",
                "citations": [{"evidence_id": "chunk-1", "snippet": "annual security awareness training"}],
            }
        ],
        "evidence_store": store,
        "conflicts": [conflict_dict],
        "verification_report": {
            "claim_results": [
                {
                    "claim_index": 0,
                    "claim_text": "Employees must complete security training annually.",
                    "status": VerificationStatus.SUPPORTED.value,
                }
            ],
        },
        "research_cycle_count": 1,
    }

    result = validator_node(state)
    resp = result["response"]
    assert len(resp["conflicts"]) == 1
    assert resp["conflicts"][0]["description"] == "Discrepancy in training deadlines between manuals."
    assert resp["conflicts"][0]["resolution"] == "unresolved"



def test_validator_rejects_claim_when_claim_text_mismatches_verifier_result():
    """Requirement C1: A SUPPORTED result for claim A cannot approve claim B when claim text or ordering differs."""
    store = _make_store()
    state: ResearchState = {
        "claims": [
            {
                # Claim at index 0 in claims list
                "claim_text": "Employees must complete security training annually.",
                "citations": [{"evidence_id": "chunk-1", "snippet": "annual security awareness training"}],
            }
        ],
        "evidence_store": store,
        "verification_report": {
            "claim_results": [
                {
                    "claim_index": 0,
                    # Stale / reordered verification text for claim A at index 0
                    "claim_text": "Discrepant claim text: bonuses are paid semi-annually.",
                    "status": VerificationStatus.SUPPORTED.value,
                }
            ],
            "all_claims_supported": True,
            "all_citations_valid": True,
        },
        "research_cycle_count": 1,
    }

    result = validator_node(state)
    # The claim must be rejected due to text mismatch, resulting in INSUFFICIENT_EVIDENCE
    assert result["outcome"] == WorkflowOutcome.INSUFFICIENT_EVIDENCE.value
    assert result["response"]["claims"] == []


def test_validator_rejects_claim_on_duplicate_or_conflicting_results():
    """Requirement C2: Duplicate/conflicting verification results cannot approve a claim."""
    store = _make_store()
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Employees must complete security training annually.",
                "citations": [{"evidence_id": "chunk-1", "snippet": "annual security awareness training"}],
            }
        ],
        "evidence_store": store,
        "verification_report": {
            "claim_results": [
                {
                    "claim_index": 0,
                    "claim_text": "Employees must complete security training annually.",
                    "status": VerificationStatus.SUPPORTED.value,
                },
                {
                    "claim_index": 0,
                    "claim_text": "Employees must complete security training annually.",
                    "status": VerificationStatus.UNSUPPORTED.value,
                },
            ],
            "all_claims_supported": True,
            "all_citations_valid": True,
        },
        "research_cycle_count": 1,
    }

    result = validator_node(state)
    # Ambiguous duplicate results must NOT be resolved by picking first or last -> rejected
    assert result["outcome"] == WorkflowOutcome.INSUFFICIENT_EVIDENCE.value
    assert result["response"]["claims"] == []


def test_validator_rejects_claim_when_verification_result_is_missing():
    """Requirement C3: Missing verification results are rejected."""
    store = _make_store()
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Employees must complete security training annually.",
                "citations": [{"evidence_id": "chunk-1", "snippet": "annual security awareness training"}],
            }
        ],
        "evidence_store": store,
        "verification_report": {
            "claim_results": [],  # No verification result for index 0
            "all_claims_supported": False,
            "all_citations_valid": True,
        },
        "research_cycle_count": 1,
    }

    result = validator_node(state)
    assert result["outcome"] == WorkflowOutcome.INSUFFICIENT_EVIDENCE.value
    assert result["response"]["claims"] == []
