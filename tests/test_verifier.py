"""Tests for Verifier node: citation integrity, support classification, and neutral conflict detection."""

from asterline.providers import FakeLLMProvider
from asterline.state import ResearchState, VerificationStatus
from asterline.verifier import verifier_node


def _make_evidence_store():
    return {
        "chunk-101": {
            "chunk_id": "chunk-101",
            "title": "Data Governance Standard",
            "content": "All customer records are retained for five calendar years following account closure.",
        },
        "chunk-102": {
            "chunk_id": "chunk-102",
            "title": "Legacy Record Policy",
            "content": "Customer records must be retained indefinitely pending system migration.",
        },
    }


def test_verifier_supported_claim_deterministic():
    """Verifier identifies a claim supported by cited corpus evidence snippet."""
    store = _make_evidence_store()
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Customer records are retained for five calendar years.",
                "citations": [
                    {
                        "evidence_id": "chunk-101",
                        "snippet": "retained for five calendar years",
                    }
                ],
            }
        ],
        "evidence_store": store,
        "research_cycle_count": 1,
        "max_research_cycles": 2,
    }

    result = verifier_node(state, provider=None)
    report = result["verification_report"]

    assert report["all_citations_valid"] is True
    assert report["all_claims_supported"] is True
    assert len(report["claim_results"]) == 1
    assert report["claim_results"][0]["status"] == VerificationStatus.SUPPORTED.value
    assert report["recommendation"] == "accept"


def test_verifier_rejects_fabricated_evidence_id():
    """Verifier marks claim UNSUPPORTED when citation references non-existent evidence ID."""
    store = _make_evidence_store()
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Customer records are retained for five calendar years.",
                "citations": [
                    {
                        "evidence_id": "fabricated-chunk-999",
                        "snippet": "retained for five calendar years",
                    }
                ],
            }
        ],
        "evidence_store": store,
        "research_cycle_count": 1,
        "max_research_cycles": 2,
    }

    result = verifier_node(state, provider=None)
    report = result["verification_report"]

    assert report["all_citations_valid"] is False
    assert report["all_claims_supported"] is False
    assert report["claim_results"][0]["status"] == VerificationStatus.UNSUPPORTED.value
    assert "fabricated" in report["claim_results"][0]["note"].lower()


def test_verifier_rejects_mismatched_snippet():
    """Verifier marks claim UNSUPPORTED when snippet is not found in corpus chunk content."""
    store = _make_evidence_store()
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Customer records are retained for five calendar years.",
                "citations": [
                    {
                        "evidence_id": "chunk-101",
                        "snippet": "completely fabricated snippet not in content",
                    }
                ],
            }
        ],
        "evidence_store": store,
        "research_cycle_count": 1,
        "max_research_cycles": 2,
    }

    result = verifier_node(state, provider=None)
    report = result["verification_report"]

    assert report["all_citations_valid"] is False
    assert report["all_claims_supported"] is False
    assert report["claim_results"][0]["status"] == VerificationStatus.UNSUPPORTED.value


def test_verifier_handles_claims_with_no_citations():
    """Claims without citations are classified as UNSUPPORTED with invalid citations."""
    store = _make_evidence_store()
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "An unsubstantiated claim without any citations.",
                "citations": [],
            }
        ],
        "evidence_store": store,
        "research_cycle_count": 1,
        "max_research_cycles": 2,
    }

    result = verifier_node(state, provider=None)
    report = result["verification_report"]

    assert report["all_citations_valid"] is False
    assert report["all_claims_supported"] is False
    assert report["claim_results"][0]["status"] == VerificationStatus.UNSUPPORTED.value


def test_verifier_llm_structured_decisions():
    """Verifier correctly records explicit LLM status decisions: supported, unsupported, uncertain."""
    store = _make_evidence_store()

    # Test UNCERTAIN
    uncertain_llm = (
        '{\n'
        '  "status": "uncertain",\n'
        '  "note": "Ambiguous timeframe mentioned in text.",\n'
        '  "has_conflict": false\n'
        '}'
    )
    provider_uncertain = FakeLLMProvider(default_text=uncertain_llm)
    state_uncertain: ResearchState = {
        "claims": [
            {
                "claim_text": "Records are retained for some period.",
                "citations": [{"evidence_id": "chunk-101", "snippet": "retained for five"}],
            }
        ],
        "evidence_store": store,
        "research_cycle_count": 1,
    }
    res_uncertain = verifier_node(state_uncertain, provider=provider_uncertain)
    assert res_uncertain["verification_report"]["claim_results"][0]["status"] == VerificationStatus.UNCERTAIN.value

    # Test UNSUPPORTED
    unsupported_llm = (
        '{\n'
        '  "status": "unsupported",\n'
        '  "note": "Evidence does not mention ten years.",\n'
        '  "has_conflict": false\n'
        '}'
    )
    provider_unsupported = FakeLLMProvider(default_text=unsupported_llm)
    state_unsupported: ResearchState = {
        "claims": [
            {
                "claim_text": "Records are retained for ten years.",
                "citations": [{"evidence_id": "chunk-101", "snippet": "retained for five"}],
            }
        ],
        "evidence_store": store,
        "research_cycle_count": 1,
    }
    res_unsupported = verifier_node(state_unsupported, provider=provider_unsupported)
    assert res_unsupported["verification_report"]["claim_results"][0]["status"] == VerificationStatus.UNSUPPORTED.value


def test_verifier_neutral_source_conflict_reporting():
    """Verifier reports conflicting source statements neutrally without inventing precedence."""
    store = _make_evidence_store()
    conflict_llm = (
        '{\n'
        '  "status": "conflict",\n'
        '  "note": "Document A states 5 years while Document B states indefinitely.",\n'
        '  "has_conflict": true,\n'
        '  "conflict_description": "Data Governance Standard specifies 5 years whereas Legacy Policy specifies indefinite retention."\n'
        '}'
    )
    provider = FakeLLMProvider(default_text=conflict_llm)

    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Retention policy is contradictory between standards.",
                "citations": [
                    {"evidence_id": "chunk-101", "snippet": "retained for five"},
                    {"evidence_id": "chunk-102", "snippet": "retained indefinitely"},
                ],
            }
        ],
        "evidence_store": store,
        "research_cycle_count": 1,
        "max_research_cycles": 2,
    }

    result = verifier_node(state, provider=provider)
    conflicts = result["conflicts"]

    assert len(conflicts) == 1
    assert "Data Governance Standard specifies 5 years" in conflicts[0]["description"]
    # Critical invariant: must be unresolved; no authority inferred from timestamps/version
    assert conflicts[0]["resolution"] == "unresolved"
    assert "chunk-101" in conflicts[0]["evidence_ids"]
    assert "chunk-102" in conflicts[0]["evidence_ids"]



def test_verifier_fails_closed_on_provider_auth_error():
    """Requirement C4: Injected verifier provider raising auth error cannot produce supported decision."""
    from asterline.config import Settings
    from asterline.providers import AuthenticationError

    store = _make_evidence_store()
    cfg = Settings(_env_file=None, api_key="secret-api-key-value-12345")

    class FailingAuthProvider(FakeLLMProvider):
        def invoke(self, prompt: str) -> str:
            raise AuthenticationError("Invalid API key secret-api-key-value-12345 provided")

    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Customer records are retained for five calendar years.",
                "citations": [{"evidence_id": "chunk-101", "snippet": "retained for five calendar years"}],
            }
        ],
        "evidence_store": store,
        "research_cycle_count": 1,
        "max_research_cycles": 2,
    }

    result = verifier_node(state, settings=cfg, provider=FailingAuthProvider())
    report = result["verification_report"]

    # Invariant: Must fail closed to UNCERTAIN, never SUPPORTED
    assert report["all_claims_supported"] is False
    assert report["claim_results"][0]["status"] == VerificationStatus.UNCERTAIN.value
    # Safe diagnostic with secret redaction
    assert "secret-api-key-value-12345" not in report["claim_results"][0]["note"]
    assert "[REDACTED]" in report["claim_results"][0]["note"]


def test_verifier_fails_closed_on_malformed_output_and_retry_exhaustion():
    """Requirement C5: Malformed verifier output and retry exhaustion cannot promote claims to SUPPORTED."""
    store = _make_evidence_store()

    # Provider returning unparseable text
    provider_malformed = FakeLLMProvider(default_text="<<< Invalid unparseable non-JSON text >>>")

    state: ResearchState = {
        "claims": [
            {
                # Note: lexical word overlap with snippet is high (100%), but provider returned garbage
                "claim_text": "retained for five calendar years",
                "citations": [{"evidence_id": "chunk-101", "snippet": "retained for five calendar years"}],
            }
        ],
        "evidence_store": store,
        "research_cycle_count": 1,
        "max_research_cycles": 2,
    }

    result = verifier_node(state, provider=provider_malformed)
    report = result["verification_report"]

    # Invariant: Must fail closed to UNCERTAIN, never SUPPORTED via lexical overlap
    assert report["all_claims_supported"] is False
    assert report["claim_results"][0]["status"] == VerificationStatus.UNCERTAIN.value
    assert report["recommendation"] in ("replan", "reject")



def test_deterministic_fallback_rejects_asymmetric_negation():
    """Deterministic fallback: affirmative claim against negated snippet must NOT be SUPPORTED."""
    store = {
        "chunk-neg": {
            "chunk_id": "chunk-neg",
            "title": "Pricing FAQ",
            "content": "Official announcement: Launch does not cost $59.",
        }
    }
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Launch costs $59.",
                "citations": [{"evidence_id": "chunk-neg", "snippet": "Launch does not cost $59"}],
            }
        ],
        "evidence_store": store,
        "research_cycle_count": 1,
        "max_research_cycles": 2,
    }

    result = verifier_node(state, provider=None)
    report = result["verification_report"]

    assert report["all_claims_supported"] is False
    assert report["claim_results"][0]["status"] != VerificationStatus.SUPPORTED.value
    assert report["claim_results"][0]["status"] in (
        VerificationStatus.UNSUPPORTED.value,
        VerificationStatus.UNCERTAIN.value,
    )


def test_deterministic_fallback_rejects_subject_object_inversion():
    """Deterministic fallback: reversed semantic roles (Company A acquired B vs B acquired A) must NOT be SUPPORTED."""
    store = {
        "chunk-acq": {
            "chunk_id": "chunk-acq",
            "title": "Corporate Press Release",
            "content": "In a major transaction, Company B acquired Company A yesterday.",
        }
    }
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Company A acquired Company B.",
                "citations": [{"evidence_id": "chunk-acq", "snippet": "Company B acquired Company A"}],
            }
        ],
        "evidence_store": store,
        "research_cycle_count": 1,
        "max_research_cycles": 2,
    }

    result = verifier_node(state, provider=None)
    report = result["verification_report"]

    assert report["all_claims_supported"] is False
    assert report["claim_results"][0]["status"] != VerificationStatus.SUPPORTED.value
    assert report["claim_results"][0]["status"] in (
        VerificationStatus.UNSUPPORTED.value,
        VerificationStatus.UNCERTAIN.value,
    )


def test_deterministic_fallback_rejects_misassociated_numbers():
    """Deterministic fallback: claim assigning wrong number to an entity must NOT be SUPPORTED."""
    store = {
        "chunk-models": {
            "chunk_id": "chunk-models",
            "title": "Product Catalog",
            "content": "Standard Model X costs $50 while upgraded Model Y costs $100.",
        }
    }
    state: ResearchState = {
        "claims": [
            {
                # False claim: pairs Model X with $100 instead of $50
                "claim_text": "Model X costs $100.",
                "citations": [
                    {
                        "evidence_id": "chunk-models",
                        "snippet": "Model X costs $50 while upgraded Model Y costs $100",
                    }
                ],
            }
        ],
        "evidence_store": store,
        "research_cycle_count": 1,
        "max_research_cycles": 2,
    }

    result = verifier_node(state, provider=None)
    report = result["verification_report"]

    assert report["all_claims_supported"] is False
    assert report["claim_results"][0]["status"] != VerificationStatus.SUPPORTED.value
    assert report["claim_results"][0]["status"] in (
        VerificationStatus.UNSUPPORTED.value,
        VerificationStatus.UNCERTAIN.value,
    )


def test_deterministic_fallback_accepts_straightforward_exact_match():
    """Deterministic fallback: exact normalized phrase match can still be conservative SUPPORTED."""
    store = {
        "chunk-exact": {
            "chunk_id": "chunk-exact",
            "title": "Corporate Retention Schedule",
            "content": "All customer records are retained for five calendar years following account closure.",
        }
    }
    state: ResearchState = {
        "claims": [
            {
                "claim_text": "Customer records are retained for five calendar years.",
                "citations": [
                    {
                        "evidence_id": "chunk-exact",
                        "snippet": "customer records are retained for five calendar years",
                    }
                ],
            }
        ],
        "evidence_store": store,
        "research_cycle_count": 1,
        "max_research_cycles": 2,
    }

    result = verifier_node(state, provider=None)
    report = result["verification_report"]

    assert report["all_citations_valid"] is True
    assert report["all_claims_supported"] is True
    assert report["claim_results"][0]["status"] == VerificationStatus.SUPPORTED.value
