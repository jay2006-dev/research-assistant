"""Unit and regression tests for claim synthesis and snippet validation."""

from asterline.state import EvidenceItem
from asterline.synthesizer import _is_snippet_verified, _parse_claims_response


def _sample_evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            evidence_id="CH-001",
            chunk_id="CH-001",
            doc_id="DOC-01",
            title="Environmental Control Systems",
            content="Asterline station employs a dual-redundant catalytic carbon dioxide scrubber.",
            distance=0.15,
            is_relevant=True,
        )
    ]


def test_is_snippet_verified():
    """_is_snippet_verified matches exact and whitespace-normalized substrings."""
    content = "Asterline station employs a dual-redundant catalytic carbon dioxide scrubber."
    assert _is_snippet_verified("dual-redundant catalytic carbon dioxide scrubber", content) is True
    assert _is_snippet_verified("Dual-Redundant  Catalytic", content) is True
    assert _is_snippet_verified("completely fabricated alien technology", content) is False
    assert _is_snippet_verified("", content) is False


def test_synthesizer_rejects_claim_when_citations_invalid():
    """Claims whose citations reference only invalid evidence IDs are rejected."""
    available = _sample_evidence()
    llm_output = """{
        "claims": [
            {
                "claim_text": "Supported claim with real evidence ID",
                "citations": [
                    {"evidence_id": "CH-001", "chunk_id": "CH-001", "snippet": "dual-redundant catalytic"}
                ]
            },
            {
                "claim_text": "Fabricated claim with fake evidence ID",
                "citations": [
                    {"evidence_id": "CH-999", "chunk_id": "CH-999", "snippet": "some fake snippet"}
                ]
            }
        ]
    }"""

    claims = _parse_claims_response(llm_output, available)
    assert len(claims) == 1, "Fabricated claim survived despite invalid citation ID"
    assert claims[0].claim_text == "Supported claim with real evidence ID"
    assert claims[0].citations[0].evidence_id == "CH-001"


def test_synthesizer_rejects_citation_when_snippet_not_in_evidence():
    """Option A: Citation is rejected when the snippet cannot be found in evidence text."""
    available = _sample_evidence()
    llm_output = """{
        "claims": [
            {
                "claim_text": "Claim citing real ID but with hallucinated quote",
                "citations": [
                    {
                        "evidence_id": "CH-001",
                        "chunk_id": "CH-001",
                        "snippet": "fabricated quote not in content text"
                    }
                ]
            }
        ]
    }"""

    claims = _parse_claims_response(llm_output, available)
    assert len(claims) == 0, (
        "Claim must be rejected when its citation has an unverified snippet under Option A"
    )
