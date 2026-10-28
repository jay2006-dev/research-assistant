"""Deterministic Validator node: independently validates claims, citations, and support status.

Design decisions
----------------
- Runs independently of LLM generation: applies strict, deterministic rules only.
- Validates referential integrity: every evidence_id must exist in evidence_store.
- Validates snippet matching: cited snippets must exist verbatim in corporate corpus content.
- Rejects personal memory or external claims: only corporate corpus items in evidence_store qualify.
- Strictly requires BOTH an explicit SUPPORTED verifier decision AND valid citations:
  snippet matching alone is never sufficient to accept an unsupported claim.
- If zero claims survive, emits an honest INSUFFICIENT_EVIDENCE response—never invents text.
- Reconstructs the final terminal ResearchResponse exclusively from surviving validated claims.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from asterline.state import (
    AnswerClaim,
    Citation,
    ClaimVerificationResult,
    ResearchResponse,
    ResearchState,
    SourceConflict,
    VerificationReport,
    VerificationStatus,
    WorkflowOutcome,
)

logger = logging.getLogger(__name__)


def validator_node(state: ResearchState) -> dict[str, Any]:
    """LangGraph node: deterministically validates and assembles the final ResearchResponse.

    Parameters
    ----------
    state : ResearchState
        Current workflow state containing claims, evidence_store, and verification_report.

    Returns
    -------
    dict[str, Any]
        State updates containing terminal response dict and outcome.
    """
    raw_cycles = state.get("research_cycle_count", 1)
    cycles = raw_cycles if isinstance(raw_cycles, int) and raw_cycles >= 0 else 0

    deadline = state.get("deadline")
    if deadline is not None and time.time() >= deadline:
        logger.warning("Workflow execution deadline exceeded at validator entry.")
        response = ResearchResponse(
            answer="Workflow execution deadline exceeded.",
            claims=[],
            citations=[],
            conflicts=[],
            outcome=WorkflowOutcome.INSUFFICIENT_EVIDENCE,
            confidence_note="Workflow execution deadline exceeded before validation.",
            cycles_used=cycles,
            iterations_used=cycles,
        )
        return {
            "response": response.model_dump(),
            "outcome": WorkflowOutcome.INSUFFICIENT_EVIDENCE.value,
        }

    evidence_store = state.get("evidence_store", {})
    raw_claims = state.get("claims", [])
    report_dict = state.get("verification_report", {})

    # Extract conflicts
    conflicts_data = state.get("conflicts", [])
    existing_conflicts: list[SourceConflict] = []
    for c in conflicts_data:
        try:
            existing_conflicts.append(SourceConflict(**c))
        except Exception:
            pass

    # Build strict verification result index mapping: claim_index -> list of ClaimVerificationResult
    from collections import defaultdict

    results_by_index: dict[int, list[ClaimVerificationResult]] = defaultdict(list)
    if report_dict:
        try:
            report = VerificationReport(**report_dict)
            for res in report.claim_results:
                results_by_index[res.claim_index].append(res)
                if report.conflicts:
                    for conf in report.conflicts:
                        if conf not in existing_conflicts:
                            existing_conflicts.append(conf)
        except Exception as err:
            logger.warning("Could not parse verification report: %s", err)

    surviving_claims: list[AnswerClaim] = []
    all_valid_citations: list[Citation] = []
    seen_citation_keys: set[tuple[str, str]] = set()

    for idx, claim_data in enumerate(raw_claims):
        claim_text = claim_data.get("claim_text", "").strip()
        if not claim_text:
            continue

        # Invariant 1: Claim MUST have an unambiguous, matching, explicit SUPPORTED verifier decision
        # 1a. Reject missing verification results
        matching_results = results_by_index.get(idx, [])
        if not matching_results:
            logger.info("Validator rejecting claim %d: missing verification result", idx)
            continue

        # 1b. Reject duplicate or conflicting verification results for the same index (never pick first or last)
        if len(matching_results) > 1:
            logger.warning(
                "Validator rejecting claim %d: duplicate/conflicting verification results found (%d results)",
                idx, len(matching_results),
            )
            continue

        res = matching_results[0]

        # 1c. Reject claim text/identity mismatch (e.g. reordered, stale, or misaligned verification results)
        norm_claim = " ".join(claim_text.lower().split())
        norm_res_text = " ".join(res.claim_text.lower().split())
        if norm_claim != norm_res_text:
            logger.warning(
                "Validator rejecting claim %d: claim text mismatch between claim and verification report ("
                "claim: %r vs verifier: %r)",
                idx, claim_text[:50], res.claim_text[:50],
            )
            continue

        # 1d. Reject non-SUPPORTED status (snippet matching alone NEVER proves semantic support)
        if res.status != VerificationStatus.SUPPORTED:
            logger.info("Validator rejecting claim %d: status is %s (not SUPPORTED)", idx, res.status)
            continue

        # Invariant 2: Validate each citation's referential and snippet integrity
        raw_citations = claim_data.get("citations", [])
        valid_citations_for_claim: list[Citation] = []

        for cit_dict in raw_citations:
            eid = cit_dict.get("evidence_id") or cit_dict.get("chunk_id", "")
            snippet = cit_dict.get("snippet", "").strip()

            # Rule 2a: Reject fabricated or missing evidence IDs
            if not eid or eid not in evidence_store:
                logger.warning("Validator rejected citation: fabricated or missing evidence ID '%s'", eid)
                continue

            # Rule 2b: Reject items from personal memory / untrusted origin
            evidence_item = evidence_store[eid]
            if evidence_item.get("category") == "personal_memory":
                logger.warning("Validator rejected citation: personal memory cannot be company evidence")
                continue

            # Rule 2c: Reject empty or mismatched snippets
            if not snippet:
                logger.warning("Validator rejected citation for '%s': empty snippet", eid)
                continue

            content = evidence_item.get("content", "")
            norm_snippet = " ".join(snippet.lower().split())
            norm_content = " ".join(content.lower().split())
            if norm_snippet not in norm_content:
                logger.warning("Validator rejected citation for '%s': snippet not found in corpus content", eid)
                continue

            citation_obj = Citation(
                evidence_id=eid,
                chunk_id=evidence_item.get("chunk_id", eid),
                snippet=snippet,
            )
            valid_citations_for_claim.append(citation_obj)

            cit_key = (eid, snippet)
            if cit_key not in seen_citation_keys:
                seen_citation_keys.add(cit_key)
                all_valid_citations.append(citation_obj)

        # Invariant 3: Reject claim if zero valid citations survive
        if not valid_citations_for_claim:
            logger.info("Validator rejecting claim %d: zero valid citations survived integrity checks", idx)
            continue

        surviving_claims.append(
            AnswerClaim(
                claim_text=claim_text,
                citations=valid_citations_for_claim,
                verification_status=VerificationStatus.SUPPORTED,
                verification_note="Independently validated by deterministic citation and snippet checks.",
            )
        )

    # Invariant 4: If zero claims survive, return honest insufficient-evidence response
    if not surviving_claims:
        logger.info("Zero claims survived validation. Returning INSUFFICIENT_EVIDENCE response.")
        response = ResearchResponse(
            answer="Insufficient evidence to answer the query with verified support.",
            claims=[],
            citations=[],
            conflicts=existing_conflicts,
            outcome=WorkflowOutcome.INSUFFICIENT_EVIDENCE,
            confidence_note="No synthesized claims survived independent citation and support validation.",
            cycles_used=cycles,
            iterations_used=cycles,
        )
        return {
            "response": response.model_dump(),
            "outcome": WorkflowOutcome.INSUFFICIENT_EVIDENCE.value,
        }

    # Invariant 5: Reconstruct final answer text strictly from surviving validated claims
    answer_text = "\n\n".join(claim.claim_text for claim in surviving_claims)
    confidence_note = (
        f"Verified with {len(surviving_claims)} claim(s) backed by "
        f"{len(all_valid_citations)} citation(s) from corporate corpus evidence."
    )

    response = ResearchResponse(
        answer=answer_text,
        claims=surviving_claims,
        citations=all_valid_citations,
        conflicts=existing_conflicts,
        outcome=WorkflowOutcome.SUCCESS,
        confidence_note=confidence_note,
        cycles_used=cycles,
        iterations_used=cycles,
    )

    logger.info("Validator accepted %d claims with outcome SUCCESS.", len(surviving_claims))
    return {
        "response": response.model_dump(),
        "outcome": WorkflowOutcome.SUCCESS.value,
    }
