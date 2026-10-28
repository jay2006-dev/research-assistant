"""Verifier node: evaluates factual claims against cited corpus evidence.

Design decisions
----------------
- Evaluates each claim independently against its cited corpus evidence.
- Explicit structured support decisions: SUPPORTED, UNSUPPORTED, UNCERTAIN, PARTIAL, CONFLICT.
- Keeps deterministic citation integrity (evidence_id exists, snippet matches content)
  strictly separate from semantic support assessment.
- Reports conflicting source statements neutrally; never infers precedence from version
  numbers or publication dates.
- Produces a structured VerificationReport consumed by routing and the deterministic validator.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from asterline.config import Settings
from asterline.providers import LLMProvider, MalformedOutputError, ProviderError, redact_secrets
from asterline.state import (
    ClaimVerificationResult,
    ResearchState,
    SourceConflict,
    VerificationReport,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


def _check_citation_integrity(
    citations: list[dict[str, Any]],
    evidence_store: dict[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Deterministically check if citations reference real evidence items and matching snippets.

    Returns (is_valid, checked_evidence_ids).
    """
    if not citations:
        return False, []

    checked_ids = []
    for cit in citations:
        eid = cit.get("evidence_id") or cit.get("chunk_id", "")
        if not eid or eid not in evidence_store:
            return False, checked_ids

        checked_ids.append(eid)
        snippet = cit.get("snippet", "").strip()
        if not snippet:
            return False, checked_ids

        content = evidence_store[eid].get("content", "")
        # Normalized substring match (case and whitespace insensitive)
        norm_snippet = " ".join(snippet.lower().split())
        norm_content = " ".join(content.lower().split())
        if norm_snippet not in norm_content:
            return False, checked_ids

    return True, checked_ids


def _build_verification_prompt(
    claim_text: str,
    cited_excerpts: list[tuple[str, str]],  # (evidence_id, content)
) -> str:
    """Build the prompt for semantic entailment verification."""
    excerpts_str = "\n\n".join(
        f"--- Evidence [{eid}] ---\n{content}" for eid, content in cited_excerpts
    )
    return (
        "You are a rigorous factual verifier evaluating whether an assertion is "
        "substantiated by corporate evidence.\n\n"
        f"Claim to verify:\n\"{claim_text}\"\n\n"
        f"Cited Evidence Excerpts:\n{excerpts_str}\n\n"
        "Instructions:\n"
        "1. Classify the support status as one of:\n"
        "   - 'supported': The claim is directly and accurately entailed by the evidence.\n"
        "   - 'unsupported': The claim is not substantiated, contradicted, or makes claims not in evidence.\n"
        "   - 'uncertain': The claim is ambiguous, partially supported, or lacks full factual confirmation.\n"
        "   - 'conflict': The evidence contains mutually contradictory statements regarding this claim.\n"
        "2. If a source conflict is detected, report it neutrally without asserting which source is newer or authoritative.\n\n"
        "Respond in valid JSON only:\n"
        "{\n"
        '  "status": "supported" | "unsupported" | "uncertain" | "conflict",\n'
        '  "note": "Short explanation of the evaluation",\n'
        '  "has_conflict": true | false,\n'
        '  "conflict_description": "Neutral description of conflict if any"\n'
        "}"
    )


def _evaluate_semantic_support(
    claim_text: str,
    citations: list[dict[str, Any]],
    evidence_store: dict[str, dict[str, Any]],
    provider: Optional[LLMProvider],
    secrets: list[str],
    deadline: Optional[float] = None,
) -> tuple[VerificationStatus, str, Optional[SourceConflict]]:
    """Evaluate whether evidence semantically entails the claim."""
    # Gather evidence excerpts
    excerpts: list[tuple[str, str]] = []
    checked_ids: list[str] = []
    for cit in citations:
        eid = cit.get("evidence_id") or cit.get("chunk_id", "")
        if eid in evidence_store:
            checked_ids.append(eid)
            excerpts.append((eid, evidence_store[eid].get("content", "")))

    if not excerpts:
        return VerificationStatus.UNSUPPORTED, "No valid evidence available to verify claim.", None

    if provider is not None:
        try:
            from asterline.providers import _extract_json

            prompt = _build_verification_prompt(claim_text, excerpts)
            try:
                response_text = provider.invoke(prompt, deadline=deadline)
            except TypeError:
                response_text = provider.invoke(prompt)
            data = json.loads(_extract_json(response_text))

            raw_status = str(data.get("status", "")).strip().lower()
            note = str(data.get("note", "LLM verification evaluation.")).strip()

            if raw_status == "supported":
                status = VerificationStatus.SUPPORTED
            elif raw_status == "uncertain":
                status = VerificationStatus.UNCERTAIN
            elif raw_status == "conflict":
                status = VerificationStatus.CONFLICT
            elif raw_status == "partial":
                status = VerificationStatus.PARTIAL
            else:
                status = VerificationStatus.UNSUPPORTED

            conflict = None
            if data.get("has_conflict"):
                desc = str(data.get("conflict_description", "Conflicting evidence detected.")).strip()
                conflict = SourceConflict(
                    description=desc,
                    evidence_ids=checked_ids,
                    resolution="unresolved",
                )

            return status, note, conflict
        except Exception as err:
            # FAIL CLOSED: When a configured semantic provider fails (auth error, provider error,
            # malformed output, or retry exhaustion), NEVER fall back to lexical overlap.
            # Return non-approving UNCERTAIN status with safe redacted diagnostic.
            safe_msg = redact_secrets(str(err), secrets)
            logger.warning("Semantic verification provider failed (failing closed to UNCERTAIN): %s", safe_msg)
            return (
                VerificationStatus.UNCERTAIN,
                f"Semantic verification provider error: {safe_msg}",
                None,
            )

    # ── Deterministic fallback (ONLY when provider is None) ──────────────────
    # Limitations: Cannot perform genuine semantic entailment or detect subtle contradictions.
    # Used solely for deterministic offline testing when no semantic provider is configured.
    # Strict conservative policy: only returns SUPPORTED for an exact, normalized sequence match
    # within the cited snippet/content, with strict bidirectional negation and numerical consistency.
    snippets = [cit.get("snippet", "").strip() for cit in citations if cit.get("snippet")]
    if not snippets:
        return VerificationStatus.UNSUPPORTED, "No cited snippets present.", None

    claim_lower = claim_text.lower()
    combined_snippets = " ".join(snippets).lower()
    combined_content = " ".join(content for _, content in excerpts).lower()

    # 1. Strict bidirectional negation consistency check
    negations = {"not", "never", "no", "neither", "nor", "none", "without", "hardly", "barely"}
    claim_words = set(re.findall(r"\w+", claim_lower))
    snippet_words = set(re.findall(r"\w+", combined_snippets))
    if not claim_words:
        return VerificationStatus.UNSUPPORTED, "Empty claim text.", None

    claim_has_neg = bool(claim_words & negations)
    snippet_has_neg = bool(snippet_words & negations)
    if claim_has_neg != snippet_has_neg:
        return (
            VerificationStatus.UNSUPPORTED,
            "Polarity mismatch: negation in claim does not match cited evidence snippet.",
            None,
        )

    # 2. Strict numerical consistency check
    claim_numbers = set(re.findall(r"\b\d+\b", claim_lower))
    snippet_numbers = set(re.findall(r"\b\d+\b", combined_snippets))
    if not claim_numbers.issubset(snippet_numbers):
        return (
            VerificationStatus.UNSUPPORTED,
            "Claim contains numerical figures not substantiated by cited evidence snippets.",
            None,
        )

    # 3. Exact normalized sequence matching
    norm_claim = " ".join(re.sub(r"[^\w\s]", " ", claim_lower).split())
    norm_snippet = " ".join(re.sub(r"[^\w\s]", " ", combined_snippets).split())
    norm_content = " ".join(re.sub(r"[^\w\s]", " ", combined_content).split())

    # Supported ONLY if the claim text is an exact normalized phrase within the snippet,
    # or within the cited content where the snippet is part of the claim
    exact_in_snippet = norm_claim in norm_snippet
    exact_in_content = norm_claim in norm_content and norm_snippet in norm_claim

    if exact_in_snippet or exact_in_content:
        return VerificationStatus.SUPPORTED, "Claim verified from exact normalized snippet match.", None

    # Never force SUPPORTED on partial or bag-of-words overlap
    return (
        VerificationStatus.UNCERTAIN,
        "Conservative verification fallback: claim is not an exact sequence match in cited snippet; "
        "semantic entailment cannot be verified without an LLM provider.",
        None,
    )


def verifier_node(
    state: ResearchState,
    *,
    settings: Optional[Settings] = None,
    provider: Optional[LLMProvider] = None,
) -> dict[str, Any]:
    """LangGraph node: verifies synthesized claims against retrieved corpus evidence.

    Parameters
    ----------
    state : ResearchState
        Current workflow state with claims and evidence_store.
    settings : Settings, optional
        Application settings.
    provider : LLMProvider, optional
        LLM provider for semantic entailment checks.

    Returns
    -------
    dict[str, Any]
        State updates containing verification_report and conflicts.
    """
    cfg = settings or Settings(_env_file=None)
    secrets = [s for s in [cfg.api_key, cfg.langsmith_api_key] if s]

    claims = state.get("claims", [])
    evidence_store = state.get("evidence_store", {})
    cycles = state.get("research_cycle_count", 0)
    max_cycles = state.get("max_research_cycles", 2)

    deadline = state.get("deadline")
    if deadline is not None and time.time() >= deadline:
        logger.warning("Workflow execution deadline exceeded at verifier entry.")
        report = VerificationReport(
            claim_results=[],
            conflicts=[],
            all_citations_valid=False,
            all_claims_supported=False,
            has_unsupported_content=False,
            recommendation="reject",
            summary="Workflow execution deadline exceeded.",
        )
        return {
            "verification_report": report.model_dump(),
            "conflicts": [],
        }

    if not claims:
        report = VerificationReport(
            claim_results=[],
            conflicts=[],
            all_citations_valid=False,
            all_claims_supported=False,
            has_unsupported_content=False,
            recommendation="replan" if cycles < max_cycles else "reject",
            summary="No claims available to verify.",
        )
        return {
            "verification_report": report.model_dump(),
            "conflicts": [],
        }

    claim_results: list[ClaimVerificationResult] = []
    all_conflicts: list[SourceConflict] = []
    all_citations_valid = True

    for idx, claim_dict in enumerate(claims):
        claim_text = claim_dict.get("claim_text", "")
        citations = claim_dict.get("citations", [])

        # 1. Deterministic citation integrity check
        cit_valid, checked_eids = _check_citation_integrity(citations, evidence_store)
        if not cit_valid:
            all_citations_valid = False

        # 2. Semantic support evaluation
        if not cit_valid:
            # A claim with invalid or mismatched citations cannot be semantically supported
            status = VerificationStatus.UNSUPPORTED
            note = "Citations invalid, fabricated, or snippets mismatched from corpus evidence."
            conflict = None
        else:
            status, note, conflict = _evaluate_semantic_support(
                claim_text=claim_text,
                citations=citations,
                evidence_store=evidence_store,
                provider=provider,
                secrets=secrets,
                deadline=deadline,
            )

        if conflict:
            all_conflicts.append(conflict)

        claim_results.append(
            ClaimVerificationResult(
                claim_index=idx,
                claim_text=claim_text,
                status=status,
                evidence_ids_checked=checked_eids,
                note=note,
            )
        )

    all_claims_supported = bool(claim_results) and all(
        res.status == VerificationStatus.SUPPORTED for res in claim_results
    )
    has_unsupported = any(
        res.status in (
            VerificationStatus.UNSUPPORTED,
            VerificationStatus.PARTIAL,
            VerificationStatus.UNCERTAIN,
            VerificationStatus.CONFLICT,
        )
        for res in claim_results
    )

    if all_citations_valid and all_claims_supported and not all_conflicts:
        recommendation = "accept"
        summary = f"All {len(claim_results)} claim(s) successfully verified against corpus evidence."
    elif cycles < max_cycles:
        recommendation = "replan"
        summary = (
            f"Verification found unverified or unsupported claims (supported: {sum(1 for r in claim_results if r.status == VerificationStatus.SUPPORTED)}/{len(claim_results)}). "
            f"Replanning cycle requested."
        )
    else:
        recommendation = "reject"
        summary = (
            f"Verification failed on final cycle ({cycles}/{max_cycles}). "
            f"Supported claims: {sum(1 for r in claim_results if r.status == VerificationStatus.SUPPORTED)}/{len(claim_results)}."
        )

    report = VerificationReport(
        claim_results=claim_results,
        conflicts=all_conflicts,
        all_citations_valid=all_citations_valid,
        all_claims_supported=all_claims_supported,
        has_unsupported_content=has_unsupported,
        recommendation=recommendation,
        summary=summary,
    )

    return {
        "verification_report": report.model_dump(),
        "conflicts": [c.model_dump() for c in all_conflicts],
    }
