"""Synthesizer node: produces structured answer claims grounded in retrieved evidence.

Design decisions
----------------
- Claims are generated from relevant evidence only, not from the LLM's
  parametric knowledge.
- Each claim includes explicit citations referencing real EvidenceItem IDs.
- Citation snippets must be derived from actual retrieved corpus text.
- The synthesizer does not generate a final answer; it produces claims that
  the verifier and validator will check before assembly.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from asterline.providers import (
    FakeLLMProvider,
    LLMProvider,
    MalformedOutputError,
    ProviderError,
    redact_secrets,
)
from asterline.state import (
    AnswerClaim,
    Citation,
    EvidenceItem,
    ResearchState,
    VerificationStatus,
    WorkflowOutcome,
)

logger = logging.getLogger(__name__)


def _build_synthesis_prompt(query: str, evidence: list[EvidenceItem]) -> str:
    """Build the structured claim synthesis prompt."""
    evidence_text = "\n\n".join(
        f"--- Evidence [{i}] ---\n"
        f"ID: {e.evidence_id}\n"
        f"Chunk ID: {e.chunk_id}\n"
        f"Title: {e.title}\n"
        f"Content:\n{e.content}"
        for i, e in enumerate(evidence)
    )

    return (
        "You are a research assistant synthesizing structured answer claims from "
        "retrieved evidence. Each claim must be grounded in the provided evidence.\n\n"
        f"QUERY: {query}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "INSTRUCTIONS:\n"
        "1. Generate factual claims that directly answer the query.\n"
        "2. Each claim MUST cite specific evidence by its ID.\n"
        "3. Each citation MUST include a relevant snippet from the evidence content.\n"
        "4. Do NOT include information not present in the evidence.\n"
        "5. If evidence is insufficient, say so rather than fabricating.\n\n"
        "Respond with a JSON object:\n"
        '{\n'
        '  "claims": [\n'
        '    {\n'
        '      "claim_text": "The factual assertion",\n'
        '      "citations": [\n'
        '        {\n'
        '          "evidence_id": "the_evidence_id",\n'
        '          "chunk_id": "the_chunk_id",\n'
        '          "snippet": "relevant excerpt from the evidence"\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}\n'
    )


def _is_snippet_verified(snippet: str, content: str) -> bool:
    """Verify that a citation snippet is grounded in the retrieved content text."""
    if not snippet or not snippet.strip():
        return False
    # Exact substring check
    if snippet in content:
        return True
    # Normalized whitespace and lowercase check
    norm_snippet = " ".join(snippet.split()).lower()
    norm_content = " ".join(content.split()).lower()
    return norm_snippet in norm_content


def _parse_claims_response(
    response_text: str, available_evidence: list[EvidenceItem]
) -> list[AnswerClaim]:
    """Parse the LLM's claims response into validated AnswerClaim objects.

    Only accepts citations that reference real evidence IDs and whose snippets
    are verified against actual corpus content. Drops any claim that has zero
    valid citations remaining.
    """
    evidence_map = {e.evidence_id: e for e in available_evidence}

    # Try to extract JSON from the response
    json_str = response_text
    if "```json" in response_text:
        start = response_text.index("```json") + len("```json")
        end = response_text.index("```", start)
        json_str = response_text[start:end].strip()
    elif "```" in response_text:
        start = response_text.index("```") + 3
        end = response_text.index("```", start)
        json_str = response_text[start:end].strip()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as err:
        raise MalformedOutputError(f"Cannot parse claims JSON: {err}")

    if isinstance(data, list):
        raw_claims = data
    elif isinstance(data, dict):
        raw_claims = data.get("claims", [])
    else:
        raise MalformedOutputError("Claims response must contain a 'claims' list.")

    if not isinstance(raw_claims, list):
        raise MalformedOutputError("Claims response must contain a 'claims' list.")

    parsed_claims: list[AnswerClaim] = []

    for raw_claim in raw_claims:
        if not isinstance(raw_claim, dict):
            continue

        claim_text = raw_claim.get("claim_text", "").strip()
        if not claim_text:
            continue

        raw_citations = raw_claim.get("citations", [])
        valid_citations: list[Citation] = []

        for raw_cit in raw_citations:
            if not isinstance(raw_cit, dict):
                continue

            eid = raw_cit.get("evidence_id", "").strip()
            cid = raw_cit.get("chunk_id", "").strip()
            snippet = raw_cit.get("snippet", "").strip()

            # Reject citation if evidence_id does not exist
            if eid not in evidence_map:
                logger.warning(
                    "Rejecting citation to nonexistent evidence_id '%s' in claim: %s",
                    eid,
                    claim_text[:80],
                )
                continue

            # Reject citation if snippet cannot be verified against original evidence text (Option A)
            evidence_content = evidence_map[eid].content
            if not _is_snippet_verified(snippet, evidence_content):
                logger.warning(
                    "Rejecting citation to '%s': snippet could not be verified in corpus content.",
                    eid,
                )
                continue

            valid_citations.append(
                Citation(
                    evidence_id=eid,
                    chunk_id=cid or evidence_map[eid].chunk_id,
                    snippet=snippet,
                )
            )

        # Reject the entire claim if zero valid citations remain
        if not valid_citations:
            logger.warning(
                "Rejecting claim because zero citations were valid: %s",
                claim_text[:80],
            )
            continue

        parsed_claims.append(
            AnswerClaim(
                claim_text=claim_text,
                citations=valid_citations,
                verification_status=VerificationStatus.UNSUPPORTED,
            )
        )

    return parsed_claims


def synthesizer_node(
    state: ResearchState,
    *,
    provider: Optional[LLMProvider] = None,
    secrets: Optional[list[str]] = None,
) -> dict[str, Any]:
    """LangGraph node: generates structured claims from relevant evidence.

    Parameters
    ----------
    state : ResearchState
        Current graph state with relevant evidence.
    provider : LLMProvider, optional
        LLM provider for claim generation.
    secrets : list[str], optional
        Secrets to redact from error messages.

    Returns
    -------
    dict
        State updates with claims field.
    """
    query = state.get("query", "")
    relevant_evidence_dicts = state.get("relevant_evidence", [])
    _secrets = secrets or []

    deadline = state.get("deadline")
    if deadline is not None and time.time() >= deadline:
        logger.warning("Workflow execution deadline exceeded at synthesizer entry.")
        return {
            "claims": [],
            "error_message": "Overall workflow execution deadline exceeded.",
            "outcome": WorkflowOutcome.INSUFFICIENT_EVIDENCE.value,
        }

    if not relevant_evidence_dicts:
        return {
            "claims": [],
            "error_message": "No relevant evidence available for synthesis.",
            "outcome": WorkflowOutcome.INSUFFICIENT_EVIDENCE.value,
        }

    # Reconstruct EvidenceItem objects
    evidence_items = [EvidenceItem(**d) for d in relevant_evidence_dicts]

    if provider is None:
        # No provider: create claims directly from evidence (deterministic path)
        claims = _create_direct_claims(query, evidence_items)
    else:
        try:
            prompt = _build_synthesis_prompt(query, evidence_items)
            try:
                response_text = provider.invoke(prompt, deadline=deadline)
            except TypeError:
                response_text = provider.invoke(prompt)
            claims = _parse_claims_response(response_text, evidence_items)
        except MalformedOutputError as err:
            safe_msg = redact_secrets(str(err), _secrets)
            logger.warning("Malformed synthesis output: %s", safe_msg)
            return {
                "claims": [],
                "error_message": f"Synthesis produced malformed output: {safe_msg}",
                "outcome": WorkflowOutcome.MALFORMED_OUTPUT.value,
            }
        except ProviderError as err:
            safe_msg = redact_secrets(str(err), _secrets)
            logger.error("Provider error during synthesis: %s", safe_msg)
            return {
                "claims": [],
                "error_message": f"Provider error during synthesis: {safe_msg}",
                "outcome": WorkflowOutcome.PROVIDER_ERROR.value,
            }

    if not claims:
        return {
            "claims": [],
            "error_message": "Synthesis produced no valid claims.",
            "outcome": WorkflowOutcome.INSUFFICIENT_EVIDENCE.value,
        }

    claims_dicts = [c.model_dump() for c in claims]
    logger.info("Synthesized %d claims from %d evidence items.", len(claims), len(evidence_items))

    return {
        "claims": claims_dicts,
    }


def _create_direct_claims(
    query: str, evidence: list[EvidenceItem]
) -> list[AnswerClaim]:
    """Create claims directly from evidence without an LLM.

    Used as a deterministic fallback when no provider is available.
    Each evidence item becomes a claim with a direct citation.
    """
    claims: list[AnswerClaim] = []
    for e in evidence:
        claim = AnswerClaim(
            claim_text=f"According to '{e.title}': {e.content[:300]}",
            citations=[
                Citation(
                    evidence_id=e.evidence_id,
                    chunk_id=e.chunk_id,
                    snippet=e.content[:200],
                )
            ],
            verification_status=VerificationStatus.UNSUPPORTED,
        )
        claims.append(claim)
    return claims
