"""Researcher node: retrieves corpus evidence and performs relevance filtering.

Integrates with the existing Milestone 2 vector store to retrieve evidence,
then applies both distance-threshold and LLM-based semantic relevance checks.

Design decisions
----------------
- Retrieves evidence through the existing ``VectorStore.search()`` interface.
- Preserves all source metadata and stable chunk IDs throughout the graph.
- Distance threshold (``max_relevance_distance``) is configurable in Settings.
- Combines numerical threshold with semantic relevance audit when an LLM is available.
- When numerical and semantic checks disagree, applies conservative policy:
  the stricter result wins (a relevant-looking result is not accepted solely
  because it is below the threshold, and vice versa).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from asterline.config import Settings
from asterline.providers import LLMProvider, MalformedOutputError, ProviderError, redact_secrets
from asterline.state import (
    EvidenceItem,
    ResearchState,
    WorkflowOutcome,
)
from asterline.store import (
    IndexNotInitializedError,
    VectorStore,
    VectorStoreError,
)

logger = logging.getLogger(__name__)


def _search_result_to_evidence(result: Any) -> EvidenceItem:
    """Convert a VectorStore SearchResult to an EvidenceItem."""
    return EvidenceItem(
        evidence_id=result.chunk_id,
        chunk_id=result.chunk_id,
        doc_id=result.doc_id,
        title=result.title,
        content=result.content,
        category=result.category,
        source_url=result.source_url,
        published=result.published,
        version=result.version,
        distance=result.distance,
        is_relevant=False,  # Will be set by relevance filtering
    )


def _apply_distance_filter(
    evidence: list[EvidenceItem], threshold: float
) -> list[EvidenceItem]:
    """Mark evidence items as relevant if their distance is below the threshold."""
    for item in evidence:
        item.is_relevant = item.distance <= threshold
    return evidence


def _is_explicitly_negative(label: str) -> bool:
    """Determine if a relevance label is an explicit negative assessment.

    Uses normalized word-boundary regex matching. Protects against false positives
    from words like 'known', 'innovative', 'normal', and phrases like 'no issue'
    or double negatives like 'not unrelated'. Ambiguous or unrecognized labels
    are NOT classified as negative.
    """
    clean = " " + " ".join(re.sub(r"[^\w\s]", " ", label.lower()).split()) + " "

    # Do not demote double negatives or explicitly positive qualifications
    if " not unrelated " in clean or " no issue " in clean:
        return False

    # Explicit negative expressions using word boundaries
    negative_patterns = [
        r"\birrelevant\b",
        r"\bnot\s+relevant\b",
        r"\bnot_relevant\b",
        r"\bunrelated\b",
        r"\bnon\s*relevant\b",
        r"^\s*no\s*$",  # Standalone "no" only
    ]
    for pat in negative_patterns:
        if re.search(pat, clean):
            return True

    return False


def _apply_semantic_relevance(
    query: str,
    evidence: list[EvidenceItem],
    provider: Optional[LLMProvider],
    secrets: list[str],
    deadline: Optional[float] = None,
) -> list[EvidenceItem]:
    """Apply LLM-based semantic relevance check as a secondary filter.

    When the LLM is available, evidence that passed the distance threshold is
    verified for actual semantic relevance. Evidence that the LLM labels as
    irrelevant is demoted. Evidence that failed the distance threshold is not
    promoted by the LLM alone (conservative policy).

    When no LLM is available, only the distance filter is used.
    """
    if provider is None:
        return evidence

    # Only check items that passed distance filter
    candidates = [e for e in evidence if e.is_relevant]
    if not candidates:
        return evidence

    try:
        # Build compact relevance-check prompt
        items_text = "\n".join(
            f"[{i}] (ID: {e.evidence_id}) Title: {e.title}\n"
            f"Content preview: {e.content[:200]}..."
            for i, e in enumerate(candidates)
        )

        prompt = (
            "You are evaluating whether retrieved evidence is relevant to a research query.\n\n"
            f"Query: {query}\n\n"
            f"Evidence items:\n{items_text}\n\n"
            "For each item, respond with its index number and 'relevant' or 'irrelevant'.\n"
            "Format: one item per line, e.g., '0: relevant'\n"
            "Only mark items as relevant if they directly address the query topic."
        )

        try:
            response = provider.invoke(prompt, deadline=deadline)
        except TypeError:
            response = provider.invoke(prompt)

        # Parse the relevance response
        for line in response.strip().splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            parts = line.split(":", 1)
            try:
                clean_idx = parts[0].strip().strip("*[]#-").strip()
                idx = int(clean_idx)
                label = parts[1].strip()
                if 0 <= idx < len(candidates):
                    if _is_explicitly_negative(label):
                        candidates[idx].is_relevant = False
                        logger.info(
                            "Semantic check demoted evidence %s as irrelevant to query.",
                            candidates[idx].evidence_id,
                        )
            except (ValueError, IndexError):
                continue

    except (ProviderError, MalformedOutputError) as err:
        safe_msg = redact_secrets(str(err), secrets)
        logger.warning(
            "Semantic relevance check failed, falling back to distance-only: %s",
            safe_msg,
        )
        # Fall back to distance-only filter without failing the workflow

    return evidence


def researcher_node(
    state: ResearchState,
    *,
    store: Optional[VectorStore] = None,
    settings: Optional[Settings] = None,
    provider: Optional[LLMProvider] = None,
) -> dict[str, Any]:
    """LangGraph node: retrieves and filters corpus evidence for the query.

    Enforces hard cycle bounds (max 2 attempts) and deduplicates evidence
    by stable corpus chunk ID.
    """
    raw_cycle = state.get("research_cycle_count", 0)

    # Validate counter integrity: must be an integer, non-negative, and < 2
    if not isinstance(raw_cycle, int) or raw_cycle < 0:
        logger.error("Malformed internal counter state in research_cycle_count: %r", raw_cycle)
        return {
            "error_message": f"Malformed internal research_cycle_count: {raw_cycle!r}",
            "outcome": WorkflowOutcome.MALFORMED_OUTPUT.value,
            "route_decision": "fallback",
        }
    if raw_cycle >= 2:
        logger.error("Researcher cycle limit exceeded (%d >= 2). Halting execution.", raw_cycle)
        return {
            "error_message": "Maximum research cycle limit reached.",
            "outcome": WorkflowOutcome.ITERATION_LIMIT.value,
            "route_decision": "fallback",
        }

    # Increment exactly once at Researcher entry
    cycle_count = raw_cycle + 1

    cfg = settings or Settings()
    query = state.get("query", "").strip()

    # Retrieve existing state store for deduplication
    existing_store = dict(state.get("evidence_store", {}))
    existing_relevant_ids = list(state.get("relevant_chunk_ids", []))

    deadline = state.get("deadline")
    if deadline is not None and time.time() >= deadline:
        logger.warning("Workflow execution deadline exceeded at researcher entry.")
        return {
            "research_cycle_count": cycle_count,
            "iteration_count": cycle_count,
            "evidence_store": existing_store,
            "relevant_chunk_ids": existing_relevant_ids,
            "new_chunk_ids": [],
            "new_chunk_count": 0,
            "evidence": list(existing_store.values()),
            "relevant_evidence": [existing_store[cid] for cid in existing_relevant_ids if cid in existing_store],
            "error_message": "Overall workflow execution deadline exceeded.",
            "route_decision": "fallback",
            "outcome": WorkflowOutcome.INSUFFICIENT_EVIDENCE.value,
        }

    if not query:
        return {
            "research_cycle_count": cycle_count,
            "iteration_count": cycle_count,
            "evidence_store": existing_store,
            "relevant_chunk_ids": existing_relevant_ids,
            "new_chunk_ids": [],
            "new_chunk_count": 0,
            "evidence": list(existing_store.values()),
            "relevant_evidence": [existing_store[cid] for cid in existing_relevant_ids if cid in existing_store],
            "error_message": "Empty or invalid query.",
            "outcome": WorkflowOutcome.IRRELEVANT_QUERY.value,
        }

    # Collect secrets for redaction
    secrets = [s for s in [cfg.api_key, cfg.langsmith_api_key] if s]

    vs = store or VectorStore(settings=cfg)

    try:
        results = vs.search(query=query, top_k=cfg.retrieval_top_k)
    except IndexNotInitializedError as err:
        logger.error("Index not initialized: %s", err)
        return {
            "research_cycle_count": cycle_count,
            "iteration_count": cycle_count,
            "evidence_store": existing_store,
            "relevant_chunk_ids": existing_relevant_ids,
            "new_chunk_ids": [],
            "new_chunk_count": 0,
            "evidence": list(existing_store.values()),
            "relevant_evidence": [existing_store[cid] for cid in existing_relevant_ids if cid in existing_store],
            "error_message": "Vector index is not initialized. Run 'asterline index' first.",
            "outcome": WorkflowOutcome.INSUFFICIENT_EVIDENCE.value,
        }
    except VectorStoreError as err:
        safe_msg = redact_secrets(str(err), secrets)
        logger.error("Vector store error: %s", safe_msg)
        return {
            "research_cycle_count": cycle_count,
            "iteration_count": cycle_count,
            "evidence_store": existing_store,
            "relevant_chunk_ids": existing_relevant_ids,
            "new_chunk_ids": [],
            "new_chunk_count": 0,
            "evidence": list(existing_store.values()),
            "relevant_evidence": [existing_store[cid] for cid in existing_relevant_ids if cid in existing_store],
            "error_message": f"Retrieval error: {safe_msg}",
            "outcome": WorkflowOutcome.PROVIDER_ERROR.value,
        }

    if not results:
        logger.info("No retrieval results for query: %s", query[:100])
        return {
            "research_cycle_count": cycle_count,
            "iteration_count": cycle_count,
            "evidence_store": existing_store,
            "relevant_chunk_ids": existing_relevant_ids,
            "new_chunk_ids": [],
            "new_chunk_count": 0,
            "evidence": list(existing_store.values()),
            "relevant_evidence": [existing_store[cid] for cid in existing_relevant_ids if cid in existing_store],
            "error_message": "No evidence found for this query.",
            "outcome": WorkflowOutcome.INSUFFICIENT_EVIDENCE.value,
        }

    # Convert to EvidenceItems
    evidence_items = [_search_result_to_evidence(r) for r in results]

    # Apply distance-threshold filter
    threshold = getattr(cfg, "max_relevance_distance", 0.70)
    evidence_items = _apply_distance_filter(evidence_items, threshold)

    # Apply semantic relevance check (if provider available)
    evidence_items = _apply_semantic_relevance(
        query, evidence_items, provider, secrets, deadline=deadline
    )

    # Track newly discovered chunks without duplicates
    new_chunk_ids = []
    for item in evidence_items:
        cid = item.chunk_id
        if cid not in existing_store:
            existing_store[cid] = item.model_dump()
            new_chunk_ids.append(cid)
        if item.is_relevant and cid not in existing_relevant_ids:
            existing_relevant_ids.append(cid)

    new_chunk_count = len(new_chunk_ids)

    # Convenience lists for downstream nodes
    all_evidence_dicts = list(existing_store.values())
    relevant_dicts = [existing_store[cid] for cid in existing_relevant_ids if cid in existing_store]

    if not relevant_dicts:
        logger.info(
            "Retrieved %d results but none passed relevance filtering (threshold=%.2f).",
            len(evidence_items),
            threshold,
        )
        return {
            "research_cycle_count": cycle_count,
            "iteration_count": cycle_count,
            "evidence_store": existing_store,
            "relevant_chunk_ids": existing_relevant_ids,
            "new_chunk_ids": new_chunk_ids,
            "new_chunk_count": new_chunk_count,
            "evidence": all_evidence_dicts,
            "relevant_evidence": [],
            "error_message": (
                f"Retrieved {len(evidence_items)} results but none met the relevance "
                f"threshold ({threshold}). The query may not be answerable from the corpus."
            ),
            "outcome": WorkflowOutcome.INSUFFICIENT_EVIDENCE.value,
        }

    logger.info(
        "Retrieved %d results (%d new), %d total pass relevance filtering.",
        len(evidence_items),
        new_chunk_count,
        len(relevant_dicts),
    )

    return {
        "research_cycle_count": cycle_count,
        "iteration_count": cycle_count,
        "evidence_store": existing_store,
        "relevant_chunk_ids": existing_relevant_ids,
        "new_chunk_ids": new_chunk_ids,
        "new_chunk_count": new_chunk_count,
        "evidence": all_evidence_dicts,
        "relevant_evidence": relevant_dicts,
    }
