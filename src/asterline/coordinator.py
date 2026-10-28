"""Coordinator node: formulates research plans and generates/refines search sub-queries.

Coordinates the research workflow by:
- Analyzing user queries to produce initial research plans and retrieval sub-queries.
- During replanning cycles, identifying evidence gaps and unsupported claims from the
  verification report to formulate targeted follow-up queries.
- Preserving stable evidence IDs, existing retrieved evidence, and counter state.
- Enforcing that personal memory or model hallucinations are never treated as corporate evidence.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from asterline.config import Settings
from asterline.providers import (
    LLMProvider,
    MalformedOutputError,
    ProviderError,
    redact_secrets,
    suppress_tracing_boundary,
)
from asterline.state import ResearchState, VerificationReport, WorkflowOutcome

logger = logging.getLogger(__name__)


def _build_initial_plan_prompt(query: str, memories: list[dict[str, Any]] | None = None) -> str:
    """Prompt for decomposing a user query into a research plan and sub-queries.

    Includes user preferences/constraints as contextual clarification when present,
    while explicitly instructing that memory cannot establish corporate facts.
    """
    memory_section = ""
    if memories:
        bullets = []
        for m in memories:
            cat = m.get("category", "preference")
            k = m.get("key", "")
            cnt = m.get("content", "")
            if cnt:
                bullets.append(f"- [{cat}] {k}: {cnt}")
        if bullets:
            memory_section = (
                "\nUser Preferences & Constraints (context only; NEVER cite as corporate facts):\n"
                + "\n".join(bullets)
                + "\n"
            )

    return (
        "You are a research coordinator for an enterprise research assistant.\n"
        "Analyze the following query and produce a research plan with 1 to 3 search sub-queries "
        "designed to retrieve factual corporate evidence.\n\n"
        f"Query: {query}\n"
        f"{memory_section}\n"
        "Instructions:\n"
        "1. Focus search sub-queries strictly on retrieving evidence from the corporate corpus.\n"
        "2. User preferences may guide focus (e.g., plan tier or region), but user memory is NOT evidence.\n"
        "3. Never cite personal memory as factual truth about Asterline products or policies.\n\n"
        "Respond in valid JSON format only, with the following structure:\n"
        "{\n"
        '  "research_plan": "Brief description of the research approach",\n'
        '  "sub_queries": ["sub query 1", "sub query 2"]\n'
        "}"
    )


def _build_replan_prompt(
    original_query: str,
    unsupported_claims: list[str],
    evidence_titles: list[str],
) -> str:
    """Prompt for formulating refined queries targeting unaddressed evidence gaps."""
    unsupported_text = "\n".join(f"- {c}" for c in unsupported_claims) or "None specified."
    titles_text = "\n".join(f"- {t}" for t in evidence_titles) or "No evidence retrieved yet."
    return (
        "You are a research coordinator refining an enterprise research query.\n"
        "A previous research cycle failed to fully support all claims with corporate evidence.\n\n"
        f"Original Query: {original_query}\n\n"
        f"Evidence already retrieved:\n{titles_text}\n\n"
        f"Claims lacking sufficient verification or evidence:\n{unsupported_text}\n\n"
        "Formulate a refined research plan and 1 to 3 targeted search sub-queries to fill these gaps.\n"
        "Do not invent facts or cite external memory; focus strictly on corporate corpus retrieval.\n\n"
        "Respond in valid JSON format only:\n"
        "{\n"
        '  "research_plan": "Description of the refined research strategy",\n'
        '  "sub_queries": ["targeted query 1"]\n'
        "}"
    )


def coordinator_node(
    state: ResearchState,
    *,
    settings: Optional[Settings] = None,
    provider: Optional[LLMProvider] = None,
) -> dict[str, Any]:
    """LangGraph node: formulates initial research plan or refines queries upon replan.

    Parameters
    ----------
    state : ResearchState
        Current workflow state.
    settings : Settings, optional
        Application settings.
    provider : LLMProvider, optional
        LLM provider for query decomposition and refinement.

    Returns
    -------
    dict[str, Any]
        State updates containing research_plan, sub_queries, and optionally updated query.
    """
    query = state.get("query", "").strip()
    if not query:
        logger.warning("Empty query provided to coordinator.")
        return {
            "sub_queries": [],
            "research_plan": "Aborted: empty query provided.",
            "error_message": "Query cannot be empty.",
            "outcome": WorkflowOutcome.IRRELEVANT_QUERY.value,
        }

    cfg = settings or Settings(_env_file=None)
    secrets = [s for s in [cfg.api_key, cfg.langsmith_api_key] if s]

    # Cycle awareness: cycle count must NOT be reset or incremented by coordinator
    cycle_count = state.get("research_cycle_count", 0)
    is_replan = cycle_count > 0

    # Collect gaps from existing verification report if replanning
    unsupported_claims: list[str] = []
    if is_replan:
        report_dict = state.get("verification_report", {})
        if report_dict:
            try:
                report = VerificationReport(**report_dict)
                for res in report.claim_results:
                    if res.status.value != "supported":
                        unsupported_claims.append(res.claim_text)
            except Exception:
                pass

        # Also inspect raw claims if report had no specifics
        if not unsupported_claims:
            for c in state.get("claims", []):
                if c.get("verification_status") != "supported":
                    unsupported_claims.append(c.get("claim_text", ""))

    evidence_store = state.get("evidence_store", {})
    evidence_titles = [
        item.get("title", "") for item in evidence_store.values() if isinstance(item, dict)
    ]

    deadline = state.get("deadline")
    if deadline is not None and time.time() >= deadline:
        logger.warning("Workflow execution deadline exceeded at coordinator entry.")
        return {
            "sub_queries": [],
            "research_plan": "Aborted: deadline exceeded.",
            "error_message": "Overall workflow execution deadline exceeded.",
            "route_decision": "fallback",
            "outcome": WorkflowOutcome.INSUFFICIENT_EVIDENCE.value,
        }

    # Attempt LLM-assisted query formulation if provider is available
    memories = state.get("user_memories", [])
    memory_secrets = [
        str(m.get("content", "")).strip()
        for m in memories
        if m.get("content") and len(str(m.get("content", "")).strip()) > 3
    ]
    all_secrets = secrets + memory_secrets

    if provider is not None:
        try:
            if is_replan:
                prompt = _build_replan_prompt(query, unsupported_claims, evidence_titles)
            else:
                prompt = _build_initial_plan_prompt(query, memories=memories)

            # When user memories are present in the prompt, isolate the LLM call
            # within a safe tracing boundary so private memory is never emitted to LangSmith child runs.
            if memories:
                with suppress_tracing_boundary():
                    try:
                        response_text = provider.invoke(prompt, deadline=deadline)
                    except TypeError:
                        response_text = provider.invoke(prompt)
            else:
                try:
                    response_text = provider.invoke(prompt, deadline=deadline)
                except TypeError:
                    response_text = provider.invoke(prompt)

            # Parse JSON response
            from asterline.providers import _extract_json

            extracted = _extract_json(response_text)
            data = json.loads(extracted)
            plan = data.get("research_plan", "").strip()
            sub_queries = [
                q.strip()
                for q in data.get("sub_queries", [])
                if isinstance(q, str) and q.strip()
            ]
            if plan and sub_queries:
                return {
                    "research_plan": plan,
                    "sub_queries": sub_queries,
                    # When replanning, align primary query to the most specific sub-query
                    "query": sub_queries[0] if is_replan else query,
                    "user_memories": [],  # Cleanse raw memory from downstream state immediately
                }
        except (ProviderError, MalformedOutputError, json.JSONDecodeError, Exception) as err:
            safe_msg = redact_secrets(str(err), all_secrets)
            logger.warning("Provider query formulation failed, falling back to deterministic plan: %s", safe_msg)

    # Deterministic fallback when no provider is present or invocation fails
    if is_replan:
        gap_suffix = f" (focusing on: {unsupported_claims[0]})" if unsupported_claims else ""
        plan = f"Refined research plan (Cycle {cycle_count + 1}): target evidence gaps for '{query}'{gap_suffix}."
        refined_sub = [f"{query} details"] if not unsupported_claims else [f"{query} {unsupported_claims[0]}"]
        return {
            "research_plan": plan,
            "sub_queries": refined_sub,
            "query": refined_sub[0],
            "user_memories": [],
        }

    return {
        "research_plan": f"Initial research plan: retrieve corporate evidence for '{query}'.",
        "sub_queries": [query],
        "user_memories": [],
    }
