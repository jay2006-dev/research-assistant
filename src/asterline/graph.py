"""Milestone 3 Stage 1: Graph skeleton and factory with explicit dependency injection.

Provides:
- ``create_research_graph`` factory binding settings, vector store, and LLM provider.
- Structured conditional routing enforcing the hard maximum of 2 research cycles.
- Minimal skeleton node interfaces for graph compilation and offline verification.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from asterline.config import Settings
from asterline.coordinator import coordinator_node
from asterline.memory import MemoryStorageError, UserMemoryStore
from asterline.providers import LLMProvider, SafeGeminiProvider, create_provider_from_settings
from asterline.researcher import researcher_node
from asterline.state import (
    ResearchResponse,
    ResearchState,
    VerificationReport,
    WorkflowOutcome,
)
from asterline.store import VectorStore
from asterline.synthesizer import synthesizer_node
from asterline.validator import validator_node
from asterline.verifier import verifier_node

logger = logging.getLogger(__name__)


# ── Skeleton / Minimal Node Interfaces for Stage 1 ───────────────────────────

def coordinator_skeleton(
    state: ResearchState,
    *,
    settings: Optional[Settings] = None,
    provider: Optional[LLMProvider] = None,
) -> dict[str, Any]:
    """Minimal Coordinator skeleton for graph construction."""
    query = state.get("query", "")
    return {
        "research_plan": f"Research plan for: {query}",
        "sub_queries": [query],
    }


def verifier_skeleton(
    state: ResearchState,
    *,
    settings: Optional[Settings] = None,
    provider: Optional[LLMProvider] = None,
) -> dict[str, Any]:
    """Minimal Verifier skeleton for graph construction."""
    claims = state.get("claims", [])
    # In skeleton: verify based on existing claim list
    report = VerificationReport(
        all_citations_valid=bool(claims),
        all_claims_supported=bool(claims),
        has_unsupported_content=False,
        recommendation="accept" if claims else "reject",
    )
    return {
        "verification_report": report.model_dump(),
        "conflicts": [],
    }


def validator_skeleton(
    state: ResearchState,
) -> dict[str, Any]:
    """Minimal Validator skeleton for graph construction."""
    claims = state.get("claims", [])
    raw_cycles = state.get("research_cycle_count", 1)
    cycles = raw_cycles if isinstance(raw_cycles, int) and raw_cycles >= 0 else 0
    outcome_str = state.get("outcome", WorkflowOutcome.SUCCESS.value)
    try:
        outcome = WorkflowOutcome(outcome_str)
    except ValueError:
        outcome = WorkflowOutcome.SUCCESS

    resp = ResearchResponse(
        answer=f"Verified answer derived from {len(claims)} claim(s).",
        claims=[],
        citations=[],
        conflicts=[],
        outcome=outcome,
        cycles_used=cycles,
        iterations_used=cycles,
    )
    return {"response": resp.model_dump()}


def fallback_skeleton(
    state: ResearchState,
) -> dict[str, Any]:
    """Minimal Fallback skeleton for graph construction."""
    raw_cycles = state.get("research_cycle_count", 1)
    cycles = raw_cycles if isinstance(raw_cycles, int) and raw_cycles >= 0 else 0
    err = state.get("error_message", "Evidence insufficient or cycle limit reached.")
    outcome_str = state.get("outcome", WorkflowOutcome.INSUFFICIENT_EVIDENCE.value)
    try:
        outcome = WorkflowOutcome(outcome_str)
    except ValueError:
        outcome = WorkflowOutcome.INSUFFICIENT_EVIDENCE

    resp = ResearchResponse(
        answer=f"Safe incomplete response: {err}",
        claims=[],
        citations=[],
        conflicts=[],
        outcome=outcome,
        cycles_used=cycles,
        iterations_used=cycles,
    )
    return {"response": resp.model_dump()}


# ── Routing Invariant ────────────────────────────────────────────────────────

def route_after_research(state: ResearchState) -> str:
    """Conditional router after the researcher node.

    Routes to ``fallback`` when the researcher signals early exit (malformed
    counter, exhausted cycles, or any explicit ``route_decision="fallback"``),
    or when the workflow deadline has been exceeded.
    Otherwise continues to ``synthesizer``.
    """
    deadline = state.get("deadline")
    if deadline is not None and time.time() >= deadline:
        logger.info("Workflow deadline exceeded after research. Routing to fallback.")
        return "fallback"
    if state.get("route_decision") == "fallback":
        return "fallback"
    return "synthesizer"


def route_after_verification(state: ResearchState) -> str:
    """Conditional router enforcing hard cycle ceiling of 2 and deadline bounds.

    Guarantees that no graph path can execute the researcher node more than twice.
    """
    deadline = state.get("deadline")
    if deadline is not None and time.time() >= deadline:
        logger.info("Workflow deadline exceeded after verification. Routing to fallback.")
        return "fallback"
    cycles = state.get("research_cycle_count", 0)
    max_cycles = state.get("max_research_cycles", 2)
    new_chunks = state.get("new_chunk_count", 0)
    report_dict = state.get("verification_report", {})
    report = VerificationReport(**report_dict) if report_dict else None

    # Invariant: Hard termination when cycle limit is reached or exceeded
    if cycles >= max_cycles:
        logger.info("Cycle ceiling reached (%d >= %d). Routing to fallback.", cycles, max_cycles)
        return "fallback"

    # Termination: Accepted claims with all citations and support verified
    if report and report.all_citations_valid and report.all_claims_supported:
        return "validator"

    # Invariant: Zero new chunks on a subsequent cycle halts immediately
    if cycles > 1 and new_chunks == 0:
        logger.info("Zero new evidence chunks discovered on cycle %d. Routing to fallback.", cycles)
        return "fallback"

    # Otherwise: Permit one bounded replan cycle
    return "coordinator"


def sanitize_terminal_state_node(state: ResearchState) -> dict[str, Any]:
    """Sanitizes sensitive fields from terminal state outputs.

    Wipes raw `user_memories` from graph state before final completion, ensuring
    private user memory content is never retained in terminal run outputs, checkpoints,
    or LangSmith trace payloads. Preserves `memory_status` ('loaded'/'empty'/'unavailable')
    and user-agnostic execution metadata.
    """
    return {
        "user_memories": [],
    }


# ── Graph Factory ────────────────────────────────────────────────────────────

def memory_loader_node(
    state: ResearchState,
    *,
    memory_store: Optional[UserMemoryStore] = None,
) -> dict[str, Any]:
    """LangGraph node: loads active user memory for personalization if available.

    Invariants:
    - Scoped strictly to state.get("user_id", "default").
    - If user_memories is already populated in state (e.g. from caller), preserves it.
    - If memory_store is None or unavailable, fails safely with memory_status="empty" or "unavailable".
    - Storage failures do not crash the workflow; corporate research proceeds normally.
    """
    user_id = state.get("user_id") or "default"
    existing_memories = state.get("user_memories")
    if existing_memories is not None:
        status = state.get("memory_status") or ("loaded" if existing_memories else "empty")
        return {"user_id": user_id, "user_memories": existing_memories, "memory_status": status}

    if memory_store is None:
        return {"user_id": user_id, "user_memories": [], "memory_status": "empty"}

    try:
        active_items = memory_store.recall(user_id=user_id)
        serialized = [m.model_dump() for m in active_items]
        status = "loaded" if serialized else "empty"
        return {"user_id": user_id, "user_memories": serialized, "memory_status": status}
    except Exception as err:
        logger.warning("User memory recall failed; proceeding with corporate research: %s", type(err).__name__)
        return {"user_id": user_id, "user_memories": [], "memory_status": "unavailable"}


def create_research_graph(
    settings: Optional[Settings] = None,
    store: Optional[VectorStore] = None,
    provider: Optional[LLMProvider] = None,
    memory_store: Optional[UserMemoryStore] = None,
) -> Any:
    """Factory creating the LangGraph research workflow with explicit dependency injection.

    Parameters
    ----------
    settings : Settings, optional
        Application configuration. Loaded from environment if None.
    store : VectorStore, optional
        Vector retrieval store instance.
    provider : LLMProvider, optional
        LLM provider. If None, SafeGeminiProvider.from_settings(settings) is used.
    memory_store : UserMemoryStore, optional
        Persistent user memory store for read-only user context personalization.

    Returns
    -------
    CompiledStateGraph
        The compiled LangGraph execution graph.
    """
    cfg = settings or Settings(_env_file=None)

    # Explicit dependency injection rule:
    # When no provider is injected, use create_provider_from_settings(cfg)
    if provider is None:
        active_provider = create_provider_from_settings(cfg)
    else:
        active_provider = provider

    # Bind dependencies to nodes explicitly via functools.partial
    bound_memory_loader = functools.partial(memory_loader_node, memory_store=memory_store)
    bound_coordinator = functools.partial(coordinator_node, settings=cfg, provider=active_provider)
    bound_researcher = functools.partial(researcher_node, settings=cfg, store=store, provider=active_provider)
    bound_synthesizer = functools.partial(
        synthesizer_node,
        provider=active_provider,
        secrets=[s for s in [cfg.api_key, cfg.groq_api_key, cfg.langsmith_api_key] if s],
    )
    bound_verifier = functools.partial(verifier_node, settings=cfg, provider=active_provider)

    builder = StateGraph(ResearchState)

    # Register nodes
    builder.add_node("memory_loader", bound_memory_loader)
    builder.add_node("coordinator", bound_coordinator)
    builder.add_node("researcher", bound_researcher)
    builder.add_node("synthesizer", bound_synthesizer)
    builder.add_node("verifier", bound_verifier)
    builder.add_node("validator", validator_node)
    builder.add_node("fallback", fallback_skeleton)
    builder.add_node("sanitize_terminal", sanitize_terminal_state_node)

    # Register deterministic edges
    builder.add_edge(START, "memory_loader")
    builder.add_edge("memory_loader", "coordinator")
    builder.add_edge("coordinator", "researcher")
    builder.add_conditional_edges(
        "researcher",
        route_after_research,
        {
            "synthesizer": "synthesizer",
            "fallback": "fallback",
        },
    )
    builder.add_edge("synthesizer", "verifier")

    # Register conditional routing edge
    builder.add_conditional_edges(
        "verifier",
        route_after_verification,
        {
            "validator": "validator",
            "fallback": "fallback",
            "coordinator": "coordinator",
        },
    )

    # Terminal sanitation edges: both validator and fallback pass through sanitizer before END
    builder.add_edge("validator", "sanitize_terminal")
    builder.add_edge("fallback", "sanitize_terminal")
    builder.add_edge("sanitize_terminal", END)

    return builder.compile()
