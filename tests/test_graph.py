"""Milestone 3 Stage 1: Tests for graph skeleton, factory, and cycle routing invariants."""

import functools
from unittest.mock import MagicMock, patch

from langgraph.graph import END, START, StateGraph

from asterline.config import Settings
from asterline.graph import (
    coordinator_skeleton,
    create_research_graph,
    fallback_skeleton,
    route_after_research,
    route_after_verification,
    validator_skeleton,
    verifier_skeleton,
)
from asterline.providers import FakeLLMProvider
from asterline.researcher import researcher_node
from asterline.state import ResearchState, VerificationReport, WorkflowOutcome
from asterline.store import SearchResult
from asterline.synthesizer import synthesizer_node


def test_graph_compilation_with_injected_dependencies():
    """Graph compiles cleanly with explicitly injected store and provider."""
    cfg = Settings(_env_file=None)
    mock_store = MagicMock()
    fake_provider = FakeLLMProvider(default_text="test response")

    graph = create_research_graph(settings=cfg, store=mock_store, provider=fake_provider)
    assert graph is not None
    # Verify node names in the compiled graph
    assert "coordinator" in graph.nodes
    assert "researcher" in graph.nodes
    assert "synthesizer" in graph.nodes
    assert "verifier" in graph.nodes
    assert "validator" in graph.nodes
    assert "fallback" in graph.nodes


def test_graph_factory_uses_safe_gemini_from_settings_by_default():
    """When provider=None, graph factory explicitly invokes SafeGeminiProvider.from_settings."""
    cfg = Settings(_env_file=None, api_key="test-gemini-key", langsmith_api_key="test-ls-key")
    mock_store = MagicMock()

    with patch("asterline.graph.SafeGeminiProvider.from_settings") as mock_from_settings:
        mock_from_settings.return_value = FakeLLMProvider()
        graph = create_research_graph(settings=cfg, store=mock_store, provider=None)
        assert graph is not None
        mock_from_settings.assert_called_once_with(cfg)


def test_routing_invariant_hard_ceiling_at_two_cycles():
    """route_after_verification strictly enforces the hard maximum of 2 research cycles."""
    # Cycle 1: Supported claims route to validator
    valid_report = VerificationReport(
        all_citations_valid=True,
        all_claims_supported=True,
        has_unsupported_content=False,
    )
    state_cycle1_valid: ResearchState = {
        "research_cycle_count": 1,
        "max_research_cycles": 2,
        "new_chunk_count": 3,
        "verification_report": valid_report.model_dump(),
    }
    assert route_after_verification(state_cycle1_valid) == "validator"

    # Cycle 1: Unsupported claims route back to coordinator for replanning
    invalid_report = VerificationReport(
        all_citations_valid=False,
        all_claims_supported=False,
        has_unsupported_content=True,
    )
    state_cycle1_invalid: ResearchState = {
        "research_cycle_count": 1,
        "max_research_cycles": 2,
        "new_chunk_count": 3,
        "verification_report": invalid_report.model_dump(),
    }
    assert route_after_verification(state_cycle1_invalid) == "coordinator"

    # Cycle 2: Unsupported claims MUST halt to fallback; cannot replan
    state_cycle2_invalid: ResearchState = {
        "research_cycle_count": 2,
        "max_research_cycles": 2,
        "new_chunk_count": 2,
        "verification_report": invalid_report.model_dump(),
    }
    assert route_after_verification(state_cycle2_invalid) == "fallback"

    # Cycle 2: Zero new chunks halts immediately to fallback
    state_cycle2_zero_new: ResearchState = {
        "research_cycle_count": 2,
        "max_research_cycles": 2,
        "new_chunk_count": 0,
        "verification_report": invalid_report.model_dump(),
    }
    assert route_after_verification(state_cycle2_zero_new) == "fallback"

    # Exceeded cycles (> 2) unconditionally route to fallback
    state_cycle3: ResearchState = {
        "research_cycle_count": 3,
        "max_research_cycles": 2,
        "verification_report": invalid_report.model_dump(),
    }
    assert route_after_verification(state_cycle3) == "fallback"


def test_researcher_counter_guards_reject_corrupted_and_exceeded_state():
    """researcher_node rejects corrupted counter types and blocks execution when cycle count >= 2."""
    from asterline.researcher import researcher_node

    mock_store = MagicMock()

    # Corrupted: string counter
    corrupt_str_state: ResearchState = {"query": "test", "research_cycle_count": "one"}  # type: ignore[typeddict-item]
    result_str = researcher_node(corrupt_str_state, store=mock_store)
    assert result_str["outcome"] == WorkflowOutcome.MALFORMED_OUTPUT.value
    assert mock_store.search.call_count == 0

    # Corrupted: negative counter
    corrupt_neg_state: ResearchState = {"query": "test", "research_cycle_count": -1}
    result_neg = researcher_node(corrupt_neg_state, store=mock_store)
    assert result_neg["outcome"] == WorkflowOutcome.MALFORMED_OUTPUT.value
    assert mock_store.search.call_count == 0

    # Exceeded: cycle count 2 blocks third attempt
    exceeded_state: ResearchState = {"query": "test", "research_cycle_count": 2}
    result_exceeded = researcher_node(exceeded_state, store=mock_store)
    assert result_exceeded["outcome"] == WorkflowOutcome.ITERATION_LIMIT.value
    assert result_exceeded["route_decision"] == "fallback"
    assert mock_store.search.call_count == 0


# ── Helper: build a test graph with an injectable verifier ────────────────────

def _make_search_result(chunk_id: str, distance: float = 0.30, content: str | None = None) -> SearchResult:
    """Create a deterministic SearchResult for test graphs."""
    chunk_content = content or (
        f"Test evidence content for chunk {chunk_id}. "
        f"This is deterministic test content used for offline verification."
    )
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        title=f"Title for {chunk_id}",
        category="test",
        owner="test-owner",
        source_url="https://example.com",
        published="2025-01-01",
        version=1,
        content=chunk_content,
        distance=distance,
    )


def _rejecting_verifier(state: ResearchState, **_kwargs) -> dict:
    """Verifier that always recommends replanning (forces a second cycle)."""
    report = VerificationReport(
        all_citations_valid=False,
        all_claims_supported=False,
        has_unsupported_content=True,
        recommendation="replan",
    )
    return {
        "verification_report": report.model_dump(),
        "conflicts": [],
    }


def _build_test_graph(
    settings: Settings,
    store,
    provider,
    verifier_fn=None,
):
    """Build a graph with the production topology but an injectable verifier.

    Uses the same nodes and edges as ``create_research_graph`` so the test
    exercises the actual routing logic, but allows overriding the verifier
    to force replan cycles.
    """
    cfg = settings
    verifier = verifier_fn or functools.partial(
        verifier_skeleton,
        settings=cfg,
        provider=provider,
    )

    bound_coordinator = functools.partial(coordinator_skeleton, settings=cfg, provider=provider)
    bound_researcher = functools.partial(researcher_node, settings=cfg, store=store, provider=provider)
    bound_synthesizer = functools.partial(
        synthesizer_node, provider=None, secrets=[],
    )

    builder = StateGraph(ResearchState)
    builder.add_node("coordinator", bound_coordinator)
    builder.add_node("researcher", bound_researcher)
    builder.add_node("synthesizer", bound_synthesizer)
    builder.add_node("verifier", verifier)
    builder.add_node("validator", validator_skeleton)
    builder.add_node("fallback", fallback_skeleton)

    builder.add_edge(START, "coordinator")
    builder.add_edge("coordinator", "researcher")
    builder.add_conditional_edges(
        "researcher",
        route_after_research,
        {"synthesizer": "synthesizer", "fallback": "fallback"},
    )
    builder.add_edge("synthesizer", "verifier")
    builder.add_conditional_edges(
        "verifier",
        route_after_verification,
        {"validator": "validator", "fallback": "fallback", "coordinator": "coordinator"},
    )
    builder.add_edge("validator", END)
    builder.add_edge("fallback", END)

    return builder.compile()


# ── Integration tests ────────────────────────────────────────────────────────

def test_route_after_research_routes_fallback_on_decision():
    """route_after_research routes to fallback when route_decision is set."""
    assert route_after_research({"route_decision": "fallback"}) == "fallback"
    assert route_after_research({}) == "synthesizer"
    assert route_after_research({"route_decision": "continue"}) == "synthesizer"


def test_compiled_graph_respects_two_cycle_limit_with_replan():
    """End-to-end: compiled graph executes the researcher at most twice
    when the verifier always requests replanning.

    Proves:
    - store.search is called no more than twice during the entire execution.
    - research_cycle_count reaches exactly 2.
    - The graph terminates at fallback (not validator) because the verifier
      never accepts and the cycle limit prevents a third attempt.
    """
    cfg = Settings(_env_file=None, max_relevance_distance=0.90)
    mock_store = MagicMock()
    # Return distinct results per cycle so new_chunk_count > 0 on cycle 2
    mock_store.search.side_effect = [
        [_make_search_result("chunk-a1"), _make_search_result("chunk-a2")],
        [_make_search_result("chunk-b1"), _make_search_result("chunk-b2")],
    ]
    fake_provider = FakeLLMProvider(default_text="fake response")

    graph = _build_test_graph(
        settings=cfg,
        store=mock_store,
        provider=fake_provider,
        verifier_fn=_rejecting_verifier,
    )

    initial_state: ResearchState = {
        "query": "What is the company research policy?",
        "max_research_cycles": 2,
    }

    result = graph.invoke(initial_state)

    # Core assertions: cycle bound - prove exactly 2 cycles executed searches
    assert mock_store.search.call_count == 2, (
        f"store.search called {mock_store.search.call_count} times; expected exactly 2 cycles"
    )
    assert result["research_cycle_count"] == 2, (
        f"Expected exactly 2 cycles, got {result['research_cycle_count']}"
    )

    # Terminal outcome: must be fallback path (insufficient evidence or iteration limit)
    response = result.get("response", {})
    assert response, "Expected a terminal response from fallback node"
    outcome = response.get("outcome", "")
    assert outcome in (
        WorkflowOutcome.INSUFFICIENT_EVIDENCE.value,
        WorkflowOutcome.ITERATION_LIMIT.value,
    ), f"Unexpected terminal outcome: {outcome}"


def test_compiled_graph_malformed_counter_routes_to_fallback():
    """End-to-end: malformed research_cycle_count routes directly to
    fallback without calling synthesizer, verifier, or store.search.

    Proves the researcher early-exit path is correctly wired through
    route_after_research to the fallback node.
    """
    cfg = Settings(_env_file=None)
    mock_store = MagicMock()
    fake_provider = FakeLLMProvider(default_text="should not be called")

    graph = _build_test_graph(
        settings=cfg,
        store=mock_store,
        provider=fake_provider,
    )

    initial_state: ResearchState = {
        "query": "test query",
        "research_cycle_count": "corrupted",  # type: ignore[typeddict-item]
        "max_research_cycles": 2,
    }

    result = graph.invoke(initial_state)

    # store.search must never be called for corrupted state
    assert mock_store.search.call_count == 0, (
        f"store.search called {mock_store.search.call_count} times with corrupted counter"
    )

    # Must reach fallback and produce a terminal response
    response = result.get("response", {})
    assert response, "Expected a terminal response from fallback node"
    outcome = response.get("outcome", "")
    assert outcome in (
        WorkflowOutcome.MALFORMED_OUTPUT.value,
        WorkflowOutcome.INSUFFICIENT_EVIDENCE.value,
    ), f"Unexpected outcome for corrupted counter: {outcome}"


def test_compiled_graph_exhausted_counter_routes_to_fallback():
    """End-to-end: research_cycle_count >= max_research_cycles routes
    directly to fallback without calling store.search.
    """
    cfg = Settings(_env_file=None)
    mock_store = MagicMock()
    fake_provider = FakeLLMProvider(default_text="should not be called")

    graph = _build_test_graph(
        settings=cfg,
        store=mock_store,
        provider=fake_provider,
    )

    initial_state: ResearchState = {
        "query": "test query",
        "research_cycle_count": 2,  # Already at max
        "max_research_cycles": 2,
    }

    result = graph.invoke(initial_state)

    assert mock_store.search.call_count == 0, (
        f"store.search called {mock_store.search.call_count} times with exhausted counter"
    )

    response = result.get("response", {})
    assert response, "Expected a terminal response from fallback node"
    outcome = response.get("outcome", "")
    assert outcome in (
        WorkflowOutcome.ITERATION_LIMIT.value,
        WorkflowOutcome.INSUFFICIENT_EVIDENCE.value,
    ), f"Unexpected outcome for exhausted counter: {outcome}"



def test_compiled_graph_full_success_lifecycle():
    """End-to-end lifecycle: Coordinator -> Researcher -> Synthesizer -> Verifier -> Validator -> SUCCESS."""
    from asterline.providers import FakeLLMResponse

    cfg = Settings(_env_file=None, max_relevance_distance=0.70)
    mock_store = MagicMock()
    mock_store.search.return_value = [
        _make_search_result("chunk-policy-1", distance=0.25)
    ]

    coordinator_json = (
        '{\n'
        '  "research_plan": "Plan for policy lookup",\n'
        '  "sub_queries": ["chunk-policy-1"]\n'
        '}'
    )
    synthesis_json = (
        '{\n'
        '  "claims": [\n'
        '    {\n'
        '      "claim_text": "Test evidence content for chunk chunk-policy-1.",\n'
        '      "citations": [\n'
        '        {\n'
        '          "evidence_id": "chunk-policy-1",\n'
        '          "chunk_id": "chunk-policy-1",\n'
        '          "snippet": "Test evidence content for chunk chunk-policy-1."\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    verifier_json = (
        '{\n'
        '  "status": "supported",\n'
        '  "note": "Fully supported",\n'
        '  "has_conflict": false\n'
        '}'
    )

    fake_provider = FakeLLMProvider(
        responses=[
            FakeLLMResponse(text=coordinator_json),  # Coordinator
            FakeLLMResponse(text="0: relevant\n"),   # Researcher semantic filter
            FakeLLMResponse(text=synthesis_json),    # Synthesizer
            FakeLLMResponse(text=verifier_json),     # Verifier
        ]
    )

    graph = create_research_graph(settings=cfg, store=mock_store, provider=fake_provider)
    initial_state: ResearchState = {
        "query": "What is the policy for chunk-policy-1?",
        "max_research_cycles": 2,
    }

    result = graph.invoke(initial_state)

    # Vector store was queried exactly once
    assert mock_store.search.call_count == 1
    assert result["research_cycle_count"] == 1

    # Terminal output was produced by Validator
    response = result.get("response", {})
    assert response.get("outcome") == WorkflowOutcome.SUCCESS.value
    assert "chunk-policy-1" in response.get("answer", "")
    assert len(response.get("claims", [])) == 1
    assert len(response.get("citations", [])) == 1


def test_compiled_graph_routes_to_fallback_when_retrieval_empty():
    """End-to-end: Empty retrieval results route to fallback and yield INSUFFICIENT_EVIDENCE."""
    cfg = Settings(_env_file=None)
    mock_store = MagicMock()
    mock_store.search.return_value = []  # No chunks found

    fake_provider = FakeLLMProvider(default_text="{}")

    graph = create_research_graph(settings=cfg, store=mock_store, provider=fake_provider)
    initial_state: ResearchState = {
        "query": "Non-existent subject in corpus",
        "max_research_cycles": 2,
    }

    result = graph.invoke(initial_state)

    # Cycle 1 finds no evidence -> triggers replan -> Cycle 2 searches again -> hits 2-cycle limit -> fallback
    assert mock_store.search.call_count == 2
    assert result["research_cycle_count"] == 2
    response = result.get("response", {})
    assert response.get("outcome") == WorkflowOutcome.INSUFFICIENT_EVIDENCE.value



def test_compiled_graph_rejects_claim_when_semantic_verification_fails():
    """Requirement C6: Compiled graph does not return SUCCESS when semantic verification fails."""
    from asterline.providers import FakeLLMResponse

    cfg = Settings(_env_file=None, max_relevance_distance=0.70)
    mock_store = MagicMock()
    mock_store.search.return_value = [
        _make_search_result("chunk-fail-verify", distance=0.20)
    ]

    coordinator_json = '{"research_plan": "Plan", "sub_queries": ["chunk-fail-verify"]}'
    synthesis_json = (
        '{\n'
        '  "claims": [\n'
        '    {\n'
        '      "claim_text": "Test evidence content for chunk chunk-fail-verify.",\n'
        '      "citations": [\n'
        '        {\n'
        '          "evidence_id": "chunk-fail-verify",\n'
        '          "snippet": "Test evidence content for chunk chunk-fail-verify."\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    # Verifier returns an explicit UNSUPPORTED decision
    verifier_unsupported_json = '{"status": "unsupported", "note": "Failed entailment", "has_conflict": false}'

    fake_provider = FakeLLMProvider(
        responses=[
            FakeLLMResponse(text=coordinator_json),
            FakeLLMResponse(text="0: relevant\n"),
            FakeLLMResponse(text=synthesis_json),
            FakeLLMResponse(text=verifier_unsupported_json),  # Cycle 1 verifier fails
            FakeLLMResponse(text=coordinator_json),
            FakeLLMResponse(text="0: relevant\n"),
            FakeLLMResponse(text=synthesis_json),
            FakeLLMResponse(text=verifier_unsupported_json),  # Cycle 2 verifier fails
        ]
    )

    graph = create_research_graph(settings=cfg, store=mock_store, provider=fake_provider)
    initial_state: ResearchState = {
        "query": "What is the policy for chunk-fail-verify?",
        "max_research_cycles": 2,
    }

    result = graph.invoke(initial_state)

    # Invariant: Must NOT return SUCCESS with unverified claims
    response = result.get("response", {})
    assert response.get("outcome") != WorkflowOutcome.SUCCESS.value
    assert response.get("outcome") in (
        WorkflowOutcome.INSUFFICIENT_EVIDENCE.value,
        WorkflowOutcome.ITERATION_LIMIT.value,
    )
    assert response.get("claims", []) == []


# ── Read-Only User Memory Integration Tests ─────────────────────────────────

def test_graph_user_memory_reaches_coordinator_and_distinguishes_users(tmp_path):
    """1, 2 & 3: Selected user's active memories reach coordinator; user isolation holds; empty status is distinct."""
    from asterline.memory import UserMemoryStore
    from asterline.providers import FakeLLMProvider, FakeLLMResponse

    mem_db = tmp_path / "graph_memory_test.db"
    mem_store = UserMemoryStore(mem_db)
    mem_store.remember("user_alice", "preference", "tier_focus", "Focus on Launch tier")
    mem_store.remember("user_bob", "preference", "tier_focus", "Focus on Enterprise tier")

    mock_store = MagicMock()
    mock_store.search.return_value = [
        _make_search_result("chunk-pricing-1", content="Launch costs $59 per month.")
    ]

    coordinator_calls = []

    class CapturingProvider(FakeLLMProvider):
        def invoke(self, prompt: str, **kwargs) -> str:
            if "research coordinator" in prompt.lower():
                coordinator_calls.append(prompt)
                return '{"research_plan": "Plan with user context", "sub_queries": ["pricing"]}'
            elif "synthesizing" in prompt.lower():
                return (
                    '{"claims": [{"claim_text": "Launch costs $59.", "citations": '
                    '[{"evidence_id": "chunk-pricing-1", "snippet": "Launch costs $59 per month."}]}]}'
                )
            elif "verifier" in prompt.lower() or "fact-checking" in prompt.lower():
                return '{"status": "supported", "note": "ok", "has_conflict": false}'
            return "0: relevant\n"

    cfg = Settings(_env_file=None)
    graph = create_research_graph(settings=cfg, store=mock_store, provider=CapturingProvider(), memory_store=mem_store)

    # Execute for Alice
    res_alice = graph.invoke({"query": "Pricing overview", "user_id": "user_alice", "max_research_cycles": 1})
    assert res_alice["memory_status"] == "loaded"
    # Terminal sanitation node wipes user_memories from terminal state for privacy
    assert res_alice["user_memories"] == []
    # But coordinator received Alice's memories during execution
    assert "Focus on Launch tier" in coordinator_calls[0]
    assert "Enterprise" not in coordinator_calls[0]

    # Execute for Charlie (has no memories)
    coordinator_calls.clear()
    res_charlie = graph.invoke({"query": "Pricing overview", "user_id": "user_charlie", "max_research_cycles": 1})
    assert res_charlie["memory_status"] == "empty"
    assert res_charlie["user_memories"] == []
    assert "User Preferences & Constraints" not in coordinator_calls[0]


def test_graph_memory_store_error_fails_safely_to_corporate_research(tmp_path):
    """4: Memory-store failures set memory_status='unavailable' and corporate research proceeds cleanly."""
    import sqlite3
    from asterline.memory import UserMemoryStore
    from asterline.providers import FakeLLMProvider

    mem_db = tmp_path / "broken_memory_test.db"
    mem_store = UserMemoryStore(mem_db)

    # Monkeypatch recall to raise an error
    def broken_recall(user_id):
        raise sqlite3.OperationalError("Simulated SQLite disk corruption")

    mem_store.recall = broken_recall

    mock_store = MagicMock()
    mock_store.search.return_value = [
        _make_search_result("chunk-1", content="Official corpus evidence content.")
    ]

    coordinator_json = '{"research_plan": "Corporate plan", "sub_queries": ["evidence"]}'
    synthesis_json = (
        '{\n'
        '  "claims": [\n'
        '    {\n'
        '      "claim_text": "Official corpus evidence content.",\n'
        '      "citations": [\n'
        '        {\n'
        '          "evidence_id": "chunk-1",\n'
        '          "chunk_id": "chunk-1",\n'
        '          "snippet": "Official corpus evidence content."\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    verifier_json = '{"status": "supported", "note": "ok", "has_conflict": false}'

    from asterline.providers import FakeLLMResponse
    fake_provider = FakeLLMProvider(
        responses=[
            FakeLLMResponse(text=coordinator_json),
            FakeLLMResponse(text="0: relevant\n"),
            FakeLLMResponse(text=synthesis_json),
            FakeLLMResponse(text=verifier_json),
        ]
    )

    cfg = Settings(_env_file=None, max_relevance_distance=0.70)
    graph = create_research_graph(settings=cfg, store=mock_store, provider=fake_provider, memory_store=mem_store)

    result = graph.invoke({"query": "Evidence query", "user_id": "user_alice", "max_research_cycles": 2})

    # Memory failed safely with "unavailable", but research completed successfully
    assert result["memory_status"] == "unavailable"
    assert result["user_memories"] == []
    response = result.get("response", {})
    assert response.get("outcome") == WorkflowOutcome.SUCCESS.value
    assert len(response.get("claims", [])) == 1


def test_graph_memory_cannot_validate_company_price_without_corpus_evidence(tmp_path):
    """6 & 7: A remembered but unsupported company price/policy does not become an accepted claim."""
    from asterline.memory import UserMemoryStore
    from asterline.providers import FakeLLMProvider

    mem_db = tmp_path / "unsupported_memory_test.db"
    mem_store = UserMemoryStore(mem_db)
    # User erroneously remembers Launch tier costs $10
    mem_store.remember("user_alice", "preference", "launch_price", "User believes Launch costs $10")

    mock_store = MagicMock()
    # Real corpus returns $59
    mock_store.search.return_value = [
        _make_search_result("chunk-price", content="Launch costs $59 per month.")
    ]

    # Synthesizer hallucinates a claim based on the $10 memory rather than corpus
    class HallucinatingProvider(FakeLLMProvider):
        def invoke(self, prompt: str, **kwargs) -> str:
            if "coordinator" in prompt.lower():
                return '{"research_plan": "Plan", "sub_queries": ["launch price"]}'
            elif "synthesizing" in prompt.lower():
                # Synthesizer produces claim "$10" citing memory ID or chunk with mismatched snippet
                return (
                    '{"claims": [{"claim_text": "Launch costs $10 per month.", "citations": '
                    '[{"evidence_id": "mem:launch_price", "snippet": "User believes Launch costs $10"}]}]}'
                )
            elif "verifier" in prompt.lower():
                # Even if LLM verifier was tricked into saying supported
                return '{"status": "supported", "note": "verified", "has_conflict": false}'
            return "0: relevant\n"

    cfg = Settings(_env_file=None)
    graph = create_research_graph(settings=cfg, store=mock_store, provider=HallucinatingProvider(), memory_store=mem_store)

    result = graph.invoke({"query": "How much does Launch cost?", "user_id": "user_alice", "max_research_cycles": 1})

    # Validator must reject mem:launch_price because it is NOT in evidence_store
    response = result.get("response", {})
    assert response.get("outcome") != WorkflowOutcome.SUCCESS.value
    assert "$10" not in response.get("answer", "")
    assert len(response.get("claims", [])) == 0


def test_graph_raw_memory_content_not_leaked_into_validator_output(tmp_path):
    """8: Raw memory content is not leaked into verified findings or claims."""
    from asterline.memory import UserMemoryStore
    from asterline.providers import FakeLLMProvider

    mem_db = tmp_path / "leak_test.db"
    mem_store = UserMemoryStore(mem_db)
    confidential_memory = "CONFIDENTIAL_USER_TOKEN_SECRET_98765"
    mem_store.remember("user_alice", "constraint", "token", confidential_memory)

    mock_store = MagicMock()
    mock_store.search.return_value = [
        _make_search_result("chunk-overview", content="Asterline platform overview.")
    ]

    class SafeProvider(FakeLLMProvider):
        def invoke(self, prompt: str, **kwargs) -> str:
            if "coordinator" in prompt.lower():
                return '{"research_plan": "Plan", "sub_queries": ["overview"]}'
            elif "synthesizing" in prompt.lower():
                return (
                    '{"claims": [{"claim_text": "Asterline platform overview.", "citations": '
                    '[{"evidence_id": "chunk-overview", "snippet": "Asterline platform overview."}]}]}'
                )
            elif "verifier" in prompt.lower():
                return '{"status": "supported", "note": "ok", "has_conflict": false}'
            return "0: relevant\n"

    cfg = Settings(_env_file=None)
    graph = create_research_graph(settings=cfg, store=mock_store, provider=SafeProvider(), memory_store=mem_store)

    result = graph.invoke({"query": "Platform overview", "user_id": "user_alice", "max_research_cycles": 1})

    response_text = result.get("response", {}).get("answer", "")
    # The confidential memory must not be in the final corporate response
    assert confidential_memory not in response_text


def test_graph_memory_privacy_and_langsmith_trace_sanitation(tmp_path):
    """Stage 7.4.5: Synthetic secret placed in user memory never appears in captured LangSmith trace payloads or state output."""
    from langchain_core.tracers.context import collect_runs
    from asterline.memory import UserMemoryStore
    from asterline.providers import FakeLLMProvider, FakeLLMResponse

    mem_db = tmp_path / "trace_sanitation_test.db"
    mem_store = UserMemoryStore(mem_db)
    synthetic_secret = "SYNTHETIC_PRIVATE_USER_SECRET_TOKEN_42"
    mem_store.remember("user_alice", "constraint", "auth_token", synthetic_secret)

    mock_store = MagicMock()
    mock_store.search.return_value = [
        _make_search_result("chunk-c1", content="Corpus overview content.")
    ]

    coordinator_json = '{"research_plan": "Plan", "sub_queries": ["overview"]}'
    synthesis_json = (
        '{\n'
        '  "claims": [\n'
        '    {\n'
        '      "claim_text": "Corpus overview content.",\n'
        '      "citations": [\n'
        '        {\n'
        '          "evidence_id": "chunk-c1",\n'
        '          "chunk_id": "chunk-c1",\n'
        '          "snippet": "Corpus overview content."\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    verifier_json = '{"status": "supported", "note": "ok", "has_conflict": false}'

    provider = FakeLLMProvider(
        responses=[
            FakeLLMResponse(text=coordinator_json),
            FakeLLMResponse(text="0: relevant\n"),
            FakeLLMResponse(text=synthesis_json),
            FakeLLMResponse(text=verifier_json),
        ]
    )

    cfg = Settings(_env_file=None, max_relevance_distance=0.70)
    graph = create_research_graph(settings=cfg, store=mock_store, provider=provider, memory_store=mem_store)

    with collect_runs() as cb:
        result = graph.invoke({
            "query": "Corporate overview query",
            "user_id": "user_alice",
            "max_research_cycles": 2,
        })
        runs = cb.traced_runs

    # 1. Terminal state has user_memories wiped cleanly
    assert result.get("user_memories") == []
    assert result.get("memory_status") == "loaded"

    # 2. Captured trace payloads do not retain raw user_memories in graph outputs
    root_run = runs[0]
    assert root_run.outputs.get("user_memories") == []
    assert synthetic_secret not in str(root_run.outputs)

    # 3. Overall response output has zero trace of the synthetic secret
    assert synthetic_secret not in str(result.get("response", {}))


def test_graph_user_isolation_bob_never_receives_alice_memory(tmp_path):
    """Stage 7.4.5: Alice's memory never enters Bob's Coordinator prompt, state, or output."""
    from asterline.memory import UserMemoryStore
    from asterline.providers import FakeLLMProvider

    mem_db = tmp_path / "isolation_test.db"
    mem_store = UserMemoryStore(mem_db)
    alice_secret = "ALICE_CONFIDENTIAL_ROLE_PREFERENCE_888"
    bob_secret = "BOB_PUBLIC_ROLE_PREFERENCE_111"

    mem_store.remember("user_alice", "preference", "role", alice_secret)
    mem_store.remember("user_bob", "preference", "role", bob_secret)

    coordinator_prompts = []

    class PromptCapturingProvider(FakeLLMProvider):
        def invoke(self, prompt: str, **kwargs) -> str:
            if "coordinator" in prompt.lower():
                coordinator_prompts.append(prompt)
                return '{"research_plan": "Plan", "sub_queries": ["sub"]}'
            elif "synthesiz" in prompt.lower():
                return '{"claims": []}'
            return "{}"

    mock_store = MagicMock()
    mock_store.search.return_value = []

    cfg = Settings(_env_file=None)
    graph = create_research_graph(settings=cfg, store=mock_store, provider=PromptCapturingProvider(), memory_store=mem_store)

    # Invoke for Bob
    res_bob = graph.invoke({"query": "Check role", "user_id": "user_bob", "max_research_cycles": 1})

    # Assert Bob's coordinator prompt received ONLY Bob's preference, never Alice's
    assert len(coordinator_prompts) == 1
    assert bob_secret in coordinator_prompts[0]
    assert alice_secret not in coordinator_prompts[0]

    # Assert Bob's state contains only Bob's context (wiped on exit)
    assert res_bob["memory_status"] == "loaded"
    assert alice_secret not in str(res_bob)


def test_graph_chat_model_child_run_tracing_boundary_never_records_user_memory(tmp_path):
    """End-to-end regression: Child LLM runs never capture private user-memory markers in traces."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tracers.context import collect_runs

    from asterline.memory import UserMemoryStore
    from asterline.providers import LLMProvider

    mem_db = tmp_path / "chat_model_trace_test.db"
    mem_store = UserMemoryStore(mem_db)
    synthetic_secret = "SYNTHETIC_TOP_SECRET_PREFERENCE_TOKEN_444"
    mem_store.remember("user_alice", "preference", "tier", synthetic_secret)

    coordinator_json = '{"research_plan": "Customized for Token 444", "sub_queries": ["overview"]}'
    synthesis_json = (
        '{\n'
        '  "claims": [\n'
        '    {\n'
        '      "claim_text": "Corpus overview content.",\n'
        '      "citations": [\n'
        '        {\n'
        '          "evidence_id": "chunk-c1",\n'
        '          "chunk_id": "chunk-c1",\n'
        '          "snippet": "Corpus overview content."\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    verifier_json = '{"status": "supported", "note": "ok", "has_conflict": false}'

    class RealChatModelProvider(LLMProvider):
        """Provider backed by LangChain BaseChatModel to ensure realistic child run trace behavior."""

        def __init__(self):
            responses = [coordinator_json, "0: relevant\n", synthesis_json, verifier_json]

            class TracedChat(BaseChatModel):
                def __init__(self, res_list):
                    super().__init__()
                    self._res_list = list(res_list)

                def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                    text = self._res_list.pop(0) if self._res_list else "{}"
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

                @property
                def _llm_type(self):
                    return "traced-chat-model"

            self._chat = TracedChat(responses)
            self.captured_prompts = []

        def invoke(self, prompt: str, **kwargs) -> str:
            self.captured_prompts.append(prompt)
            msg = self._chat.invoke(prompt)
            return msg.content

    provider = RealChatModelProvider()

    mock_store = MagicMock()
    mock_store.search.return_value = [
        _make_search_result("chunk-c1", content="Corpus overview content.")
    ]

    cfg = Settings(_env_file=None, max_relevance_distance=0.70)
    graph = create_research_graph(
        settings=cfg,
        store=mock_store,
        provider=provider,
        memory_store=mem_store,
    )

    with collect_runs() as cb:
        result = graph.invoke({
            "query": "Corporate overview query",
            "user_id": "user_alice",
            "max_research_cycles": 1,
        })
        all_traced_runs = list(cb.traced_runs)

    # 1. Personalization succeeded at runtime: Coordinator LLM prompt received the private memory
    assert any(synthetic_secret in p for p in provider.captured_prompts)

    # 2. Terminal response and final state do not retain the synthetic secret or raw memories
    assert result.get("user_memories") == []
    assert synthetic_secret not in str(result.get("response", {}))

    # 3. Across ALL recorded runs (root graph and all child LLM runs), the secret marker is completely absent
    assert len(all_traced_runs) > 0  # Tracing was active (e.g. LangGraph root run was collected)
    for run in all_traced_runs:
        run_dict = run.model_dump()
        run_str = str(run_dict)
        assert synthetic_secret not in run_str, (
            f"Synthetic secret leaked in traced run '{run.name}' (type={run.run_type})"
        )
