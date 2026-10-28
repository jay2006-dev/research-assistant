"""Tests for Coordinator node: query formulation, replanning, and counter preservation."""

from unittest.mock import MagicMock

from asterline.config import Settings
from asterline.coordinator import coordinator_node
from asterline.providers import FakeLLMProvider, ProviderError
from asterline.state import ResearchState, VerificationReport, WorkflowOutcome


def test_coordinator_initial_cycle_deterministic_plan():
    """Coordinator generates initial research plan and sub_queries deterministically without provider."""
    state: ResearchState = {
        "query": "What is the Asterline retention policy?",
        "research_cycle_count": 0,
        "max_research_cycles": 2,
    }

    result = coordinator_node(state, provider=None)

    assert "Asterline retention policy" in result["research_plan"]
    assert result["sub_queries"] == ["What is the Asterline retention policy?"]
    # Invariant: Coordinator must not touch or reset research_cycle_count
    assert "research_cycle_count" not in result


def test_coordinator_empty_query_rejected():
    """Coordinator rejects empty or whitespace query with IRRELEVANT_QUERY outcome."""
    state: ResearchState = {
        "query": "   ",
        "research_cycle_count": 0,
    }

    result = coordinator_node(state)
    assert result["outcome"] == WorkflowOutcome.IRRELEVANT_QUERY.value
    assert "empty" in result["error_message"].lower()
    assert result["sub_queries"] == []


def test_coordinator_replan_cycle_identifies_gaps():
    """On replan cycle, coordinator targets evidence gaps and refines sub_queries."""
    invalid_report = VerificationReport(
        all_citations_valid=False,
        all_claims_supported=False,
        has_unsupported_content=True,
        claim_results=[
            {
                "claim_index": 0,
                "claim_text": "Retention period is exactly 7 years",
                "status": "unsupported",
                "note": "Corpus does not specify 7 years",
            }
        ],
    )
    state: ResearchState = {
        "query": "Asterline data retention",
        "research_cycle_count": 1,  # Cycle 2 entry
        "max_research_cycles": 2,
        "verification_report": invalid_report.model_dump(),
        "evidence_store": {
            "chunk-1": {"title": "General Retention Overview", "content": "Retention policies..."}
        },
    }

    result = coordinator_node(state, provider=None)

    assert "Refined research plan (Cycle 2)" in result["research_plan"]
    assert any("Retention period is exactly 7 years" in q for q in result["sub_queries"])
    assert result["query"] == result["sub_queries"][0]
    # Invariant: research_cycle_count must remain untouched
    assert "research_cycle_count" not in result


def test_coordinator_llm_assisted_decomposition():
    """Coordinator parses structured LLM response into research plan and sub_queries."""
    llm_response = (
        '{\n'
        '  "research_plan": "Decompose into tenure rules and exceptions.",\n'
        '  "sub_queries": ["Asterline tenure rules", "Asterline retention exceptions"]\n'
        '}'
    )
    fake_provider = FakeLLMProvider(default_text=llm_response)
    state: ResearchState = {
        "query": "Asterline tenure policy",
        "research_cycle_count": 0,
    }

    result = coordinator_node(state, provider=fake_provider)

    assert result["research_plan"] == "Decompose into tenure rules and exceptions."
    assert result["sub_queries"] == ["Asterline tenure rules", "Asterline retention exceptions"]


def test_coordinator_falls_back_gracefully_on_provider_error():
    """When LLM provider raises an error, coordinator falls back to deterministic plan."""
    mock_provider = MagicMock()
    mock_provider.invoke.side_effect = ProviderError("LLM rate limit")

    state: ResearchState = {
        "query": "Asterline audit procedures",
        "research_cycle_count": 0,
    }

    result = coordinator_node(state, provider=mock_provider)

    assert "Initial research plan" in result["research_plan"]
    assert result["sub_queries"] == ["Asterline audit procedures"]


def test_coordinator_with_user_memories_suppresses_child_run_tracing_and_wipes_state():
    """Coordinator isolates memory-bearing LLM calls from trace collectors and cleanses state."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tracers.context import collect_runs

    from asterline.providers import LLMProvider

    synthetic_secret = "CONFIDENTIAL_USER_TIER_PREFERENCE_555"

    class TracedChatProvider(LLMProvider):
        def __init__(self):
            class InternalChat(BaseChatModel):
                def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                    return ChatResult(
                        generations=[
                            ChatGeneration(
                                message=AIMessage(
                                    content=(
                                        '{"research_plan": "Tailored for Tier 555", '
                                        '"sub_queries": ["Asterline Tier 555 pricing"]}'
                                    )
                                )
                            )
                        ]
                    )

                @property
                def _llm_type(self):
                    return "traced-chat"

            self._chat = InternalChat()
            self.captured_prompts = []

        def invoke(self, prompt: str, **kwargs) -> str:
            self.captured_prompts.append(prompt)
            msg = self._chat.invoke(prompt)
            return msg.content

    provider = TracedChatProvider()
    state: ResearchState = {
        "query": "What is my plan cost?",
        "research_cycle_count": 0,
        "user_memories": [
            {"category": "preference", "key": "tier", "content": synthetic_secret}
        ],
    }

    with collect_runs() as cb:
        result = coordinator_node(state, provider=provider)

    # 1. Personalization succeeded at runtime: prompt received the memory context
    assert len(provider.captured_prompts) == 1
    assert synthetic_secret in provider.captured_prompts[0]
    assert result["research_plan"] == "Tailored for Tier 555"
    assert result["sub_queries"] == ["Asterline Tier 555 pricing"]

    # 2. Raw user_memories wiped from node return to prevent retention in downstream state
    assert result["user_memories"] == []

    # 3. Tracing boundary prevented child LLM run from being recorded in trace collector
    assert len(cb.traced_runs) == 0


def test_coordinator_redacts_memory_tokens_on_provider_error(caplog):
    """Coordinator redacts user memory content from error logs and messages upon provider failure."""
    import logging

    synthetic_secret = "SECRET_MEMORY_KEY_PAYLOAD_888"
    mock_provider = MagicMock()
    mock_provider.invoke.side_effect = ProviderError(f"API failed processing: {synthetic_secret}")

    state: ResearchState = {
        "query": "Check pricing",
        "research_cycle_count": 0,
        "user_memories": [
            {"category": "preference", "key": "secret", "content": synthetic_secret}
        ],
    }

    with caplog.at_level(logging.WARNING):
        result = coordinator_node(state, provider=mock_provider)

    # Deterministic fallback returned
    assert "Initial research plan" in result["research_plan"]
    assert result["user_memories"] == []

    # Warning log MUST NOT leak synthetic_secret
    log_text = caplog.text
    assert synthetic_secret not in log_text
    assert "[REDACTED]" in log_text
