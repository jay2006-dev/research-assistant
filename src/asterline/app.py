"""Streamlit user interface for Asterline Research Assistant.

Provides an enterprise research interface featuring:
- User-scoped memory management (namespace isolation, inspection, correction, forgetting).
- Multi-step LangGraph research workflow invocation.
- Deterministic validator-approved findings and corpus citation/snippet display.
- Observability and metadata display without leaking secrets or private reasoning.
- Graceful error handling for missing indices, provider errors, and memory store issues.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import streamlit as st

from asterline.config import Settings
from asterline.graph import create_research_graph
from asterline.memory import (
    MemoryError,
    MemoryNotFoundError,
    MemoryValidationError,
    UserMemoryStore,
)
from asterline.providers import (
    LLMProvider,
    ProviderError,
    create_provider_from_settings,
    redact_secrets,
)
from asterline.state import ResearchState, WorkflowOutcome
from asterline.store import (
    IndexCorruptedError,
    IndexNotInitializedError,
    VectorStore,
)

logger = logging.getLogger(__name__)


def get_settings() -> Settings:
    """Retrieve application settings."""
    return Settings()


def get_memory_store(settings: Optional[Settings] = None) -> UserMemoryStore:
    """Retrieve persistent user memory store instance."""
    cfg = settings or get_settings()
    return UserMemoryStore(cfg.memory_db_abs_path)


def get_vector_store(settings: Optional[Settings] = None) -> VectorStore:
    """Retrieve persistent ChromaDB vector store instance."""
    cfg = settings or get_settings()
    return VectorStore(settings=cfg)


def execute_research(
    query: str,
    user_id: str,
    settings: Optional[Settings] = None,
    provider: Optional[LLMProvider] = None,
    store: Optional[VectorStore] = None,
    memory_store: Optional[UserMemoryStore] = None,
) -> Dict[str, Any]:
    """Execute the multi-step research workflow and return validated findings and metadata."""
    cfg = settings or get_settings()
    vs = store or get_vector_store(cfg)
    mem_store = memory_store or get_memory_store(cfg)

    # 1. Pre-flight check on vector index
    try:
        vs.get_active_collection()
    except IndexNotInitializedError as err:
        return {
            "outcome": "error",
            "error_type": "IndexNotInitializedError",
            "error_message": f"Knowledge base index is uninitialized: {err}. Please run 'uv run python -m asterline.cli index' first.",
        }
    except Exception as err:
        return {
            "outcome": "error",
            "error_type": type(err).__name__,
            "error_message": f"Failed to access knowledge base index: {err}",
        }

    # 2. Configure model provider
    active_provider = provider
    secrets = [s for s in [cfg.api_key, cfg.groq_api_key, cfg.langsmith_api_key] if s]

    if active_provider is None:
        provider_name = getattr(cfg, "model_provider", "gemini").lower()
        if provider_name == "groq" and not cfg.groq_api_key:
            return {
                "outcome": "error",
                "error_type": "MissingConfigurationError",
                "error_message": "Groq API key is not configured. Set GROQ_API_KEY in .env.",
            }
        elif provider_name != "groq" and not cfg.api_key:
            return {
                "outcome": "error",
                "error_type": "MissingConfigurationError",
                "error_message": "Gemini API key is not configured. Set GEMINI_API_KEY or API_KEY in .env.",
            }

        try:
            active_provider = create_provider_from_settings(cfg)
        except Exception as err:
            safe_err = redact_secrets(str(err), secrets)
            return {
                "outcome": "error",
                "error_type": type(err).__name__,
                "error_message": f"Failed to initialize model provider: {safe_err}",
            }

    # 3. Execute research graph
    workflow_timeout = getattr(cfg, "workflow_timeout", 120.0)
    deadline = time.time() + workflow_timeout

    try:
        graph = create_research_graph(
            settings=cfg,
            store=vs,
            provider=active_provider,
            memory_store=mem_store,
        )
        initial_state: ResearchState = {
            "query": query,
            "user_id": user_id,
            "max_research_cycles": 2,
            "deadline": deadline,
        }
        result = graph.invoke(initial_state)
    except Exception as err:
        safe_err = redact_secrets(str(err), secrets)
        return {
            "outcome": "error",
            "error_type": type(err).__name__,
            "error_message": f"Research workflow execution failed: {safe_err}",
        }

    # 4. Extract validated findings
    resp = result.get("response", {})
    raw_outcome = resp.get("outcome") or result.get("outcome", "unknown")
    outcome_str = raw_outcome.value if hasattr(raw_outcome, "value") else str(raw_outcome)
    if outcome_str.startswith("WorkflowOutcome."):
        outcome_str = outcome_str.split(".", 1)[1]
    outcome_str = outcome_str.lower()

    return {
        "outcome": outcome_str,
        "answer": resp.get("answer", "").strip(),
        "claims": resp.get("claims", []),
        "citations": resp.get("citations", []),
        "conflicts": resp.get("conflicts", []),
        "confidence_note": resp.get("confidence_note", ""),
        "cycles_used": resp.get("cycles_used", result.get("research_cycle_count", 1)),
        "memory_status": result.get("memory_status", "empty"),
        "relevant_chunks": [e.get("chunk_id") if isinstance(e, dict) else getattr(e, "chunk_id", str(e)) for e in result.get("evidence", [])],
        "user_id": user_id,
        "query": query,
    }


def render_sidebar(settings: Settings, mem_store: UserMemoryStore) -> str:
    """Render the sidebar with user identity and explicit memory management."""
    st.sidebar.title("Asterline Assistant")
    st.sidebar.caption("Grounded Corporate Research & Personal Context")

    # User namespace selector
    st.sidebar.subheader("👤 User Identity")
    user_options = ["default", "alice", "bob", "custom"]
    selected_option = st.sidebar.selectbox(
        "Active User Namespace:",
        options=user_options,
        index=0,
        help="Local user selector scopes memory and research context.",
    )

    if selected_option == "custom":
        user_id = st.sidebar.text_input("Custom User ID:", value="analyst_1").strip()
        if not user_id:
            user_id = "default"
    else:
        user_id = selected_option

    st.sidebar.info(
        "ℹ️ **Namespace Notice:** The user selector defines a scoped memory namespace for local demos, "
        "not cryptographic authentication."
    )

    # Provider & Model Info
    st.sidebar.subheader("⚙️ System Configuration")
    provider_name = getattr(settings, "model_provider", "gemini").upper()
    model_name = settings.groq_model_name if provider_name == "GROQ" else settings.model_name
    st.sidebar.text(f"Provider: {provider_name}")
    st.sidebar.text(f"Model:    {model_name}")

    # Memory Management Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Persistent User Memory")
    st.sidebar.caption(
        "User context is retained for query personalization. "
        "Memories are never treated or cited as corporate evidence."
    )

    # Load active memories for selected user
    active_memories = []
    try:
        active_memories = mem_store.recall(user_id=user_id)
    except Exception as err:
        st.sidebar.error(f"Failed to access memory store: {err}")

    # Memory tab / action selection
    mem_action = st.sidebar.radio(
        "Memory Actions:",
        options=["View Active", "Add Context", "Correct Context", "Forget Context"],
        index=0,
    )

    if mem_action == "View Active":
        if active_memories:
            st.sidebar.write(f"**{len(active_memories)} Active Memory Item(s):**")
            for m in active_memories:
                with st.sidebar.expander(f"🔑 {m.key} ({m.category})", expanded=False):
                    st.write(m.content)
                    st.caption(f"Updated: {time.strftime('%Y-%m-%d %H:%M', time.localtime(m.updated_at))}")
        else:
            st.sidebar.write(f"No active memories for user `{user_id}`.")

    elif mem_action == "Add Context":
        with st.sidebar.form("add_memory_form", clear_on_submit=True):
            category = st.selectbox("Category:", ["preference", "constraint", "role", "project"], index=0)
            key = st.text_input("Memory Key (e.g. target_tier):").strip()
            content = st.text_area("Memory Content (e.g. Interested in Launch plan details):").strip()
            submitted = st.form_submit_button("Save Memory")
            if submitted:
                if not key or not content:
                    st.sidebar.error("Key and content cannot be empty.")
                else:
                    try:
                        mem_store.remember(user_id=user_id, category=category, key=key, content=content)
                        st.sidebar.success(f"Remembered '{key}' for user '{user_id}'.")
                        st.rerun()
                    except Exception as err:
                        st.sidebar.error(f"Failed to save: {err}")

    elif mem_action == "Correct Context":
        if not active_memories:
            st.sidebar.info(f"No active memories to correct for user `{user_id}`.")
        else:
            keys = [m.key for m in active_memories]
            with st.sidebar.form("correct_memory_form", clear_on_submit=True):
                target_key = st.selectbox("Select Key to Correct:", keys)
                new_content = st.text_area("Updated Content:").strip()
                submitted = st.form_submit_button("Apply Correction")
                if submitted:
                    if not new_content:
                        st.sidebar.error("Content cannot be empty.")
                    else:
                        try:
                            mem_store.correct(user_id=user_id, key=target_key, new_content=new_content)
                            st.sidebar.success(f"Updated '{target_key}' for user '{user_id}'.")
                            st.rerun()
                        except Exception as err:
                            st.sidebar.error(f"Failed to update: {err}")

    elif mem_action == "Forget Context":
        if not active_memories:
            st.sidebar.info(f"No active memories to forget for user `{user_id}`.")
        else:
            keys = [m.key for m in active_memories]
            with st.sidebar.form("forget_memory_form", clear_on_submit=True):
                target_key = st.selectbox("Select Key to Forget:", keys)
                st.caption("⚠️ Wipes all current and historical versions for this user.")
                submitted = st.form_submit_button("Forget Permanently")
                if submitted:
                    try:
                        count = mem_store.forget(user_id=user_id, key=target_key)
                        st.sidebar.success(f"Forgot '{target_key}' ({count} records wiped).")
                        st.rerun()
                    except Exception as err:
                        st.sidebar.error(f"Failed to forget: {err}")

    return user_id


def render_response(finding: Dict[str, Any]) -> None:
    """Render a structured research result with outcome indicators, citations, and metadata."""
    outcome = finding.get("outcome", "unknown")
    query = finding.get("query", "")
    answer = finding.get("answer", "")
    claims = finding.get("claims", [])
    citations = finding.get("citations", [])
    conflicts = finding.get("conflicts", [])
    cycles_used = finding.get("cycles_used", 1)
    memory_status = finding.get("memory_status", "empty")
    relevant_chunks = finding.get("relevant_chunks", [])

    st.markdown(f"### Research Query: *\"{query}\"*")

    # 1. Outcome Status Badge & Primary Answer Area
    if outcome == "success":
        st.success("✅ **Verified Research Findings (Corporate Evidence Confirmed)**")
        st.markdown(f"#### {answer}")

        # Structured accepted findings & citations
        if claims:
            st.markdown("##### 📌 Accepted Claims and Corpus Evidence:")
            for i, claim in enumerate(claims, 1):
                c_text = claim.get("claim_text") if isinstance(claim, dict) else getattr(claim, "claim_text", "")
                c_citations = claim.get("citations", []) if isinstance(claim, dict) else getattr(claim, "citations", [])
                st.markdown(f"**Claim {i}:** {c_text}")
                for cit in c_citations:
                    cid = cit.get("chunk_id") if isinstance(cit, dict) else getattr(cit, "chunk_id", "")
                    snippet = cit.get("snippet") if isinstance(cit, dict) else getattr(cit, "snippet", "")
                    st.info(f"📄 **Citation `[{cid}]`:** *\"{snippet}\"*")

    elif outcome == "insufficient_evidence":
        st.warning("⚠️ **Insufficient Evidence in Corporate Knowledge Base**")
        if answer:
            st.markdown(f"**Assistant Response:** {answer}")
        else:
            st.markdown(
                "The Asterline knowledge base does not contain verified corporate documentation to answer "
                "this query. Parametric assertions and unsupported guesses are strictly rejected."
            )

    elif outcome == "conflict_detected":
        st.error("⚠️ **Unresolved Conflict in Corporate Evidence**")
        st.markdown(f"**Status:** {answer or 'Conflicting evidence detected across retrieved corpus documents.'}")
        if conflicts:
            st.markdown("**Conflicting Statements Detected:**")
            for conf in conflicts:
                st.markdown(f"- {conf}")

    elif outcome == "error":
        st.error(f"🚨 **System Error ({finding.get('error_type', 'ExecutionError')}):**")
        st.code(finding.get("error_message", "An unexpected error occurred."))

    else:
        st.info(f"Outcome: {outcome.upper()}")
        if answer:
            st.write(answer)

    # 2. Agent Visibility & Observability Panel (No secret leaks, no hidden reasoning)
    with st.expander("🔍 Research Graph Details & Observability", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Workflow Outcome", outcome.upper())
        with col2:
            st.metric("Research Cycles", f"{cycles_used} / 2")
        with col3:
            st.metric("User Memory Status", memory_status.upper())

        if relevant_chunks:
            st.write(f"**Corpus Chunks Evaluated ({len(relevant_chunks)}):** `{', '.join(relevant_chunks)}`")

        st.caption(
            "🔒 **Privacy & Safety Guarantee:** Raw provider credentials, API keys, and internal model reasoning "
            "are strictly isolated and never exposed in observability panels."
        )


def main() -> None:
    """Streamlit application main entry point."""
    settings = get_settings()
    st.set_page_config(
        page_title=settings.app_name,
        page_icon="🔍",
        layout="wide",
    )

    mem_store = get_memory_store(settings)
    active_user = render_sidebar(settings, mem_store)

    st.title("🔍 Asterline Enterprise Research Assistant")
    st.caption(
        "Grounded multi-step research assistant backed by Asterline's authoritative corporate knowledge base."
    )

    # Initialize session state for conversation history
    if "research_history" not in st.session_state:
        st.session_state["research_history"] = []

    # Clear / Start New Question action
    header_col1, header_col2 = st.columns([6, 2])
    with header_col2:
        if st.button("🔄 New Question / Clear View", use_container_width=True):
            st.session_state["research_history"] = []
            st.rerun()

    # Question Input Form
    with st.form("research_query_form", clear_on_submit=False):
        user_query = st.text_input(
            "Ask a research question about Asterline products, pricing, policies, or architecture:",
            placeholder="e.g. How much does the Launch plan cost, and how long are its run records retained?",
        ).strip()
        submit_btn = st.form_submit_button("Submit Research Query", use_container_width=True)

    if submit_btn:
        if not user_query:
            st.warning("Please enter a research question before submitting.")
        else:
            with st.spinner(f"Executing multi-step research workflow for user '{active_user}'..."):
                finding = execute_research(query=user_query, user_id=active_user, settings=settings)
                # Append to session history (session state is conversational display, NOT persistent memory)
                st.session_state["research_history"].insert(0, finding)

    # Render History / Active Findings
    if st.session_state["research_history"]:
        st.markdown("---")
        for finding in st.session_state["research_history"]:
            render_response(finding)
            st.markdown("---")
    else:
        st.markdown("---")
        st.subheader("💡 Example Queries to Explore:")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                """
                - **Direct Pricing:**
                  *“How much does the Launch plan cost per month?”*
                - **Multi-Part Synthesis:**
                  *“How much does the Launch plan cost, and how long are its run records retained?”*
                """
            )
        with col_b:
            st.markdown(
                """
                - **Absent Topic (Safe Abstention):**
                  *“What are the quantum computing integration specifications for Asterline?”*
                - **Policy Supersession (Time-Sensitive):**
                  *“How long are raw run records held in cold recovery storage under the effective policy?”*
                """
            )


if __name__ == "__main__":
    main()
