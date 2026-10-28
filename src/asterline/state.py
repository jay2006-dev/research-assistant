"""Structured data contracts for the Asterline research workflow.

Defines the shared state, evidence, citation, claim, conflict, verification,
and response models used across all graph nodes.

Design decisions
----------------
- ``ResearchState`` is a ``TypedDict`` for LangGraph ``StateGraph`` compatibility.
- All externally supplied and LLM-generated structured data uses Pydantic models
  with strict validation to reject malformed fields at the boundary.
- ``AnswerClaim.evidence_ids`` references real ``EvidenceItem.evidence_id`` values;
  the deterministic validator enforces referential integrity.
- ``VerificationReport`` is the verifier's structured output. It records support,
  relevance, citation validity, and verification failures per claim.
- ``ResearchResponse`` is the validated terminal output returned to the caller.
  It is reconstructed exclusively from validated claims—never from an unconstrained
  LLM-generated draft.
"""

from __future__ import annotations

import enum
from typing import Any, List, Optional, TypedDict

from pydantic import BaseModel, Field, computed_field


# ── Enumerations ─────────────────────────────────────────────────────────────

class VerificationStatus(str, enum.Enum):
    """Outcome of claim-level verification."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    PARTIAL = "partial"
    UNCERTAIN = "uncertain"
    CONFLICT = "conflict"


class WorkflowOutcome(str, enum.Enum):
    """Terminal outcome of the research workflow."""

    SUCCESS = "success"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROVIDER_ERROR = "provider_error"
    MALFORMED_OUTPUT = "malformed_output"
    ITERATION_LIMIT = "iteration_limit"
    IRRELEVANT_QUERY = "irrelevant_query"


# ── Evidence and Citation Models ─────────────────────────────────────────────

class EvidenceItem(BaseModel):
    """Retrieved corpus evidence with stable source and chunk identifiers."""

    evidence_id: str = Field(
        ...,
        description="Unique identifier for this evidence item (typically the corpus chunk_id).",
    )
    chunk_id: str = Field(
        ...,
        description="Original corpus chunk ID from the vector store.",
    )
    doc_id: str = Field(..., description="Document ID from corpus metadata.")
    title: str = Field(..., description="Title from corpus metadata.")
    content: str = Field(..., description="Full content text of the evidence chunk.")
    category: str = Field(default="", description="Category from corpus metadata.")
    source_url: str = Field(default="", description="Source URL from corpus metadata.")
    published: str = Field(default="", description="Publication date from corpus metadata.")
    version: int = Field(default=0, description="Version number from corpus metadata.")
    distance: float = Field(
        ...,
        description="Cosine distance from query (lower = more relevant).",
    )
    is_relevant: bool = Field(
        default=False,
        description="Whether this evidence passed relevance validation.",
    )


class Citation(BaseModel):
    """Identifies the corpus evidence supporting a claim."""

    evidence_id: str = Field(
        ...,
        description="Reference to an EvidenceItem.evidence_id.",
    )
    chunk_id: str = Field(
        ...,
        description="Original corpus chunk_id for traceability.",
    )
    snippet: str = Field(
        default="",
        description=(
            "Relevant excerpt from the evidence content. "
            "Must be derived from actual retrieved corpus text."
        ),
    )


class AnswerClaim(BaseModel):
    """An individual factual assertion with its supporting citations."""

    claim_text: str = Field(
        ...,
        description="The factual assertion text.",
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Citations supporting this claim.",
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNSUPPORTED,
        description="Current verification status of this claim.",
    )
    verification_note: str = Field(
        default="",
        description="Explanation of verification result.",
    )

    @computed_field
    @property
    def evidence_ids(self) -> list[str]:
        """Return all evidence IDs referenced by this claim's citations."""
        return [c.evidence_id for c in self.citations]


class SourceConflict(BaseModel):
    """Records materially inconsistent evidence without inventing precedence."""

    description: str = Field(
        ...,
        description="Neutral description of the conflict.",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Evidence item IDs involved in the conflict.",
    )
    resolution: str = Field(
        default="unresolved",
        description=(
            "How the conflict was handled. Must not infer authority from "
            "version numbers or publication dates without an explicit policy."
        ),
    )


# ── Verification Report ─────────────────────────────────────────────────────

class ClaimVerificationResult(BaseModel):
    """Verification result for a single claim."""

    claim_index: int = Field(..., description="Index of the claim in the claims list.")
    claim_text: str = Field(..., description="The claim text being verified.")
    status: VerificationStatus = Field(..., description="Verification outcome.")
    evidence_ids_checked: List[str] = Field(
        default_factory=list,
        description="Evidence IDs that were checked for support.",
    )
    note: str = Field(default="", description="Explanation of the verification outcome.")


class VerificationReport(BaseModel):
    """Records support, relevance, citation validity, and verification failures."""

    claim_results: List[ClaimVerificationResult] = Field(
        default_factory=list,
        description="Per-claim verification results.",
    )
    conflicts: List[SourceConflict] = Field(
        default_factory=list,
        description="Detected source conflicts.",
    )
    all_citations_valid: bool = Field(
        default=False,
        description="Whether all citation IDs reference real evidence items.",
    )
    all_claims_supported: bool = Field(
        default=False,
        description="Whether all claims have sufficient evidence support.",
    )
    has_unsupported_content: bool = Field(
        default=False,
        description="Whether unsupported draft text was detected.",
    )
    recommendation: str = Field(
        default="reject",
        description="accept, replan, or reject.",
    )
    summary: str = Field(
        default="",
        description="Human-readable summary of verification results.",
    )


# ── Research Response ────────────────────────────────────────────────────────

class ResearchResponse(BaseModel):
    """Validated terminal output returned to the caller.

    Reconstructed exclusively from validated claims—never from an unconstrained
    LLM-generated draft.
    """

    answer: str = Field(
        ...,
        description="Final answer text, reconstructed from validated claims.",
    )
    claims: List[AnswerClaim] = Field(
        default_factory=list,
        description="Validated claims composing the answer.",
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="All citations used in the answer.",
    )
    conflicts: List[SourceConflict] = Field(
        default_factory=list,
        description="Source conflicts detected during research.",
    )
    outcome: WorkflowOutcome = Field(
        ...,
        description="Terminal outcome of the research workflow.",
    )
    confidence_note: str = Field(
        default="",
        description="Honest assessment of answer confidence and limitations.",
    )
    cycles_used: int = Field(
        default=0,
        description="Number of research cycles performed (max 2).",
    )
    iterations_used: int = Field(
        default=0,
        description="Alias for cycles_used for backwards compatibility.",
    )


# ── LangGraph State ─────────────────────────────────────────────────────────

class ResearchState(TypedDict, total=False):
    """LangGraph state dictionary tracking the full research workflow.

    Uses TypedDict for StateGraph compatibility. All fields are optional
    (total=False) since different nodes populate different fields.
    """

    # ── Input ────────────────────────────────────────────────────
    user_id: str                               # Local identity namespace boundary (defaults to "default")
    user_memories: list[dict[str, Any]]        # Serialized active MemoryItem dicts for user context
    memory_status: str                         # "loaded", "empty", or "unavailable"
    query: str
    sub_queries: list[str]
    research_plan: str
    deadline: float                            # Absolute timestamp (time.time()) when workflow must complete

    # ── Evidence ─────────────────────────────────────────────────
    evidence_store: dict[str, dict[str, Any]]  # Keyed by stable corpus chunk_id
    relevant_chunk_ids: list[str]              # Unique chunk IDs meeting relevance criteria
    new_chunk_ids: list[str]                   # Chunk IDs retrieved in the current cycle
    new_chunk_count: int                       # Count of newly discovered chunks in this cycle
    evidence: list[dict[str, Any]]             # All EvidenceItem dicts (convenience list)
    relevant_evidence: list[dict[str, Any]]    # Relevant EvidenceItem dicts (convenience list)

    # ── Synthesis ────────────────────────────────────────────────
    claims: list[dict[str, Any]]               # Serialized AnswerClaim dicts
    draft_answer: str

    # ── Verification ─────────────────────────────────────────────
    verification_report: dict[str, Any]        # Serialized VerificationReport
    conflicts: list[dict[str, Any]]            # Serialized SourceConflict dicts

    # ── Control Flow ─────────────────────────────────────────────
    research_cycle_count: int                  # Number of research attempts started (max 2)
    max_research_cycles: int                   # Hard ceiling (default 2)
    iteration_count: int                       # Alias for backwards compatibility
    max_iterations: int                        # Alias for backwards compatibility
    route_decision: str                        # "coordinator", "researcher", "synthesizer", "verifier", "validator", "fallback"
    error_message: str
    outcome: str                               # WorkflowOutcome value

    # ── Terminal Output ──────────────────────────────────────────
    response: dict[str, Any]                   # Serialized ResearchResponse
