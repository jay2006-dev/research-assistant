"""Command-line interface for Asterline knowledge base indexing, status, and research.

Usage:
    python -m asterline.cli index [--rebuild]
    python -m asterline.cli status
    python -m asterline.cli search "query string" [--top-k N]
    python -m asterline.cli research "research question" [--max-cycles N]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from typing import Optional

from asterline.config import Settings
from asterline.corpus import parse_corpus
from asterline.graph import create_research_graph
from asterline.providers import (
    LLMProvider,
    SafeGeminiProvider,
    create_provider_from_settings,
    redact_secrets,
)
from asterline.state import ResearchState
from asterline.store import (
    IndexBuildLockedError,
    IndexCorruptedError,
    IndexNotInitializedError,
    PointerUpdateError,
    VectorStore,
)


def handle_index(args: argparse.Namespace, settings: Settings) -> int:
    """Handle the 'index' command."""
    print("=" * 60)
    print(f"Asterline Indexer — {'Rebuild' if args.rebuild else 'Idempotent Sync'}")
    print("=" * 60)
    print(f"Corpus Path: {settings.corpus_abs_path}")
    print(f"Chroma Directory: {settings.chroma_abs_dir}")

    # Inspect parse results first for user visibility
    try:
        parse_result = parse_corpus(settings.corpus_abs_path)
        print(f"Corpus parsed: {len(parse_result.chunks)} valid chunks.")
        print(f"Pagination artifacts removed: {parse_result.removed_artifacts_count}.")
        if parse_result.diagnostics:
            print(f"Parser diagnostics ({len(parse_result.diagnostics)}):")
            for d in parse_result.diagnostics:
                print(f"  [{d.severity.upper()}] Line {d.line_number}: {d.message}")
    except Exception as err:
        print(f"\n[ERROR] Corpus pre-flight validation failed: {err}", file=sys.stderr)
        return 1

    store = VectorStore(settings=settings)
    try:
        pointer = store.build_index(rebuild=args.rebuild)
        print("\n[SUCCESS] Indexing complete.")
        print(f"  Active Collection: {pointer.collection_name}")
        print(f"  Indexed Chunks:    {pointer.chunks_count}")
        print(f"  Created At:        {pointer.created_at}")
        print(f"  Distance Metric:   {pointer.distance_metric} (lower = closer)")
        return 0
    except IndexBuildLockedError as err:
        print(f"\n[ERROR] Build lock active: {err}", file=sys.stderr)
        return 1
    except PointerUpdateError as err:
        print(f"\n[ERROR] Pointer update failed: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"\n[ERROR] Index build failed: {err}", file=sys.stderr)
        return 1


def handle_status(args: argparse.Namespace, settings: Settings) -> int:
    """Handle the 'status' command."""
    print("=" * 60)
    print("Asterline Vector Index Status")
    print("=" * 60)
    store = VectorStore(settings=settings)
    try:
        collection = store.get_active_collection()
        pointer = store._get_active_pointer()
        print("Status: Active")
        print(f"Pointer File:      {settings.active_pointer_abs_path}")
        print(f"Active Collection: {pointer.collection_name if pointer else 'N/A'}")
        print(f"Document Count:    {collection.count()}")
        print(f"Distance Metric:   {pointer.distance_metric if pointer else 'cosine'}")
        return 0
    except IndexNotInitializedError as err:
        print("Status: Uninitialized")
        print(f"Detail: {err}")
        return 1
    except IndexCorruptedError as err:
        print("Status: Corrupted")
        print(f"Detail: {err}")
        return 1
    except Exception as err:
        print(f"Status: Error - {err}", file=sys.stderr)
        return 1


def handle_search(args: argparse.Namespace, settings: Settings) -> int:
    """Handle the 'search' command."""
    store = VectorStore(settings=settings)
    try:
        results = store.search(query=args.query, top_k=args.top_k)
        if not results:
            print(f"No results found for query: '{args.query}'")
            return 0

        print(f"Retrieved {len(results)} results (ranked by ascending cosine distance):")
        print("-" * 60)
        for idx, r in enumerate(results, 1):
            print(f"#{idx} [{r.chunk_id}] (distance: {r.distance:.4f}) - {r.title}")
            print(f"    Category: {r.category} | Source: {r.source_url} | Published: {r.published}")
            snippet = r.content.replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:120] + "..."
            print(f"    Content: {snippet}\n")
        return 0
    except Exception as err:
        print(f"[ERROR] Search failed: {err}", file=sys.stderr)
        return 1


def handle_research(
    args: argparse.Namespace,
    settings: Settings,
    provider: Optional[LLMProvider] = None,
    store: Optional[VectorStore] = None,
) -> int:
    """Handle the 'research' command: executes the LangGraph research workflow."""
    query = args.query.strip() if args.query else ""
    if not query:
        print("[ERROR] Research query cannot be empty or whitespace.", file=sys.stderr)
        return 1

    # 1. Pre-flight check on vector index
    vs = store or VectorStore(settings=settings)
    try:
        vs.get_active_collection()
    except IndexNotInitializedError as err:
        print(f"[ERROR] Vector index is not initialized: {err}", file=sys.stderr)
        print("Run 'python -m asterline.cli index' to build the index first.", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"[ERROR] Failed to access vector index: {err}", file=sys.stderr)
        return 1

    # 2. Configure LLM provider
    secrets = [
        s for s in [settings.api_key, settings.groq_api_key, settings.langsmith_api_key] if s
    ]
    active_provider = provider
    if active_provider is None:
        provider_name = getattr(settings, "model_provider", "gemini").lower()
        if provider_name == "groq" and not settings.groq_api_key:
            print(
                "[ERROR] Groq API key is not configured. "
                "Set GROQ_API_KEY in your environment or .env file.",
                file=sys.stderr,
            )
            return 1
        elif provider_name != "groq" and not settings.api_key:
            print(
                "[ERROR] Gemini API key is not configured. "
                "Set GEMINI_API_KEY or API_KEY in your environment or .env file.",
                file=sys.stderr,
            )
            return 1
        try:
            active_provider = create_provider_from_settings(settings)
        except Exception as err:
            safe_err = redact_secrets(str(err), secrets)
            print(f"[ERROR] Failed to initialize provider: {safe_err}", file=sys.stderr)
            return 1

    # 3. Compile and execute research graph
    max_cycles = getattr(args, "max_cycles", 2)
    user_id = getattr(args, "user", "default")
    workflow_timeout = getattr(settings, "workflow_timeout", 120.0)
    deadline = time.time() + workflow_timeout
    try:
        from asterline.memory import UserMemoryStore

        mem_store = UserMemoryStore(settings.memory_db_abs_path)
        graph = create_research_graph(
            settings=settings,
            store=vs,
            provider=active_provider,
            memory_store=mem_store,
        )
        initial_state: ResearchState = {
            "query": query,
            "user_id": user_id,
            "max_research_cycles": max_cycles,
            "deadline": deadline,
        }
        trace_url = None
        if getattr(settings, "langsmith_tracing", False):
            try:
                from langchain_core.tracers.context import tracing_v2_enabled

                with tracing_v2_enabled(project_name=settings.langsmith_project) as tracer:
                    result = graph.invoke(initial_state)
                    try:
                        trace_url = tracer.get_run_url()
                    except Exception:
                        trace_url = None
            except Exception:
                result = graph.invoke(initial_state)
        else:
            result = graph.invoke(initial_state)
    except Exception as err:
        safe_err = redact_secrets(str(err), secrets)
        print(f"[ERROR] Research workflow execution failed: {safe_err}", file=sys.stderr)
        return 1

    # 4. Format and present verified findings
    response_dict = result.get("response", {})
    raw_outcome = response_dict.get("outcome") or result.get("outcome", "unknown")
    outcome_str = raw_outcome.value if hasattr(raw_outcome, "value") else str(raw_outcome)
    if outcome_str.startswith("WorkflowOutcome."):
        outcome_str = outcome_str.split(".", 1)[1]
    outcome_str = outcome_str.lower()

    answer = response_dict.get("answer", "").strip()
    claims = response_dict.get("claims", [])
    conflicts = response_dict.get("conflicts", [])
    confidence_note = response_dict.get("confidence_note", "")
    cycles_used = response_dict.get("cycles_used", result.get("research_cycle_count", 1))

    print("=" * 60)
    print("Asterline Research Assistant — Research Findings")
    print("=" * 60)
    print(f"Query:       {query}")
    print(f"Outcome:     {outcome_str.upper()}")
    print(f"Cycles Used: {cycles_used} / {max_cycles}")
    if trace_url:
        print(f"Trace URL:   {trace_url}")
    print("-" * 60)

    if outcome_str == "success":
        print("\n[VERIFIED ANSWER]")
        print(answer)

        if claims:
            print("\n[SUPPORTED FINDINGS & CITATIONS]")
            for i, c in enumerate(claims, 1):
                c_text = (
                    c.get("claim_text") if isinstance(c, dict) else getattr(c, "claim_text", "")
                )
                print(f"  {i}. {c_text}")
                c_citations = (
                    c.get("citations", []) if isinstance(c, dict) else getattr(c, "citations", [])
                )
                for cit in c_citations:
                    eid = (
                        cit.get("evidence_id")
                        if isinstance(cit, dict)
                        else getattr(cit, "evidence_id", "")
                    )
                    snip = (
                        cit.get("snippet")
                        if isinstance(cit, dict)
                        else getattr(cit, "snippet", "")
                    )
                    print(f"     - [{eid}] \"{snip}\"")

        if conflicts:
            print("\n[SOURCE CONFLICTS]")
            for conf in conflicts:
                desc = (
                    conf.get("description")
                    if isinstance(conf, dict)
                    else getattr(conf, "description", "")
                )
                res = (
                    conf.get("resolution")
                    if isinstance(conf, dict)
                    else getattr(conf, "resolution", "")
                )
                eids = (
                    conf.get("evidence_ids")
                    if isinstance(conf, dict)
                    else getattr(conf, "evidence_ids", [])
                )
                print(f"  • {desc} (Resolution: {res}; Evidence: {', '.join(eids)})")

        if confidence_note:
            print(f"\n[NOTE]\n{confidence_note}")
        return 0

    elif outcome_str == "insufficient_evidence":
        print("\n[INSUFFICIENT EVIDENCE]")
        print(answer or "The query could not be verified from corporate corpus evidence.")

        # Display any unverified/uncertain findings discovered during research
        raw_claims = result.get("claims", [])
        if raw_claims:
            print("\n[UNCERTAIN / UNVERIFIED FINDINGS]")
            for i, c in enumerate(raw_claims, 1):
                c_text = (
                    c.get("claim_text") if isinstance(c, dict) else getattr(c, "claim_text", "")
                )
                print(f"  {i}. {c_text} (Status: UNVERIFIED)")

        if conflicts:
            print("\n[SOURCE CONFLICTS]")
            for conf in conflicts:
                desc = (
                    conf.get("description")
                    if isinstance(conf, dict)
                    else getattr(conf, "description", "")
                )
                print(f"  • {desc}")
        return 0

    else:
        print("\n[WORKFLOW FAILURE]")
        print(f"Outcome: {outcome_str.upper()}")
        detail = (
            result.get("error_message")
            or answer
            or "Workflow terminated due to provider error."
        )
        print(f"Detail:  {detail}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="asterline",
        description="Asterline Research Assistant CLI — indexing and retrieval management.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser("index", help="Build or rebuild the knowledge base index.")
    index_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Perform a clean staged rebuild into a fresh versioned collection.",
    )

    # Status command
    subparsers.add_parser("status", help="Check vector index status and active collection.")

    # Search command
    search_parser = subparsers.add_parser("search", help="Execute test retrieval probe.")
    search_parser.add_argument("query", type=str, help="Search query string.")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")

    # Research command
    research_parser = subparsers.add_parser(
        "research",
        help="Execute multi-step grounded research workflow using LangGraph.",
    )
    research_parser.add_argument("query", type=str, help="Research question or topic.")
    research_parser.add_argument(
        "--user",
        type=str,
        default="default",
        help="Local user identity namespace boundary (default: 'default').",
    )
    research_parser.add_argument(
        "--max-cycles",
        type=int,
        default=2,
        help="Maximum research cycles allowed (default: 2).",
    )

    # Eval command
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run reproducible offline benchmark evaluation on research and memory quality.",
    )
    eval_parser.add_argument(
        "--benchmark-file",
        type=str,
        default="data/benchmark_cases.json",
        help="Path to benchmark JSON cases (default: data/benchmark_cases.json).",
    )
    eval_parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output evaluation results as raw JSON.",
    )

    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    settings: Optional[Settings] = None,
    provider: Optional[LLMProvider] = None,
    store: Optional[VectorStore] = None,
) -> int:
    """CLI entry point with dependency injection support for testing."""
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = settings or Settings()

    if args.command == "index":
        return handle_index(args, cfg)
    elif args.command == "status":
        return handle_status(args, cfg)
    elif args.command == "search":
        return handle_search(args, cfg)
    elif args.command == "research":
        return handle_research(args, cfg, provider=provider, store=store)
    elif args.command == "eval":
        from asterline.eval import main as eval_main

        eval_args = []
        if getattr(args, "benchmark_file", None):
            eval_args.extend(["--benchmark-file", args.benchmark_file])
        if getattr(args, "json_output", False):
            eval_args.append("--json-output")
        return eval_main(eval_args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
