"""Command-line interface for Asterline knowledge base indexing and status.

Usage:
    python -m asterline.cli index [--rebuild]
    python -m asterline.cli status
    python -m asterline.cli search "query string" [--top-k N]
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from asterline.config import Settings
from asterline.corpus import parse_corpus
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

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings()

    if args.command == "index":
        return handle_index(args, settings)
    elif args.command == "status":
        return handle_status(args, settings)
    elif args.command == "search":
        return handle_search(args, settings)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
