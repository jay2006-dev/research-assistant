"""Tests for the Asterline CLI (tests/test_cli.py).

Covers:
- Argument parsing for index, status, search, and invalid commands.
- Uninitialized index behavior and error handling.
- Input validation (empty query).
- Idempotent indexing, rebuilds, and status output.
- Cross-process persistence test using isolated subprocess execution with FakeEmbeddingFunction.
- Complete isolation to pytest tmp_path fixtures.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import asterline.store
from asterline.cli import main
from asterline.config import Settings
from asterline.embeddings import FakeEmbeddingFunction


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Isolate CLI storage paths to tmp_path and patch default embedder with FakeEmbeddingFunction."""
    chroma_dir = tmp_path / "chroma"
    pointer_file = tmp_path / "active_pointer.json"
    lock_file = tmp_path / "build.lock"

    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(chroma_dir))
    monkeypatch.setenv("ACTIVE_POINTER_FILE", str(pointer_file))
    monkeypatch.setenv("BUILD_LOCK_FILE", str(lock_file))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "cli_test_collection")

    # Intercept VectorStore instantiation to supply FakeEmbeddingFunction if not provided
    orig_init = asterline.store.VectorStore.__init__

    def patched_init(self, settings=None, embedder=None):
        orig_init(self, settings=settings, embedder=embedder or FakeEmbeddingFunction())

    monkeypatch.setattr(asterline.store.VectorStore, "__init__", patched_init)
    return tmp_path


def test_cli_uninitialized_status(cli_env, capsys):
    """'status' on an uninitialized index reports Uninitialized and exits with code 1."""
    code = main(["status"])
    captured = capsys.readouterr()

    assert code == 1
    assert "Status: Uninitialized" in captured.out


def test_cli_uninitialized_search(cli_env, capsys):
    """'search' on an uninitialized index fails with clear diagnostic and exits with code 1."""
    code = main(["search", "pricing options"])
    captured = capsys.readouterr()

    assert code == 1
    assert "Search failed: Index has not been initialized" in captured.err


def test_cli_invalid_arguments():
    """Missing or invalid subcommands exit with standard argparse error code."""
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 2

    with pytest.raises(SystemExit) as exc_info:
        main(["unknown-cmd"])
    assert exc_info.value.code == 2


def test_cli_empty_query_rejected(cli_env, capsys):
    """'search' with whitespace or empty query text fails with validation error."""
    code = main(["search", "   "])
    captured = capsys.readouterr()

    assert code == 1
    assert "Search failed: Query text cannot be empty or whitespace" in captured.err


def test_cli_index_and_status(cli_env, capsys):
    """'index' creates an active index and 'status' reports Active with 75 documents."""
    # Step 1: Run index
    code_index = main(["index"])
    captured_index = capsys.readouterr()

    assert code_index == 0
    assert "Corpus parsed: 75 valid chunks." in captured_index.out
    assert "Pagination artifacts removed: 18." in captured_index.out
    assert "[SUCCESS] Indexing complete." in captured_index.out
    assert "Indexed Chunks:    75" in captured_index.out

    # Step 2: Run status
    code_status = main(["status"])
    captured_status = capsys.readouterr()

    assert code_status == 0
    assert "Status: Active" in captured_status.out
    assert "Document Count:    75" in captured_status.out
    assert "Distance Metric:   cosine" in captured_status.out


def test_cli_index_rebuild(cli_env, capsys):
    """'index --rebuild' creates a fresh versioned collection and activates it."""
    code_initial = main(["index"])
    assert code_initial == 0

    code_rebuild = main(["index", "--rebuild"])
    captured_rebuild = capsys.readouterr()

    assert code_rebuild == 0
    assert "Asterline Indexer — Rebuild" in captured_rebuild.out
    assert "[SUCCESS] Indexing complete." in captured_rebuild.out
    assert "Indexed Chunks:    75" in captured_rebuild.out


def test_cli_search_output(cli_env, capsys):
    """'search' retrieves results ranked by ascending distance and formats snippet."""
    code_index = main(["index"])
    assert code_index == 0
    capsys.readouterr()  # clear buffer

    code_search = main(["search", "Company overview", "--top-k", "2"])
    captured = capsys.readouterr()

    assert code_search == 0
    assert "Retrieved 2 results (ranked by ascending cosine distance):" in captured.out
    assert "#1 [" in captured.out
    assert "#2 [" in captured.out
    assert "(distance:" in captured.out
    assert "Category:" in captured.out
    assert "Source:" in captured.out
    assert "Published:" in captured.out


def test_cli_cross_process_persistence(tmp_path):
    """Prove that an index created by one CLI process can be reopened and searched by a separate OS process.

    Uses sys.executable to spawn two independent OS processes with separate PIDs:
    Process 1: Runs 'asterline index' in an isolated subprocess.
    Process 2: Runs 'asterline search' in an independent second subprocess.
    Both processes point to the same tmp_path storage and use deterministic FakeEmbeddingFunction.
    """
    chroma_dir = tmp_path / "chroma"
    pointer_file = tmp_path / "active_pointer.json"
    lock_file = tmp_path / "build.lock"

    # Python runner script that configures environment, injects FakeEmbeddingFunction, and runs main()
    runner_script = (
        "import sys\n"
        "from unittest.mock import patch\n"
        "import asterline.store\n"
        "from asterline.embeddings import FakeEmbeddingFunction\n"
        "from asterline.cli import main\n"
        "\n"
        "orig_init = asterline.store.VectorStore.__init__\n"
        "def patched_init(self, settings=None, embedder=None):\n"
        "    orig_init(self, settings=settings, embedder=embedder or FakeEmbeddingFunction())\n"
        "asterline.store.VectorStore.__init__ = patched_init\n"
        "\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )

    env = os.environ.copy()
    env["CHROMA_PERSIST_DIR"] = str(chroma_dir)
    env["ACTIVE_POINTER_FILE"] = str(pointer_file)
    env["BUILD_LOCK_FILE"] = str(lock_file)
    env["CHROMA_COLLECTION_NAME"] = "subprocess_kb"
    # Ensure src is on PYTHONPATH
    project_root = Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = str(project_root / "src")

    # ── Process 1: Build Index ──────────────────────────────────────
    proc_1 = subprocess.run(
        [sys.executable, "-c", runner_script, "index"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(project_root),
    )
    assert proc_1.returncode == 0, f"Process 1 (index) failed: {proc_1.stderr}"
    assert "[SUCCESS] Indexing complete." in proc_1.stdout
    assert pointer_file.is_file(), "Active pointer file must exist after process 1 completes"

    # ── Process 2: Search Persisted Index from Fresh Process ─────────
    proc_2 = subprocess.run(
        [sys.executable, "-c", runner_script, "search", "Company overview", "--top-k", "3"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(project_root),
    )
    assert proc_2.returncode == 0, f"Process 2 (search) failed: {proc_2.stderr}"
    assert "Retrieved 3 results (ranked by ascending cosine distance):" in proc_2.stdout
    assert "#1 [" in proc_2.stdout
    assert "#2 [" in proc_2.stdout
    assert "#3 [" in proc_2.stdout
