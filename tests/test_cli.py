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



def test_cli_research_uninitialized_index(cli_env, capsys):
    """'research' on an uninitialized index reports error and exits with code 1."""
    code = main(["research", "What is the Asterline retention policy?"])
    captured = capsys.readouterr()

    assert code == 1
    assert "Vector index is not initialized" in captured.err
    assert "Run 'python -m asterline.cli index' to build the index first." in captured.err


def test_cli_research_empty_query_rejected(cli_env, capsys):
    """'research' with empty or whitespace query fails with validation error and exits with code 1."""
    code = main(["research", "   "])
    captured = capsys.readouterr()

    assert code == 1
    assert "Research query cannot be empty or whitespace" in captured.err


def test_cli_research_missing_api_key(cli_env, capsys):
    """'research' without configured API key reports guidance and exits with code 1."""
    # Build index first so pre-flight passes
    assert main(["index"]) == 0
    capsys.readouterr()

    # Settings with empty api_key and no injected provider
    cfg = Settings(_env_file=None, api_key="")
    code = main(["research", "What is the Launch plan cost?"], settings=cfg, provider=None)
    captured = capsys.readouterr()

    assert code == 1
    assert "Gemini API key is not configured" in captured.err
    assert "Set GEMINI_API_KEY or API_KEY in your environment or .env file" in captured.err


def test_cli_research_successful_lifecycle(cli_env, capsys):
    """'research' with an active index and fake provider outputs verified findings and citations."""
    from asterline.providers import FakeLLMProvider, FakeLLMResponse

    # 1. Build index
    assert main(["index"]) == 0
    capsys.readouterr()

    # 2. Configure FakeLLMProvider for complete lifecycle
    coordinator_json = '{"research_plan": "Search for Launch plan costs", "sub_queries": ["pricing"]}'
    # Valid corpus chunk title & content: pricing:0 has Launch plan details
    synthesis_json = (
        '{\n'
        '  "claims": [\n'
        '    {\n'
        '      "claim_text": "Launch costs $59 per month and includes 4 seats.",\n'
        '      "citations": [\n'
        '        {\n'
        '          "evidence_id": "pricing:0",\n'
        '          "chunk_id": "pricing:0",\n'
        '          "snippet": "Launch costs $59 per month and includes 4 seats"\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    verifier_json = '{"status": "supported", "note": "Fully verified in corpus", "has_conflict": false}'

    fake_provider = FakeLLMProvider(
        responses=[
            FakeLLMResponse(text=coordinator_json),  # Coordinator
            FakeLLMResponse(text="0: relevant\n"),   # Researcher relevance filter
            FakeLLMResponse(text=synthesis_json),    # Synthesizer
            FakeLLMResponse(text=verifier_json),     # Verifier
        ]
    )

    code = main(["research", "How much does the Launch plan cost?"], provider=fake_provider)
    captured = capsys.readouterr()

    assert code == 0
    assert "Asterline Research Assistant — Research Findings" in captured.out
    assert "Outcome:     SUCCESS" in captured.out
    assert "[VERIFIED ANSWER]" in captured.out
    assert "Launch costs $59 per month and includes 4 seats." in captured.out
    assert "[SUPPORTED FINDINGS & CITATIONS]" in captured.out
    assert "pricing:0" in captured.out
    assert "Launch costs $59 per month and includes 4 seats" in captured.out


def test_cli_research_insufficient_evidence_output(cli_env, capsys):
    """'research' when claims are rejected formats honest INSUFFICIENT_EVIDENCE report."""
    from asterline.providers import FakeLLMProvider, FakeLLMResponse

    assert main(["index"]) == 0
    capsys.readouterr()

    # Verifier rejects all claims
    coordinator_json = '{"research_plan": "Search plan", "sub_queries": ["unknown topic"]}'
    synthesis_json = (
        '{\n'
        '  "claims": [\n'
        '    {\n'
        '      "claim_text": "Unverified assertion about space travel.",\n'
        '      "citations": [\n'
        '        {\n'
        '          "evidence_id": "pricing:0",\n'
        '          "snippet": "Launch costs $59 per month and includes 4 seats"\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    verifier_reject_json = '{"status": "unsupported", "note": "No support in evidence", "has_conflict": false}'

    fake_provider = FakeLLMProvider(
        responses=[
            FakeLLMResponse(text=coordinator_json),
            FakeLLMResponse(text="0: relevant\n"),
            FakeLLMResponse(text=synthesis_json),
            FakeLLMResponse(text=verifier_reject_json),  # Cycle 1
            FakeLLMResponse(text=coordinator_json),
            FakeLLMResponse(text="0: relevant\n"),
            FakeLLMResponse(text=synthesis_json),
            FakeLLMResponse(text=verifier_reject_json),  # Cycle 2
        ]
    )

    code = main(["research", "What are the space exploration rules?"], provider=fake_provider)
    captured = capsys.readouterr()

    assert code == 0
    assert "Outcome:     INSUFFICIENT_EVIDENCE" in captured.out
    assert "[INSUFFICIENT EVIDENCE]" in captured.out


def test_cli_research_provider_error_handling_and_redaction(cli_env, capsys):
    """'research' redacts secrets and exits with code 1 when provider fails."""
    from asterline.providers import FakeLLMProvider, ProviderError

    assert main(["index"]) == 0
    capsys.readouterr()

    class FailingProvider(FakeLLMProvider):
        def invoke(self, prompt: str) -> str:
            raise ProviderError("API call failed for key AIzaSyDfakeSecretKey123456789012345")

    cfg = Settings(_env_file=None, api_key="AIzaSyDfakeSecretKey123456789012345")
    code = main(
        ["research", "General query"],
        settings=cfg,
        provider=FailingProvider(),
    )
    captured = capsys.readouterr()

    assert code == 1
    # Check that error is indicated in output/stderr
    combined_output = captured.out + captured.err
    assert "WORKFLOW FAILURE" in combined_output or "[ERROR]" in combined_output
    # Invariant: Secret key must be redacted, never logged in raw form
    assert "AIzaSyDfakeSecretKey123456789012345" not in combined_output
    assert "[REDACTED]" in combined_output


def test_cli_research_workflow_deadline_enforcement(cli_env, capsys):
    """'research' halts cleanly and safely when workflow deadline expires without inventing answers."""
    from asterline.providers import FakeLLMProvider, FakeLLMResponse

    assert main(["index"]) == 0
    capsys.readouterr()

    # Provider that sleeps to simulate long operation
    class SlowProvider(FakeLLMProvider):
        def invoke(self, prompt: str, *, deadline=None) -> str:
            # Check deadline or simulate exceeding deadline
            import time
            if deadline is not None and time.time() >= deadline:
                from asterline.providers import TimeoutError as AsterlineTimeoutError
                raise AsterlineTimeoutError("Overall workflow execution deadline exceeded.")
            return '{"research_plan": "plan", "sub_queries": ["query"]}'

    # Workflow timeout is ultra-short (0.001s)
    cfg = Settings(_env_file=None, api_key="test-key", workflow_timeout=0.001)
    code = main(
        ["research", "How much does Launch cost?"],
        settings=cfg,
        provider=SlowProvider(),
    )
    captured = capsys.readouterr()

    # Must fail or return insufficient evidence, never fabricate a successful answer
    assert "Launch costs $59" not in captured.out
    assert "WORKFLOW FAILURE" in captured.out or "[ERROR]" in captured.err or "INSUFFICIENT" in captured.out


def test_cli_research_groq_failure_never_falls_back_to_gemini(cli_env, capsys, monkeypatch):
    """'research' with Groq fails with exit code 1 and never invokes or falls back to Gemini."""
    from asterline.providers import SafeGeminiProvider, SafeGroqProvider, ProviderError

    assert main(["index"]) == 0
    capsys.readouterr()

    gemini_invocations = []
    original_gemini_init = SafeGeminiProvider.__init__

    def tracked_gemini_init(self, *args, **kwargs):
        gemini_invocations.append(True)
        return original_gemini_init(self, *args, **kwargs)

    monkeypatch.setattr(SafeGeminiProvider, "__init__", tracked_gemini_init)

    # Mock SafeGroqProvider._call_llm or invoke to raise an auth or provider error
    def failing_invoke(self, prompt, *, deadline=None):
        raise ProviderError("Groq 401 Unauthorized: Invalid API key")

    monkeypatch.setattr(SafeGroqProvider, "invoke", failing_invoke)

    cfg = Settings(
        _env_file=None,
        model_provider="groq",
        groq_api_key="gsk-cli-isolation-key-12345",
        api_key="gemini-key-should-never-be-touched",
    )
    code = main(
        ["research", "How much does Launch cost?"],
        settings=cfg,
    )
    captured = capsys.readouterr()

    # 1. Exit code must be 1 on provider failure
    assert code == 1
    # 2. Output must indicate workflow or execution failure
    combined = captured.out + captured.err
    assert "Groq 401 Unauthorized" in combined or "[ERROR]" in combined
    # 3. Gemini must never have been instantiated or invoked as fallback
    assert len(gemini_invocations) == 0
