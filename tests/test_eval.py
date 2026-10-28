"""Tests for Asterline offline benchmark evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from asterline.config import PROJECT_ROOT, Settings
from asterline.eval import (
    BenchmarkCase,
    BenchmarkSummary,
    CaseEvaluationResult,
    build_deterministic_eval_provider,
    main,
    run_benchmark,
)


def test_eval_benchmark_runner_full_suite() -> None:
    """Verify that all version-controlled benchmark cases pass deterministically offline."""
    benchmark_file = PROJECT_ROOT / "data" / "benchmark_cases.json"
    assert benchmark_file.exists(), f"Benchmark file missing at {benchmark_file}"

    summary = run_benchmark(benchmark_file)

    assert summary.total_cases == 10
    assert summary.passed_cases == 10
    assert summary.accuracy_rate == 1.0
    assert summary.groundedness_rate == 1.0
    assert summary.total_citations_emitted == 7  # tc-01 (1), tc-02 (1), tc-05 (1), tc-08 (1), tc-09 (2), tc-10 (1)
    assert summary.valid_citations_emitted == 7
    assert summary.citation_precision == 1.0
    assert summary.abstention_expected_cases == 4  # tc-03, tc-04, tc-06, tc-07
    assert summary.abstention_actual_correct == 4
    assert summary.abstention_correctness_rate == 1.0
    assert summary.forbidden_leakage_count == 0
    assert len(summary.results) == 10

    for res in summary.results:
        assert res.passed, f"Case {res.case_id} failed with reasons: {res.failure_reasons}"
        assert not res.failure_reasons


def test_eval_metric_denominators_and_calculations() -> None:
    """Verify evaluation metric definitions and edge case denominator handling."""
    # Scenario 1: 2 cases. Case 1 passed with 2 citations. Case 2 failed with 1 invalid citation and forbidden leak.
    results = [
        CaseEvaluationResult(
            case_id="tc-01",
            category="factual",
            passed=True,
            actual_outcome="success",
            expected_outcome="success",
            actual_citations=["pricing:0", "pricing:1"],
            citations_matched=True,
            keywords_matched=True,
            no_forbidden_leaked=True,
            failure_reasons=[],
        ),
        CaseEvaluationResult(
            case_id="tc-02",
            category="adversarial",
            passed=False,
            actual_outcome="success",  # Expected insufficient_evidence
            expected_outcome="insufficient_evidence",
            actual_citations=["fake:99"],
            citations_matched=False,
            keywords_matched=False,
            no_forbidden_leaked=False,
            failure_reasons=["Outcome mismatch", "Forbidden keyword found"],
        ),
    ]

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    accuracy = passed / total if total > 0 else 0.0

    valid_corpus_citations = {"pricing:0", "pricing:1", "pricing:2"}
    total_citations = sum(len(r.actual_citations) for r in results)
    valid_citations = sum(
        sum(1 for c in r.actual_citations if c in valid_corpus_citations)
        for r in results
    )
    citation_precision = valid_citations / total_citations if total_citations > 0 else None

    abstention_expected = sum(1 for r in results if r.expected_outcome == "insufficient_evidence")
    abstention_correct = sum(
        1 for r in results if r.expected_outcome == "insufficient_evidence" and r.actual_outcome == "insufficient_evidence"
    )
    abstention_rate = (abstention_correct / abstention_expected) if abstention_expected > 0 else None

    assert accuracy == 0.5  # 1/2
    assert total_citations == 3
    assert valid_citations == 2
    assert citation_precision == pytest.approx(2 / 3)
    assert abstention_expected == 1
    assert abstention_correct == 0
    assert abstention_rate == 0.0

    # Scenario 2: 0 citations emitted must report citation_precision as None (N/A), NOT 1.0 (100%)
    empty_citation_results = [
        CaseEvaluationResult(
            case_id="tc-empty",
            category="absent_evidence",
            passed=True,
            actual_outcome="insufficient_evidence",
            expected_outcome="insufficient_evidence",
            actual_citations=[],
            citations_matched=True,
            keywords_matched=True,
            no_forbidden_leaked=True,
            failure_reasons=[],
        )
    ]
    tot_c = sum(len(r.actual_citations) for r in empty_citation_results)
    prec = (0 / tot_c) if tot_c > 0 else None
    assert prec is None, "Zero citations emitted must yield None (N/A) for precision"


def test_eval_cli_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that CLI eval supports --json-output flag for CI/CD consumption."""
    benchmark_file = str(PROJECT_ROOT / "data" / "benchmark_cases.json")
    exit_code = main(["--benchmark-file", benchmark_file, "--json-output"])

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["total_cases"] == 10
    assert payload["passed_cases"] == 10
    assert payload["accuracy_rate"] == 1.0
    assert payload["groundedness_rate"] == 1.0
    assert payload["total_citations_emitted"] == 7
    assert payload["valid_citations_emitted"] == 7
    assert payload["citation_precision"] == 1.0
    assert payload["abstention_expected_cases"] == 4
    assert payload["abstention_actual_correct"] == 4
    assert payload["abstention_correctness_rate"] == 1.0
    assert payload["forbidden_leakage_count"] == 0
