"""CLI regression: negatives do not inflate recall and configuration is explicit."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _evaluate(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_retrieval.py", "--top-k", "3", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_extended_evaluation_separates_negatives():
    report = _evaluate("--dataset", "scripts/retrieval_cases_extended.json")
    assert report["documents"] == 12
    assert report["answerable_questions"] == 14
    assert report["negative_questions"] == 4
    assert report["embedder"] == "hashing"
    assert all(r["recall_at_k"] is None for r in report["results"] if not r["answerable"])
    assert all(r["precision_at_k"] is None for r in report["results"] if not r["answerable"])


def test_hashing_baseline_is_stable_at_cutoff_zero():
    """Same numbers as scripts/retrieval_baseline_extended.json: a regression guard."""
    report = _evaluate("--dataset", "scripts/retrieval_cases_extended.json", "--summary")
    assert round(report["mean_recall_at_k"], 4) == 0.8571
    assert round(report["mean_reciprocal_rank"], 4) == 0.8095
    assert report["negative_context_rate"] == 1.0
    assert "results" not in report


def test_calibration_trades_recall_for_abstention_on_negatives():
    baseline = _evaluate("--dataset", "scripts/retrieval_cases_extended.json", "--summary")
    calibrated = _evaluate(
        "--dataset", "scripts/retrieval_cases_extended.json", "--calibrate", "--summary"
    )
    assert calibrated["calibrated_on_this_dataset"] is True
    assert calibrated["min_score_exclusive"] > 0.0
    assert calibrated["negative_context_rate"] < baseline["negative_context_rate"]
    assert calibrated["mean_precision_at_k"] >= baseline["mean_precision_at_k"]


def test_heldout_dataset_reports_categories():
    report = _evaluate("--dataset", "scripts/retrieval_cases_heldout.json", "--summary")
    assert report["questions"] == 16
    assert report["negative_questions"] == 4
    assert set(report["by_category"]) == {
        "cross_language",
        "answerable",
        "partial",
        "ambiguous",
        "out_of_domain",
        "insufficient_context",
    }
