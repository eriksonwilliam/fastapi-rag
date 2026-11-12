"""CLI regression: negatives do not inflate recall and configuration is explicit."""

import json
import subprocess
import sys
from pathlib import Path


def test_extended_evaluation_separates_negatives():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_retrieval.py",
            "--dataset",
            "scripts/retrieval_cases_extended.json",
            "--top-k",
            "3",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["documents"] == 12
    assert report["answerable_questions"] == 14
    assert report["negative_questions"] == 4
    assert report["embedder"] == "hashing"
    assert all(r["recall_at_k"] is None for r in report["results"] if not r["answerable"])
