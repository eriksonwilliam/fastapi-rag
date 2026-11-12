"""Evaluate document retrieval on a small, reproducible PT/EN fixture."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import default_service  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path(__file__).with_name("retrieval_cases.json")
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not dataset["questions"]:
        parser.error("dataset must contain at least one question")
    document_ids = {document["id"] for document in dataset["documents"]}
    for case in dataset["questions"]:
        if not case["expected_documents"] or not set(case["expected_documents"]) <= document_ids:
            parser.error("each question must reference at least one existing document")
    service = default_service()
    for document in dataset["documents"]:
        service.ingest(document["id"], document["text"])
    results = []
    for case in dataset["questions"]:
        hits = service.retrieve(case["question"], args.top_k)
        retrieved = list(dict.fromkeys(hit.chunk.doc_id for hit in hits))
        relevant = set(case["expected_documents"])
        matches = relevant.intersection(retrieved)
        reciprocal_rank = next(
            (1 / rank for rank, doc_id in enumerate(retrieved, 1) if doc_id in relevant), 0.0
        )
        results.append(
            {
                "question": case["question"],
                "retrieved_documents": retrieved,
                "recall_at_k": len(matches) / len(relevant),
                "reciprocal_rank": reciprocal_rank,
            }
        )
    count = len(results)
    print(
        json.dumps(
            {
                "scope": "document retrieval on a small PT/EN fixture",
                "top_k_chunks": args.top_k,
                "questions": count,
                "mean_recall_at_k": sum(r["recall_at_k"] for r in results) / count,
                "mean_reciprocal_rank": sum(r["reciprocal_rank"] for r in results) / count,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
