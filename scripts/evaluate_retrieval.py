"""Evaluate document retrieval on a small, reproducible PT/EN fixture."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.application.rag_service import RagService  # noqa: E402
from app.infrastructure.hashing_embedder import HashingEmbedder  # noqa: E402
from app.infrastructure.llm_stub import StubLLM  # noqa: E402
from app.infrastructure.memory_store import InMemoryVectorStore  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path(__file__).with_name("retrieval_cases.json")
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embedder", choices=["hashing", "fastembed"], default="hashing")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Fixed retrieval cutoff; calibrate on a separate development set",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not dataset["questions"]:
        parser.error("dataset must contain at least one question")
    document_ids = {document["id"] for document in dataset["documents"]}
    for case in dataset["questions"]:
        if not set(case["expected_documents"]) <= document_ids:
            parser.error(
                "expected_documents must reference existing documents or be empty for negatives"
            )
    embedder = HashingEmbedder()
    if args.embedder == "fastembed":
        from app.infrastructure.fastembed_embedder import FastEmbedEmbedder

        embedder = FastEmbedEmbedder(args.model)
    service = RagService(InMemoryVectorStore(), StubLLM(), embedder)
    for document in dataset["documents"]:
        service.ingest(document["id"], document["text"])
    results = []
    for case in dataset["questions"]:
        hits = [
            h for h in service.retrieve(case["question"], args.top_k) if h.score > args.min_score
        ]
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
                "answerable": bool(relevant),
                "recall_at_k": len(matches) / len(relevant) if relevant else None,
                "negative_has_context": bool(retrieved) if not relevant else None,
                "reciprocal_rank": reciprocal_rank,
            }
        )
    count = len(results)
    positives = [r for r in results if r["answerable"]]
    negatives = [r for r in results if not r["answerable"]]
    print(
        json.dumps(
            {
                "scope": "document retrieval on a small PT/EN fixture",
                "top_k_chunks": args.top_k,
                "embedder": args.embedder,
                "model": args.model if args.embedder == "fastembed" else "offline-hashing",
                "min_score_exclusive": args.min_score,
                "documents": len(document_ids),
                "answerable_questions": len(positives),
                "negative_questions": len(negatives),
                "negative_context_rate": (
                    sum(r["negative_has_context"] for r in negatives) / len(negatives)
                )
                if negatives
                else None,
                "questions": count,
                "mean_recall_at_k": sum(r["recall_at_k"] for r in positives) / len(positives)
                if positives
                else None,
                "mean_reciprocal_rank": sum(r["reciprocal_rank"] for r in positives)
                / len(positives)
                if positives
                else None,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
