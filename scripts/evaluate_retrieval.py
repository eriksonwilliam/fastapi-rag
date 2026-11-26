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


def build_embedder(name, model):
    if name == "fastembed":
        from app.infrastructure.fastembed_embedder import FastEmbedEmbedder

        return FastEmbedEmbedder(model)
    return HashingEmbedder()


def scored_retrievals(dataset, embedder, top_k):
    """Run retrieval once with no cutoff; cutoffs are applied afterwards.

    Embedding is the expensive step with a real model, and calibration needs to
    try many cutoffs over the same retrieval.
    """
    service = RagService(InMemoryVectorStore(), StubLLM(), embedder, min_score=float("-inf"))
    for document in dataset["documents"]:
        service.ingest(document["id"], document["text"])
    return [
        [(hit.chunk.doc_id, hit.score) for hit in service.retrieve(case["question"], top_k)]
        for case in dataset["questions"]
    ]


def evaluate(dataset, retrievals, min_score):
    results = []
    for case, scored in zip(dataset["questions"], retrievals, strict=True):
        retrieved = list(dict.fromkeys(doc_id for doc_id, score in scored if score > min_score))
        relevant = set(case["expected_documents"])
        matches = relevant.intersection(retrieved)
        reciprocal_rank = next(
            (1 / rank for rank, doc_id in enumerate(retrieved, 1) if doc_id in relevant), 0.0
        )
        results.append(
            {
                "question": case["question"],
                "category": case.get("category", "answerable" if relevant else "negative"),
                "retrieved_documents": retrieved,
                "top_score": round(max((s for _, s in scored), default=0.0), 4),
                "answerable": bool(relevant),
                "abstained": not retrieved,
                "recall_at_k": len(matches) / len(relevant) if relevant else None,
                "precision_at_k": (len(matches) / len(retrieved) if retrieved else 0.0)
                if relevant
                else None,
                "negative_has_context": bool(retrieved) if not relevant else None,
                "reciprocal_rank": reciprocal_rank,
            }
        )
    positives = [r for r in results if r["answerable"]]
    negatives = [r for r in results if not r["answerable"]]

    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else None

    by_category = {}
    for result in results:
        bucket = by_category.setdefault(result["category"], {"questions": 0, "abstained": 0})
        bucket["questions"] += 1
        bucket["abstained"] += int(result["abstained"])
    return {
        "min_score_exclusive": min_score,
        "answerable_questions": len(positives),
        "negative_questions": len(negatives),
        "mean_recall_at_k": mean(r["recall_at_k"] for r in positives),
        "mean_precision_at_k": mean(r["precision_at_k"] for r in positives),
        "mean_reciprocal_rank": mean(r["reciprocal_rank"] for r in positives),
        "answerable_abstention_rate": mean(r["abstained"] for r in positives),
        "negative_context_rate": mean(r["negative_has_context"] for r in negatives),
        "by_category": by_category,
        "results": results,
    }


def calibrate(dataset, retrievals):
    """Pick the cutoff that maximizes recall on answerables + abstention on negatives.

    Candidates are the observed scores themselves (a cutoff between two observed
    scores behaves like the lower one). Ties go to the lowest cutoff, which
    abstains least on answerable questions.
    """
    candidates = sorted({0.0, *(score for scored in retrievals for _, score in scored)})
    best = None
    for cutoff in candidates:
        report = evaluate(dataset, retrievals, cutoff)
        objective = (report["mean_recall_at_k"] or 0.0) + (
            1 - (report["negative_context_rate"] or 0.0)
        )
        if best is None or objective > best[0]:
            best = (objective, cutoff)
    return best[1]


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
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Choose the cutoff on this dataset (use only the development set) and report it",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Omit per-question results from the output"
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
    retrievals = scored_retrievals(dataset, build_embedder(args.embedder, args.model), args.top_k)
    min_score = calibrate(dataset, retrievals) if args.calibrate else args.min_score
    report = evaluate(dataset, retrievals, min_score)
    if args.summary:
        report.pop("results")
    print(
        json.dumps(
            {
                "scope": "document retrieval on a small PT/EN fixture",
                "dataset": args.dataset.name,
                "top_k_chunks": args.top_k,
                "embedder": args.embedder,
                "model": args.model if args.embedder == "fastembed" else "offline-hashing",
                "calibrated_on_this_dataset": args.calibrate,
                "documents": len(document_ids),
                "questions": len(dataset["questions"]),
                **report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
