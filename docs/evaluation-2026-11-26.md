# Retrieval evaluation, 2026-11-26

First run against real embedding models, plus a held-out set and an
abstention cutoff chosen on a separate development set.

## Setup

| Item | Value |
| --- | --- |
| Development set | `scripts/retrieval_cases_extended.json`: 12 documents, 14 answerable + 4 negative questions |
| Held-out set | `scripts/retrieval_cases_heldout.json`: same 12 documents, 16 new questions written after the development results |
| Held-out categories | 5 cross-language, 4 answerable, 2 partial, 1 ambiguous, 2 out-of-domain, 2 insufficient-context |
| Top-k | 3 chunks |
| Cutoff | chosen with `--calibrate` on the development set only, then applied unchanged to held-out |
| Calibration objective | `mean recall@3 + (1 − negative context rate)`, ties to the lowest cutoff |
| Runtime | Python 3.14.10, fastembed 0.9.1, onnxruntime 1.32.0, Windows 11, CPU |
| Raw output | `scripts/results/2026-11-26/*.json` (one file per model × set × cutoff) |

Metrics are document-level after deduplicating chunks. Recall, precision and
MRR are computed over answerable questions only. `negative_context_rate` is
the share of negatives that still retrieve any chunk above the cutoff.
`answerable_abstention_rate` is the share of answerable questions that
retrieve nothing above it.

## Results

### Development set

| Embedder | Cutoff | R@3 | P@3 | MRR | Answerable abstained | Negatives with context |
| --- | --- | --- | --- | --- | --- | --- |
| hashing (offline) | 0 | 0.857 | 0.286 | 0.810 | 0% | 100% |
| hashing (offline) | 0.436 (calibrated) | 0.714 | 0.631 | 0.714 | 29% | 25% |
| `BAAI/bge-small-en-v1.5` | 0 | 1.000 | 0.333 | 0.917 | 0% | 100% |
| `BAAI/bge-small-en-v1.5` | 0.654 (calibrated) | 0.929 | 0.440 | 0.893 | 7% | 0% |
| `paraphrase-multilingual-MiniLM-L12-v2` | 0 | 1.000 | 0.333 | 0.964 | 0% | 100% |
| `paraphrase-multilingual-MiniLM-L12-v2` | 0.444 (calibrated) | 0.929 | 0.738 | 0.929 | 0% | 0% |

### Held-out set, development cutoff applied

| Embedder | Cutoff | R@3 | P@3 | MRR | Answerable abstained | Negatives with context |
| --- | --- | --- | --- | --- | --- | --- |
| hashing (offline) | 0 | 0.500 | 0.167 | 0.500 | 0% | 100% |
| hashing (offline) | 0.436 | 0.250 | 0.194 | 0.250 | 58% | 25% |
| `BAAI/bge-small-en-v1.5` | 0 | 0.778 | 0.278 | 0.750 | 0% | 100% |
| `BAAI/bge-small-en-v1.5` | 0.654 | 0.694 | 0.556 | 0.667 | 8% | 25% |
| `paraphrase-multilingual-MiniLM-L12-v2` | 0 | 0.972 | 0.361 | 0.944 | 0% | 100% |
| `paraphrase-multilingual-MiniLM-L12-v2` | 0.444 | 0.861 | 0.750 | 0.917 | 8% | 0% |

## Reading

- **Without a cutoff, every embedder retrieves context for every negative.**
  Top-k always returns something; the score has to decide whether it counts.
- **The hashing cutoff does not transfer.** It looked reasonable on the
  development set and then abstained on all 5 cross-language held-out
  questions. Hashing is lexical; a Portuguese question shares no tokens with
  an English document.
- **The multilingual model is the only one that held up on held-out**: all 5
  cross-language questions answered, all 4 negatives abstained, precision
  doubled against cutoff 0.
- **The margins are thin.** With MiniLM the lowest answerable held-out
  question scored 0.519 and the highest insufficient-context one 0.387,
  against a cutoff of 0.444. One more question could cross either side.
- **Insufficient-context questions are not a retrieval problem in general.**
  "Which Kafka partition key does the outbox use?" shares vocabulary with the
  outbox document. Here the score happened to fall below the cutoff;
  catching this reliably needs an answer-level check against the retrieved
  text, which this project does not have yet.
- One "partial" question (how many retries a payment service should attempt)
  abstained with every model. That is arguably the right outcome: the corpus
  describes retries but gives no number.

## What this does not show

- 16 held-out questions and 12 one-sentence documents. These are diagnostic
  fixtures, not a benchmark; no claim about production accuracy follows.
- Generated answers were not evaluated. The stub generator was used; no LLM
  ran. Faithfulness and citation quality remain unmeasured.
- No conflicting-documents case: the corpus has no contradictions to test.
- Scores are model-specific. A cutoff calibrated for one embedder means
  nothing for another; `RAG_MIN_SCORE` has to be recalibrated when
  `EMBEDDER` changes.

## Reproduce

```bash
pip install -r requirements-real.txt   # downloads ONNX models on first run
for model in BAAI/bge-small-en-v1.5 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2; do
  python scripts/evaluate_retrieval.py --dataset scripts/retrieval_cases_extended.json \
    --embedder fastembed --model "$model" --calibrate --summary
  # use the printed min_score_exclusive below
  python scripts/evaluate_retrieval.py --dataset scripts/retrieval_cases_heldout.json \
    --embedder fastembed --model "$model" --min-score <cutoff> --summary
done
```
