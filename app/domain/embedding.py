"""Embedding deterministico e sem dependencias (hashing bag-of-words).

Nao e um modelo semantico de ponta; e leve, estavel e testavel — suficiente
para demonstrar a pipeline de RAG. Troque por um modelo real via um adapter.
"""

import hashlib
import math

DIM = 64


def embed(text: str) -> list[float]:
    """Vetor L2-normalizado de dimensao DIM."""
    vector = [0.0] * DIM
    for token in _tokenize(text):
        bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % DIM
        vector[bucket] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def cosine(a: list[float], b: list[float]) -> float:
    """Similaridade de cosseno (assume vetores ja normalizados)."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]
