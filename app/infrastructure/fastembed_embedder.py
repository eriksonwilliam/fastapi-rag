"""Embedder semantico real via fastembed (ONNX, sem torch).

Requer `pip install fastembed`; baixa um modelo pequeno em runtime. Fora da
cobertura unitaria (dependencia opcional + download), como os demais adapters.
"""

import math

from fastembed import TextEmbedding


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model = TextEmbedding(model_name=model_name)

    def embed(self, text: str) -> list[float]:
        vector = [float(value) for value in next(iter(self._model.embed([text])))]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
