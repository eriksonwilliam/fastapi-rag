from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StoredChunk:
    doc_id: str
    chunk_index: int
    text: str
    vector: list[float]


@dataclass(frozen=True)
class SearchHit:
    chunk: StoredChunk
    score: float


class VectorStore(Protocol):
    """Port de saida: indice vetorial."""

    def add(self, chunks: list[StoredChunk]) -> None: ...

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]: ...


class LLM(Protocol):
    """Port de saida: geracao de texto a partir de uma pergunta + contexto."""

    def generate(self, question: str, context: list[str]) -> str: ...
