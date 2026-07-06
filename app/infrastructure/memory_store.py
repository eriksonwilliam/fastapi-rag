from app.application.ports import SearchHit, StoredChunk
from app.domain.embedding import cosine


class InMemoryVectorStore:
    """Indice vetorial em memoria (busca linear por cosseno)."""

    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []

    def add(self, chunks: list[StoredChunk]) -> None:
        self._chunks.extend(chunks)

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        scored = [
            SearchHit(chunk=chunk, score=cosine(vector, chunk.vector)) for chunk in self._chunks
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]
