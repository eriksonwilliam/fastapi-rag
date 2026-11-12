from threading import Lock

from app.application.ports import SearchHit, StoredChunk
from app.domain.embedding import cosine


class InMemoryVectorStore:
    """Indice vetorial em memoria (busca linear por cosseno)."""

    def __init__(self) -> None:
        self._chunks: list[StoredChunk] = []
        self._lock = Lock()

    def add(self, chunks: list[StoredChunk]) -> None:
        """Replace the complete chunk set for each document in this batch atomically."""
        document_ids = {chunk.doc_id for chunk in chunks}
        with self._lock:
            self._chunks = [
                chunk for chunk in self._chunks if chunk.doc_id not in document_ids
            ] + list(chunks)

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        with self._lock:
            chunks = self._chunks
        scored = [SearchHit(chunk=chunk, score=cosine(vector, chunk.vector)) for chunk in chunks]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]
