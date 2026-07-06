from app.application.ports import LLM, SearchHit, StoredChunk, VectorStore
from app.domain.chunking import chunk_text
from app.domain.embedding import embed


class RagService:
    """Orquestra a pipeline de RAG: ingestao e consulta."""

    def __init__(self, store: VectorStore, llm: LLM, chunk_size: int = 40, overlap: int = 10):
        self._store = store
        self._llm = llm
        self._chunk_size = chunk_size
        self._overlap = overlap

    def ingest(self, doc_id: str, text: str) -> int:
        pieces = chunk_text(text, self._chunk_size, self._overlap)
        chunks = [
            StoredChunk(doc_id=doc_id, chunk_index=index, text=piece, vector=embed(piece))
            for index, piece in enumerate(pieces)
        ]
        self._store.add(chunks)
        return len(chunks)

    def query(self, question: str, top_k: int = 4) -> tuple[str, list[SearchHit]]:
        hits = self._store.search(embed(question), top_k)
        context = [hit.chunk.text for hit in hits]
        answer = self._llm.generate(question, context)
        return answer, hits
