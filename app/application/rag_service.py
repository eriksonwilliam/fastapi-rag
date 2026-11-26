from dataclasses import dataclass

from app.application.ports import LLM, Embedder, SearchHit, StoredChunk, VectorStore
from app.domain.chunking import chunk_text

ABSTENTION_ANSWER = "Nao ha evidencia suficiente nos documentos para responder."


@dataclass(frozen=True)
class QueryResult:
    answer: str
    hits: list[SearchHit]
    abstained: bool


class RagService:
    """Orquestra a pipeline de RAG: ingestao e consulta."""

    def __init__(
        self,
        store: VectorStore,
        llm: LLM,
        embedder: Embedder,
        chunk_size: int = 40,
        overlap: int = 10,
        min_score: float = 0.0,
    ):
        self._store = store
        self._llm = llm
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._min_score = min_score

    def ingest(self, doc_id: str, text: str) -> int:
        pieces = chunk_text(text, self._chunk_size, self._overlap)
        chunks = [
            StoredChunk(
                doc_id=doc_id, chunk_index=index, text=piece, vector=self._embedder.embed(piece)
            )
            for index, piece in enumerate(pieces)
        ]
        self._store.add(chunks)
        return len(chunks)

    def retrieve(self, question: str, top_k: int = 4) -> list[SearchHit]:
        """Retrieve context independently of generation, including for evaluation.

        Hits at or below `min_score` are dropped: a score that low is treated as
        absence of evidence, not as weak evidence.
        """
        hits = self._store.search(self._embedder.embed(question), top_k)
        return [hit for hit in hits if hit.score > self._min_score]

    def query(self, question: str, top_k: int = 4) -> QueryResult:
        hits = self.retrieve(question, top_k)
        if not hits:
            # Sem evidencia o LLM nao e chamado: um modelo com contexto vazio
            # tende a responder do proprio conhecimento, que e exatamente o
            # que um RAG precisa evitar.
            return QueryResult(answer=ABSTENTION_ANSWER, hits=[], abstained=True)
        answer = self._llm.generate(question, [hit.chunk.text for hit in hits])
        return QueryResult(answer=answer, hits=hits, abstained=False)
