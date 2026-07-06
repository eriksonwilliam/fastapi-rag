from app.application.rag_service import RagService
from app.infrastructure.hashing_embedder import HashingEmbedder
from app.infrastructure.llm_stub import StubLLM
from app.infrastructure.memory_store import InMemoryVectorStore


def _service() -> RagService:
    return RagService(
        InMemoryVectorStore(), StubLLM(), HashingEmbedder(), chunk_size=40, overlap=10
    )


def test_ingest_returns_chunk_count():
    service = _service()
    assert service.ingest("doc1", " ".join(str(i) for i in range(100))) >= 2


def test_query_after_ingest_returns_answer_and_hits():
    service = _service()
    service.ingest("doc1", "python e uma linguagem de programacao muito popular")

    answer, hits = service.query("linguagem de programacao", top_k=3)

    assert hits
    assert answer


def test_query_without_documents_has_no_context():
    answer, hits = _service().query("qualquer coisa", top_k=3)
    assert hits == []
    assert "Nao encontrei" in answer
