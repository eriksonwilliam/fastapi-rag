from app.application.rag_service import ABSTENTION_ANSWER, RagService
from app.infrastructure.hashing_embedder import HashingEmbedder
from app.infrastructure.llm_stub import StubLLM
from app.infrastructure.memory_store import InMemoryVectorStore


class SpyLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def generate(self, question: str, context: list[str]) -> str:
        self.calls.append((question, context))
        return "resposta"


def _service(llm=None, min_score: float = 0.0) -> RagService:
    return RagService(
        InMemoryVectorStore(),
        llm or StubLLM(),
        HashingEmbedder(),
        chunk_size=40,
        overlap=10,
        min_score=min_score,
    )


def test_ingest_returns_chunk_count():
    service = _service()
    assert service.ingest("doc1", " ".join(str(i) for i in range(100))) >= 2


def test_query_after_ingest_returns_answer_and_hits():
    service = _service()
    service.ingest("doc1", "python e uma linguagem de programacao muito popular")

    result = service.query("linguagem de programacao", top_k=3)

    assert result.hits
    assert result.answer
    assert result.abstained is False


def test_query_without_documents_abstains_without_calling_llm():
    llm = SpyLLM()

    result = _service(llm).query("qualquer coisa", top_k=3)

    assert result.abstained is True
    assert result.hits == []
    assert result.answer == ABSTENTION_ANSWER
    assert llm.calls == []


def test_query_with_no_shared_terms_abstains():
    llm = SpyLLM()
    service = _service(llm)
    service.ingest("doc1", "outbox grava eventos na mesma transacao")

    result = service.query("cotacao do dolar hoje", top_k=3)

    assert result.abstained is True
    assert llm.calls == []


def test_min_score_drops_weak_hits_and_abstains():
    llm = SpyLLM()
    service = _service(llm, min_score=0.99)
    service.ingest("doc1", "outbox grava eventos pendentes na mesma transacao do banco")

    result = service.query("outbox eventos", top_k=3)

    assert result.abstained is True
    assert llm.calls == []


def test_llm_receives_only_hits_above_min_score():
    llm = SpyLLM()
    service = _service(llm, min_score=0.1)
    service.ingest("forte", "outbox eventos transacao")
    service.ingest("fraco", "cache prazo invalidacao")

    result = service.query("outbox eventos transacao", top_k=5)

    assert [hit.chunk.doc_id for hit in result.hits] == ["forte"]
    assert llm.calls == [("outbox eventos transacao", ["outbox eventos transacao"])]
    assert all(hit.score > 0.1 for hit in result.hits)
