from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.application.ports import StoredChunk
from app.application.rag_service import RagService
from app.domain.embedding import embed
from app.infrastructure.hashing_embedder import HashingEmbedder
from app.infrastructure.llm_stub import StubLLM
from app.infrastructure.memory_store import InMemoryVectorStore


def _chunk(doc_id: str, index: int, text: str) -> StoredChunk:
    return StoredChunk(doc_id=doc_id, chunk_index=index, text=text, vector=embed(text))


def _service() -> RagService:
    return RagService(InMemoryVectorStore(), StubLLM(), HashingEmbedder())


def _client() -> TestClient:
    return TestClient(create_app(_service()))


def test_reingest_replaces_old_chunks_and_preserves_other_documents():
    store = InMemoryVectorStore()
    store.add([_chunk("d", 0, "antigo"), _chunk("d", 1, "obsoleto")])
    store.add([_chunk("other", 0, "outro documento")])
    store.add([_chunk("d", 0, "novo")])
    hits = store.search(embed("novo"), top_k=20)
    assert {(hit.chunk.doc_id, hit.chunk.text) for hit in hits} == {
        ("d", "novo"),
        ("other", "outro documento"),
    }


def test_empty_batch_preserves_existing_documents():
    store = InMemoryVectorStore()
    store.add([_chunk("d", 0, "existente")])
    store.add([])
    assert len(store.search(embed("existente"), top_k=20)) == 1


def test_concurrent_reingest_keeps_one_complete_version():
    store = InMemoryVectorStore()
    batches = [
        [_chunk("d", 0, f"version {version}"), _chunk("d", 1, f"version {version}")]
        for version in range(20)
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(store.add, batches))
    hits = store.search(embed("version"), top_k=20)
    assert len(hits) == 2
    assert len({hit.chunk.text for hit in hits}) == 1


def test_reingest_same_document_does_not_duplicate_sources():
    service = _service()
    service.ingest("doc1", "MCP conecta agentes a ferramentas")
    service.ingest("doc1", "MCP conecta agentes a ferramentas")
    _, hits = service.query("MCP agentes", top_k=20)
    assert len(hits) == 1


def test_failed_embedding_preserves_previous_document():
    class FailingEmbedder:
        def embed(self, text):
            raise RuntimeError("embedding unavailable")

    store = InMemoryVectorStore()
    service = RagService(store, StubLLM(), HashingEmbedder(), chunk_size=40, overlap=10)
    service.ingest("doc1", "versao anterior")
    failing = RagService(store, StubLLM(), FailingEmbedder())
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        failing.ingest("doc1", "nova versao")
    assert service.retrieve("versao", top_k=20)[0].chunk.text == "versao anterior"


def test_reingest_replaces_document_through_http_api():
    client = _client()
    client.post("/documents", json={"doc_id": "d", "text": "conteudo antigo"})
    updated = client.post("/documents", json={"doc_id": "d", "text": "conteudo atualizado"})
    assert updated.status_code == 201
    sources = client.post("/query", json={"question": "conteudo", "top_k": 20}).json()["sources"]
    assert len(sources) == 1
    assert sources[0]["text"] == "conteudo atualizado"
