from fastapi.testclient import TestClient

from app.api.app import create_app
from app.application.rag_service import RagService
from app.infrastructure.llm_stub import StubLLM
from app.infrastructure.memory_store import InMemoryVectorStore


def _client() -> TestClient:
    return TestClient(create_app(RagService(InMemoryVectorStore(), StubLLM())))


def test_health():
    assert _client().get("/health").json() == {"status": "ok"}


def test_openapi_available():
    assert _client().get("/openapi.json").status_code == 200


def test_ingest_and_query_flow():
    client = _client()

    ingest = client.post(
        "/documents",
        json={"text": "python e uma linguagem de programacao", "doc_id": "doc1"},
    )
    assert ingest.status_code == 201
    assert ingest.json()["doc_id"] == "doc1"
    assert ingest.json()["chunks"] >= 1

    query = client.post("/query", json={"question": "linguagem de programacao", "top_k": 3})
    assert query.status_code == 200
    body = query.json()
    assert body["sources"]
    assert body["answer"]


def test_ingest_generates_doc_id_when_absent():
    response = _client().post("/documents", json={"text": "algum texto de exemplo aqui"})
    assert response.status_code == 201
    assert response.json()["doc_id"]


def test_ingest_empty_text_returns_400():
    response = _client().post("/documents", json={"text": "   "})
    assert response.status_code == 400
    assert response.json()["error"] == "ValidationError"


def test_query_empty_question_returns_422():
    response = _client().post("/query", json={"question": "", "top_k": 3})
    assert response.status_code == 422
