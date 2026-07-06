import uvicorn

from app.api.app import create_app
from app.application.rag_service import RagService
from app.infrastructure.llm_stub import StubLLM
from app.infrastructure.memory_store import InMemoryVectorStore


def default_service() -> RagService:
    # Troque StubLLM por OllamaLLM (app.infrastructure.llm_ollama) para um LLM real.
    return RagService(InMemoryVectorStore(), StubLLM())


app = create_app(default_service())


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(app, host="0.0.0.0", port=8000)
