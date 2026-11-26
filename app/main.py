import os

import uvicorn

from app.api.app import create_app
from app.application.ports import LLM, Embedder, VectorStore
from app.application.rag_service import RagService
from app.infrastructure.hashing_embedder import HashingEmbedder
from app.infrastructure.llm_stub import StubLLM
from app.infrastructure.memory_store import InMemoryVectorStore


def build_embedder() -> Embedder:
    if os.getenv("EMBEDDER") == "fastembed":  # pragma: no cover
        from app.infrastructure.fastembed_embedder import FastEmbedEmbedder

        return FastEmbedEmbedder()
    return HashingEmbedder()


def build_llm() -> LLM:
    if os.getenv("LLM") == "ollama":  # pragma: no cover
        from app.infrastructure.llm_ollama import OllamaLLM

        return OllamaLLM()
    return StubLLM()


def build_store() -> VectorStore:
    if os.getenv("VECTOR_STORE") == "sqlite":  # pragma: no cover
        from app.infrastructure.sqlite_store import SqliteVectorStore

        return SqliteVectorStore(os.getenv("SQLITE_PATH", "rag.sqlite3"))
    return InMemoryVectorStore()


def default_service() -> RagService:
    # O limiar depende do embedder (escalas de score diferentes): calibre com
    # scripts/evaluate_retrieval.py --calibrate no conjunto de desenvolvimento.
    min_score = float(os.getenv("RAG_MIN_SCORE", "0.0"))
    return RagService(build_store(), build_llm(), build_embedder(), min_score=min_score)


app = create_app(default_service())


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(app, host="0.0.0.0", port=8000)
