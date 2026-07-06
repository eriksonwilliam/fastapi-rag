import os

import uvicorn

from app.api.app import create_app
from app.application.ports import LLM, Embedder
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


def default_service() -> RagService:
    return RagService(InMemoryVectorStore(), build_llm(), build_embedder())


app = create_app(default_service())


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(app, host="0.0.0.0", port=8000)
