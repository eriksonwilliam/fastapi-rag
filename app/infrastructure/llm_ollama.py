"""Adapter de LLM real usando Ollama (local). Fora da cobertura unitaria
porque depende de um servico externo; validado manualmente / em integracao."""

import os

import httpx


class OllamaLLM:
    def __init__(self, host: str | None = None, model: str = "llama3"):
        self._host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._model = model

    def generate(self, question: str, context: list[str]) -> str:
        prompt = (
            "Use apenas o contexto abaixo para responder a pergunta.\n\n"
            f"Contexto:\n{chr(10).join(context)}\n\n"
            f"Pergunta: {question}\nResposta:"
        )
        response = httpx.post(
            f"{self._host}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"]
