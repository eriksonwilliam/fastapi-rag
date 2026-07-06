class StubLLM:
    """LLM offline e deterministico — permite rodar/testar sem um modelo real.

    Em producao, troque por um adapter que fale com um LLM de verdade
    (veja OllamaLLM em llm_ollama.py).
    """

    def generate(self, question: str, context: list[str]) -> str:
        if not context:
            return "Nao encontrei contexto relevante para responder."
        joined = " ".join(context)
        return f"Com base nos documentos: {joined[:300]}"
