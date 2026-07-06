from app.infrastructure.llm_stub import StubLLM


def test_stub_without_context():
    assert "Nao encontrei" in StubLLM().generate("pergunta", [])


def test_stub_with_context_grounds_the_answer():
    answer = StubLLM().generate("pergunta", ["um trecho relevante"])
    assert "um trecho relevante" in answer
