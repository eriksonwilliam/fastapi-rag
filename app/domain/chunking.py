"""Divisao de um texto em pedacos (chunks) com sobreposicao."""

from app.domain.errors import ValidationError


def chunk_text(text: str, size: int = 40, overlap: int = 10) -> list[str]:
    """Quebra o texto em janelas de `size` palavras, avancando `size - overlap`."""
    words = text.split()
    if not words:
        raise ValidationError("texto nao pode ser vazio")

    step = size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + size]))
        start += step
    return chunks
