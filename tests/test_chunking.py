import pytest

from app.domain.chunking import chunk_text
from app.domain.errors import ValidationError


def test_chunk_short_text_single_chunk():
    assert chunk_text("uma frase curta") == ["uma frase curta"]


def test_chunk_long_text_multiple_chunks():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, size=40, overlap=10)
    assert len(chunks) > 1


def test_chunk_empty_text_raises():
    with pytest.raises(ValidationError):
        chunk_text("   ")
