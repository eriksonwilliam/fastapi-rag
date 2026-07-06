import math

from app.domain.embedding import DIM, cosine, embed


def test_embedding_is_normalized():
    vector = embed("python e uma linguagem de programacao")
    assert len(vector) == DIM
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-9)


def test_empty_text_yields_zero_vector():
    assert embed("") == [0.0] * DIM


def test_cosine_similar_beats_different():
    a = embed("banco de dados relacional")
    b = embed("banco de dados relacional")
    c = embed("receita de bolo de chocolate")
    assert cosine(a, b) > cosine(a, c)
