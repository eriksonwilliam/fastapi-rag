from app.domain.embedding import embed


class HashingEmbedder:
    """Embedder leve e deterministico (hashing bag-of-words). Padrao offline."""

    def embed(self, text: str) -> list[float]:
        return embed(text)
