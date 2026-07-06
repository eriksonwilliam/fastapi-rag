from app.application.ports import StoredChunk
from app.domain.embedding import embed
from app.infrastructure.memory_store import InMemoryVectorStore


def _chunk(doc_id: str, index: int, text: str) -> StoredChunk:
    return StoredChunk(doc_id=doc_id, chunk_index=index, text=text, vector=embed(text))


def test_add_and_search_orders_by_score():
    store = InMemoryVectorStore()
    store.add([_chunk("d", 0, "gatos e cachorros"), _chunk("d", 1, "bolsa de valores")])

    hits = store.search(embed("gatos e cachorros"), top_k=2)

    assert len(hits) == 2
    assert hits[0].chunk.text == "gatos e cachorros"


def test_search_on_empty_store_returns_empty():
    assert InMemoryVectorStore().search(embed("qualquer"), top_k=5) == []
