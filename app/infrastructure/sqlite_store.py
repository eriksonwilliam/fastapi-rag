"""Optional durable local vector store; exhaustive cosine search, not an ANN index."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from app.application.ports import SearchHit, StoredChunk
from app.domain.embedding import cosine


class SqliteVectorStore:
    def __init__(self, path: str | Path):
        self._path = str(path)
        if self._path in {"", ":memory:"}:
            raise ValueError("Use a file path for the durable SQLite store")
        with closing(sqlite3.connect(self._path, timeout=10)) as connection:
            with connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS chunks ("
                    "doc_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, "
                    "text TEXT NOT NULL, vector TEXT NOT NULL, "
                    "PRIMARY KEY (doc_id, chunk_index))"
                )

    def add(self, chunks: list[StoredChunk]) -> None:
        # Serialize before taking a database write lock. A failed insert rolls the
        # whole replacement back, including deletes of the previous document.
        rows = [(c.doc_id, c.chunk_index, c.text, json.dumps(c.vector)) for c in chunks]
        with closing(sqlite3.connect(self._path, timeout=10)) as connection:
            with connection:
                connection.executemany(
                    "DELETE FROM chunks WHERE doc_id = ?",
                    [(doc_id,) for doc_id in {c.doc_id for c in chunks}],
                )
                connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", rows)

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        with closing(sqlite3.connect(self._path, timeout=10)) as connection:
            rows = connection.execute(
                "SELECT doc_id, chunk_index, text, vector FROM chunks ORDER BY doc_id, chunk_index"
            ).fetchall()
        hits = [
            SearchHit(
                chunk=StoredChunk(doc_id, index, text, json.loads(stored_vector)),
                score=cosine(vector, json.loads(stored_vector)),
            )
            for doc_id, index, text, stored_vector in rows
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]
