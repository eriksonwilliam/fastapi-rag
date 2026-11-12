import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.application.ports import StoredChunk
from app.application.rag_service import RagService
from app.infrastructure.hashing_embedder import HashingEmbedder
from app.infrastructure.llm_stub import StubLLM
from app.infrastructure.sqlite_store import SqliteVectorStore


def chunk(doc, index, text):
    return StoredChunk(doc, index, text, [1.0, 0.0])


def test_restart_replacement_and_other_documents(tmp_path):
    path = tmp_path / "rag.sqlite3"
    first = SqliteVectorStore(path)
    first.add([chunk("a", 0, "old"), chunk("a", 1, "stale"), chunk("b", 0, "keep")])
    restarted = SqliteVectorStore(path)
    assert len(restarted.search([1.0, 0.0], 10)) == 3
    restarted.add([chunk("a", 0, "new")])
    hits = SqliteVectorStore(path).search([1.0, 0.0], 10)
    assert [(h.chunk.doc_id, h.chunk.text) for h in hits] == [("a", "new"), ("b", "keep")]
    assert hits[0].score == 1.0
    assert len(restarted.search([1.0, 0.0], 1)) == 1


def test_failed_batch_rolls_back_deleted_documents(tmp_path):
    store = SqliteVectorStore(tmp_path / "rag.sqlite3")
    store.add([chunk("a", 0, "old")])
    with pytest.raises(sqlite3.IntegrityError):
        store.add([chunk("a", 0, "replacement"), chunk("a", 0, "duplicate index")])
    assert [h.chunk.text for h in store.search([1.0, 0.0], 10)] == ["old"]


def test_two_instances_replace_complete_batches_without_mixing(tmp_path):
    path = tmp_path / "rag.sqlite3"
    stores = [SqliteVectorStore(path), SqliteVectorStore(path)]
    batches = [[chunk("a", i, version) for i in range(3)] for version in ["v1", "v2"]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(store.add, batch) for store, batch in zip(stores, batches, strict=True)
        ]
        for future in futures:
            future.result()
    result = SqliteVectorStore(path).search([1.0, 0.0], 10)
    assert len(result) == 3
    assert {h.chunk.text for h in result} in [{"v1"}, {"v2"}]


def test_empty_store_and_empty_batch(tmp_path):
    store = SqliteVectorStore(tmp_path / "rag.sqlite3")
    assert store.search([1.0, 0.0], 10) == []
    store.add([])
    store.add([chunk("a", 0, "keep")])
    store.add([])
    assert len(store.search([1.0, 0.0], 10)) == 1


@pytest.mark.parametrize("path", ["", ":memory:"])
def test_reject_in_memory_connection(path):
    with pytest.raises(ValueError, match="file path"):
        SqliteVectorStore(path)


def test_service_retrieves_after_reconstruction_and_preserves_on_embedding_failure(tmp_path):
    path = tmp_path / "rag.sqlite3"
    service = RagService(SqliteVectorStore(path), StubLLM(), HashingEmbedder())
    service.ingest("guide", "Kafka commits offsets after processing")
    reconstructed = RagService(SqliteVectorStore(path), StubLLM(), HashingEmbedder())
    assert reconstructed.retrieve("Kafka")[0].chunk.doc_id == "guide"

    class BrokenEmbedder:
        def embed(self, text):
            raise RuntimeError("embedder unavailable")

    broken = RagService(SqliteVectorStore(path), StubLLM(), BrokenEmbedder())
    with pytest.raises(RuntimeError, match="embedder unavailable"):
        broken.ingest("guide", "changed source")
    assert reconstructed.retrieve("Kafka")[0].chunk.text == "Kafka commits offsets after processing"


def test_data_survives_a_new_python_process(tmp_path):
    env = {
        **os.environ,
        "VECTOR_STORE": "sqlite",
        "SQLITE_PATH": str(tmp_path / "process.sqlite3"),
        "EMBEDDER": "hashing",
        "LLM": "stub",
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.main import default_service; "
            "default_service().ingest('restart', 'Persistent Kafka context')",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    recovered = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.main import default_service; "
            "print(default_service().retrieve('Kafka')[0].chunk.doc_id)",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert recovered.stdout.strip() == "restart"
