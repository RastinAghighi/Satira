import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from satira.temporal.index_manager import CachedRetriever, FAISSIndexManager


DIM = 32


def _embeddings(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, DIM)).astype(np.float32)


def _metadata(n: int, prefix: str = "a") -> list[dict]:
    return [
        {
            "article_id": f"{prefix}{i}",
            "title": f"title-{i}",
            "source": "src",
            "timestamp": i,
        }
        for i in range(n)
    ]


# --- build_index / search ----------------------------------------------
def test_build_index_and_search_returns_k_results() -> None:
    mgr = FAISSIndexManager(dim=DIM, index_type="IVFFlat", nlist=4)
    embs = _embeddings(20, seed=1)
    mgr.build_index(embs, _metadata(20))

    out = mgr.search(embs[3], k=5)

    assert len(out) == 5
    # Each result carries the metadata fields plus a distance.
    for row in out:
        assert {"article_id", "title", "source", "timestamp", "distance"} <= row.keys()
    # Searching for a vector that's in the index should surface itself first.
    assert out[0]["article_id"] == "a3"
    assert out[0]["distance"] == pytest.approx(0.0, abs=1e-4)


def test_search_returns_empty_when_index_empty() -> None:
    mgr = FAISSIndexManager(dim=DIM, index_type="Flat")
    assert mgr.search(np.zeros(DIM, dtype=np.float32), k=5) == []


# --- hot_reload --------------------------------------------------------
def test_hot_reload_swaps_index_without_error(tmp_path) -> None:
    mgr = FAISSIndexManager(dim=DIM, index_type="IVFFlat", nlist=4)
    original_embs = _embeddings(20, seed=2)
    mgr.build_index(original_embs, _metadata(20, prefix="a"))

    save_path = str(tmp_path / "idx.faiss")
    mgr.save(save_path)

    # Replace the in-memory index with a different one.
    other_embs = _embeddings(16, seed=99)
    mgr.build_index(other_embs, _metadata(16, prefix="x"))
    assert mgr.search(other_embs[0], k=1)[0]["article_id"].startswith("x")

    # Hot-reload the saved snapshot and confirm the original metadata is back.
    mgr.hot_reload(save_path)
    out = mgr.search(original_embs[0], k=1)
    assert out[0]["article_id"] == "a0"


# --- WAL ---------------------------------------------------------------
def test_append_to_wal_items_are_searchable() -> None:
    mgr = FAISSIndexManager(dim=DIM, index_type="IVFFlat", nlist=4)
    mgr.build_index(_embeddings(20, seed=3), _metadata(20))

    # A vector deliberately far from the existing distribution.
    new_emb = np.full(DIM, 50.0, dtype=np.float32)
    mgr.append_to_wal(
        new_emb,
        {"article_id": "wal1", "title": "breaking", "source": "wire", "timestamp": 999},
    )

    assert mgr.get_index_stats()["wal_size"] == 1
    out = mgr.search(new_emb, k=3)
    assert "wal1" in [row["article_id"] for row in out]

    mgr.merge_wal()
    stats = mgr.get_index_stats()
    assert stats["wal_size"] == 0
    assert stats["total_vectors"] == 21

    # Still searchable after merge — now via the main index.
    out_after = mgr.search(new_emb, k=3)
    assert "wal1" in [row["article_id"] for row in out_after]


# --- CachedRetriever ---------------------------------------------------
def test_cached_retriever_returns_same_results_as_direct() -> None:
    mgr = FAISSIndexManager(dim=DIM, index_type="IVFFlat", nlist=4)
    embs = _embeddings(30, seed=4)
    mgr.build_index(embs, _metadata(30))

    cached = CachedRetriever(mgr, cache_size=100)
    q = embs[7]

    direct = mgr.search(q, k=5)
    via_cache = cached.retrieve(q, k=5)

    assert [row["article_id"] for row in direct] == [row["article_id"] for row in via_cache]


def test_cache_hit_rate_increases_with_repeated_queries() -> None:
    mgr = FAISSIndexManager(dim=DIM, index_type="IVFFlat", nlist=4)
    embs = _embeddings(20, seed=5)
    mgr.build_index(embs, _metadata(20))

    cached = CachedRetriever(mgr, cache_size=50)
    q = embs[0]

    cached.retrieve(q, k=5)  # miss
    after_first = cached.cache_stats()
    assert after_first["misses"] == 1
    assert after_first["hits"] == 0

    for _ in range(3):
        cached.retrieve(q, k=5)

    stats = cached.cache_stats()
    assert stats["hits"] == 3
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.75)


# --- get_index_stats ---------------------------------------------------
def test_get_index_stats_reports_expected_fields() -> None:
    mgr = FAISSIndexManager(dim=DIM, index_type="IVFFlat", nlist=4)
    mgr.build_index(_embeddings(20, seed=6), _metadata(20))

    stats = mgr.get_index_stats()
    assert stats["total_vectors"] == 20
    assert stats["wal_size"] == 0
    assert stats["index_type"] == "IVFFlat"
    assert stats["memory_bytes"] == 20 * DIM * 4
