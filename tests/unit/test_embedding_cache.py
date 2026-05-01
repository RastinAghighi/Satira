import pytest
import torch

from satira.graph.embedding_cache import GraphEmbeddingCache, GraphEmbeddingVersionStore


DIM = 8


def _emb(*values: float) -> torch.Tensor:
    if len(values) == DIM:
        return torch.tensor(values, dtype=torch.float32)
    if len(values) == 1:
        return torch.full((DIM,), values[0], dtype=torch.float32)
    raise ValueError("provide either 1 fill value or DIM values")


# --- set / get ----------------------------------------------------------
def test_set_and_get_round_trip() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    vec = torch.arange(DIM, dtype=torch.float32)

    cache.set("e1", vec)

    fetched = cache.get("e1")
    assert fetched is not None
    assert torch.equal(fetched, vec)


def test_get_missing_returns_none() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    assert cache.get("ghost") is None


def test_set_rejects_wrong_shape() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    with pytest.raises(ValueError):
        cache.set("e1", torch.zeros(DIM + 1))
    with pytest.raises(ValueError):
        cache.set("e1", torch.zeros(2, DIM))


def test_set_stores_a_copy_independent_of_caller() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    vec = torch.zeros(DIM)
    cache.set("e1", vec)
    vec[0] = 99.0

    fetched = cache.get("e1")
    assert fetched is not None
    assert fetched[0].item() == 0.0


# --- mget ---------------------------------------------------------------
def test_mget_returns_none_for_missing_keys() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    cache.set("e1", _emb(1.0))
    cache.set("e3", _emb(3.0))

    out = cache.mget(["e1", "e2", "e3", "e4"])

    assert out[0] is not None and torch.equal(out[0], _emb(1.0))
    assert out[1] is None
    assert out[2] is not None and torch.equal(out[2], _emb(3.0))
    assert out[3] is None


def test_mget_empty_input() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    assert cache.mget([]) == []


# --- attention_pool -----------------------------------------------------
def test_attention_pool_single_embedding_returns_that_embedding() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    e = _emb(2.0)
    out = cache.attention_pool([e], [1.0])
    assert out.shape == (DIM,)
    assert torch.allclose(out, e)


def test_attention_pool_shape_with_varying_input_count() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    for n in (1, 2, 5, 10):
        embeddings = [torch.randn(DIM) for _ in range(n)]
        weights = [1.0] * n
        out = cache.attention_pool(embeddings, weights)
        assert out.shape == (DIM,)


def test_attention_pool_empty_returns_zero_vector() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    out = cache.attention_pool([], [])
    assert out.shape == (DIM,)
    assert torch.equal(out, torch.zeros(DIM))


def test_attention_pool_higher_weight_means_more_influence() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    a = _emb(1.0)
    b = _emb(0.0)

    biased_to_a = cache.attention_pool([a, b], [0.9, 0.1])
    biased_to_b = cache.attention_pool([a, b], [0.1, 0.9])

    # When a's weight dominates, the pool should land closer to a (1.0)
    # than to b (0.0); and vice versa.
    assert biased_to_a.mean().item() > biased_to_b.mean().item()
    assert biased_to_a.mean().item() > 0.5
    assert biased_to_b.mean().item() < 0.5


def test_attention_pool_weights_normalize() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    a = _emb(1.0)
    b = _emb(3.0)

    # Equal weights → simple average (= 2.0)
    out = cache.attention_pool([a, b], [1.0, 1.0])
    assert torch.allclose(out, _emb(2.0))

    # Doubled weights should still produce the same average
    out2 = cache.attention_pool([a, b], [2.0, 2.0])
    assert torch.allclose(out2, _emb(2.0))


def test_attention_pool_zero_weights_returns_zero() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    out = cache.attention_pool([_emb(5.0), _emb(7.0)], [0.0, 0.0])
    assert torch.equal(out, torch.zeros(DIM))


def test_attention_pool_length_mismatch_raises() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    with pytest.raises(ValueError):
        cache.attention_pool([_emb(1.0)], [1.0, 2.0])


# --- snapshot / load ----------------------------------------------------
def test_snapshot_load_round_trip_preserves_embeddings() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    vecs = {
        "e1": torch.arange(DIM, dtype=torch.float32),
        "e2": torch.linspace(-1.0, 1.0, DIM),
        "e3": torch.zeros(DIM),
    }
    for nid, v in vecs.items():
        cache.set(nid, v)

    snap = cache.snapshot()

    restored = GraphEmbeddingCache(embedding_dim=DIM)
    restored.load_snapshot(snap)

    assert len(restored) == len(vecs)
    for nid, original in vecs.items():
        retrieved = restored.get(nid)
        assert retrieved is not None
        assert torch.equal(retrieved, original)


def test_snapshot_includes_distribution_stats() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    cache.set("e1", _emb(1.0))
    cache.set("e2", _emb(2.0))

    snap = cache.snapshot()
    assert "stats" in snap
    assert "embedding_dim" in snap
    assert snap["embedding_dim"] == DIM


def test_load_snapshot_replaces_existing_state() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    cache.set("old", _emb(7.0))

    other = GraphEmbeddingCache(embedding_dim=DIM)
    other.set("new", _emb(3.0))
    snap = other.snapshot()

    cache.load_snapshot(snap)

    assert cache.get("old") is None
    new = cache.get("new")
    assert new is not None
    assert torch.equal(new, _emb(3.0))


# --- compute_distribution_stats -----------------------------------------
def test_compute_distribution_stats_returns_expected_keys() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    for i in range(5):
        cache.set(f"e{i}", torch.randn(DIM))

    stats = cache.compute_distribution_stats()

    for key in ("count", "mean", "std", "norm_mean", "norm_std", "top_singular_values"):
        assert key in stats

    assert stats["count"] == 5
    assert stats["mean"].shape == (DIM,)
    assert stats["std"].shape == (DIM,)
    assert isinstance(stats["norm_mean"], float)
    assert isinstance(stats["norm_std"], float)
    assert stats["top_singular_values"].numel() <= 10


def test_compute_distribution_stats_empty_cache() -> None:
    cache = GraphEmbeddingCache(embedding_dim=DIM)
    stats = cache.compute_distribution_stats()
    assert stats["count"] == 0
    assert stats["norm_mean"] == 0.0
    assert stats["norm_std"] == 0.0


# --- version store ------------------------------------------------------
def test_version_store_save_load_round_trip() -> None:
    store = GraphEmbeddingVersionStore()
    embeddings = {"e1": _emb(1.0), "e2": _emb(2.0)}

    vid = store.save_snapshot(embeddings, gnn_version="gnn-v1")

    snap = store.load_snapshot(vid)
    assert snap["gnn_version"] == "gnn-v1"
    assert torch.equal(snap["embeddings"]["e1"], _emb(1.0))
    assert torch.equal(snap["embeddings"]["e2"], _emb(2.0))


def test_version_store_load_unknown_raises() -> None:
    store = GraphEmbeddingVersionStore()
    with pytest.raises(KeyError):
        store.load_snapshot("missing")


def test_version_store_list_versions_filters_by_age() -> None:
    from datetime import datetime, timedelta, timezone

    store = GraphEmbeddingVersionStore()
    v_recent = store.save_snapshot({"e1": _emb(1.0)}, gnn_version="gnn-v1")
    v_old = store.save_snapshot({"e1": _emb(2.0)}, gnn_version="gnn-v0")

    # Backdate the second snapshot to 30 days ago
    store._snapshots[v_old]["saved_at"] = datetime.now(timezone.utc) - timedelta(days=30)

    listed_recent = store.list_versions(last_n_days=21)
    assert v_recent in listed_recent
    assert v_old not in listed_recent

    listed_all = store.list_versions(last_n_days=60)
    assert v_recent in listed_all
    assert v_old in listed_all
