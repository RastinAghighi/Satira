import time

import numpy as np
import pytest
import torch

faiss = pytest.importorskip("faiss")

from satira.temporal.index_manager import FAISSIndexManager
from satira.temporal.retriever import TemporalContextRetriever


DIM = 32


class FixedEncoder:
    """Encodes any text to a fixed numpy embedding."""

    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector.astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        return self._vector


class SlowEncoder:
    """Encoder that sleeps before returning, used to force the timeout path."""

    def __init__(self, vector: np.ndarray, delay_s: float) -> None:
        self._vector = vector.astype(np.float32)
        self._delay = delay_s

    def encode(self, text: str) -> np.ndarray:
        time.sleep(self._delay)
        return self._vector


def _build_index(n: int = 20, seed: int = 0) -> FAISSIndexManager:
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((n, DIM)).astype(np.float32)
    metadata = [
        {"article_id": f"a{i}", "embedding": embs[i].tolist()}
        for i in range(n)
    ]
    mgr = FAISSIndexManager(dim=DIM, index_type="IVFFlat", nlist=4)
    mgr.build_index(embs, metadata)
    return mgr


# --- shape -------------------------------------------------------------
async def test_retrieve_returns_tensor_of_index_dim() -> None:
    mgr = _build_index()
    encoder = FixedEncoder(np.zeros(DIM, dtype=np.float32))
    retriever = TemporalContextRetriever(mgr, text_encoder=encoder, top_k=5)

    out = await retriever.retrieve("breaking news")

    assert isinstance(out, torch.Tensor)
    assert out.shape == (DIM,)


async def test_retrieve_pools_top_k_neighbour_embeddings() -> None:
    mgr = _build_index()
    encoder = FixedEncoder(np.zeros(DIM, dtype=np.float32))
    retriever = TemporalContextRetriever(mgr, text_encoder=encoder, top_k=3)

    out = await retriever.retrieve("any text")

    # Mean-pooled vector should not be the zero default — the index has data.
    assert not torch.equal(out, torch.zeros(DIM))


# --- timeout -----------------------------------------------------------
async def test_timeout_fallback_returns_default_embedding() -> None:
    mgr = _build_index()
    default = torch.full((DIM,), 0.5, dtype=torch.float32)
    encoder = SlowEncoder(np.zeros(DIM, dtype=np.float32), delay_s=1.0)
    retriever = TemporalContextRetriever(
        mgr,
        default_embedding=default,
        timeout_ms=20.0,
        text_encoder=encoder,
    )

    out = await retriever.retrieve("anything")

    assert torch.equal(out, default)


async def test_default_embedding_defaults_to_zeros_of_index_dim() -> None:
    mgr = _build_index()
    encoder = SlowEncoder(np.zeros(DIM, dtype=np.float32), delay_s=1.0)
    retriever = TemporalContextRetriever(mgr, timeout_ms=20.0, text_encoder=encoder)

    out = await retriever.retrieve("anything")

    assert torch.equal(out, torch.zeros(DIM))


# --- cache hits --------------------------------------------------------
async def test_cache_hit_avoids_underlying_index_search() -> None:
    mgr = _build_index()
    encoder = FixedEncoder(np.zeros(DIM, dtype=np.float32))
    retriever = TemporalContextRetriever(mgr, text_encoder=encoder)

    # First call: cache miss; populates the LRU.
    first_out, first_was_hit = await retriever.retrieve_with_timeout("hello")
    assert first_was_hit is False
    assert first_out.shape == (DIM,)

    # Replace the underlying FAISS search so any second call would blow up —
    # the cache hit path must bypass it entirely.
    def _boom(*_args, **_kwargs):
        raise AssertionError("FAISS index.search should not be called on cache hit")

    mgr.search = _boom  # type: ignore[assignment]

    second_out, second_was_hit = await retriever.retrieve_with_timeout("hello")

    assert second_was_hit is True
    assert second_out.shape == (DIM,)
    # Same query → identical pooled output across the two calls.
    assert torch.equal(first_out, second_out)
