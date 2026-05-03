import time

import numpy as np
import pytest
import torch

faiss = pytest.importorskip("faiss")

from satira.graph.embedding_cache import GraphEmbeddingCache
from satira.graph.entity_resolution import MentionNormalizer
from satira.inference.batcher import InferenceRequest
from satira.inference.context_resolver import ContextResolver
from satira.temporal.index_manager import FAISSIndexManager
from satira.temporal.retriever import TemporalContextRetriever


TEMPORAL_DIM = 32
GRAPH_DIM = 16
VISION_SHAPE = (10, 64)
TEXT_SHAPE = (12, 64)


# --- fakes -------------------------------------------------------------
class FakeOCR:
    def __init__(self, text: str = "", delay_s: float = 0.0) -> None:
        self._text = text
        self._delay = delay_s

    def extract_text(self, image_bytes: bytes) -> str:
        if self._delay:
            time.sleep(self._delay)
        return self._text


class FakeVisionEncoder:
    def __init__(self, shape: tuple[int, ...] = VISION_SHAPE, delay_s: float = 0.0) -> None:
        self._shape = shape
        self._delay = delay_s

    def encode(self, image_bytes: bytes) -> torch.Tensor:
        if self._delay:
            time.sleep(self._delay)
        return torch.zeros(*self._shape)


class FakeTextEncoder:
    """Returns a (n_tokens, dim) tensor for the model and a flat
    embedding for the temporal retriever, depending on which API is
    called."""

    def __init__(
        self,
        token_shape: tuple[int, ...] = TEXT_SHAPE,
        retrieval_vector: np.ndarray | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self._token_shape = token_shape
        self._retrieval_vector = (
            retrieval_vector
            if retrieval_vector is not None
            else np.zeros(TEMPORAL_DIM, dtype=np.float32)
        )
        self._delay = delay_s

    def encode(self, text: str):
        if self._delay:
            time.sleep(self._delay)
        return torch.zeros(*self._token_shape)


class TemporalQueryEncoder:
    """Used inside the TemporalContextRetriever — returns a numpy vector
    of the FAISS index dim."""

    def __init__(self, vector: np.ndarray, delay_s: float = 0.0) -> None:
        self._vector = vector.astype(np.float32)
        self._delay = delay_s

    def encode(self, text: str) -> np.ndarray:
        if self._delay:
            time.sleep(self._delay)
        return self._vector


class SlowNormalizer(MentionNormalizer):
    def __init__(self, delay_s: float, entity_id: str = "E1") -> None:
        super().__init__()
        self._delay = delay_s
        self._entity_id = entity_id

    def normalize(self, raw_mention: str) -> tuple[str | None, float]:
        time.sleep(self._delay)
        return (self._entity_id, 1.0)


# --- helpers -----------------------------------------------------------
def _build_temporal_index(n: int = 20, seed: int = 0) -> FAISSIndexManager:
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((n, TEMPORAL_DIM)).astype(np.float32)
    metadata = [
        {"article_id": f"a{i}", "embedding": embs[i].tolist()} for i in range(n)
    ]
    mgr = FAISSIndexManager(dim=TEMPORAL_DIM, index_type="IVFFlat", nlist=4)
    mgr.build_index(embs, metadata)
    return mgr


def _build_resolver(
    *,
    ocr_text: str = "Acme",
    ocr_delay_s: float = 0.0,
    vision_delay_s: float = 0.0,
    text_delay_s: float = 0.0,
    temporal_query_delay_s: float = 0.0,
    temporal_timeout_ms: float = 200.0,
    normalizer: MentionNormalizer | None = None,
    graph_cache: GraphEmbeddingCache | None = None,
    graph_timeout_s: float = 0.5,
) -> ContextResolver:
    if normalizer is None:
        normalizer = MentionNormalizer()
        normalizer.register_alias("Acme", "E1")
    if graph_cache is None:
        graph_cache = GraphEmbeddingCache(embedding_dim=GRAPH_DIM)
        graph_cache.set("E1", torch.ones(GRAPH_DIM))

    temporal = TemporalContextRetriever(
        _build_temporal_index(),
        timeout_ms=temporal_timeout_ms,
        text_encoder=TemporalQueryEncoder(
            np.zeros(TEMPORAL_DIM, dtype=np.float32),
            delay_s=temporal_query_delay_s,
        ),
        top_k=3,
    )

    return ContextResolver(
        mention_normalizer=normalizer,
        graph_cache=graph_cache,
        temporal_retriever=temporal,
        text_encoder=FakeTextEncoder(delay_s=text_delay_s),
        vision_encoder=FakeVisionEncoder(delay_s=vision_delay_s),
        ocr_engine=FakeOCR(text=ocr_text, delay_s=ocr_delay_s),
        graph_timeout_s=graph_timeout_s,
    )


# --- shape -------------------------------------------------------------
async def test_resolve_returns_inference_request_with_correct_shapes() -> None:
    resolver = _build_resolver()

    request = await resolver.resolve(b"fake-image-bytes")

    assert isinstance(request, InferenceRequest)
    assert request.vision_emb.shape == VISION_SHAPE
    assert request.text_emb.shape == TEXT_SHAPE
    assert request.temporal_emb.shape == (TEMPORAL_DIM,)
    assert request.graph_emb.shape == (GRAPH_DIM,)
    # An "Acme" mention resolves to E1 → cache returns ones, pooled is ones.
    assert torch.allclose(request.graph_emb, torch.ones(GRAPH_DIM))


# --- parallelism -------------------------------------------------------
async def test_parallel_streams_run_concurrently() -> None:
    DELAY_S = 0.15

    normalizer = SlowNormalizer(delay_s=DELAY_S, entity_id="E1")
    normalizer.register_alias("Acme", "E1")
    graph_cache = GraphEmbeddingCache(embedding_dim=GRAPH_DIM)
    graph_cache.set("E1", torch.ones(GRAPH_DIM))

    resolver = _build_resolver(
        ocr_text="Acme",
        vision_delay_s=DELAY_S,
        text_delay_s=DELAY_S,
        temporal_query_delay_s=DELAY_S,
        temporal_timeout_ms=2000.0,
        normalizer=normalizer,
        graph_cache=graph_cache,
        graph_timeout_s=2.0,
    )

    start = time.monotonic()
    request = await resolver.resolve(b"image")
    elapsed = time.monotonic() - start

    # Sequential lower bound would be ~4 * DELAY_S; parallel should sit
    # close to a single DELAY_S (plus slack for thread pool scheduling).
    assert elapsed < DELAY_S * 2.5, (
        f"resolve took {elapsed:.3f}s — looks serialized (4x={4*DELAY_S:.3f}s)"
    )
    assert elapsed >= DELAY_S * 0.8

    assert request.vision_emb.shape == VISION_SHAPE
    assert request.graph_emb.shape == (GRAPH_DIM,)


# --- graph fallback ----------------------------------------------------
async def test_graph_context_fallback_when_no_entities_found() -> None:
    # OCR text has no matching aliases, so nothing resolves.
    resolver = _build_resolver(ocr_text="zzz qqq xxx unknown")

    request = await resolver.resolve(b"image")

    assert torch.equal(request.graph_emb, torch.zeros(GRAPH_DIM))


async def test_graph_context_fallback_when_text_is_empty() -> None:
    resolver = _build_resolver(ocr_text="")

    request = await resolver.resolve(b"image")

    assert torch.equal(request.graph_emb, torch.zeros(GRAPH_DIM))


async def test_graph_context_fallback_on_timeout() -> None:
    DELAY_S = 0.20

    normalizer = SlowNormalizer(delay_s=DELAY_S, entity_id="E1")
    normalizer.register_alias("Acme", "E1")
    graph_cache = GraphEmbeddingCache(embedding_dim=GRAPH_DIM)
    graph_cache.set("E1", torch.ones(GRAPH_DIM))

    resolver = _build_resolver(
        ocr_text="Acme",
        normalizer=normalizer,
        graph_cache=graph_cache,
        graph_timeout_s=0.02,
    )

    request = await resolver.resolve(b"image")

    assert torch.equal(request.graph_emb, torch.zeros(GRAPH_DIM))


# --- temporal fallback -------------------------------------------------
async def test_temporal_context_fallback_on_timeout() -> None:
    resolver = _build_resolver(
        temporal_query_delay_s=0.5,
        temporal_timeout_ms=20.0,
    )

    request = await resolver.resolve(b"image")

    # Default fallback is zero-vector of TEMPORAL_DIM.
    assert torch.equal(request.temporal_emb, torch.zeros(TEMPORAL_DIM))


async def test_resolve_returns_pooled_graph_embedding_with_two_mentions() -> None:
    normalizer = MentionNormalizer()
    normalizer.register_alias("Acme", "E1")
    normalizer.register_alias("Globex", "E2")

    graph_cache = GraphEmbeddingCache(embedding_dim=GRAPH_DIM)
    graph_cache.set("E1", torch.ones(GRAPH_DIM) * 2.0)
    graph_cache.set("E2", torch.ones(GRAPH_DIM) * 4.0)

    resolver = _build_resolver(
        ocr_text="Acme Globex",
        normalizer=normalizer,
        graph_cache=graph_cache,
    )

    request = await resolver.resolve(b"image")

    # Both score 1.0 → exact_alias weight 1.0; pooled = mean(2, 4) = 3.
    assert torch.allclose(request.graph_emb, torch.full((GRAPH_DIM,), 3.0))
