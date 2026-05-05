"""End-to-end integration tests for the inference pipeline.

Wires real ContextResolver, DynamicBatcher, InferencePipeline, and
SatireDetectionEngine together and plugs in mock encoders so the tests
exercise the full coordination flow without depending on heavyweight
vision/text models or external services.
"""
import asyncio

import numpy as np
import pytest
import torch

faiss = pytest.importorskip("faiss")

from satira.config import Settings
from satira.graph.embedding_cache import GraphEmbeddingCache
from satira.graph.entity_resolution import MentionNormalizer
from satira.inference.context_resolver import ContextResolver
from satira.inference.pipeline import ClassificationResult, InferencePipeline
from satira.models.engine import SatireDetectionEngine
from satira.temporal.index_manager import FAISSIndexManager
from satira.temporal.retriever import TemporalContextRetriever


D_MODEL = 64
NUM_HEADS = 4
VISION_DIM = 128
TEXT_DIM = 96
TEMPORAL_DIM = 96
GRAPH_DIM = 48
VISION_PATCHES = 8
TEXT_TOKENS = 10


def _make_settings(
    max_batch_size: int = 16,
    batch_timeout_ms: float = 100.0,
) -> Settings:
    return Settings(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        vision_dim=VISION_DIM,
        text_dim=TEXT_DIM,
        temporal_dim=TEMPORAL_DIM,
        graph_dim=GRAPH_DIM,
        num_reasoning_layers=1,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
    )


# --- mock encoders -----------------------------------------------------
class MockOCR:
    def __init__(self, text: str = "Acme Globex News") -> None:
        self._text = text

    def extract_text(self, image_bytes: bytes) -> str:
        return self._text


class MockVisionEncoder:
    def encode(self, image_bytes: bytes) -> torch.Tensor:
        return torch.randn(VISION_PATCHES, VISION_DIM)


class MockTextEncoder:
    def encode(self, text: str) -> torch.Tensor:
        return torch.randn(TEXT_TOKENS, TEXT_DIM)


class MockTemporalQueryEncoder:
    def encode(self, text: str) -> np.ndarray:
        return np.random.randn(TEMPORAL_DIM).astype(np.float32)


# --- helpers -----------------------------------------------------------
def _build_temporal_index(populated: bool, n: int = 16, seed: int = 0) -> FAISSIndexManager:
    mgr = FAISSIndexManager(dim=TEMPORAL_DIM, index_type="IVFFlat", nlist=4)
    if populated:
        rng = np.random.default_rng(seed)
        embs = rng.standard_normal((n, TEMPORAL_DIM)).astype(np.float32)
        metadata = [
            {"article_id": f"a{i}", "embedding": embs[i].tolist()} for i in range(n)
        ]
        mgr.build_index(embs, metadata)
    return mgr


def _build_pipeline(
    *,
    populate_temporal: bool = True,
    populate_graph: bool = True,
    ocr_text: str = "Acme Globex News",
    config: Settings | None = None,
) -> InferencePipeline:
    if config is None:
        config = _make_settings()

    normalizer = MentionNormalizer()
    normalizer.register_alias("Acme", "E1")
    normalizer.register_alias("Globex", "E2")

    graph_cache = GraphEmbeddingCache(embedding_dim=GRAPH_DIM)
    if populate_graph:
        graph_cache.set("E1", torch.randn(GRAPH_DIM))
        graph_cache.set("E2", torch.randn(GRAPH_DIM))

    temporal_index = _build_temporal_index(populated=populate_temporal)
    temporal = TemporalContextRetriever(
        temporal_index,
        timeout_ms=500.0,
        text_encoder=MockTemporalQueryEncoder(),
        top_k=3,
    )

    resolver = ContextResolver(
        mention_normalizer=normalizer,
        graph_cache=graph_cache,
        temporal_retriever=temporal,
        text_encoder=MockTextEncoder(),
        vision_encoder=MockVisionEncoder(),
        ocr_engine=MockOCR(text=ocr_text),
        graph_timeout_s=2.0,
    )

    engine = SatireDetectionEngine(config)
    engine.eval()

    return InferencePipeline(
        config=config,
        context_resolver=resolver,
        model=engine,
        device="cpu",
    )


# --- tests -------------------------------------------------------------
async def test_full_pipeline_mock() -> None:
    config = _make_settings()
    pipeline = _build_pipeline(config=config)
    await pipeline.initialize()
    try:
        result = await pipeline.classify(b"fake-image-bytes")
    finally:
        await pipeline.shutdown()

    assert isinstance(result, ClassificationResult)
    assert isinstance(result.class_name, str)
    assert isinstance(result.class_index, int)
    assert 0 <= result.class_index < len(config.CLASS_NAMES)
    assert result.class_name == config.CLASS_NAMES[result.class_index]

    assert 0.0 <= result.confidence <= 1.0
    assert set(result.all_probabilities.keys()) == set(config.CLASS_NAMES)
    assert abs(sum(result.all_probabilities.values()) - 1.0) < 1e-4

    assert result.latency_ms > 0.0

    # Attention fields exist on the result; the current pipeline surfaces
    # them as None until the batcher is extended to plumb the cross-attn
    # weights through. The type contract still allows np.ndarray.
    assert result.t2v_attention is None or isinstance(result.t2v_attention, np.ndarray)
    assert result.v2t_attention is None or isinstance(result.v2t_attention, np.ndarray)

    assert isinstance(result.graph_confidence, float)
    assert isinstance(result.temporal_cache_hit, bool)


async def test_dynamic_batching() -> None:
    pipeline = _build_pipeline(
        config=_make_settings(max_batch_size=16, batch_timeout_ms=150.0),
    )
    await pipeline.initialize()
    try:
        results = await asyncio.gather(
            *[pipeline.classify(b"image-%d" % i) for i in range(10)]
        )
        # Capture stats before shutdown — shutdown() nulls _batcher.
        stats = pipeline._batcher.stats()
    finally:
        await pipeline.shutdown()

    assert len(results) == 10
    assert all(isinstance(r, ClassificationResult) for r in results)
    assert all(0.0 <= r.confidence <= 1.0 for r in results)

    assert stats["total_requests"] == 10
    assert stats["total_batches"] >= 1
    # Concurrent submissions inside the batch_timeout window should
    # coalesce into batches larger than 1.
    assert stats["avg_batch_size"] > 1.0


async def test_context_fallback() -> None:
    # Empty FAISS index → temporal retriever falls back to its default
    # (zero) embedding instead of failing.
    pipeline = _build_pipeline(populate_temporal=False)
    await pipeline.initialize()
    try:
        result = await pipeline.classify(b"image")
    finally:
        await pipeline.shutdown()

    assert isinstance(result, ClassificationResult)
    assert 0.0 <= result.confidence <= 1.0
    assert abs(sum(result.all_probabilities.values()) - 1.0) < 1e-4
    # No FAISS hit means the LRU cache never serves a real result.
    assert result.temporal_cache_hit is False


async def test_graph_context_missing() -> None:
    # OCR text contains "Acme" and "Globex" — both resolve to entity IDs
    # via the normalizer, but the empty graph cache yields no embeddings,
    # so the resolver falls back to a zero graph vector.
    pipeline = _build_pipeline(populate_graph=False)
    await pipeline.initialize()
    try:
        result = await pipeline.classify(b"image")
    finally:
        await pipeline.shutdown()

    assert isinstance(result, ClassificationResult)
    assert 0.0 <= result.confidence <= 1.0
    assert abs(sum(result.all_probabilities.values()) - 1.0) < 1e-4
    # With no graph hits the resolver returns a zero embedding; no
    # graph evidence is reflected in the final confidence.
    assert isinstance(result.graph_confidence, float)
    assert result.graph_confidence <= 0.0
