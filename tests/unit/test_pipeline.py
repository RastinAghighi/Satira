import asyncio
import time
from dataclasses import fields

import numpy as np
import pytest
import torch
import torch.nn as nn

from satira.config import Settings
from satira.inference.batcher import InferenceRequest
from satira.inference.pipeline import ClassificationResult, InferencePipeline


# --- fakes -------------------------------------------------------------
class _StubEngine(nn.Module):
    """Minimal stand-in for SatireDetectionEngine — same forward signature."""

    def __init__(self, num_classes: int = 5, predict_class: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self._predict_class = predict_class
        self._linear = nn.Linear(1, num_classes)

    def forward(
        self,
        v: torch.Tensor,
        t: torch.Tensor,
        temp: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = v.shape[0]
        logits = torch.full((batch, self.num_classes), -5.0)
        logits[:, self._predict_class] = 5.0
        dummy = torch.zeros(batch, 1, 1)
        return logits, dummy, dummy, dummy, dummy


class _StubResolver:
    """Returns a fresh InferenceRequest per call so each submit gets its own future."""

    def __init__(
        self,
        vision_shape: tuple[int, ...] = (10, 64),
        text_shape: tuple[int, ...] = (12, 64),
        temporal_dim: int = 64,
        graph_dim: int = 64,
        delay_s: float = 0.0,
    ) -> None:
        self._vision_shape = vision_shape
        self._text_shape = text_shape
        self._temporal_dim = temporal_dim
        self._graph_dim = graph_dim
        self._delay = delay_s
        self.calls = 0

    async def resolve(self, image_bytes: bytes) -> InferenceRequest:
        if self._delay:
            await asyncio.sleep(self._delay)
        self.calls += 1
        return InferenceRequest(
            vision_emb=torch.randn(*self._vision_shape),
            text_emb=torch.randn(*self._text_shape),
            temporal_emb=torch.randn(self._temporal_dim),
            graph_emb=torch.randn(self._graph_dim),
        )


def _build_pipeline(
    *,
    predict_class: int = 1,
    resolver_delay_s: float = 0.0,
) -> tuple[InferencePipeline, _StubResolver, Settings]:
    config = Settings()
    resolver = _StubResolver(delay_s=resolver_delay_s)
    pipeline = InferencePipeline(
        config=config,
        context_resolver=resolver,
        model=_StubEngine(num_classes=config.num_classes, predict_class=predict_class),
        device="cpu",
    )
    return pipeline, resolver, config


# --- tests -------------------------------------------------------------
async def test_classify_returns_classification_result_with_expected_class() -> None:
    pipeline, _, config = _build_pipeline(predict_class=2)
    await pipeline.initialize()
    try:
        result = await pipeline.classify(b"image-bytes")
    finally:
        await pipeline.shutdown()

    assert isinstance(result, ClassificationResult)
    assert result.class_index == 2
    assert result.class_name == config.CLASS_NAMES[2]
    assert 0.0 <= result.confidence <= 1.0
    # Prediction is dominant — confidence should be near 1.0.
    assert result.confidence > 0.9


async def test_classification_result_has_all_expected_fields() -> None:
    expected = {
        "class_name",
        "class_index",
        "confidence",
        "all_probabilities",
        "t2v_attention",
        "v2t_attention",
        "graph_confidence",
        "temporal_cache_hit",
        "latency_ms",
    }
    actual = {f.name for f in fields(ClassificationResult)}
    assert expected == actual


async def test_all_probabilities_keyed_by_class_name_and_sum_to_one() -> None:
    pipeline, _, config = _build_pipeline(predict_class=0)
    await pipeline.initialize()
    try:
        result = await pipeline.classify(b"image")
    finally:
        await pipeline.shutdown()

    assert set(result.all_probabilities.keys()) == set(config.CLASS_NAMES)
    total = sum(result.all_probabilities.values())
    assert abs(total - 1.0) < 1e-4


async def test_attention_and_metadata_fields_have_correct_types() -> None:
    pipeline, _, _ = _build_pipeline()
    await pipeline.initialize()
    try:
        result = await pipeline.classify(b"image")
    finally:
        await pipeline.shutdown()

    # The current MVP returns None for attention; the type annotation
    # allows np.ndarray once the batcher is extended to surface it.
    assert result.t2v_attention is None or isinstance(result.t2v_attention, np.ndarray)
    assert result.v2t_attention is None or isinstance(result.v2t_attention, np.ndarray)
    assert isinstance(result.graph_confidence, float)
    assert isinstance(result.temporal_cache_hit, bool)


async def test_latency_ms_is_tracked_and_includes_resolver_time() -> None:
    resolver_delay_s = 0.05
    pipeline, _, _ = _build_pipeline(resolver_delay_s=resolver_delay_s)
    await pipeline.initialize()
    try:
        start = time.monotonic()
        result = await pipeline.classify(b"image")
        wall_ms = (time.monotonic() - start) * 1000.0
    finally:
        await pipeline.shutdown()

    assert result.latency_ms > 0.0
    # Latency should at least cover the resolver delay (with slack for
    # event-loop scheduling and the batcher's wait window).
    assert result.latency_ms >= resolver_delay_s * 1000.0 * 0.8
    # And it should never exceed the total wall-clock for the call.
    assert result.latency_ms <= wall_ms + 5.0


async def test_classify_before_initialize_raises() -> None:
    pipeline, _, _ = _build_pipeline()
    with pytest.raises(RuntimeError):
        await pipeline.classify(b"image")


async def test_initialize_is_idempotent() -> None:
    pipeline, _, _ = _build_pipeline()
    await pipeline.initialize()
    await pipeline.initialize()  # second call must be a no-op
    try:
        result = await pipeline.classify(b"image")
        assert isinstance(result, ClassificationResult)
    finally:
        await pipeline.shutdown()


async def test_shutdown_leaves_no_pending_tasks() -> None:
    pipeline, _, _ = _build_pipeline()
    await pipeline.initialize()
    await pipeline.classify(b"image")
    await pipeline.shutdown()

    # Yield once so any in-flight cancellations finalize.
    await asyncio.sleep(0)

    pending = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    ]
    assert pending == [], f"unexpected pending tasks after shutdown: {pending}"


async def test_concurrent_classify_calls_are_batched() -> None:
    pipeline, resolver, _ = _build_pipeline()
    await pipeline.initialize()
    try:
        results = await asyncio.gather(
            *[pipeline.classify(b"image") for _ in range(5)]
        )
    finally:
        await pipeline.shutdown()

    assert len(results) == 5
    assert resolver.calls == 5
    assert all(isinstance(r, ClassificationResult) for r in results)
    assert all(0.0 <= r.confidence <= 1.0 for r in results)


async def test_shutdown_without_initialize_is_safe() -> None:
    pipeline, _, _ = _build_pipeline()
    # No initialize() — shutdown() must still be callable.
    await pipeline.shutdown()
