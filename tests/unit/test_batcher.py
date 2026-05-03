import asyncio
import time

import torch
import torch.nn as nn

from satira.inference.batcher import DynamicBatcher, InferenceRequest


class _StubEngine(nn.Module):
    """Minimal stand-in for SatireDetectionEngine — same forward signature."""

    def __init__(self, num_classes: int = 5, forward_delay_s: float = 0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.forward_delay_s = forward_delay_s
        # parameters so .train()/.eval()/.to() behave like a real module
        self._linear = nn.Linear(1, num_classes)

    def forward(
        self,
        v: torch.Tensor,
        t: torch.Tensor,
        temp: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.forward_delay_s:
            time.sleep(self.forward_delay_s)
        batch = v.shape[0]
        logits = torch.zeros(batch, self.num_classes)
        for i in range(batch):
            logits[i, i % self.num_classes] = 5.0
        dummy = torch.zeros(batch, 1, 1)
        return logits, dummy, dummy, dummy, dummy


def _make_request(seed: int = 0) -> InferenceRequest:
    return InferenceRequest(
        vision_emb=torch.randn(10, 64),
        text_emb=torch.randn(12, 64),
        temporal_emb=torch.randn(64),
        graph_emb=torch.randn(64),
        request_id=f"req-{seed}",
    )


async def test_single_request_completes() -> None:
    batcher = DynamicBatcher(model=_StubEngine(), max_batch=4, max_wait_ms=20.0, device="cpu")
    batcher.start()
    try:
        result = await batcher.submit(_make_request(0))
    finally:
        await batcher.stop()

    assert result["request_id"] == "req-0"
    assert isinstance(result["logits"], torch.Tensor)
    assert isinstance(result["probs"], torch.Tensor)
    assert result["logits"].shape == (5,)
    assert 0 <= result["predicted_class"] < 5
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["latency_ms"] >= 0.0


async def test_concurrent_requests_are_batched_together() -> None:
    batcher = DynamicBatcher(model=_StubEngine(), max_batch=8, max_wait_ms=100.0, device="cpu")
    batcher.start()
    try:
        results = await asyncio.gather(
            *[batcher.submit(_make_request(i)) for i in range(5)]
        )
    finally:
        await batcher.stop()

    assert len(results) == 5
    ids = {r["request_id"] for r in results}
    assert ids == {f"req-{i}" for i in range(5)}

    stats = batcher.stats()
    assert stats["total_requests"] == 5
    # Five concurrent submissions should coalesce into far fewer than 5 batches
    assert stats["total_batches"] < 5
    assert stats["avg_batch_size"] > 1.0


async def test_max_batch_is_respected() -> None:
    batcher = DynamicBatcher(model=_StubEngine(), max_batch=3, max_wait_ms=200.0, device="cpu")
    batcher.start()
    try:
        results = await asyncio.gather(
            *[batcher.submit(_make_request(i)) for i in range(7)]
        )
    finally:
        await batcher.stop()

    assert len(results) == 7
    stats = batcher.stats()
    assert stats["total_requests"] == 7
    # 7 requests / 3 max_batch → at least 3 batches
    assert stats["total_batches"] >= 3
    assert stats["avg_batch_size"] <= 3.0


async def test_timeout_triggers_with_partial_batch() -> None:
    wait_ms = 40.0
    batcher = DynamicBatcher(
        model=_StubEngine(), max_batch=64, max_wait_ms=wait_ms, device="cpu"
    )
    batcher.start()
    try:
        start = time.monotonic()
        result = await batcher.submit(_make_request(0))
        elapsed_ms = (time.monotonic() - start) * 1000.0
    finally:
        await batcher.stop()

    assert result["request_id"] == "req-0"
    # A lone request must wait at least max_wait_ms before the loop dispatches it
    assert elapsed_ms >= wait_ms * 0.8

    stats = batcher.stats()
    assert stats["total_batches"] == 1
    assert stats["total_requests"] == 1
    assert stats["avg_batch_size"] == 1.0


async def test_stats_are_tracked_correctly() -> None:
    batcher = DynamicBatcher(model=_StubEngine(), max_batch=4, max_wait_ms=20.0, device="cpu")

    initial = batcher.stats()
    assert initial == {
        "total_requests": 0,
        "total_batches": 0,
        "avg_batch_size": 0.0,
        "avg_latency_ms": 0.0,
        "queue_depth": 0,
    }

    batcher.start()
    try:
        await asyncio.gather(*[batcher.submit(_make_request(i)) for i in range(6)])
    finally:
        await batcher.stop()

    stats = batcher.stats()
    assert stats["total_requests"] == 6
    assert stats["total_batches"] >= 1
    assert stats["total_batches"] <= 6
    assert stats["avg_batch_size"] == 6 / stats["total_batches"]
    assert stats["avg_latency_ms"] > 0.0
    assert stats["queue_depth"] == 0
