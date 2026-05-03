"""Dynamic batcher for inference requests.

Collects async inference requests across submitters and runs a single batched
GPU forward pass per cycle. Replaces TorchServe so we keep full control over
the four-stream input format and avoid extra abstraction layers.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import torch

from satira.models.engine import SatireDetectionEngine


@dataclass
class InferenceRequest:
    """Single inference request with pre-processed tensors and a future for the result."""

    vision_emb: torch.Tensor
    text_emb: torch.Tensor
    temporal_emb: torch.Tensor
    graph_emb: torch.Tensor
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    future: Optional[asyncio.Future] = None
    timestamp: float = 0.0


class DynamicBatcher:
    """Async dynamic batcher that fans submissions into batched forward passes.

    During low traffic the loop waits up to ``max_wait_ms`` for additional
    requests, then runs whatever has accumulated. During surges, the batch
    fills before the timeout and dispatches immediately.
    """

    def __init__(
        self,
        model: SatireDetectionEngine,
        max_batch: int = 32,
        max_wait_ms: float = 50.0,
        device: str = "cuda",
    ) -> None:
        self._model = model
        self._max_batch = max_batch
        self._max_wait_ms = max_wait_ms
        self._device = device

        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._running = False

        self._total_requests = 0
        self._total_batches = 0
        self._total_latency_ms = 0.0
        self._batch_sizes_sum = 0

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            await self._task
            self._task = None

    async def submit(self, request: InferenceRequest) -> dict:
        if request.future is None:
            request.future = asyncio.get_running_loop().create_future()
        if request.timestamp == 0.0:
            request.timestamp = time.monotonic()
        await self._queue.put(request)
        return await request.future

    async def _batch_loop(self) -> None:
        try:
            while self._running:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
        finally:
            await self._drain()

    async def _collect_batch(self) -> list[InferenceRequest]:
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            return []

        batch = [first]
        deadline = time.monotonic() + self._max_wait_ms / 1000.0

        while len(batch) < self._max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(req)
            except asyncio.TimeoutError:
                break

        return batch

    async def _drain(self) -> None:
        batch: list[InferenceRequest] = []
        while True:
            try:
                req = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            batch.append(req)
            if len(batch) >= self._max_batch:
                await self._process_batch(batch)
                batch = []
        if batch:
            await self._process_batch(batch)

    async def _process_batch(self, batch: list[InferenceRequest]) -> None:
        try:
            logits, probs = await asyncio.to_thread(self._forward, batch)
        except Exception as exc:
            for req in batch:
                if req.future is not None and not req.future.done():
                    req.future.set_exception(exc)
            return

        now = time.monotonic()
        for i, req in enumerate(batch):
            latency_ms = (now - req.timestamp) * 1000.0
            self._total_latency_ms += latency_ms
            result = {
                "request_id": req.request_id,
                "logits": logits[i],
                "probs": probs[i],
                "predicted_class": int(probs[i].argmax().item()),
                "confidence": float(probs[i].max().item()),
                "latency_ms": latency_ms,
            }
            if req.future is not None and not req.future.done():
                req.future.set_result(result)

        self._total_requests += len(batch)
        self._total_batches += 1
        self._batch_sizes_sum += len(batch)

    def _forward(self, batch: list[InferenceRequest]) -> tuple[torch.Tensor, torch.Tensor]:
        v = torch.stack([r.vision_emb for r in batch]).to(self._device)
        t = torch.stack([r.text_emb for r in batch]).to(self._device)
        temp = torch.stack([r.temporal_emb for r in batch]).to(self._device)
        g = torch.stack([r.graph_emb for r in batch]).to(self._device)

        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad():
                logits, *_ = self._model(v, t, temp, g)
        finally:
            if was_training:
                self._model.train()

        probs = torch.softmax(logits, dim=-1)
        return logits.detach().cpu(), probs.detach().cpu()

    def stats(self) -> dict:
        avg_batch = (
            self._batch_sizes_sum / self._total_batches if self._total_batches else 0.0
        )
        avg_lat = (
            self._total_latency_ms / self._total_requests if self._total_requests else 0.0
        )
        return {
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "avg_batch_size": avg_batch,
            "avg_latency_ms": avg_lat,
            "queue_depth": self._queue.qsize(),
        }
