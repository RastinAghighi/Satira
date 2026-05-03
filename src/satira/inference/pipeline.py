"""Top-level inference pipeline that orchestrates the full classification flow.

ContextResolver fans out the four-stream preprocessing concurrently,
DynamicBatcher coalesces submissions into batched GPU forward passes,
and this pipeline ties the two together behind a single ``classify``
entry point that the API layer can call.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from satira.config import Settings
from satira.inference.batcher import DynamicBatcher
from satira.inference.context_resolver import ContextResolver
from satira.models.engine import SatireDetectionEngine


@dataclass
class ClassificationResult:
    class_name: str
    class_index: int
    confidence: float
    all_probabilities: dict[str, float]
    t2v_attention: np.ndarray | None
    v2t_attention: np.ndarray | None
    graph_confidence: float
    temporal_cache_hit: bool
    latency_ms: float


class InferencePipeline:
    """Orchestrates ``image bytes -> ClassificationResult`` end to end.

    The constructor takes already-built sub-components so the pipeline
    stays decoupled from how the heavy resources (model weights, FAISS
    index, OCR engine) are loaded; ``initialize`` only spins up the
    DynamicBatcher's background loop.
    """

    def __init__(
        self,
        config: Settings,
        context_resolver: ContextResolver,
        model: SatireDetectionEngine | Any,
        device: str = "cpu",
    ) -> None:
        self._config = config
        self._context_resolver = context_resolver
        self._model = model
        self._device = device
        self._batcher: DynamicBatcher | None = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._batcher = DynamicBatcher(
            model=self._model,
            max_batch=self._config.max_batch_size,
            max_wait_ms=self._config.batch_timeout_ms,
            device=self._device,
        )
        self._batcher.start()
        self._initialized = True

    async def classify(self, image_bytes: bytes) -> ClassificationResult:
        if not self._initialized or self._batcher is None:
            raise RuntimeError("pipeline not initialized; call initialize() first")

        start = time.monotonic()
        request = await self._context_resolver.resolve(image_bytes)
        raw = await self._batcher.submit(request)
        latency_ms = (time.monotonic() - start) * 1000.0

        return self._build_result(raw, latency_ms)

    async def shutdown(self) -> None:
        if self._batcher is not None:
            await self._batcher.stop()
            self._batcher = None
        self._initialized = False

    def _build_result(self, raw: dict, latency_ms: float) -> ClassificationResult:
        names = list(self._config.CLASS_NAMES)
        probs = raw["probs"]
        class_index = int(raw["predicted_class"])
        all_probs = {names[i]: float(probs[i].item()) for i in range(len(names))}
        return ClassificationResult(
            class_name=names[class_index],
            class_index=class_index,
            confidence=float(raw["confidence"]),
            all_probabilities=all_probs,
            t2v_attention=None,
            v2t_attention=None,
            graph_confidence=0.0,
            temporal_cache_hit=False,
            latency_ms=latency_ms,
        )
