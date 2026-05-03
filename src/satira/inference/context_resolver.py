"""CPU-bound preprocessing for inference requests (Tier B).

Translates a raw image into the four embedding streams the inference
engine expects (vision, text, temporal, graph). Vision/text encoding,
graph context resolution, and temporal retrieval all run concurrently
so wall-clock latency tracks the slowest stream rather than their sum.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import torch

from satira.graph.embedding_cache import GraphEmbeddingCache
from satira.graph.entity_resolution import EntityResolutionResult, MentionNormalizer
from satira.inference.batcher import InferenceRequest
from satira.temporal.retriever import TemporalContextRetriever


_DEFAULT_GRAPH_TIMEOUT_S = 0.05
_MAX_NGRAM = 3
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")


class ContextResolver:
    """Preprocesses incoming images into the four-stream input format.

    Runs OCR first because the extracted text feeds three of the four
    downstream tasks (text encoding, entity resolution, temporal
    retrieval). After OCR, vision encoding, text encoding, graph
    context lookup, and temporal retrieval fan out concurrently.
    """

    def __init__(
        self,
        mention_normalizer: MentionNormalizer,
        graph_cache: GraphEmbeddingCache,
        temporal_retriever: TemporalContextRetriever,
        text_encoder: Any = None,
        vision_encoder: Any = None,
        ocr_engine: Any = None,
        graph_timeout_s: float = _DEFAULT_GRAPH_TIMEOUT_S,
    ) -> None:
        self._normalizer = mention_normalizer
        self._graph_cache = graph_cache
        self._temporal = temporal_retriever
        self._text_encoder = text_encoder
        self._vision_encoder = vision_encoder
        self._ocr_engine = ocr_engine
        self._graph_timeout_s = graph_timeout_s

    async def resolve(self, image_bytes: bytes) -> InferenceRequest:
        extracted_text = await self._extract_text(image_bytes)

        vision_task = asyncio.create_task(self._encode_vision(image_bytes))
        text_task = asyncio.create_task(self._encode_text(extracted_text))
        graph_task = asyncio.create_task(self._resolve_graph_context(extracted_text))
        temporal_task = asyncio.create_task(self._resolve_temporal_context(extracted_text))

        vision_emb, text_emb, graph_result, temporal_emb = await asyncio.gather(
            vision_task, text_task, graph_task, temporal_task
        )
        graph_emb, _confidence = graph_result

        return InferenceRequest(
            vision_emb=vision_emb,
            text_emb=text_emb,
            temporal_emb=temporal_emb,
            graph_emb=graph_emb,
            timestamp=time.monotonic(),
        )

    async def _extract_text(self, image_bytes: bytes) -> str:
        if self._ocr_engine is None:
            return ""
        return await asyncio.to_thread(self._ocr_engine.extract_text, image_bytes)

    async def _encode_vision(self, image_bytes: bytes) -> torch.Tensor:
        if self._vision_encoder is None:
            raise ValueError("vision_encoder is required")
        result = await asyncio.to_thread(self._vision_encoder.encode, image_bytes)
        return _to_tensor(result)

    async def _encode_text(self, text: str) -> torch.Tensor:
        if self._text_encoder is None:
            raise ValueError("text_encoder is required")
        result = await asyncio.to_thread(self._text_encoder.encode, text)
        return _to_tensor(result)

    async def _resolve_temporal_context(self, extracted_text: str) -> torch.Tensor:
        # The retriever already enforces its own timeout and falls back
        # to a learned default embedding, so no extra wait_for here.
        return await self._temporal.retrieve(extracted_text)

    async def _resolve_graph_context(self, extracted_text: str) -> tuple[torch.Tensor, float]:
        """Resolve mentions, fetch graph embeddings, attention-pool them.

        Returns a zero vector with 0.0 confidence whenever no candidate
        produces a usable graph hit, or when the bounded timeout fires.
        The downstream model is trained with graph dropout, so a zero
        fallback is a known-handled case.
        """
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._sync_resolve_graph, extracted_text),
                timeout=self._graph_timeout_s,
            )
        except asyncio.TimeoutError:
            return self._graph_fallback()

    def _sync_resolve_graph(self, extracted_text: str) -> tuple[torch.Tensor, float]:
        candidates = _candidate_mentions(extracted_text)
        if not candidates:
            return self._graph_fallback()

        seen: set[str] = set()
        embeddings: list[torch.Tensor] = []
        weights: list[float] = []
        confidences: list[float] = []

        for mention in candidates:
            entity_id, score = self._normalizer.normalize(mention)
            if entity_id is None or entity_id in seen:
                continue
            emb = self._graph_cache.get(entity_id)
            if emb is None:
                continue

            resolution_type = "exact_alias" if score >= 1.0 else "high_confidence"
            weight = EntityResolutionResult(
                entity_id=entity_id,
                confidence=score,
                resolution_type=resolution_type,
            ).graph_weight
            if weight <= 0.0:
                continue

            seen.add(entity_id)
            embeddings.append(emb)
            weights.append(weight)
            confidences.append(score)

        if not embeddings:
            return self._graph_fallback()

        pooled = self._graph_cache.attention_pool(embeddings, weights)
        avg_confidence = sum(confidences) / len(confidences)
        return pooled, avg_confidence

    def _graph_fallback(self) -> tuple[torch.Tensor, float]:
        return torch.zeros(self._graph_cache.embedding_dim), 0.0


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach()
    return torch.as_tensor(value)


def _candidate_mentions(text: str) -> list[str]:
    """Token n-grams (1..3) as candidate entity mentions.

    Keeps the hot path NER-free; the MentionNormalizer's confidence
    threshold filters out garbage candidates.
    """
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return []
    out: list[str] = []
    for n in range(1, _MAX_NGRAM + 1):
        for i in range(len(tokens) - n + 1):
            out.append(" ".join(tokens[i : i + n]))
    return out
