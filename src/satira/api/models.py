"""Pydantic request/response schemas for the Satira inference HTTP API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ClassificationResponse(BaseModel):
    class_name: str
    class_index: int
    confidence: float
    all_probabilities: dict[str, float]
    t2v_attention: Optional[list] = None
    v2t_attention: Optional[list] = None
    graph_confidence: float
    temporal_cache_hit: bool
    latency_ms: float
    request_id: str


class BatchClassificationResponse(BaseModel):
    results: list[ClassificationResponse]
    request_id: str
    count: int


class FAISSIndexStats(BaseModel):
    total_vectors: int
    wal_size: int
    index_type: str
    memory_bytes: int


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' if all subsystems are ready, else 'degraded'")
    model_loaded: bool
    batcher_running: bool
    queue_depth: int
    faiss_index: Optional[FAISSIndexStats] = None


class CacheStats(BaseModel):
    hits: int
    misses: int
    hit_rate: float


class BatcherStats(BaseModel):
    total_requests: int
    total_batches: int
    avg_batch_size: float
    avg_latency_ms: float
    queue_depth: int


class MetricsResponse(BaseModel):
    batcher: BatcherStats
    temporal_cache: Optional[CacheStats] = None
    avg_latency_ms: float
    throughput_rps: float
    uptime_s: float


class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None
