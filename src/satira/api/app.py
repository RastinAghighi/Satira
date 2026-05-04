"""FastAPI application exposing the Satira inference pipeline over HTTP.

The app is created via ``create_app`` so tests can inject a stub pipeline
without spinning up real model weights, FAISS indices, or OCR engines.
The lifespan context only calls ``initialize`` / ``shutdown`` on whatever
pipeline lives on ``app.state.pipeline``.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from satira.api.models import (
    BatchClassificationResponse,
    BatcherStats,
    CacheStats,
    ClassificationResponse,
    FAISSIndexStats,
    HealthResponse,
    MetricsResponse,
)
from satira.inference.pipeline import ClassificationResult, InferencePipeline


REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a stable request id to every request and echo it on the response."""

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = rid
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        return response


def _attention_to_list(arr: Optional[np.ndarray]):
    if arr is None:
        return None
    return arr.tolist()


def _serialize_result(result: ClassificationResult, request_id: str) -> ClassificationResponse:
    return ClassificationResponse(
        class_name=result.class_name,
        class_index=result.class_index,
        confidence=result.confidence,
        all_probabilities=result.all_probabilities,
        t2v_attention=_attention_to_list(result.t2v_attention),
        v2t_attention=_attention_to_list(result.v2t_attention),
        graph_confidence=result.graph_confidence,
        temporal_cache_hit=result.temporal_cache_hit,
        latency_ms=result.latency_ms,
        request_id=request_id,
    )


def _pipeline_ready(p: Optional[InferencePipeline]) -> bool:
    return p is not None and getattr(p, "_initialized", False) and getattr(p, "_batcher", None) is not None


def get_pipeline(request: Request) -> InferencePipeline:
    """Dependency that returns the active pipeline or 503s."""
    p: Optional[InferencePipeline] = getattr(request.app.state, "pipeline", None)
    if not _pipeline_ready(p):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="inference pipeline is not initialized",
        )
    return p  # type: ignore[return-value]


def _faiss_stats(p: InferencePipeline) -> Optional[FAISSIndexStats]:
    temporal = getattr(getattr(p, "_context_resolver", None), "_temporal", None)
    index_manager = getattr(temporal, "index_manager", None)
    if index_manager is None:
        return None
    try:
        return FAISSIndexStats(**index_manager.get_index_stats())
    except Exception:
        return None


def _cache_stats(p: InferencePipeline) -> Optional[CacheStats]:
    temporal = getattr(getattr(p, "_context_resolver", None), "_temporal", None)
    cached = getattr(temporal, "cached_retriever", None)
    if cached is None:
        return None
    try:
        return CacheStats(**cached.cache_stats())
    except Exception:
        return None


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"empty upload for file '{file.filename or 'unknown'}'",
        )
    return data


def create_app(
    pipeline: Optional[InferencePipeline] = None,
    pipeline_factory: Optional[Callable[[], InferencePipeline]] = None,
    *,
    cors_allow_origins: Optional[list[str]] = None,
) -> FastAPI:
    """Build a FastAPI app wired to ``pipeline`` (or one built by ``pipeline_factory``)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "pipeline", None) is None and pipeline_factory is not None:
            app.state.pipeline = pipeline_factory()
        p: Optional[InferencePipeline] = getattr(app.state, "pipeline", None)
        if p is not None:
            await p.initialize()
        app.state.start_time = time.monotonic()
        try:
            yield
        finally:
            p = getattr(app.state, "pipeline", None)
            if p is not None:
                await p.shutdown()

    app = FastAPI(title="Satira Inference API", version="0.1.0", lifespan=lifespan)
    app.state.pipeline = pipeline
    app.state.start_time = time.monotonic()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        p: Optional[InferencePipeline] = getattr(request.app.state, "pipeline", None)
        model_loaded = p is not None and getattr(p, "_model", None) is not None
        running = _pipeline_ready(p)

        queue_depth = 0
        if p is not None and getattr(p, "_batcher", None) is not None:
            queue_depth = int(p._batcher.stats().get("queue_depth", 0))

        faiss = _faiss_stats(p) if p is not None else None
        return HealthResponse(
            status="ok" if (model_loaded and running) else "degraded",
            model_loaded=model_loaded,
            batcher_running=running,
            queue_depth=queue_depth,
            faiss_index=faiss,
        )

    @app.get("/api/v1/metrics", response_model=MetricsResponse)
    async def metrics(
        request: Request,
        p: InferencePipeline = Depends(get_pipeline),
    ) -> MetricsResponse:
        raw = p._batcher.stats()  # type: ignore[union-attr]
        elapsed = max(time.monotonic() - request.app.state.start_time, 1e-9)
        throughput = raw["total_requests"] / elapsed
        return MetricsResponse(
            batcher=BatcherStats(**raw),
            temporal_cache=_cache_stats(p),
            avg_latency_ms=float(raw["avg_latency_ms"]),
            throughput_rps=float(throughput),
            uptime_s=float(elapsed),
        )

    @app.post("/api/v1/classify", response_model=ClassificationResponse)
    async def classify(
        request: Request,
        file: UploadFile = File(...),
        p: InferencePipeline = Depends(get_pipeline),
    ) -> ClassificationResponse:
        image_bytes = await _read_upload(file)
        result = await p.classify(image_bytes)
        return _serialize_result(result, request.state.request_id)

    @app.post("/api/v1/classify/batch", response_model=BatchClassificationResponse)
    async def classify_batch(
        request: Request,
        files: list[UploadFile] = File(...),
        p: InferencePipeline = Depends(get_pipeline),
    ) -> BatchClassificationResponse:
        if not files:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="no files uploaded",
            )
        all_bytes = await asyncio.gather(*[_read_upload(f) for f in files])
        results = await asyncio.gather(*[p.classify(b) for b in all_bytes])
        rid = request.state.request_id
        return BatchClassificationResponse(
            results=[_serialize_result(r, rid) for r in results],
            request_id=rid,
            count=len(results),
        )

    return app


app = create_app()
