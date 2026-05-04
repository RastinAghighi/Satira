import asyncio
from typing import Optional

import pytest
from httpx import ASGITransport, AsyncClient

from satira.api.app import create_app
from satira.inference.pipeline import ClassificationResult


# --- stubs -------------------------------------------------------------
class _StubBatcher:
    def __init__(self) -> None:
        self.total_requests = 0
        self.total_batches = 0

    def stats(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "avg_batch_size": (
                self.total_requests / self.total_batches if self.total_batches else 0.0
            ),
            "avg_latency_ms": 4.2,
            "queue_depth": 0,
        }


class _StubIndexManager:
    def get_index_stats(self) -> dict:
        return {
            "total_vectors": 100,
            "wal_size": 0,
            "index_type": "IVFFlat",
            "memory_bytes": 100 * 768 * 4,
        }


class _StubCachedRetriever:
    def __init__(self) -> None:
        self.hits = 7
        self.misses = 3

    def cache_stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }


class _StubTemporal:
    def __init__(self) -> None:
        self.index_manager = _StubIndexManager()
        self.cached_retriever = _StubCachedRetriever()


class _StubContextResolver:
    def __init__(self) -> None:
        self._temporal = _StubTemporal()


class StubPipeline:
    """Quacks like InferencePipeline for the API layer (no torch / no FAISS)."""

    def __init__(self, *, classify_delay_s: float = 0.0, predict_class: int = 1) -> None:
        self._initialized = False
        self._model: Optional[object] = object()
        self._batcher: Optional[_StubBatcher] = None
        self._context_resolver: Optional[_StubContextResolver] = None
        self._classify_delay = classify_delay_s
        self._predict_class = predict_class
        self.classify_calls = 0

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._batcher = _StubBatcher()
        self._context_resolver = _StubContextResolver()
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False
        self._batcher = None

    async def classify(self, image_bytes: bytes) -> ClassificationResult:
        if self._classify_delay:
            await asyncio.sleep(self._classify_delay)
        self.classify_calls += 1
        if self._batcher is not None:
            self._batcher.total_requests += 1
            self._batcher.total_batches += 1
        names = ["authentic", "satire", "parody", "misleading_context", "fabricated"]
        probs = [0.05, 0.05, 0.05, 0.03, 0.02]
        probs[self._predict_class] = 0.85
        return ClassificationResult(
            class_name=names[self._predict_class],
            class_index=self._predict_class,
            confidence=0.85,
            all_probabilities=dict(zip(names, probs)),
            t2v_attention=None,
            v2t_attention=None,
            graph_confidence=0.7,
            temporal_cache_hit=False,
            latency_ms=12.3,
        )


# --- fixtures ----------------------------------------------------------
def _client(pipeline: Optional[StubPipeline]) -> AsyncClient:
    app = create_app(pipeline=pipeline)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def initialized_pipeline():
    pipeline = StubPipeline()
    await pipeline.initialize()
    yield pipeline
    await pipeline.shutdown()


@pytest.fixture
async def client(initialized_pipeline):
    async with _client(initialized_pipeline) as c:
        yield c


def _png(name: str = "img.png", payload: bytes = b"\x89PNG\r\n\x1a\nfakepayload"):
    return ("file", (name, payload, "image/png"))


# --- tests -------------------------------------------------------------
async def test_health_returns_200_when_ready(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["batcher_running"] is True
    assert body["queue_depth"] == 0
    assert body["faiss_index"]["total_vectors"] == 100
    assert body["faiss_index"]["index_type"] == "IVFFlat"


async def test_health_reports_degraded_without_pipeline() -> None:
    async with _client(pipeline=None) as c:
        response = await c.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["batcher_running"] is False
    assert body["faiss_index"] is None


async def test_classify_returns_classification_result(client: AsyncClient) -> None:
    response = await client.post("/api/v1/classify", files=[_png()])

    assert response.status_code == 200
    body = response.json()
    assert body["class_name"] == "satire"
    assert body["class_index"] == 1
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["all_probabilities"].keys()) == {
        "authentic", "satire", "parody", "misleading_context", "fabricated",
    }
    assert body["latency_ms"] > 0.0
    assert body["request_id"]
    # Request id is echoed back via header
    assert response.headers["X-Request-ID"] == body["request_id"]


async def test_classify_propagates_request_id_from_header(client: AsyncClient) -> None:
    rid = "client-supplied-id-123"
    response = await client.post(
        "/api/v1/classify",
        files=[_png()],
        headers={"X-Request-ID": rid},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == rid
    assert response.json()["request_id"] == rid


async def test_classify_rejects_empty_upload(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/classify",
        files=[("file", ("empty.png", b"", "image/png"))],
    )
    assert response.status_code == 422


async def test_classify_rejects_missing_file(client: AsyncClient) -> None:
    response = await client.post("/api/v1/classify")
    assert response.status_code == 422


async def test_batch_endpoint_processes_multiple_images(client: AsyncClient) -> None:
    files = [
        ("files", (f"img{i}.png", b"\x89PNG\r\n\x1a\npayload" + bytes([i]), "image/png"))
        for i in range(4)
    ]
    response = await client.post("/api/v1/classify/batch", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 4
    assert len(body["results"]) == 4
    for item in body["results"]:
        assert item["class_name"] == "satire"
        assert 0.0 <= item["confidence"] <= 1.0
        assert item["request_id"] == body["request_id"]


async def test_batch_endpoint_runs_concurrently(initialized_pipeline) -> None:
    delay = 0.1
    initialized_pipeline._classify_delay = delay

    async with _client(initialized_pipeline) as c:
        files = [
            ("files", (f"img{i}.png", b"payload-bytes", "image/png"))
            for i in range(5)
        ]
        loop = asyncio.get_event_loop()
        start = loop.time()
        response = await c.post("/api/v1/classify/batch", files=files)
        elapsed = loop.time() - start

    assert response.status_code == 200
    assert response.json()["count"] == 5
    # Sequential lower bound is 5*delay; concurrent should sit close to a single delay.
    assert elapsed < delay * 5 * 0.7, (
        f"batch took {elapsed:.3f}s — looks serialized (5x={5*delay:.3f}s)"
    )


async def test_batch_rejects_when_any_file_is_empty(client: AsyncClient) -> None:
    files = [
        ("files", ("ok.png", b"data", "image/png")),
        ("files", ("bad.png", b"", "image/png")),
    ]
    response = await client.post("/api/v1/classify/batch", files=files)
    assert response.status_code == 422


async def test_classify_returns_503_when_pipeline_not_initialized() -> None:
    pipeline = StubPipeline()  # not initialized
    async with _client(pipeline) as c:
        response = await c.post("/api/v1/classify", files=[_png()])

    assert response.status_code == 503
    assert "not initialized" in response.json()["detail"].lower()


async def test_batch_returns_503_when_pipeline_not_initialized() -> None:
    async with _client(pipeline=None) as c:
        response = await c.post(
            "/api/v1/classify/batch",
            files=[("files", ("a.png", b"data", "image/png"))],
        )
    assert response.status_code == 503


async def test_metrics_returns_503_when_pipeline_not_initialized() -> None:
    async with _client(pipeline=None) as c:
        response = await c.get("/api/v1/metrics")
    assert response.status_code == 503


async def test_metrics_returns_batcher_and_cache_stats(client: AsyncClient) -> None:
    # Drive a couple of classifications so the batcher has non-zero counters.
    await client.post("/api/v1/classify", files=[_png()])
    await client.post("/api/v1/classify", files=[_png("img2.png")])

    response = await client.get("/api/v1/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["batcher"]["total_requests"] == 2
    assert body["batcher"]["total_batches"] == 2
    assert body["avg_latency_ms"] == pytest.approx(4.2)
    assert body["throughput_rps"] >= 0.0
    assert body["uptime_s"] > 0.0
    assert body["temporal_cache"]["hits"] == 7
    assert body["temporal_cache"]["misses"] == 3
    assert body["temporal_cache"]["hit_rate"] == pytest.approx(0.7)


async def test_cors_headers_present(client: AsyncClient) -> None:
    response = await client.get(
        "/api/v1/health",
        headers={"Origin": "https://example.com"},
    )
    assert response.status_code == 200
    allow_origin = response.headers.get("access-control-allow-origin")
    assert allow_origin in ("*", "https://example.com")
