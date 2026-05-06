"""Unit tests for BaseScraper.

All HTTP traffic is faked through ``httpx.MockTransport`` so the tests
run hermetically. Each test installs its own handler so we can shape
the response sequence (success, 503-then-200, repeated failures, …)
that exercises the retry and rate-limit logic.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Callable

import httpx
import pytest

from satira.ingest.base_scraper import BaseScraper, ScrapedItem


# --- helpers ----------------------------------------------------------------
def _ok(body: bytes | str = b"hello", content_type: str = "text/html") -> httpx.Response:
    return httpx.Response(200, content=body, headers={"content-type": content_type})


def _empty_robots() -> httpx.Response:
    # An empty robots.txt is "allow everything".
    return httpx.Response(200, content=b"", headers={"content-type": "text/plain"})


def _disallow_all_robots() -> httpx.Response:
    body = b"User-agent: *\nDisallow: /\n"
    return httpx.Response(200, content=body, headers={"content-type": "text/plain"})


class _Handler:
    """Stateful request handler for ``httpx.MockTransport``.

    Routes ``/robots.txt`` to one queue and everything else to another so
    tests can shape the response sequence for the URL under test
    independently of the robots fetch.
    """

    def __init__(
        self,
        robots: httpx.Response | None = None,
        responses: list[httpx.Response] | None = None,
    ) -> None:
        self.robots = robots if robots is not None else _empty_robots()
        self.responses = list(responses) if responses else []
        self.calls: list[str] = []
        self.call_times: list[float] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(str(request.url))
        self.call_times.append(time.monotonic())
        if request.url.path == "/robots.txt":
            return self.robots
        if not self.responses:
            raise AssertionError(f"unexpected request: {request.url}")
        return self.responses.pop(0)


def _make_scraper(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    rate_limit_per_minute: int = 6000,  # ~10ms spacing — fast for tests
    respect_robots: bool = True,
) -> BaseScraper:
    """Build a concrete BaseScraper subclass wired to a MockTransport."""

    class _Concrete(BaseScraper):
        async def scrape(self, **kwargs: Any) -> AsyncIterator[ScrapedItem]:
            yield ScrapedItem(
                source_url="x",
                image_url=None,
                title="t",
                text="b",
                timestamp=datetime(2026, 1, 1),
                source_domain="x",
                metadata={},
            )

    scraper = _Concrete(
        rate_limit_per_minute=rate_limit_per_minute,
        respect_robots=respect_robots,
    )
    scraper._client = httpx.AsyncClient(
        headers={"User-Agent": scraper.user_agent},
        timeout=scraper.timeout,
        transport=httpx.MockTransport(handler),
    )
    return scraper


# --- ScrapedItem ------------------------------------------------------------
def test_scraped_item_metadata_defaults_to_empty_dict() -> None:
    item = ScrapedItem(
        source_url="https://example.com/a",
        image_url=None,
        title="t",
        text="body",
        timestamp=datetime(2026, 1, 1),
        source_domain="example.com",
    )
    assert item.metadata == {}


# --- abstract enforcement ---------------------------------------------------
def test_base_scraper_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseScraper()  # type: ignore[abstract]


# --- fetch: success ---------------------------------------------------------
async def test_fetch_returns_body_on_200() -> None:
    handler = _Handler(responses=[_ok(b"<html>ok</html>")])
    scraper = _make_scraper(handler)
    try:
        body = await scraper.fetch("https://example.com/page")
    finally:
        await scraper.close()

    assert body == "<html>ok</html>"
    assert scraper.stats.requests_succeeded == 1
    assert scraper.stats.requests_failed == 0
    assert scraper.stats.retries == 0
    assert scraper.stats.bytes_downloaded == len(b"<html>ok</html>")


# --- fetch: retry on transient errors ---------------------------------------
async def test_fetch_retries_on_503_then_succeeds() -> None:
    handler = _Handler(
        responses=[
            httpx.Response(503),
            httpx.Response(503),
            _ok(b"finally"),
        ]
    )
    scraper = _make_scraper(handler)
    try:
        body = await scraper.fetch("https://example.com/flaky")
    finally:
        await scraper.close()

    assert body == "finally"
    assert scraper.stats.retries == 2
    assert scraper.stats.requests_succeeded == 1
    # 2 failed attempts + 1 success.
    assert scraper.stats.requests_failed == 2


async def test_fetch_retries_on_transport_error_then_succeeds() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise httpx.ConnectError("boom", request=request)
        return _ok(b"recovered")

    scraper = _make_scraper(handler)
    try:
        body = await scraper.fetch("https://example.com/x")
    finally:
        await scraper.close()

    assert body == "recovered"
    assert scraper.stats.retries == 1


async def test_fetch_gives_up_after_exhausting_retries() -> None:
    handler = _Handler(responses=[httpx.Response(503) for _ in range(10)])
    scraper = _make_scraper(handler)
    try:
        body = await scraper.fetch("https://example.com/dead")
    finally:
        await scraper.close()

    assert body is None
    # 1 initial + 3 retries = 4 attempts.
    assert scraper.stats.requests_sent == 4
    assert scraper.stats.retries == 3
    assert scraper.stats.requests_succeeded == 0


# --- fetch: non-retryable errors --------------------------------------------
async def test_fetch_does_not_retry_on_404() -> None:
    handler = _Handler(responses=[httpx.Response(404)])
    scraper = _make_scraper(handler)
    try:
        body = await scraper.fetch("https://example.com/missing")
    finally:
        await scraper.close()

    assert body is None
    # No retries — single attempt.
    assert scraper.stats.requests_sent == 1
    assert scraper.stats.retries == 0


# --- fetch_image ------------------------------------------------------------
async def test_fetch_image_returns_bytes_for_image_content_type() -> None:
    handler = _Handler(responses=[_ok(b"\x89PNGdata", content_type="image/png")])
    scraper = _make_scraper(handler)
    try:
        data = await scraper.fetch_image("https://example.com/cat.png")
    finally:
        await scraper.close()

    assert data == b"\x89PNGdata"
    assert scraper.stats.requests_succeeded == 1


async def test_fetch_image_rejects_non_image_content_type() -> None:
    handler = _Handler(responses=[_ok(b"<html>not an image</html>", content_type="text/html")])
    scraper = _make_scraper(handler)
    try:
        data = await scraper.fetch_image("https://example.com/fake.jpg")
    finally:
        await scraper.close()

    assert data is None
    # The HTTP request still succeeded — only the content-type check failed.
    assert scraper.stats.requests_succeeded == 1


# --- robots.txt -------------------------------------------------------------
async def test_robots_txt_blocks_disallowed_url() -> None:
    handler = _Handler(robots=_disallow_all_robots(), responses=[_ok()])
    scraper = _make_scraper(handler)
    try:
        body = await scraper.fetch("https://example.com/secret")
    finally:
        await scraper.close()

    assert body is None
    assert scraper.stats.robots_blocked == 1
    # Only the robots.txt fetch hit the server — the actual URL was skipped.
    assert any(call.endswith("/robots.txt") for call in handler.calls)
    assert not any(call.endswith("/secret") for call in handler.calls)


async def test_respect_robots_false_bypasses_check() -> None:
    handler = _Handler(robots=_disallow_all_robots(), responses=[_ok(b"got it")])
    scraper = _make_scraper(handler, respect_robots=False)
    try:
        body = await scraper.fetch("https://example.com/secret")
    finally:
        await scraper.close()

    assert body == "got it"
    assert scraper.stats.robots_blocked == 0
    # robots.txt should never have been requested.
    assert not any(call.endswith("/robots.txt") for call in handler.calls)


async def test_robots_txt_cached_per_host() -> None:
    handler = _Handler(responses=[_ok(b"a"), _ok(b"b")])
    scraper = _make_scraper(handler)
    try:
        await scraper.fetch("https://example.com/one")
        await scraper.fetch("https://example.com/two")
    finally:
        await scraper.close()

    robots_calls = [c for c in handler.calls if c.endswith("/robots.txt")]
    assert len(robots_calls) == 1


async def test_robots_txt_missing_treated_as_allow_all() -> None:
    handler = _Handler(robots=httpx.Response(404), responses=[_ok(b"ok")])
    scraper = _make_scraper(handler)
    try:
        body = await scraper.fetch("https://example.com/page")
    finally:
        await scraper.close()

    assert body == "ok"
    assert scraper.stats.robots_blocked == 0


# --- rate limiting ----------------------------------------------------------
async def test_rate_limit_enforces_minimum_spacing() -> None:
    # 600/min => 100ms minimum spacing. Three sequential requests must
    # take at least ~200ms because the *first* doesn't wait.
    handler = _Handler(responses=[_ok(b"a"), _ok(b"b"), _ok(b"c")])
    scraper = _make_scraper(handler, rate_limit_per_minute=600)
    try:
        start = time.monotonic()
        for path in ("a", "b", "c"):
            assert await scraper.fetch(f"https://example.com/{path}") is not None
        elapsed = time.monotonic() - start
    finally:
        await scraper.close()

    # 2 enforced gaps of 100ms each. Allow some slack for slow CI but
    # require clearly more than zero — a broken limiter would be ~0ms.
    assert elapsed >= 0.18, f"expected >= 0.18s, got {elapsed:.3f}s"


async def test_rate_limit_serializes_concurrent_requests() -> None:
    handler = _Handler(responses=[_ok(b"a"), _ok(b"b"), _ok(b"c")])
    scraper = _make_scraper(handler, rate_limit_per_minute=600)
    try:
        start = time.monotonic()
        await asyncio.gather(
            scraper.fetch("https://example.com/a"),
            scraper.fetch("https://example.com/b"),
            scraper.fetch("https://example.com/c"),
        )
        elapsed = time.monotonic() - start
    finally:
        await scraper.close()

    # Concurrency must not let the limiter be bypassed: 3 calls at
    # 100ms spacing still need at least ~200ms.
    assert elapsed >= 0.18


def test_rate_limit_rejects_non_positive() -> None:
    class _C(BaseScraper):
        async def scrape(self, **kwargs: Any) -> AsyncIterator[ScrapedItem]:
            if False:
                yield  # pragma: no cover

    with pytest.raises(ValueError):
        _C(rate_limit_per_minute=0)


# --- stats counter ----------------------------------------------------------
async def test_stats_counter_tracks_full_lifecycle() -> None:
    handler = _Handler(
        robots=_empty_robots(),
        responses=[
            _ok(b"first"),
            httpx.Response(503),
            _ok(b"second"),
            httpx.Response(404),  # non-retryable
        ],
    )
    scraper = _make_scraper(handler)
    try:
        assert await scraper.fetch("https://example.com/a") == "first"
        assert await scraper.fetch("https://example.com/b") == "second"
        assert await scraper.fetch("https://example.com/c") is None
    finally:
        await scraper.close()

    s = scraper.stats
    # 3 URLs: first=1 send, second=2 sends (1 retry), third=1 send.
    assert s.requests_sent == 4
    assert s.requests_succeeded == 2
    # 1 retryable failure + 1 non-retryable failure = 2 failures.
    assert s.requests_failed == 2
    assert s.retries == 1
    assert s.bytes_downloaded == len(b"first") + len(b"second")


# --- async context manager --------------------------------------------------
async def test_async_context_manager_closes_client() -> None:
    handler = _Handler(responses=[_ok(b"x")])
    scraper = _make_scraper(handler)
    async with scraper:
        assert await scraper.fetch("https://example.com/x") == "x"
    # After exit, the client should be released and a second close is a no-op.
    assert scraper._client is None
    await scraper.close()
