"""Unit tests for the image download pipeline.

Sample images are generated in-memory with PIL so the suite stays
hermetic — no fixtures need to live on disk. HTTP traffic is faked
through ``httpx.MockTransport`` (same pattern as ``test_base_scraper``)
so we can shape the response sequence per test.
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from PIL import Image

from satira.ingest.base_scraper import BaseScraper, ScrapedItem
from satira.ingest.image_pipeline import ImageDownloader, ProcessedItem


# --- image fixture helpers --------------------------------------------------
def _img_bytes(
    fmt: str = "PNG",
    width: int = 300,
    height: int = 300,
    color: tuple[int, int, int] = (50, 100, 200),
) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _large_random_png(min_size: int = 1_500_000) -> bytes:
    # Random pixels barely compress, so a noise-filled PNG ends up
    # roughly the same size as its raw RGB buffer.
    side = int((min_size // 3) ** 0.5) + 50
    img = Image.frombytes("RGB", (side, side), os.urandom(side * side * 3))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- HTTP plumbing ----------------------------------------------------------
def _image_response(
    data: bytes, content_type: str = "image/png"
) -> httpx.Response:
    return httpx.Response(200, content=data, headers={"content-type": content_type})


def _make_scraper(
    handler: Callable[[httpx.Request], httpx.Response],
) -> BaseScraper:
    """Concrete BaseScraper wired to a MockTransport, robots disabled."""

    class _Concrete(BaseScraper):
        async def scrape(self, **_: Any) -> AsyncIterator[ScrapedItem]:
            if False:  # pragma: no cover
                yield

    scraper = _Concrete(rate_limit_per_minute=6000, respect_robots=False)
    scraper._client = httpx.AsyncClient(
        headers={"User-Agent": scraper.user_agent},
        timeout=scraper.timeout,
        transport=httpx.MockTransport(handler),
    )
    return scraper


def _item(image_url: str | None = "https://cdn.example.com/img.png") -> ScrapedItem:
    return ScrapedItem(
        source_url="https://example.com/article",
        image_url=image_url,
        title="t",
        text="b",
        timestamp=datetime(2026, 1, 1),
        source_domain="example.com",
        metadata={"label": "satire"},
    )


@pytest.fixture
def storage(tmp_path: Path) -> Path:
    return tmp_path / "images"


# --- constructor validation -------------------------------------------------
def test_rejects_non_positive_max_size(storage: Path) -> None:
    with pytest.raises(ValueError):
        ImageDownloader(storage_path=str(storage), max_size_mb=0)


def test_rejects_non_positive_min_dimensions(storage: Path) -> None:
    with pytest.raises(ValueError):
        ImageDownloader(storage_path=str(storage), min_dimensions=(0, 200))


def test_creates_storage_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "images"
    ImageDownloader(storage_path=str(target))
    assert target.is_dir()


# --- download: success paths ------------------------------------------------
async def test_download_saves_and_returns_processed_item(storage: Path) -> None:
    data = _img_bytes("PNG")
    scraper = _make_scraper(lambda r: _image_response(data, "image/png"))
    downloader = ImageDownloader(storage_path=str(storage), scraper=scraper)
    try:
        out = await downloader.download(_item())
    finally:
        await scraper.close()

    assert isinstance(out, ProcessedItem)
    assert out.image_dimensions == (300, 300)
    assert out.file_size_bytes == len(data)
    # Default phash hash_size=8 → 64 bits → 16 hex chars.
    assert len(out.perceptual_hash) == 16
    assert all(c in "0123456789abcdef" for c in out.perceptual_hash)
    assert Path(out.image_path).exists()
    assert Path(out.image_path).read_bytes() == data
    # Scraped metadata is preserved, not shared by reference.
    assert out.metadata == {"label": "satire"}
    assert out.metadata is not _item().metadata


async def test_download_handles_jpeg_format(storage: Path) -> None:
    data = _img_bytes("JPEG")
    scraper = _make_scraper(lambda r: _image_response(data, "image/jpeg"))
    downloader = ImageDownloader(storage_path=str(storage), scraper=scraper)
    try:
        out = await downloader.download(_item("https://cdn.example.com/x.jpg"))
    finally:
        await scraper.close()

    assert out is not None
    assert out.image_path.endswith(".jpg")
    assert Path(out.image_path).exists()


async def test_download_handles_webp_format(storage: Path) -> None:
    data = _img_bytes("WEBP")
    scraper = _make_scraper(lambda r: _image_response(data, "image/webp"))
    downloader = ImageDownloader(storage_path=str(storage), scraper=scraper)
    try:
        out = await downloader.download(_item("https://cdn.example.com/x.webp"))
    finally:
        await scraper.close()

    assert out is not None
    assert out.image_path.endswith(".webp")


async def test_download_uses_content_addressable_filename(storage: Path) -> None:
    """Two identical fetches produce one file on disk."""
    data = _img_bytes("PNG")
    scraper = _make_scraper(lambda r: _image_response(data, "image/png"))
    downloader = ImageDownloader(storage_path=str(storage), scraper=scraper)
    try:
        a = await downloader.download(_item("https://cdn.example.com/a.png"))
        b = await downloader.download(_item("https://cdn.example.com/b.png"))
    finally:
        await scraper.close()

    assert a is not None and b is not None
    assert a.image_path == b.image_path
    assert len(list(storage.iterdir())) == 1


# --- download: validation rejections ----------------------------------------
async def test_download_rejects_too_small_dimensions(storage: Path) -> None:
    data = _img_bytes("PNG", width=100, height=100)
    scraper = _make_scraper(lambda r: _image_response(data, "image/png"))
    downloader = ImageDownloader(
        storage_path=str(storage),
        scraper=scraper,
        min_dimensions=(200, 200),
    )
    try:
        out = await downloader.download(_item())
    finally:
        await scraper.close()

    assert out is None
    assert list(storage.iterdir()) == []


async def test_download_rejects_unsupported_format(storage: Path) -> None:
    data = _img_bytes("GIF")
    scraper = _make_scraper(lambda r: _image_response(data, "image/gif"))
    downloader = ImageDownloader(storage_path=str(storage), scraper=scraper)
    try:
        out = await downloader.download(_item())
    finally:
        await scraper.close()

    assert out is None


async def test_download_rejects_oversize_file(storage: Path) -> None:
    data = _large_random_png(min_size=1_500_000)
    assert len(data) > 1_000_000  # sanity-check the fixture
    scraper = _make_scraper(lambda r: _image_response(data, "image/png"))
    downloader = ImageDownloader(
        storage_path=str(storage), scraper=scraper, max_size_mb=1
    )
    try:
        out = await downloader.download(_item())
    finally:
        await scraper.close()

    assert out is None


async def test_download_rejects_corrupt_bytes(storage: Path) -> None:
    # Right content-type, garbage body — fetch_image accepts it,
    # PIL rejects it.
    scraper = _make_scraper(
        lambda r: _image_response(b"not really a png", "image/png")
    )
    downloader = ImageDownloader(storage_path=str(storage), scraper=scraper)
    try:
        out = await downloader.download(_item())
    finally:
        await scraper.close()

    assert out is None


# --- download: short-circuit paths ------------------------------------------
async def test_download_returns_none_when_image_url_missing(storage: Path) -> None:
    scraper = _make_scraper(lambda r: pytest.fail("should not fetch"))
    downloader = ImageDownloader(storage_path=str(storage), scraper=scraper)
    try:
        out = await downloader.download(_item(image_url=None))
    finally:
        await scraper.close()

    assert out is None


async def test_download_returns_none_on_fetch_failure(storage: Path) -> None:
    scraper = _make_scraper(lambda r: httpx.Response(404))
    downloader = ImageDownloader(storage_path=str(storage), scraper=scraper)
    try:
        out = await downloader.download(_item())
    finally:
        await scraper.close()

    assert out is None


async def test_download_returns_none_on_non_image_content_type(
    storage: Path,
) -> None:
    # fetch_image already rejects non-image content-types, so the
    # downloader never even sees the bytes.
    scraper = _make_scraper(
        lambda r: httpx.Response(
            200,
            content=b"<html>nope</html>",
            headers={"content-type": "text/html"},
        )
    )
    downloader = ImageDownloader(storage_path=str(storage), scraper=scraper)
    try:
        out = await downloader.download(_item())
    finally:
        await scraper.close()

    assert out is None


# --- download_batch ---------------------------------------------------------
async def test_download_batch_skips_individual_failures(storage: Path) -> None:
    good = _img_bytes("PNG", color=(10, 20, 30))
    other = _img_bytes("PNG", color=(200, 100, 50))
    responses = {
        "/ok1.png": _image_response(good, "image/png"),
        "/bad.png": httpx.Response(404),
        "/ok2.png": _image_response(other, "image/png"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return responses[request.url.path]

    scraper = _make_scraper(handler)
    downloader = ImageDownloader(storage_path=str(storage), scraper=scraper)
    try:
        items = [
            _item("https://cdn.example.com/ok1.png"),
            _item("https://cdn.example.com/bad.png"),
            _item("https://cdn.example.com/ok2.png"),
        ]
        out = await downloader.download_batch(items, max_concurrent=2)
    finally:
        await scraper.close()

    assert len(out) == 2
    assert {p.image_dimensions for p in out} == {(300, 300)}


async def test_download_batch_rejects_non_positive_concurrency(
    storage: Path,
) -> None:
    scraper = _make_scraper(lambda r: httpx.Response(404))
    downloader = ImageDownloader(storage_path=str(storage), scraper=scraper)
    try:
        with pytest.raises(ValueError):
            await downloader.download_batch([_item()], max_concurrent=0)
    finally:
        await scraper.close()


async def test_download_batch_handles_empty_input(storage: Path) -> None:
    scraper = _make_scraper(lambda r: pytest.fail("should not fetch"))
    downloader = ImageDownloader(storage_path=str(storage), scraper=scraper)
    try:
        out = await downloader.download_batch([])
    finally:
        await scraper.close()

    assert out == []


# --- deduplicate_by_phash ---------------------------------------------------
def _processed(phash: str, tag: str = "a") -> ProcessedItem:
    return ProcessedItem(
        source_url=f"https://example.com/{tag}",
        image_url=f"https://cdn.example.com/{tag}.png",
        title=f"title-{tag}",
        text="",
        timestamp=datetime(2026, 1, 1),
        source_domain="example.com",
        metadata={},
        image_path=f"/tmp/{tag}.png",
        image_dimensions=(300, 300),
        perceptual_hash=phash,
        file_size_bytes=100,
    )


def test_dedup_drops_near_duplicates_within_threshold(storage: Path) -> None:
    # "0…0" and "0…3" differ in 2 bits (hex 0=0000, 3=0011).
    a = _processed("0000000000000000", "a")
    b = _processed("0000000000000003", "b")
    c = _processed("ffffffffffffffff", "c")  # fully different

    out = ImageDownloader(storage_path=str(storage)).deduplicate_by_phash(
        [a, b, c], hamming_threshold=4
    )

    assert [p.source_url for p in out] == [
        "https://example.com/a",
        "https://example.com/c",
    ]


def test_dedup_keeps_distinct_when_above_threshold(storage: Path) -> None:
    a = _processed("0000000000000000", "a")
    # "00…00ff" vs "00…0000" differs in 8 bits — above threshold of 4.
    b = _processed("00000000000000ff", "b")

    out = ImageDownloader(storage_path=str(storage)).deduplicate_by_phash(
        [a, b], hamming_threshold=4
    )

    assert len(out) == 2


def test_dedup_threshold_zero_only_drops_exact_matches(storage: Path) -> None:
    a = _processed("0000000000000000", "a")
    a_exact = _processed("0000000000000000", "a-clone")
    b = _processed("0000000000000001", "b")  # 1 bit different

    out = ImageDownloader(storage_path=str(storage)).deduplicate_by_phash(
        [a, a_exact, b], hamming_threshold=0
    )

    assert [p.source_url for p in out] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_dedup_skips_invalid_perceptual_hash(storage: Path) -> None:
    bad = _processed("not-a-hex-string", "bad")
    good = _processed("0000000000000000", "good")

    out = ImageDownloader(storage_path=str(storage)).deduplicate_by_phash(
        [bad, good]
    )

    assert [p.source_url for p in out] == ["https://example.com/good"]


def test_dedup_rejects_negative_threshold(storage: Path) -> None:
    with pytest.raises(ValueError):
        ImageDownloader(storage_path=str(storage)).deduplicate_by_phash(
            [], hamming_threshold=-1
        )


def test_dedup_preserves_order(storage: Path) -> None:
    # Build distinct phashes that sit far apart in hamming space.
    items = [
        _processed("0000000000000000", "first"),
        _processed("ffff000000000000", "second"),
        _processed("ffffffff00000000", "third"),
    ]
    out = ImageDownloader(storage_path=str(storage)).deduplicate_by_phash(
        items, hamming_threshold=4
    )
    assert [p.source_url for p in out] == [
        "https://example.com/first",
        "https://example.com/second",
        "https://example.com/third",
    ]


# --- async context manager --------------------------------------------------
async def test_async_context_manager_closes_owned_scraper(storage: Path) -> None:
    async with ImageDownloader(storage_path=str(storage)) as dl:
        assert dl._owns_scraper is True
    # After exit the owned scraper's HTTP client should be released.
    assert dl._scraper._client is None


async def test_does_not_close_externally_provided_scraper(storage: Path) -> None:
    data = _img_bytes("PNG")
    scraper = _make_scraper(lambda r: _image_response(data, "image/png"))
    async with ImageDownloader(storage_path=str(storage), scraper=scraper) as dl:
        out = await dl.download(_item())
        assert out is not None
    # The external scraper's client must still be usable.
    assert scraper._client is not None
    await scraper.close()
