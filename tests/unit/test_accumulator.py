"""Unit tests for the daily corpus accumulator.

Everything is hermetic: RSS feeds and image bytes are served through
``httpx.MockTransport`` (the same trick the scraper/image-pipeline tests
use), images are generated in-memory with PIL, and the corpus lives
under pytest's ``tmp_path``. No test touches the network.

Two layers are exercised:

* the dedup predicate (``_is_duplicate`` / ``_phash_duplicate``) in
  isolation, so each axis — URL, fuzzy title, perceptual hash — is
  pinned precisely; and
* the full :meth:`CorpusAccumulator.run_daily_collection` against mocked
  registries, covering idempotency, manifest persistence, report
  counts, and the keep-text-only-on-failure behaviour.
"""
from __future__ import annotations

import json
import random
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import httpx
import imagehash
from PIL import Image

from satira.ingest.accumulator import (
    CollectionReport,
    CorpusAccumulator,
    _derive_license,
    _norm_title,
)
from satira.ingest.base_scraper import BaseScraper, ScrapedItem
from satira.ingest.image_pipeline import ProcessedItem
from satira.ingest.news_scrapers import GDELTScraper, NewsScraperRegistry, RSSNewsScraper
from satira.ingest.satire_scrapers import SatireScraperRegistry, TheOnionScraper


# --- image + feed fixtures --------------------------------------------------
def _noise_png(seed: int, width: int = 300, height: int = 300) -> bytes:
    """A deterministic noise PNG.

    Solid-colour images all hash to ~zero (no frequency content), which
    would make distinct images look like phash duplicates. Noise gives
    each image a distinct perceptual hash, so two different ``seed``\\ s
    are reliably *not* near-duplicates.
    """
    rng = random.Random(seed)
    data = rng.randbytes(width * height * 3)
    img = Image.frombytes("RGB", (width, height), data)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _rss(entries: list[dict[str, Any]], title: str = "Test Feed") -> str:
    """Build an RSS document from ``[{title, link, image?}]`` entries."""
    items = []
    for e in entries:
        media = (
            f'<media:thumbnail url="{e["image"]}" />' if e.get("image") else ""
        )
        items.append(
            f"""    <item>
      <title>{e['title']}</title>
      <link>{e['link']}</link>
      <description><![CDATA[<p>{e.get('desc', 'A sufficiently long body.')}</p>]]></description>
      <pubDate>Mon, 04 May 2026 10:00:00 +0000</pubDate>
      {media}
    </item>"""
        )
    body = "\n".join(items)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">\n'
        f"  <channel><title>{title}</title><link>https://example.com</link>"
        "<description>t</description>\n"
        f"{body}\n"
        "  </channel>\n</rss>\n"
    )


class _StubScraper(BaseScraper):
    """Concrete BaseScraper used only for its ``fetch_image`` plumbing."""

    async def scrape(self, **_: Any) -> AsyncIterator[ScrapedItem]:
        if False:  # pragma: no cover — never iterated
            yield


def _wire(scraper: BaseScraper, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    """Inject a MockTransport-backed client (pre-populating the lazy field)."""
    scraper._client = httpx.AsyncClient(
        headers={"User-Agent": scraper.user_agent},
        timeout=scraper.timeout,
        transport=httpx.MockTransport(handler),
    )


def _feed_handler(
    feed_url: str, body: str, *, status: int = 200
) -> Callable[[httpx.Request], httpx.Response]:
    """Serve ``body`` for ``feed_url``; 404 everything else (e.g. og: fetches)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == feed_url:
            return httpx.Response(
                status,
                content=body.encode("utf-8"),
                headers={"content-type": "application/rss+xml"},
            )
        return httpx.Response(404)

    return handler


def _image_handler(
    url_to_bytes: dict[str, bytes],
) -> Callable[[httpx.Request], httpx.Response]:
    """Serve PNG bytes per image URL; 404 unknown URLs."""

    def handler(request: httpx.Request) -> httpx.Response:
        data = url_to_bytes.get(str(request.url))
        if data is None:
            return httpx.Response(404)
        return httpx.Response(
            200, content=data, headers={"content-type": "image/png"}
        )

    return handler


def _make_accumulator(
    corpus_dir: Path,
    *,
    satire_entries: list[dict[str, Any]] | None = None,
    news_entries: list[dict[str, Any]] | None = None,
    images: dict[str, bytes] | None = None,
    satire_status: int = 200,
) -> tuple[CorpusAccumulator, SatireScraperRegistry, NewsScraperRegistry, BaseScraper]:
    """Wire an accumulator with fully-mocked satire/news/image transports.

    Returns the accumulator plus the three owned objects the caller must
    close (the accumulator does not close injected registries).
    """
    satire_url = TheOnionScraper.feed_url
    news_url = "https://news.example.com/rss"

    onion = TheOnionScraper(respect_robots=False, rate_limit_per_minute=6000)
    _wire(onion, _feed_handler(satire_url, _rss(satire_entries or []), status=satire_status))
    satire_registry = SatireScraperRegistry(scrapers=[onion])

    rss = RSSNewsScraper(
        feeds={"test": news_url},
        outlets={"test": ("Test News", "news.example.com")},
        respect_robots=False,
        rate_limit_per_minute=6000,
    )
    _wire(rss, _feed_handler(news_url, _rss(news_entries or [])))
    gdelt = GDELTScraper(respect_robots=False, rate_limit_per_minute=6000)
    news_registry = NewsScraperRegistry(rss_scraper=rss, gdelt_scraper=gdelt)

    image_scraper = _StubScraper(respect_robots=False, rate_limit_per_minute=6000)
    _wire(image_scraper, _image_handler(images or {}))

    accumulator = CorpusAccumulator(
        corpus_dir=str(corpus_dir),
        satire_registry=satire_registry,
        news_registry=news_registry,
        image_scraper=image_scraper,
    )
    return accumulator, satire_registry, news_registry, image_scraper


async def _close_all(*objs: Any) -> None:
    for obj in objs:
        await obj.close()


def _processed(
    *,
    url: str = "https://example.com/a",
    title: str = "A headline",
    phash: str | None = None,
    label: str = "satire",
) -> ProcessedItem:
    return ProcessedItem(
        source_url=url,
        image_url=None,
        title=title,
        text="body",
        timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
        source_domain="example.com",
        metadata={"label": label},
        perceptual_hash=phash,
    )


# --- dedup predicate: URL ---------------------------------------------------
def test_is_duplicate_by_exact_url(tmp_path: Path) -> None:
    acc = CorpusAccumulator(corpus_dir=str(tmp_path / "c"))
    urls = {"https://example.com/article"}
    # Trailing slash + case differences are normalized away.
    item = _processed(url="https://Example.com/article/", title="totally unrelated")
    assert acc._is_duplicate(item, [], urls, {}) is True

    fresh = _processed(url="https://example.com/other", title="totally unrelated")
    assert acc._is_duplicate(fresh, [], urls, {}) is False


# --- dedup predicate: title -------------------------------------------------
def test_is_duplicate_by_fuzzy_title(tmp_path: Path) -> None:
    acc = CorpusAccumulator(corpus_dir=str(tmp_path / "c"))
    kept = _norm_title("Senate Passes Sweeping New Infrastructure Budget Bill")
    titles = {kept[:2]: [kept]}

    # One trailing character differs -> ratio well above 95.
    near = _processed(
        url="https://example.com/x",
        title="Senate Passes Sweeping New Infrastructure Budget Bill!",
    )
    assert acc._is_duplicate(near, [], set(), titles) is True

    # A headline that merely shares the topic is not a duplicate.
    different = _processed(
        url="https://example.com/y",
        title="House Rejects Unrelated Defense Spending Proposal Entirely",
    )
    assert acc._is_duplicate(different, [], set(), titles) is False


# --- dedup predicate: perceptual hash ---------------------------------------
def test_phash_duplicate_respects_hamming_threshold(tmp_path: Path) -> None:
    acc = CorpusAccumulator(corpus_dir=str(tmp_path / "c"))
    base = imagehash.hex_to_hash("0000000000000000")

    identical = _processed(phash="0000000000000000")
    near = _processed(phash="0000000000000007")  # distance 3 -> dup (< 4)
    edge = _processed(phash="000000000000000f")  # distance 4 -> not a dup

    assert acc._phash_duplicate(identical, [base]) is True
    assert acc._phash_duplicate(near, [base]) is True
    assert acc._phash_duplicate(edge, [base]) is False
    # No image -> never a phash duplicate.
    assert acc._phash_duplicate(_processed(phash=None), [base]) is False


# --- license derivation -----------------------------------------------------
def test_derive_license_by_provenance() -> None:
    rss_satire = ScrapedItem(
        "u", None, "t", "b", datetime.now(timezone.utc), "d",
        metadata={"label": "satire", "feed_url": "https://x/feed"},
    )
    rss_news = ScrapedItem(
        "u", None, "t", "b", datetime.now(timezone.utc), "d",
        metadata={"label": "authentic", "source_type": "rss"},
    )
    gdelt = ScrapedItem(
        "u", None, "t", "b", datetime.now(timezone.utc), "d",
        metadata={"label": "authentic", "source_type": "gdelt"},
    )
    bare = ScrapedItem("u", None, "t", "b", datetime.now(timezone.utc), "d")

    assert _derive_license(rss_satire) == "rss-feed-excerpt"
    assert _derive_license(rss_news) == "rss-feed-excerpt"
    assert _derive_license(gdelt) == "gdelt-doc-metadata"
    assert _derive_license(bare) == "unknown"


# --- end-to-end: new items are added ----------------------------------------
async def test_new_items_are_added_to_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    acc, sr, nr, img = _make_accumulator(
        corpus,
        satire_entries=[
            {"title": "Onion One", "link": "https://o/1"},
            {"title": "Onion Two", "link": "https://o/2"},
        ],
        news_entries=[{"title": "News One", "link": "https://n/1"}],
    )
    try:
        report = await acc.run_daily_collection()
    finally:
        await _close_all(sr, nr, img)

    assert report.items_scraped == 3
    assert report.items_new == 3
    assert report.items_duplicate == 0
    assert report.errors == []
    assert report.by_label == {"satire": 2, "authentic": 1}

    corpus_records = acc.load_corpus()
    assert len(corpus_records) == 3
    assert {r["label"] for r in corpus_records} == {"satire", "authentic"}


# --- end-to-end: idempotency ------------------------------------------------
async def test_running_twice_adds_nothing_the_second_time(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    acc, sr, nr, img = _make_accumulator(
        corpus,
        satire_entries=[
            {"title": "Headline A", "link": "https://o/a"},
            {"title": "Headline B", "link": "https://o/b"},
        ],
        news_entries=[{"title": "News C", "link": "https://n/c"}],
    )
    try:
        first = await acc.run_daily_collection()
        second = await acc.run_daily_collection()
    finally:
        await _close_all(sr, nr, img)

    assert first.items_new == 3
    assert second.items_new == 0
    assert second.items_scraped == 3
    assert second.items_duplicate == 3
    # Corpus did not grow on the second run.
    assert len(acc.load_corpus()) == 3
    # And the manifest has exactly three physical lines (append-only,
    # no rewrite).
    lines = [
        ln for ln in acc.manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert len(lines) == 3


# --- end-to-end: dedup by URL across a single run ---------------------------
async def test_duplicate_url_within_run_is_dropped(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    acc, sr, nr, img = _make_accumulator(
        corpus,
        satire_entries=[
            {"title": "First", "link": "https://o/dup"},
            {"title": "Different Title Same Link", "link": "https://o/dup"},
        ],
    )
    try:
        report = await acc.run_daily_collection()
    finally:
        await _close_all(sr, nr, img)

    assert report.items_scraped == 2
    assert report.items_new == 1
    assert report.items_duplicate == 1


# --- end-to-end: dedup by perceptual hash -----------------------------------
async def test_visually_identical_images_not_added_twice(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    same_png = _noise_png(seed=1)
    img_a = "https://cdn.example.com/a.png"
    img_b = "https://cdn.example.com/b.png"
    acc, sr, nr, img = _make_accumulator(
        corpus,
        satire_entries=[
            {"title": "Cats Demand Naps", "link": "https://o/cats", "image": img_a},
            {"title": "Dogs Form A Union", "link": "https://o/dogs", "image": img_b},
        ],
        # Distinct URLs, identical bytes -> identical perceptual hash.
        images={img_a: same_png, img_b: same_png},
    )
    try:
        report = await acc.run_daily_collection()
    finally:
        await _close_all(sr, nr, img)

    # First item kept; second rejected by phash despite a fresh URL/title.
    assert report.items_new == 1
    assert report.items_duplicate == 1
    assert len(acc.load_corpus()) == 1


# --- end-to-end: dedup by fuzzy title ---------------------------------------
async def test_near_identical_titles_deduped_across_run(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    acc, sr, nr, img = _make_accumulator(
        corpus,
        news_entries=[
            {
                "title": "Local Man Declares Total War On Monday Mornings",
                "link": "https://n/1",
            },
            {
                "title": "Local Man Declares Total War On Monday Mornings.",
                "link": "https://n/2",
            },
        ],
    )
    try:
        report = await acc.run_daily_collection()
    finally:
        await _close_all(sr, nr, img)

    assert report.items_new == 1
    assert report.items_duplicate == 1


# --- end-to-end: manifest correctness ---------------------------------------
async def test_manifest_records_have_expected_fields(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    png = _noise_png(seed=7)
    img_url = "https://cdn.example.com/hero.png"
    acc, sr, nr, img = _make_accumulator(
        corpus,
        satire_entries=[
            {"title": "Imaged Satire", "link": "https://o/img", "image": img_url}
        ],
        news_entries=[{"title": "Text Only News", "link": "https://n/text"}],
        images={img_url: png},
    )
    try:
        report = await acc.run_daily_collection()
    finally:
        await _close_all(sr, nr, img)

    records = acc.load_corpus()
    assert len(records) == 2
    for rec in records:
        for key in (
            "source_url",
            "title",
            "label",
            "source",
            "license",
            "perceptual_hash",
            "collection_date",
            "timestamp",
            "metadata",
        ):
            assert key in rec
        # collection_date is parseable and equals the report's run date.
        assert (
            datetime.fromisoformat(rec["collection_date"]) == report.run_date
        )
        assert rec["license"] == "rss-feed-excerpt"

    imaged = next(r for r in records if r["label"] == "satire")
    text_only = next(r for r in records if r["label"] == "authentic")
    assert imaged["image_path"] is not None
    assert imaged["perceptual_hash"] is not None
    assert text_only["image_path"] is None
    assert text_only["perceptual_hash"] is None


# --- end-to-end: report counts ----------------------------------------------
async def test_collection_report_counts_are_accurate(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    png = _noise_png(seed=3)
    img_url = "https://cdn.example.com/p.png"
    acc, sr, nr, img = _make_accumulator(
        corpus,
        satire_entries=[
            {"title": "Unique One", "link": "https://o/1", "image": img_url},
            {"title": "Unique Two", "link": "https://o/2"},
            {"title": "Unique Three", "link": "https://o/3"},
            # Exact URL repeat of the first -> one duplicate.
            {"title": "Repeat", "link": "https://o/1"},
        ],
        images={img_url: png},
    )
    try:
        report = await acc.run_daily_collection()
    finally:
        await _close_all(sr, nr, img)

    assert report.items_scraped == 4
    assert report.items_new == 3
    assert report.items_duplicate == 1
    # Headline invariant.
    assert report.items_scraped == report.items_new + report.items_duplicate
    # One new item carried a downloaded image.
    assert report.images_downloaded == 1
    assert report.by_label == {"satire": 3}
    assert isinstance(report, CollectionReport)


# --- end-to-end: download failure kept as text-only -------------------------
async def test_download_failure_keeps_item_as_text_only(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    # The feed references an image, but the image transport 404s it.
    acc, sr, nr, img = _make_accumulator(
        corpus,
        satire_entries=[
            {
                "title": "Has A Broken Image",
                "link": "https://o/broken",
                "image": "https://cdn.example.com/missing.png",
            }
        ],
        images={},  # nothing served -> download fails
    )
    try:
        report = await acc.run_daily_collection()
    finally:
        await _close_all(sr, nr, img)

    # The item is still collected, just without an image.
    assert report.items_new == 1
    assert report.images_downloaded == 0
    rec = acc.load_corpus()[0]
    assert rec["image_path"] is None
    assert rec["perceptual_hash"] is None


# --- end-to-end: images persist to disk -------------------------------------
async def test_images_are_persisted_to_disk(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    img_a, img_b = "https://cdn.example.com/1.png", "https://cdn.example.com/2.png"
    acc, sr, nr, img = _make_accumulator(
        corpus,
        satire_entries=[
            {"title": "Distinct One", "link": "https://o/1", "image": img_a},
            {"title": "Distinct Two", "link": "https://o/2", "image": img_b},
        ],
        images={img_a: _noise_png(seed=10), img_b: _noise_png(seed=20)},
    )
    try:
        report = await acc.run_daily_collection()
    finally:
        await _close_all(sr, nr, img)

    assert report.items_new == 2
    assert report.images_downloaded == 2
    stored = list((corpus / "images").glob("*.png"))
    assert len(stored) == 2
    # Manifest image paths point at real files on disk.
    for rec in acc.load_corpus():
        assert rec["image_path"] is not None
        assert Path(rec["image_path"]).exists()


# --- persistence across instances -------------------------------------------
async def test_corpus_survives_reload(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    acc, sr, nr, img = _make_accumulator(
        corpus,
        satire_entries=[
            {"title": "Persist A", "link": "https://o/a"},
            {"title": "Persist B", "link": "https://o/b"},
        ],
    )
    try:
        await acc.run_daily_collection()
    finally:
        await _close_all(sr, nr, img)

    # A brand-new accumulator pointed at the same directory sees the
    # corpus, with no registries wired at all.
    reloaded = CorpusAccumulator(corpus_dir=str(corpus))
    records = reloaded.load_corpus()
    assert len(records) == 2
    assert {r["title"] for r in records} == {"Persist A", "Persist B"}

    stats = reloaded.get_corpus_stats()
    assert stats["total_items"] == 2

    # A second run from yet another fresh instance (same feeds) adds
    # nothing: dedup rebuilt its index from the persisted manifest.
    acc2, sr2, nr2, img2 = _make_accumulator(
        corpus,
        satire_entries=[
            {"title": "Persist A", "link": "https://o/a"},
            {"title": "Persist B", "link": "https://o/b"},
        ],
    )
    try:
        report2 = await acc2.run_daily_collection()
    finally:
        await _close_all(sr2, nr2, img2)
    assert report2.items_new == 0
    assert len(reloaded.load_corpus()) == 2


# --- stats ------------------------------------------------------------------
async def test_get_corpus_stats_summarizes_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    png = _noise_png(seed=42)
    img_url = "https://cdn.example.com/s.png"
    acc, sr, nr, img = _make_accumulator(
        corpus,
        satire_entries=[
            {"title": "Stat One", "link": "https://o/1", "image": img_url},
            {"title": "Stat Two", "link": "https://o/2"},
        ],
        news_entries=[{"title": "Stat News", "link": "https://n/1"}],
        images={img_url: png},
    )
    try:
        await acc.run_daily_collection()
    finally:
        await _close_all(sr, nr, img)

    stats = acc.get_corpus_stats()
    assert stats["total_items"] == 3
    assert stats["by_label"] == {"satire": 2, "authentic": 1}
    assert stats["unique_sources"] >= 1
    # One of three items had no image.
    assert stats["text_only"] == 2
    assert 0.0 < stats["text_only_ratio"] < 1.0
    assert stats["date_range"] is not None
    # Items were added today, so the per-day window has one entry.
    today = datetime.now(timezone.utc).date().isoformat()
    assert stats["added_per_day"].get(today) == 3


def test_get_corpus_stats_empty(tmp_path: Path) -> None:
    acc = CorpusAccumulator(corpus_dir=str(tmp_path / "empty"))
    stats = acc.get_corpus_stats()
    assert stats["total_items"] == 0
    assert stats["by_label"] == {}
    assert stats["added_per_day"] == {}


# --- load robustness --------------------------------------------------------
def test_load_corpus_skips_malformed_lines(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    manifest = corpus / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"source_url": "https://o/1", "title": "Good"})
        + "\n"
        + "{ this is not valid json\n"
        + "\n"  # blank line
        + json.dumps({"source_url": "https://o/2", "title": "Also Good"})
        + "\n",
        encoding="utf-8",
    )
    acc = CorpusAccumulator(corpus_dir=str(corpus))
    records = acc.load_corpus()
    assert len(records) == 2
    assert {r["title"] for r in records} == {"Good", "Also Good"}


# --- error surfacing --------------------------------------------------------
async def test_feed_failure_surfaces_in_report_errors(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    # Satire feed 404s (non-retryable -> fast); news feed is fine.
    acc, sr, nr, img = _make_accumulator(
        corpus,
        satire_entries=[{"title": "Never Seen", "link": "https://o/x"}],
        news_entries=[{"title": "Fine News", "link": "https://n/1"}],
        satire_status=404,
    )
    try:
        report = await acc.run_daily_collection()
    finally:
        await _close_all(sr, nr, img)

    # The satire outlet yielded nothing and had a failed request.
    assert any("Onion" in e for e in report.errors)
    # The healthy news item still made it in.
    assert report.items_new == 1
