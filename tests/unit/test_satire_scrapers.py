"""Unit tests for the satire RSS scrapers.

All HTTP traffic is faked through ``httpx.MockTransport``; feed bodies
are inline RSS fixtures so each test controls exactly what the parser
sees. A test that wanted to verify "missing image" behaviour would just
build a feed with no ``<media:thumbnail>``, etc.
"""
from __future__ import annotations

from typing import Callable

import httpx
import pytest

from satira.ingest.base_scraper import BaseScraper
from satira.ingest.satire_scrapers import (
    BabylonBeeScraper,
    ReductressScraper,
    SatireScraperRegistry,
    TheOnionScraper,
    _RSSSatireScraper,
)


# --- RSS fixtures -----------------------------------------------------------
FULL_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">
  <channel>
    <title>Test Satire Feed</title>
    <link>https://example.com</link>
    <description>Tests</description>
    <item>
      <title>Local Man Declares War On Mondays</title>
      <link>https://example.com/articles/mondays</link>
      <description><![CDATA[<p>In a stunning press conference...</p>]]></description>
      <pubDate>Mon, 04 May 2026 10:00:00 +0000</pubDate>
      <media:thumbnail url="https://example.com/img/mondays.jpg" />
    </item>
    <item>
      <title>Nation's Cats Demand Better Naps</title>
      <link>https://example.com/articles/cats</link>
      <description><![CDATA[<p>The feline lobby...</p>]]></description>
      <pubDate>Tue, 05 May 2026 09:30:00 +0000</pubDate>
      <enclosure url="https://example.com/img/cats.png" type="image/png" length="1024" />
    </item>
  </channel>
</rss>
"""

MINIMAL_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Bare Feed</title>
    <link>https://example.com</link>
    <description>Tests</description>
    <item>
      <title>Just A Title</title>
      <link>https://example.com/articles/bare</link>
    </item>
  </channel>
</rss>
"""

EMBEDDED_IMG_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Embedded Image Feed</title>
    <link>https://example.com</link>
    <description>Tests</description>
    <item>
      <title>Article With Inline Image</title>
      <link>https://example.com/articles/inline</link>
      <description><![CDATA[<p>Lead paragraph</p><img src="https://cdn.example.com/hero.jpg" alt="hero"/><p>More text</p>]]></description>
      <pubDate>Wed, 06 May 2026 12:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""

EMPTY_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Empty Feed</title>
    <link>https://example.com</link>
    <description>Tests</description>
  </channel>
</rss>
"""


# --- helpers ----------------------------------------------------------------
def _empty_robots() -> httpx.Response:
    return httpx.Response(200, content=b"", headers={"content-type": "text/plain"})


def _disallow_all_robots() -> httpx.Response:
    return httpx.Response(
        200,
        content=b"User-agent: *\nDisallow: /\n",
        headers={"content-type": "text/plain"},
    )


def _make_handler(
    feed_url: str,
    feed_body: str,
    *,
    robots: httpx.Response | None = None,
    feed_status: int = 200,
) -> tuple[Callable[[httpx.Request], httpx.Response], list[str]]:
    """Build a MockTransport handler that returns ``feed_body`` for ``feed_url``.

    Returns a (handler, calls) pair so tests can assert on the call log.
    """
    calls: list[str] = []
    robots_resp = robots if robots is not None else _empty_robots()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return robots_resp
        if str(request.url) == feed_url:
            return httpx.Response(
                feed_status,
                content=feed_body.encode("utf-8"),
                headers={"content-type": "application/rss+xml"},
            )
        raise AssertionError(f"unexpected request: {request.url}")

    return handler, calls


def _wire_transport(
    scraper: BaseScraper, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    """Inject a MockTransport-backed AsyncClient into ``scraper``.

    Same trick the base-scraper tests use: the lazy-init in
    ``_get_client`` looks at ``self._client`` first, so pre-populating
    it avoids hitting the network.
    """
    scraper._client = httpx.AsyncClient(
        headers={"User-Agent": scraper.user_agent},
        timeout=scraper.timeout,
        transport=httpx.MockTransport(handler),
    )


# --- per-outlet wiring ------------------------------------------------------
@pytest.mark.parametrize(
    ("scraper_cls", "expected_domain", "expected_outlet"),
    [
        (TheOnionScraper, "theonion.com", "The Onion"),
        (BabylonBeeScraper, "babylonbee.com", "Babylon Bee"),
        (ReductressScraper, "reductress.com", "Reductress"),
    ],
)
async def test_each_outlet_yields_items_with_satire_label(
    scraper_cls: type[_RSSSatireScraper],
    expected_domain: str,
    expected_outlet: str,
) -> None:
    scraper = scraper_cls()
    handler, _ = _make_handler(scraper.feed_url, FULL_RSS)
    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape(max_items=10)]
    finally:
        await scraper.close()

    assert len(items) == 2
    for item in items:
        assert item.metadata["label"] == "satire"
        assert item.metadata["outlet"] == expected_outlet
        assert item.source_domain == expected_domain


# --- field extraction -------------------------------------------------------
async def test_full_rss_extracts_title_text_image_and_timestamp() -> None:
    scraper = TheOnionScraper()
    handler, _ = _make_handler(scraper.feed_url, FULL_RSS)
    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape()]
    finally:
        await scraper.close()

    first = items[0]
    assert first.title == "Local Man Declares War On Mondays"
    assert first.source_url == "https://example.com/articles/mondays"
    assert first.image_url == "https://example.com/img/mondays.jpg"
    # HTML stripped from description.
    assert "<p>" not in first.text
    assert "press conference" in first.text
    assert first.timestamp.year == 2026
    assert first.timestamp.month == 5
    assert first.timestamp.day == 4

    second = items[1]
    # Image picked up via <enclosure> instead of media:thumbnail.
    assert second.image_url == "https://example.com/img/cats.png"


async def test_image_falls_back_to_inline_img_tag_in_description() -> None:
    scraper = TheOnionScraper()
    handler, _ = _make_handler(scraper.feed_url, EMBEDDED_IMG_RSS)
    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape()]
    finally:
        await scraper.close()

    assert len(items) == 1
    assert items[0].image_url == "https://cdn.example.com/hero.jpg"


async def test_missing_optional_fields_yield_graceful_defaults() -> None:
    scraper = TheOnionScraper()
    handler, _ = _make_handler(scraper.feed_url, MINIMAL_RSS)
    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape()]
    finally:
        await scraper.close()

    assert len(items) == 1
    item = items[0]
    assert item.title == "Just A Title"
    assert item.source_url == "https://example.com/articles/bare"
    # No image was provided.
    assert item.image_url is None
    # Empty description is fine — text is just empty, not missing.
    assert item.text == ""
    # Missing pubDate falls back to a real datetime, not None or a crash.
    assert item.timestamp is not None


# --- max_items / empty / failure modes --------------------------------------
async def test_max_items_caps_yielded_count() -> None:
    scraper = TheOnionScraper()
    handler, _ = _make_handler(scraper.feed_url, FULL_RSS)
    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape(max_items=1)]
    finally:
        await scraper.close()

    assert len(items) == 1


async def test_empty_feed_yields_nothing() -> None:
    scraper = TheOnionScraper()
    handler, _ = _make_handler(scraper.feed_url, EMPTY_RSS)
    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape()]
    finally:
        await scraper.close()

    assert items == []


async def test_robots_blocked_yields_nothing() -> None:
    scraper = TheOnionScraper()
    handler, calls = _make_handler(
        scraper.feed_url, FULL_RSS, robots=_disallow_all_robots()
    )
    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape()]
    finally:
        await scraper.close()

    assert items == []
    assert scraper.stats.robots_blocked == 1
    # Only robots.txt was actually fetched.
    assert any(c.endswith("/robots.txt") for c in calls)
    assert not any(c == scraper.feed_url for c in calls)


async def test_fetch_failure_yields_nothing() -> None:
    scraper = TheOnionScraper()
    handler, _ = _make_handler(scraper.feed_url, "", feed_status=500)
    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape()]
    finally:
        await scraper.close()

    assert items == []


async def test_stats_items_yielded_counter_increments() -> None:
    scraper = TheOnionScraper()
    handler, _ = _make_handler(scraper.feed_url, FULL_RSS)
    _wire_transport(scraper, handler)
    try:
        async for _ in scraper.scrape():
            pass
    finally:
        await scraper.close()

    assert scraper.stats.items_yielded == 2


# --- registry ---------------------------------------------------------------
async def test_registry_combines_items_from_all_sources() -> None:
    onion = TheOnionScraper()
    bee = BabylonBeeScraper()
    red = ReductressScraper()

    # Each scraper gets its own MockTransport keyed to its feed URL.
    for scraper in (onion, bee, red):
        handler, _ = _make_handler(scraper.feed_url, FULL_RSS)
        _wire_transport(scraper, handler)

    registry = SatireScraperRegistry(scrapers=[onion, bee, red])
    try:
        items = [item async for item in registry.scrape_all(max_items_per_source=10)]
    finally:
        await registry.close()

    # 2 items × 3 sources.
    assert len(items) == 6
    outlets = {item.metadata["outlet"] for item in items}
    assert outlets == {"The Onion", "Babylon Bee", "Reductress"}


async def test_registry_continues_when_one_source_fails() -> None:
    onion = TheOnionScraper()
    bee = BabylonBeeScraper()

    # Onion's transport raises on every call — registry should log and skip.
    def raising_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated outage", request=request)

    _wire_transport(onion, raising_handler)
    bee_handler, _ = _make_handler(bee.feed_url, FULL_RSS)
    _wire_transport(bee, bee_handler)

    registry = SatireScraperRegistry(scrapers=[onion, bee])
    try:
        items = [item async for item in registry.scrape_all()]
    finally:
        await registry.close()

    # Onion contributes nothing, Bee contributes its 2 items.
    assert len(items) == 2
    assert all(item.metadata["outlet"] == "Babylon Bee" for item in items)


async def test_registry_default_includes_three_outlets() -> None:
    registry = SatireScraperRegistry()
    try:
        outlets = {type(s).__name__ for s in registry.scrapers}
    finally:
        await registry.close()

    assert outlets == {"TheOnionScraper", "BabylonBeeScraper", "ReductressScraper"}


async def test_registry_async_context_manager_closes_scrapers() -> None:
    scraper = TheOnionScraper()
    handler, _ = _make_handler(scraper.feed_url, FULL_RSS)
    _wire_transport(scraper, handler)

    async with SatireScraperRegistry(scrapers=[scraper]) as registry:
        items = [item async for item in registry.scrape_all()]

    assert len(items) == 2
    assert scraper._client is None
