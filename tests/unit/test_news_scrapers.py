"""Unit tests for the news scrapers (GDELT + generic RSS + registry).

All HTTP traffic is faked through ``httpx.MockTransport`` so the tests
run hermetically. ``rate_limit_per_minute`` is set high on every
scraper so retry/rate-limit code paths don't make these tests slow.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Callable

import httpx
import pytest

from satira.ingest.base_scraper import BaseScraper, ScrapedItem
from satira.ingest.news_scrapers import (
    GDELTScraper,
    NewsScraperRegistry,
    RSSNewsScraper,
)


# --- fixtures ---------------------------------------------------------------
GDELT_TWO_ARTICLES: dict[str, Any] = {
    "articles": [
        {
            "url": "https://www.reuters.com/world/example-1",
            "title": "Reuters World Story",
            "seendate": "20260504T100000Z",
            "socialimage": "https://www.reuters.com/img/r1.jpg",
            "domain": "reuters.com",
            "language": "English",
            "sourcecountry": "United States",
        },
        {
            "url": "https://www.bbc.com/news/example-2",
            "title": "BBC Headline",
            "seendate": "20260504T093000Z",
            "socialimage": "",
            "domain": "bbc.com",
            "language": "English",
            "sourcecountry": "United Kingdom",
        },
    ]
}

GDELT_EMPTY: dict[str, Any] = {"articles": []}

NEWS_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:media="http://search.yahoo.com/mrss/">
  <channel>
    <title>Test News Feed</title>
    <link>https://example.com</link>
    <description>News tests</description>
    <item>
      <title>Cabinet reshuffle announced</title>
      <link>https://example.com/articles/cabinet</link>
      <description><![CDATA[<p>The cabinet reshuffle...</p>]]></description>
      <pubDate>Mon, 04 May 2026 10:00:00 +0000</pubDate>
      <media:thumbnail url="https://example.com/img/cab.jpg" />
    </item>
    <item>
      <title>Markets rally on policy news</title>
      <link>https://example.com/articles/markets</link>
      <description><![CDATA[<p>Markets reacted...</p>]]></description>
      <pubDate>Mon, 04 May 2026 11:30:00 +0000</pubDate>
    </item>
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


def _wire_transport(
    scraper: BaseScraper, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    """Inject a MockTransport-backed AsyncClient into ``scraper``."""
    scraper._client = httpx.AsyncClient(
        headers={"User-Agent": scraper.user_agent},
        timeout=scraper.timeout,
        transport=httpx.MockTransport(handler),
    )


def _json_response(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        content=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def _rss_response(body: str = NEWS_RSS) -> httpx.Response:
    return httpx.Response(
        200,
        content=body.encode("utf-8"),
        headers={"content-type": "application/rss+xml"},
    )


def _make_gdelt(**kwargs: Any) -> GDELTScraper:
    kwargs.setdefault("rate_limit_per_minute", 6000)
    return GDELTScraper(**kwargs)


def _make_rss(**kwargs: Any) -> RSSNewsScraper:
    kwargs.setdefault("rate_limit_per_minute", 6000)
    return RSSNewsScraper(**kwargs)


# --- GDELT tests ------------------------------------------------------------
async def test_gdelt_yields_items_with_authentic_label() -> None:
    scraper = _make_gdelt()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        seen.append(str(request.url))
        return _json_response(GDELT_TWO_ARTICLES)

    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape(query="climate", max_items=10)]
    finally:
        await scraper.close()

    assert len(items) == 2
    for item in items:
        assert item.metadata["label"] == "authentic"
        assert item.metadata["source_type"] == "gdelt"
        assert item.metadata["query"] == "climate"
    # source_domain comes from the article record itself.
    assert {item.source_domain for item in items} == {"reuters.com", "bbc.com"}
    # First article carries an image; second has empty socialimage → None.
    assert items[0].image_url == "https://www.reuters.com/img/r1.jpg"
    assert items[1].image_url is None
    # The query string actually appeared in the request URL.
    assert any("query=climate" in u for u in seen)


async def test_gdelt_passes_start_date_in_request() -> None:
    scraper = _make_gdelt()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        seen.append(str(request.url))
        return _json_response(GDELT_EMPTY)

    _wire_transport(scraper, handler)
    try:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        items = [
            item async for item in scraper.scrape(query="economy", start_date=start)
        ]
    finally:
        await scraper.close()

    assert items == []
    assert any("startdatetime=20260101000000" in u for u in seen)


async def test_gdelt_empty_articles_yields_nothing() -> None:
    scraper = _make_gdelt()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        return _json_response(GDELT_EMPTY)

    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape(query="x")]
    finally:
        await scraper.close()
    assert items == []


async def test_gdelt_invalid_json_yields_nothing() -> None:
    scraper = _make_gdelt()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        return httpx.Response(
            200,
            content=b"this is not json",
            headers={"content-type": "application/json"},
        )

    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape(query="x")]
    finally:
        await scraper.close()
    assert items == []


async def test_gdelt_robots_blocked_yields_nothing() -> None:
    scraper = _make_gdelt()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _disallow_all_robots()
        raise AssertionError("API should not be called when robots.txt blocks")

    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape(query="x")]
    finally:
        await scraper.close()

    assert items == []
    assert scraper.stats.robots_blocked == 1


async def test_gdelt_paginates_when_more_than_one_page_requested() -> None:
    scraper = _make_gdelt()
    # Force pagination on a small fixture.
    scraper.MAX_RECORDS_PER_CALL = 2

    pages = [
        {
            "articles": [
                {
                    "url": "https://example.com/a1",
                    "title": "A1",
                    "seendate": "20260504T120000Z",
                    "domain": "example.com",
                },
                {
                    "url": "https://example.com/a2",
                    "title": "A2",
                    "seendate": "20260504T110000Z",
                    "domain": "example.com",
                },
            ]
        },
        {
            "articles": [
                {
                    "url": "https://example.com/a3",
                    "title": "A3",
                    "seendate": "20260504T100000Z",
                    "domain": "example.com",
                },
            ]
        },
    ]
    served: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        served.append(str(request.url))
        idx = len(served) - 1
        if idx >= len(pages):
            return _json_response({"articles": []})
        return _json_response(pages[idx])

    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape(query="x", max_items=10)]
    finally:
        await scraper.close()

    # Page 1 fills the page (2 == page_size) so we paginate; page 2 returns
    # 1 (< page_size) so we stop.
    assert [i.title for i in items] == ["A1", "A2", "A3"]
    assert len(served) == 2
    # Second call's enddatetime should advance to A2's seendate.
    assert "enddatetime=20260504110000" in served[1]


async def test_gdelt_dedupes_repeated_urls_and_terminates() -> None:
    scraper = _make_gdelt()
    scraper.MAX_RECORDS_PER_CALL = 1

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        # Always return the same article.
        return _json_response(
            {
                "articles": [
                    {
                        "url": "https://example.com/dup",
                        "title": "Dup",
                        "seendate": "20260504T100000Z",
                        "domain": "example.com",
                    }
                ]
            }
        )

    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape(query="x", max_items=5)]
    finally:
        await scraper.close()

    # Only the first occurrence is yielded; pagination terminates because
    # the second page yields nothing new (dedup) → page_emitted == 0 bail.
    assert len(items) == 1


async def test_gdelt_empty_query_raises() -> None:
    scraper = _make_gdelt()
    try:
        with pytest.raises(ValueError):
            async for _ in scraper.scrape(query=""):
                pass
    finally:
        await scraper.close()


async def test_gdelt_falls_back_to_url_netloc_when_domain_missing() -> None:
    scraper = _make_gdelt()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        return _json_response(
            {
                "articles": [
                    {
                        "url": "https://www.example.org/path",
                        "title": "T",
                        "seendate": "20260504T100000Z",
                        # No "domain" field.
                    }
                ]
            }
        )

    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape(query="x")]
    finally:
        await scraper.close()

    assert len(items) == 1
    assert items[0].source_domain == "www.example.org"


# --- RSS news tests ---------------------------------------------------------
async def test_rss_news_yields_items_with_authentic_label() -> None:
    feeds = {"reuters_world": "https://feeds.reuters.com/reuters/worldNews"}
    scraper = _make_rss(feeds=feeds)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        if str(request.url) == feeds["reuters_world"]:
            return _rss_response()
        raise AssertionError(f"unexpected url: {request.url}")

    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape()]
    finally:
        await scraper.close()

    assert len(items) == 2
    for item in items:
        assert item.metadata["label"] == "authentic"
        assert item.metadata["source_type"] == "rss"
        assert item.metadata["outlet"] == "Reuters"
        assert item.metadata["feed_key"] == "reuters_world"
        assert item.source_domain == "reuters.com"


async def test_rss_news_feed_keys_filters_feeds() -> None:
    feeds = {
        "reuters_world": "https://reuters.example/feed",
        "bbc_news": "https://bbc.example/feed",
    }
    outlets = {
        "reuters_world": ("Reuters", "reuters.com"),
        "bbc_news": ("BBC", "bbc.co.uk"),
    }
    scraper = _make_rss(feeds=feeds, outlets=outlets)

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        seen.append(str(request.url))
        return _rss_response()

    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape(feed_keys=["bbc_news"])]
    finally:
        await scraper.close()

    assert len(items) == 2
    assert all(item.metadata["outlet"] == "BBC" for item in items)
    assert any(u == feeds["bbc_news"] for u in seen)
    assert not any(u == feeds["reuters_world"] for u in seen)


async def test_rss_news_unknown_feed_key_is_skipped() -> None:
    scraper = _make_rss(feeds={"known": "https://known.example/feed"})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        if str(request.url).startswith("https://known.example"):
            return _rss_response()
        raise AssertionError(f"unexpected url: {request.url}")

    _wire_transport(scraper, handler)
    try:
        items = [
            item async for item in scraper.scrape(feed_keys=["nope", "known"])
        ]
    finally:
        await scraper.close()
    assert len(items) == 2  # all came from the known feed


async def test_rss_news_max_per_feed_caps_yield() -> None:
    feeds = {"k": "https://x.example/feed"}
    outlets = {"k": ("X", "x.example")}
    scraper = _make_rss(feeds=feeds, outlets=outlets)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        return _rss_response()

    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape(max_per_feed=1)]
    finally:
        await scraper.close()
    assert len(items) == 1


async def test_rss_news_unknown_outlet_falls_back_to_netloc() -> None:
    feeds = {"weird": "https://news.weird.example/feed"}
    scraper = _make_rss(feeds=feeds, outlets={})  # no outlet metadata

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        return _rss_response()

    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape()]
    finally:
        await scraper.close()
    assert len(items) == 2
    # Falls back to (feed_key, netloc) pair.
    assert items[0].source_domain == "news.weird.example"
    assert items[0].metadata["outlet"] == "weird"


async def test_rss_news_default_feeds_cover_known_outlets() -> None:
    scraper = RSSNewsScraper()
    try:
        keys = set(scraper.feeds.keys())
    finally:
        await scraper.close()
    assert keys == {
        "reuters_world",
        "bbc_news",
        "ap_top",
        "npr_news",
        "guardian_world",
    }


async def test_rss_news_robots_blocked_yields_nothing() -> None:
    feeds = {"k": "https://k.example/feed"}
    outlets = {"k": ("Kx", "k.example")}
    scraper = _make_rss(feeds=feeds, outlets=outlets)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _disallow_all_robots()
        raise AssertionError("feed should not be fetched when robots blocks")

    _wire_transport(scraper, handler)
    try:
        items = [item async for item in scraper.scrape()]
    finally:
        await scraper.close()
    assert items == []
    assert scraper.stats.robots_blocked == 1


# --- registry tests ---------------------------------------------------------
def _wire_two_scrapers(
    rss: RSSNewsScraper,
    gdelt: GDELTScraper,
    *,
    rss_handler: Callable[[httpx.Request], httpx.Response],
    gdelt_handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    _wire_transport(rss, rss_handler)
    _wire_transport(gdelt, gdelt_handler)


async def test_registry_combines_rss_and_gdelt() -> None:
    feeds = {"k": "https://k.example/feed"}
    outlets = {"k": ("Kx", "k.example")}
    rss = _make_rss(feeds=feeds, outlets=outlets)
    gdelt = _make_gdelt()

    def rss_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        return _rss_response()

    def gdelt_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        return _json_response(GDELT_TWO_ARTICLES)

    _wire_two_scrapers(rss, gdelt, rss_handler=rss_handler, gdelt_handler=gdelt_handler)

    registry = NewsScraperRegistry(rss_scraper=rss, gdelt_scraper=gdelt)
    try:
        items = [
            item
            async for item in registry.scrape_all(
                gdelt_queries=["climate"], max_items=100
            )
        ]
    finally:
        await registry.close()

    assert len(items) == 4
    assert sum(1 for i in items if i.metadata["source_type"] == "rss") == 2
    assert sum(1 for i in items if i.metadata["source_type"] == "gdelt") == 2
    assert all(i.metadata["label"] == "authentic" for i in items)


async def test_registry_no_gdelt_queries_skips_gdelt() -> None:
    feeds = {"k": "https://k.example/feed"}
    outlets = {"k": ("Kx", "k.example")}
    rss = _make_rss(feeds=feeds, outlets=outlets)
    gdelt = _make_gdelt()

    def rss_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        return _rss_response()

    def gdelt_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("GDELT should not be invoked without queries")

    _wire_two_scrapers(rss, gdelt, rss_handler=rss_handler, gdelt_handler=gdelt_handler)

    registry = NewsScraperRegistry(rss_scraper=rss, gdelt_scraper=gdelt)
    try:
        items = [item async for item in registry.scrape_all()]
    finally:
        await registry.close()
    assert len(items) == 2


async def test_registry_max_items_caps_total_yield() -> None:
    feeds = {"k": "https://k.example/feed"}
    outlets = {"k": ("Kx", "k.example")}
    rss = _make_rss(feeds=feeds, outlets=outlets)
    gdelt = _make_gdelt()

    def rss_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        return _rss_response()

    def gdelt_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(
            "GDELT should not be invoked when budget is met by RSS"
        )

    _wire_two_scrapers(rss, gdelt, rss_handler=rss_handler, gdelt_handler=gdelt_handler)

    registry = NewsScraperRegistry(rss_scraper=rss, gdelt_scraper=gdelt)
    try:
        items = [
            item
            async for item in registry.scrape_all(
                gdelt_queries=["x"], max_items=1
            )
        ]
    finally:
        await registry.close()
    assert len(items) == 1


class _RaisingRSSScraper(RSSNewsScraper):
    """RSS stub whose ``scrape`` raises immediately — used to exercise the
    registry's per-source error tolerance without going through the
    retry/backoff machinery."""

    async def scrape(self, **_: Any) -> AsyncIterator[ScrapedItem]:
        raise RuntimeError("simulated outage")
        yield  # pragma: no cover — keeps function an async generator


async def test_registry_continues_when_rss_fails() -> None:
    rss = _RaisingRSSScraper(rate_limit_per_minute=6000)
    gdelt = _make_gdelt()

    def gdelt_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        return _json_response(GDELT_TWO_ARTICLES)

    _wire_transport(gdelt, gdelt_handler)

    registry = NewsScraperRegistry(rss_scraper=rss, gdelt_scraper=gdelt)
    try:
        items = [
            item
            async for item in registry.scrape_all(
                gdelt_queries=["x"], max_items=100
            )
        ]
    finally:
        await registry.close()
    # RSS exploded → contributed nothing; GDELT still contributes 2.
    assert len(items) == 2
    assert all(i.metadata["source_type"] == "gdelt" for i in items)


async def test_registry_async_context_manager_closes_scrapers() -> None:
    feeds = {"k": "https://k.example/feed"}
    outlets = {"k": ("Kx", "k.example")}
    rss = _make_rss(feeds=feeds, outlets=outlets)
    gdelt = _make_gdelt()

    def rss_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        return _rss_response()

    def gdelt_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        return _json_response(GDELT_EMPTY)

    _wire_two_scrapers(rss, gdelt, rss_handler=rss_handler, gdelt_handler=gdelt_handler)

    async with NewsScraperRegistry(rss_scraper=rss, gdelt_scraper=gdelt) as registry:
        async for _ in registry.scrape_all(gdelt_queries=["a"]):
            pass

    assert rss._client is None
    assert gdelt._client is None
