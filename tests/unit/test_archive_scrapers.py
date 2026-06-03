"""Unit tests for the paginated archive scrapers.

All HTTP traffic is faked through ``httpx.MockTransport`` so the tests
run hermetically — no live site is ever contacted (which matters here:
the real outlets the archive task named have opted out of AI scraping).
``requests_per_second`` is set very high on every scraper so the
1-req/sec archive politeness delay doesn't make the suite crawl.

The fixtures model a generic WordPress-style archive on ``example.com``:
a sitemap index pointing at sub-sitemaps, each sub-sitemap a urlset of
article URLs, and article pages whose hero image lives in ``og:image``
(or ``twitter:image`` / a body ``<img>`` for the priority-order tests).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Callable

import httpx
import pytest

from satira.ingest.archive_scrapers import (
    ArchiveScraperRegistry,
    FlatSitemapArchiveScraper,
    PaginatedArchiveScraper,
    SitemapIndexArchiveScraper,
    build_scraper_from_config,
    detect_ai_optout,
    parse_sitemap_xml,
)
from satira.ingest.base_scraper import BaseScraper


# --- fixtures ---------------------------------------------------------------
SITEMAP_INDEX = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/post-sitemap1.xml</loc></sitemap>
  <sitemap><loc>https://example.com/post-sitemap2.xml</loc></sitemap>
</sitemapindex>
"""

SUBSITEMAP_1 = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/news/article-1</loc></url>
  <url><loc>https://example.com/news/article-2</loc></url>
  <url><loc>https://example.com/news/article-3</loc></url>
</urlset>
"""

SUBSITEMAP_2 = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/news/article-4</loc></url>
  <url><loc>https://example.com/news/article-5</loc></url>
</urlset>
"""

FLAT_SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/news/article-1</loc></url>
  <url><loc>https://example.com/news/article-2</loc></url>
  <url><loc>https://example.com/news/article-3</loc></url>
  <url><loc>https://example.com/news/article-4</loc></url>
</urlset>
"""


def _article_html(slug: str) -> str:
    return (
        "<html><head>"
        f'<meta property="og:title" content="Headline {slug}"/>'
        f'<meta property="og:description" content="Dek for {slug} with enough words."/>'
        f'<meta property="og:image" content="https://cdn.example.com/{slug}.jpg"/>'
        '<meta property="article:published_time" content="2026-05-04T10:00:00Z"/>'
        "</head><body><p>Body</p></body></html>"
    )


LISTING_PAGE_1 = """<html><body>
  <nav><a href="https://example.com/about">About</a></nav>
  <a href="https://example.com/news/article-1">One</a>
  <a href="https://example.com/news/article-2">Two</a>
  <a href="https://othersite.com/news/x">Offsite</a>
</body></html>"""

LISTING_PAGE_2 = """<html><body>
  <a href="https://example.com/news/article-3">Three</a>
</body></html>"""


# --- helpers ----------------------------------------------------------------
def _resp(body: str, *, ctype: str) -> httpx.Response:
    return httpx.Response(200, content=body.encode("utf-8"), headers={"content-type": ctype})


def _xml(body: str) -> httpx.Response:
    return _resp(body, ctype="text/xml; charset=utf-8")


def _html(body: str) -> httpx.Response:
    return _resp(body, ctype="text/html; charset=utf-8")


def _gzip_xml(body: str) -> httpx.Response:
    return httpx.Response(
        200,
        content=gzip.compress(body.encode("utf-8")),
        headers={"content-type": "application/gzip"},
    )


def _empty_robots() -> httpx.Response:
    return httpx.Response(200, content=b"", headers={"content-type": "text/plain"})


def _make_handler(
    routes: dict[str, httpx.Response],
    *,
    robots: httpx.Response | None = None,
) -> tuple[Callable[[httpx.Request], httpx.Response], list[str]]:
    """Build a MockTransport handler routing by full URL, plus robots.txt."""
    calls: list[str] = []
    robots_resp = robots if robots is not None else _empty_robots()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if request.url.path == "/robots.txt":
            return robots_resp
        if url in routes:
            return routes[url]
        return httpx.Response(404, content=b"not found")

    return handler, calls


def _wire(scraper: BaseScraper, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    scraper._client = httpx.AsyncClient(
        headers={"User-Agent": scraper.user_agent},
        timeout=scraper.timeout,
        transport=httpx.MockTransport(handler),
    )


def _index_routes() -> dict[str, httpx.Response]:
    routes = {
        "https://example.com/sitemap_index.xml": _xml(SITEMAP_INDEX),
        "https://example.com/post-sitemap1.xml": _xml(SUBSITEMAP_1),
        "https://example.com/post-sitemap2.xml": _xml(SUBSITEMAP_2),
    }
    for i in range(1, 6):
        routes[f"https://example.com/news/article-{i}"] = _html(_article_html(f"article-{i}"))
    return routes


def _make_index_scraper(tmp_path: Path, **kwargs) -> SitemapIndexArchiveScraper:
    kwargs.setdefault("requests_per_second", 100000)
    kwargs.setdefault("name", "example-index")
    return SitemapIndexArchiveScraper(
        sitemap_url="https://example.com/sitemap_index.xml",
        source_domain="example.com",
        label="satire",
        state_dir=tmp_path,
        **kwargs,
    )


# --- detect_ai_optout -------------------------------------------------------
def test_optout_detects_ai_crawler_block() -> None:
    robots = "User-agent: GPTBot\nDisallow: /\nUser-agent: *\nDisallow: /wp-json/\n"
    verdict = detect_ai_optout(robots)
    assert verdict.opted_out is True
    assert "gptbot" in verdict.blocked_agents


def test_optout_detects_content_signal() -> None:
    robots = "User-agent: *\nContent-Signal: search=yes,ai-train=no\nAllow: /\n"
    verdict = detect_ai_optout(robots)
    assert verdict.opted_out is True
    assert verdict.content_signal_optout is True


def test_optout_permitting_robots_is_not_opted_out() -> None:
    robots = "User-agent: *\nAllow: /\nDisallow: /admin/\nSitemap: https://x.com/s.xml\n"
    verdict = detect_ai_optout(robots)
    assert verdict.opted_out is False
    assert verdict.blocked_agents == []


def test_optout_shared_useragent_group_blocks_all() -> None:
    # Consecutive user-agent lines share the rule block beneath them.
    robots = "User-agent: CCBot\nUser-agent: GPTBot\nDisallow: /\n"
    verdict = detect_ai_optout(robots)
    assert verdict.opted_out is True
    assert set(verdict.blocked_agents) == {"ccbot", "gptbot"}


def test_optout_partial_disallow_is_not_full_block() -> None:
    # GPTBot only restricted from a subtree, not the whole site.
    robots = "User-agent: GPTBot\nDisallow: /private/\n"
    verdict = detect_ai_optout(robots)
    assert verdict.opted_out is False


# --- parse_sitemap_xml ------------------------------------------------------
def test_parse_sitemap_index() -> None:
    kind, locs = parse_sitemap_xml(SITEMAP_INDEX.encode("utf-8"))
    assert kind == "index"
    assert locs == [
        "https://example.com/post-sitemap1.xml",
        "https://example.com/post-sitemap2.xml",
    ]


def test_parse_sitemap_urlset() -> None:
    kind, locs = parse_sitemap_xml(SUBSITEMAP_1.encode("utf-8"))
    assert kind == "urlset"
    assert len(locs) == 3


def test_parse_sitemap_malformed_returns_unknown() -> None:
    kind, locs = parse_sitemap_xml(b"<not-xml")
    assert kind == "unknown"
    assert locs == []


# --- image extraction priority ----------------------------------------------
def test_image_priority_prefers_og_then_twitter_then_img(tmp_path: Path) -> None:
    scraper = _make_index_scraper(tmp_path)
    base = "https://example.com/news/x"

    og = (
        '<meta property="og:image" content="https://cdn/og.jpg"/>'
        '<meta name="twitter:image" content="https://cdn/tw.jpg"/>'
        '<img src="https://cdn/body.jpg"/>'
    )
    assert scraper._extract_image(og, base) == "https://cdn/og.jpg"

    twitter = (
        '<meta name="twitter:image" content="https://cdn/tw.jpg"/>'
        '<img src="https://cdn/body.jpg"/>'
    )
    assert scraper._extract_image(twitter, base) == "https://cdn/tw.jpg"

    body_img = '<p>x</p><img src="/media/body.jpg"/>'
    assert scraper._extract_image(body_img, base) == "https://example.com/media/body.jpg"


def test_image_extraction_skips_data_uris_and_svgs(tmp_path: Path) -> None:
    scraper = _make_index_scraper(tmp_path)
    html = (
        '<img src="data:image/png;base64,xxxx"/>'
        '<img src="/icons/logo.svg"/>'
        '<img src="/media/real.jpg"/>'
    )
    assert scraper._extract_image(html, "https://example.com/a") == (
        "https://example.com/media/real.jpg"
    )


def test_image_extraction_returns_none_when_absent(tmp_path: Path) -> None:
    scraper = _make_index_scraper(tmp_path)
    assert scraper._extract_image("<html><body>no images</body></html>", "https://example.com/a") is None


# --- SitemapIndexArchiveScraper end-to-end ----------------------------------
async def test_sitemap_index_scrape_yields_imaged_items(tmp_path: Path) -> None:
    scraper = _make_index_scraper(tmp_path)
    handler, _ = _make_handler(_index_routes())
    _wire(scraper, handler)
    try:
        items = [it async for it in scraper.scrape(max_articles=100)]
    finally:
        await scraper.close()

    # 3 articles in sub-sitemap 1 + 2 in sub-sitemap 2 = 5, all imaged.
    assert len(items) == 5
    urls = {it.source_url for it in items}
    assert urls == {f"https://example.com/news/article-{i}" for i in range(1, 6)}
    for it in items:
        assert it.image_url and it.image_url.startswith("https://cdn.example.com/")
        assert it.title.startswith("Headline article-")
        assert it.source_domain == "example.com"
        assert it.metadata["label"] == "satire"
        assert it.metadata["source_type"] == "archive"
        assert it.timestamp.year == 2026


async def test_article_without_image_is_skipped(tmp_path: Path) -> None:
    routes = _index_routes()
    # Strip the image out of article-2 entirely.
    routes["https://example.com/news/article-2"] = _html(
        '<html><head><meta property="og:title" content="No Image Here"/>'
        "</head><body>text</body></html>"
    )
    scraper = _make_index_scraper(tmp_path)
    handler, _ = _make_handler(routes)
    _wire(scraper, handler)
    try:
        items = [it async for it in scraper.scrape(max_articles=100)]
    finally:
        await scraper.close()

    assert len(items) == 4
    assert "https://example.com/news/article-2" not in {it.source_url for it in items}


async def test_max_articles_caps_and_leaves_partial_page_unadvanced(tmp_path: Path) -> None:
    scraper = _make_index_scraper(tmp_path)
    handler, _ = _make_handler(_index_routes())
    _wire(scraper, handler)
    try:
        items = [it async for it in scraper.scrape(max_articles=2)]
    finally:
        await scraper.close()

    assert len(items) == 2
    # Page 1 (sub-sitemap 1) was interrupted at the budget, so last_page
    # stays at 0 while the two done URLs are recorded for the resume.
    state = json.loads((tmp_path / "example-index.json").read_text())
    assert state["last_page"] == 0
    assert len(state["scraped_urls"]) == 2


# --- resumability -----------------------------------------------------------
async def test_resume_skips_already_scraped_urls(tmp_path: Path) -> None:
    # First run: cap at 2 (partway through sub-sitemap 1).
    s1 = _make_index_scraper(tmp_path)
    handler, _ = _make_handler(_index_routes())
    _wire(s1, handler)
    try:
        first = [it async for it in s1.scrape(max_articles=2)]
    finally:
        await s1.close()
    assert len(first) == 2
    done = {it.source_url for it in first}

    # Second run resumes; must not re-yield the first two URLs.
    s2 = _make_index_scraper(tmp_path)
    handler2, _ = _make_handler(_index_routes())
    _wire(s2, handler2)
    try:
        second = [it async for it in s2.scrape(max_articles=100, resume=True)]
    finally:
        await s2.close()

    second_urls = {it.source_url for it in second}
    assert done.isdisjoint(second_urls)
    # Across both runs every distinct article is collected exactly once.
    assert done | second_urls == {f"https://example.com/news/article-{i}" for i in range(1, 6)}


async def test_fresh_ignores_saved_state(tmp_path: Path) -> None:
    s1 = _make_index_scraper(tmp_path)
    _wire(s1, _make_handler(_index_routes())[0])
    try:
        [it async for it in s1.scrape(max_articles=2)]
    finally:
        await s1.close()

    s2 = _make_index_scraper(tmp_path)
    _wire(s2, _make_handler(_index_routes())[0])
    try:
        again = [it async for it in s2.scrape(max_articles=100, resume=False)]
    finally:
        await s2.close()

    # resume=False re-collects everything from page 1.
    assert len(again) == 5


# --- opt-out + robots gates -------------------------------------------------
async def test_optout_source_is_skipped(tmp_path: Path) -> None:
    scraper = _make_index_scraper(tmp_path)
    robots = httpx.Response(
        200,
        content=b"User-agent: GPTBot\nDisallow: /\n",
        headers={"content-type": "text/plain"},
    )
    handler, _ = _make_handler(_index_routes(), robots=robots)
    _wire(scraper, handler)
    try:
        items = [it async for it in scraper.scrape(max_articles=100)]
    finally:
        await scraper.close()
    assert items == []


async def test_optout_override_allows_scrape(tmp_path: Path) -> None:
    scraper = _make_index_scraper(tmp_path, respect_optout=False)
    robots = httpx.Response(
        200,
        content=b"User-agent: GPTBot\nDisallow: /\n",
        headers={"content-type": "text/plain"},
    )
    handler, _ = _make_handler(_index_routes(), robots=robots)
    _wire(scraper, handler)
    try:
        items = [it async for it in scraper.scrape(max_articles=100)]
    finally:
        await scraper.close()
    # GPTBot is blocked but our UA (Satira-Ingest) isn't, and the explicit
    # override means the AI-opt-out guard is bypassed.
    assert len(items) == 5


async def test_robots_disallow_on_entry_skips_source(tmp_path: Path) -> None:
    scraper = _make_index_scraper(tmp_path)
    robots = httpx.Response(
        200,
        content=b"User-agent: *\nDisallow: /\n",
        headers={"content-type": "text/plain"},
    )
    handler, _ = _make_handler(_index_routes(), robots=robots)
    _wire(scraper, handler)
    try:
        items = [it async for it in scraper.scrape(max_articles=100)]
    finally:
        await scraper.close()
    assert items == []


# --- FlatSitemapArchiveScraper ----------------------------------------------
async def test_flat_sitemap_paginates_by_page_size(tmp_path: Path) -> None:
    routes = {"https://example.com/sitemap.xml": _xml(FLAT_SITEMAP)}
    for i in range(1, 5):
        routes[f"https://example.com/news/article-{i}"] = _html(_article_html(f"article-{i}"))
    scraper = FlatSitemapArchiveScraper(
        name="flat",
        sitemap_url="https://example.com/sitemap.xml",
        source_domain="example.com",
        label="satire",
        state_dir=tmp_path,
        requests_per_second=100000,
        page_size=2,
    )
    handler, _ = _make_handler(routes)
    _wire(scraper, handler)
    try:
        items = [it async for it in scraper.scrape(max_articles=100)]
    finally:
        await scraper.close()

    assert len(items) == 4
    # page_size=2 over 4 urls => 2 pages fully consumed.
    state = json.loads((tmp_path / "flat.json").read_text())
    assert state["last_page"] == 2


async def test_flat_sitemap_handles_gzip(tmp_path: Path) -> None:
    routes = {"https://example.com/sitemap.xml": _gzip_xml(FLAT_SITEMAP)}
    for i in range(1, 5):
        routes[f"https://example.com/news/article-{i}"] = _html(_article_html(f"article-{i}"))
    scraper = FlatSitemapArchiveScraper(
        name="flatgz",
        sitemap_url="https://example.com/sitemap.xml",
        source_domain="example.com",
        state_dir=tmp_path,
        requests_per_second=100000,
        page_size=50,
    )
    handler, _ = _make_handler(routes)
    _wire(scraper, handler)
    try:
        items = [it async for it in scraper.scrape(max_articles=100)]
    finally:
        await scraper.close()
    assert len(items) == 4


# --- PaginatedArchiveScraper ------------------------------------------------
async def test_paginated_scraper_walks_listing_pages(tmp_path: Path) -> None:
    routes = {
        "https://example.com/page/1/": _html(LISTING_PAGE_1),
        "https://example.com/page/2/": _html(LISTING_PAGE_2),
        # page 3 -> 404 (default), so the walk stops.
    }
    for i in range(1, 4):
        routes[f"https://example.com/news/article-{i}"] = _html(_article_html(f"article-{i}"))
    scraper = PaginatedArchiveScraper(
        name="paged",
        listing_url_template="https://example.com/page/{page}/",
        source_domain="example.com",
        article_url_regex=r"/news/",
        state_dir=tmp_path,
        requests_per_second=100000,
    )
    handler, _ = _make_handler(routes)
    _wire(scraper, handler)
    try:
        items = [it async for it in scraper.scrape(max_articles=100)]
    finally:
        await scraper.close()

    urls = {it.source_url for it in items}
    # Offsite link and the /about nav link are filtered by domain + regex.
    assert urls == {f"https://example.com/news/article-{i}" for i in range(1, 4)}


def test_paginated_requires_page_placeholder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="page"):
        PaginatedArchiveScraper(
            name="bad",
            listing_url_template="https://example.com/archive/",
            source_domain="example.com",
            state_dir=tmp_path,
        )


# --- registry ---------------------------------------------------------------
async def test_registry_combines_sources_concurrently(tmp_path: Path) -> None:
    a = _make_index_scraper(tmp_path, name="src-a")
    _wire(a, _make_handler(_index_routes())[0])

    b = FlatSitemapArchiveScraper(
        name="src-b",
        sitemap_url="https://example.com/sitemap.xml",
        source_domain="example.com",
        state_dir=tmp_path,
        requests_per_second=100000,
    )
    b_routes = {"https://example.com/sitemap.xml": _xml(FLAT_SITEMAP)}
    for i in range(1, 5):
        b_routes[f"https://example.com/news/article-{i}"] = _html(_article_html(f"a{i}"))
    _wire(b, _make_handler(b_routes)[0])

    registry = ArchiveScraperRegistry(scrapers=[a, b])
    try:
        items = [it async for it in registry.scrape_all(max_articles_per_source=100)]
    finally:
        await registry.close()

    # 5 from the index source + 4 from the flat source.
    assert len(items) == 9


async def test_registry_tolerates_a_failing_source(tmp_path: Path) -> None:
    good = _make_index_scraper(tmp_path, name="good")
    _wire(good, _make_handler(_index_routes())[0])

    bad = _make_index_scraper(tmp_path, name="bad")

    def boom(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return _empty_robots()
        raise httpx.ConnectError("simulated outage", request=request)

    _wire(bad, boom)

    registry = ArchiveScraperRegistry(scrapers=[good, bad])
    try:
        items = [it async for it in registry.scrape_all(max_articles_per_source=100)]
    finally:
        await registry.close()

    # The bad source contributes nothing; the good one still yields its 5.
    assert len(items) == 5


async def test_empty_registry_yields_nothing() -> None:
    registry = ArchiveScraperRegistry()
    items = [it async for it in registry.scrape_all(max_articles_per_source=10)]
    assert items == []


# --- build_scraper_from_config ----------------------------------------------
def test_build_from_config_creates_correct_types() -> None:
    sm = build_scraper_from_config(
        {
            "type": "sitemap_index",
            "name": "x",
            "source_domain": "example.com",
            "sitemap_url": "https://example.com/sitemap_index.xml",
        }
    )
    assert isinstance(sm, SitemapIndexArchiveScraper)

    pg = build_scraper_from_config(
        {
            "type": "paginated",
            "name": "y",
            "source_domain": "example.com",
            "listing_url_template": "https://example.com/page/{page}/",
        }
    )
    assert isinstance(pg, PaginatedArchiveScraper)


def test_build_from_config_rejects_unknown_type_and_missing_fields() -> None:
    assert build_scraper_from_config({"type": "nope", "name": "x"}) is None
    assert build_scraper_from_config({"type": "sitemap_index"}) is None  # no name
    # Missing the type-specific URL.
    assert build_scraper_from_config({"type": "paginated", "name": "z"}) is None


def test_registry_from_config_skips_invalid_entries() -> None:
    registry = ArchiveScraperRegistry.from_config(
        [
            {
                "type": "flat_sitemap",
                "name": "ok",
                "source_domain": "example.com",
                "sitemap_url": "https://example.com/sitemap.xml",
            },
            {"type": "bogus", "name": "dropme"},
        ]
    )
    assert len(registry.scrapers) == 1
    assert registry.scrapers[0].name == "ok"
