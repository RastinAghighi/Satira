"""Authentic-news scrapers for the ingest pipeline.

These are the credibility counterpart to ``satire_scrapers``: same
:class:`ScrapedItem` shape, but every record is stamped with
``metadata["label"] = "authentic"`` so downstream classifiers can
contrast them against the satire feeds.

Two flavours live here. :class:`GDELTScraper` queries the public
GDELT 2.0 DOC API — free, machine-readable, and uniform per record,
which means we never have to scrape article HTML. :class:`RSSNewsScraper`
is a generic ``feedparser`` wrapper pre-configured for major outlets
(BBC, NPR, Guardian, NYT, Al Jazeera); it mirrors the satire RSS
scrapers
so the test scaffolding and quirks (image extraction, bozo-feed
handling, …) carry over almost verbatim.

Both stamp ``source_domain`` on every item — downstream credibility
scoring and dedupe layers care about which outlet a story came from
and shouldn't have to re-parse the URL each time.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse

import feedparser

from satira.ingest.base_scraper import BaseScraper, ScrapedItem


logger = logging.getLogger(__name__)


_GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

_HTML_TAG_RE = re.compile(r"<[^>]+>")

# og:image can have property/name and content in either order. Match
# both orderings rather than a single permissive pattern that would also
# match unrelated <meta> tags between attributes.
_OG_IMAGE_PROP_FIRST = re.compile(
    r'<meta\b[^>]*?\b(?:property|name)\s*=\s*["\']og:image(?::url)?["\']'
    r'[^>]*?\bcontent\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_OG_IMAGE_CONTENT_FIRST = re.compile(
    r'<meta\b[^>]*?\bcontent\s*=\s*["\']([^"\']+)["\']'
    r'[^>]*?\b(?:property|name)\s*=\s*["\']og:image(?::url)?["\']',
    re.IGNORECASE,
)

# BBC's RSS hands out 240-px thumbnails (240x135) which sit below the
# downloader's 200x200 minimum. The same image is served at larger
# sizes by the CDN via path substitution — rewriting the URL here is
# cheaper than fetching the article HTML for og:image.
_BBC_THUMB_PATH_RE = re.compile(r"/ace/standard/240/")
_BBC_IC_THUMB_PATH_RE = re.compile(r"/images/ic/240x135/")


def _upgrade_thumbnail_url(url: str) -> str:
    """Rewrite known low-res RSS thumbnail URLs to a larger CDN size."""
    if _BBC_THUMB_PATH_RE.search(url):
        return _BBC_THUMB_PATH_RE.sub("/ace/standard/1024/", url)
    if _BBC_IC_THUMB_PATH_RE.search(url):
        return _BBC_IC_THUMB_PATH_RE.sub("/images/ic/1024x576/", url)
    return url


HtmlFetcher = Callable[[str], Awaitable[str | None]]


async def _extract_image_url(
    entry: Any,
    *,
    fetcher: HtmlFetcher | None = None,
) -> tuple[str | None, str]:
    """Pull a featured image URL from an RSS entry, trying common locations.

    Order: ``enclosure`` → ``media:thumbnail`` → ``media:content`` →
    ``og:image`` from the article page (tertiary fallback — only fired
    when ``fetcher`` is provided and the structured locations all came
    up empty, since fetching every article HTML just for an image would
    otherwise gut throughput).

    Returns ``(url_or_none, strategy)`` where ``strategy`` names the
    location the URL came from (``enclosure``, ``media_thumbnail``,
    ``media_content``, ``og_image``, or ``none``) so the caller can log
    which path each item took.
    """
    enclosures = getattr(entry, "enclosures", None) or []
    for enc in enclosures:
        etype = (enc.get("type") or "").lower()
        href = enc.get("href") or enc.get("url")
        if href and (etype.startswith("image/") or not etype):
            return href, "enclosure"

    media_thumb = getattr(entry, "media_thumbnail", None)
    if media_thumb:
        url = media_thumb[0].get("url")
        if url:
            return _upgrade_thumbnail_url(url), "media_thumbnail"

    media_content = getattr(entry, "media_content", None)
    if media_content:
        for mc in media_content:
            url = mc.get("url")
            mtype = (mc.get("type") or "").lower()
            medium = (mc.get("medium") or "").lower()
            if url and (
                mtype.startswith("image/") or medium == "image" or not mtype
            ):
                return _upgrade_thumbnail_url(url), "media_content"

    if fetcher is not None:
        link = (getattr(entry, "link", "") or "").strip()
        if link:
            og = await _extract_og_image(link, fetcher)
            if og:
                return og, "og_image"

    return None, "none"


async def _extract_og_image(article_url: str, fetcher: HtmlFetcher) -> str | None:
    """Best-effort fetch of the article page and pull of its ``og:image``.

    Failures are silent: this is a fallback, and the caller will simply
    end up with a text-only item if nothing comes back.
    """
    try:
        html = await fetcher(article_url)
    except Exception as exc:  # noqa: BLE001 — opportunistic fallback
        logger.debug("og:image fetch failed for %s: %s", article_url, exc)
        return None
    if not html:
        return None
    for pattern in (_OG_IMAGE_PROP_FIRST, _OG_IMAGE_CONTENT_FIRST):
        match = pattern.search(html)
        if match:
            url = match.group(1).strip()
            if url:
                # og:image may be a relative URL; resolve against the
                # article URL so downstream fetches don't 404.
                return urljoin(article_url, url)
    return None


def _extract_timestamp(entry: Any) -> datetime:
    """Best-effort UTC timestamp from an RSS entry, defaulting to now."""
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = getattr(entry, attr, None)
        if parsed:
            try:
                return datetime(*parsed[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    return datetime.now(timezone.utc)


def _strip_html(text: str) -> str:
    if not text:
        return ""
    return _HTML_TAG_RE.sub("", text).strip()


def _parse_gdelt_seendate(seendate: str) -> datetime:
    """Parse GDELT's ``seendate`` ('YYYYMMDDTHHMMSSZ') into a UTC datetime.

    Falls back to ``datetime.now(UTC)`` for unparseable values rather
    than dropping the record — a single odd timestamp shouldn't cost us
    an article that's otherwise fine.
    """
    try:
        return datetime.strptime(seendate, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except (TypeError, ValueError):
        logger.debug("could not parse GDELT seendate %r", seendate)
        return datetime.now(timezone.utc)


def _format_gdelt_datetime(dt: datetime) -> str:
    """Format a UTC datetime in GDELT's ``YYYYMMDDHHMMSS`` form."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y%m%d%H%M%S")


class GDELTScraper(BaseScraper):
    """Scrapes the GDELT Project's public DOC API for news articles.

    Two public entry points:

    * :meth:`search` — one DOC API call, returns the raw article list.
      Driven by ``timespan`` (e.g. ``"1d"``) plus optional language /
      country filters. Use this when you want a single shot at a
      specific topic and don't care about deep history.
    * :meth:`scrape` — async iterator over ``ScrapedItem`` records for a
      single query. Paginates backwards through history by walking the
      oldest ``seendate`` returned per page; useful when you want more
      than :attr:`MAX_RECORDS_PER_CALL` items for one query.
    * :meth:`scrape_topics` — async iterator that fans :meth:`search`
      across a list of broad topic queries (default
      :attr:`DEFAULT_QUERIES`) and dedups results by URL across the
      whole run. This is the high-volume path the Tier 1 builder uses.

    Rate limiting defaults to 1 request/second, GDELT's stated public
    cap; the base class honours this between every API call.

    API reference: https://api.gdeltproject.org/api/v2/doc/doc
    """

    API_URL = _GDELT_DOC_API
    MAX_RECORDS_PER_CALL = 250  # GDELT's hard cap.
    DEFAULT_RATE_LIMIT_PER_MINUTE = 60  # GDELT allows ~1 req/sec.

    # Broad-topic queries tuned for diverse English-language coverage.
    # Ten topics × 250 records ≈ 2500 articles per scrape_topics() pass.
    DEFAULT_QUERIES: tuple[str, ...] = (
        "politics",
        "economy",
        "technology",
        "science",
        "health",
        "climate",
        "elections",
        "business",
        "international",
        "sports",
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault(
            "rate_limit_per_minute", self.DEFAULT_RATE_LIMIT_PER_MINUTE
        )
        super().__init__(**kwargs)

    async def search(
        self,
        query: str,
        *,
        mode: str = "ArtList",
        maxrecords: int = 250,
        format: str = "json",
        timespan: str = "1d",
        sourcecountry: str | None = None,
        sourcelang: str | None = "english",
    ) -> list[dict[str, Any]]:
        """Run one DOC API call and return its raw ``articles`` list.

        Returns ``[]`` on any fetch / parse failure (logged at
        ``WARNING``) so a single bad request can't take down a wider
        :meth:`scrape_topics` loop.
        """
        if not query:
            raise ValueError("GDELTScraper.search requires a non-empty query")
        if maxrecords <= 0:
            return []

        params: dict[str, Any] = {
            "query": query,
            "mode": mode,
            "format": format,
            "maxrecords": min(maxrecords, self.MAX_RECORDS_PER_CALL),
            "timespan": timespan,
            "sort": "DateDesc",
        }
        if sourcecountry:
            params["sourcecountry"] = sourcecountry
        if sourcelang:
            params["sourcelang"] = sourcelang

        url = f"{self.API_URL}?{urlencode(params)}"
        body = await self.fetch(url)
        if body is None:
            logger.warning("GDELT search: query %r — fetch failed or blocked", query)
            return []
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.warning(
                "GDELT search: invalid JSON for query %r: %s", query, exc
            )
            return []
        articles = data.get("articles")
        if not isinstance(articles, list):
            return []
        return articles

    async def scrape(
        self,
        query: str,
        max_items: int = 500,
        start_date: datetime | None = None,
        sourcecountry: str | None = None,
        sourcelang: str | None = None,
        **_: Any,
    ) -> AsyncIterator[ScrapedItem]:
        if not query:
            raise ValueError("GDELTScraper.scrape requires a non-empty query")
        if max_items <= 0:
            return

        end_dt = datetime.now(timezone.utc)
        emitted = 0
        seen_urls: set[str] = set()

        while emitted < max_items:
            page_size = min(max_items - emitted, self.MAX_RECORDS_PER_CALL)
            url = self._build_paginated_url(
                query=query,
                page_size=page_size,
                end_dt=end_dt,
                start_date=start_date,
                sourcecountry=sourcecountry,
                sourcelang=sourcelang,
            )

            body = await self.fetch(url)
            if body is None:
                logger.warning("GDELT: query %r — fetch failed or blocked", query)
                return
            try:
                data = json.loads(body)
            except json.JSONDecodeError as exc:
                logger.warning("GDELT: invalid JSON for query %r: %s", query, exc)
                return

            articles = data.get("articles") or []
            if not articles:
                return

            page_emitted = 0
            oldest: datetime | None = None
            for art in articles:
                item = self._article_to_item(art, query)
                if item is None:
                    continue
                if item.source_url and item.source_url in seen_urls:
                    continue
                if item.source_url:
                    seen_urls.add(item.source_url)
                self.stats.items_yielded += 1
                emitted += 1
                page_emitted += 1
                yield item
                if oldest is None or item.timestamp < oldest:
                    oldest = item.timestamp
                if emitted >= max_items:
                    return

            # Pagination guard rails:
            #   * If GDELT didn't fill the page, the window is drained.
            #   * If pagination wouldn't move the cursor backwards, bail
            #     to avoid an infinite loop on identical timestamps.
            if len(articles) < page_size or page_emitted == 0 or oldest is None:
                return
            if oldest >= end_dt:
                return
            end_dt = oldest

    async def scrape_topics(
        self,
        queries: list[str] | tuple[str, ...] | None = None,
        *,
        max_per_query: int = 250,
        timespan: str = "1d",
        sourcecountry: str | None = None,
        sourcelang: str | None = "english",
    ) -> AsyncIterator[ScrapedItem]:
        """Iterate over a set of topic queries, yielding deduped items.

        URLs are deduped across the *whole* run, not per query — GDELT
        often surfaces the same article under multiple broad topics
        (e.g. an election story tagged ``politics`` and
        ``international``), and we don't want the same record in the
        dataset twice. A single ``search()`` failure for one topic
        logs and is skipped; the next topic still runs.
        """
        topics = (
            list(queries) if queries is not None else list(self.DEFAULT_QUERIES)
        )
        seen_urls: set[str] = set()

        for query in topics:
            try:
                articles = await self.search(
                    query,
                    maxrecords=max_per_query,
                    timespan=timespan,
                    sourcecountry=sourcecountry,
                    sourcelang=sourcelang,
                )
            except Exception as exc:  # noqa: BLE001 — one query mustn't kill the loop
                logger.exception(
                    "GDELT scrape_topics: search failed for %r: %s", query, exc
                )
                continue

            for art in articles:
                item = self._article_to_item(art, query)
                if item is None:
                    continue
                if item.source_url and item.source_url in seen_urls:
                    continue
                if item.source_url:
                    seen_urls.add(item.source_url)
                self.stats.items_yielded += 1
                yield item

    def _build_paginated_url(
        self,
        *,
        query: str,
        page_size: int,
        end_dt: datetime,
        start_date: datetime | None,
        sourcecountry: str | None = None,
        sourcelang: str | None = None,
    ) -> str:
        params: dict[str, Any] = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": page_size,
            "sort": "DateDesc",
            "enddatetime": _format_gdelt_datetime(end_dt),
        }
        if start_date is not None:
            params["startdatetime"] = _format_gdelt_datetime(start_date)
        if sourcecountry:
            params["sourcecountry"] = sourcecountry
        if sourcelang:
            params["sourcelang"] = sourcelang
        return f"{self.API_URL}?{urlencode(params)}"

    def _article_to_item(self, art: dict[str, Any], query: str) -> ScrapedItem | None:
        url = (art.get("url") or "").strip()
        title = (art.get("title") or "").strip()
        if not url and not title:
            return None

        domain = (art.get("domain") or "").strip().lower()
        if not domain and url:
            domain = urlparse(url).netloc.lower()

        timestamp = _parse_gdelt_seendate(art.get("seendate") or "")
        image_url = (art.get("socialimage") or "").strip() or None

        return ScrapedItem(
            source_url=url,
            image_url=image_url,
            title=title,
            text="",  # GDELT ArtList doesn't include article body text.
            timestamp=timestamp,
            source_domain=domain,
            metadata={
                "label": "authentic",
                "source_type": "gdelt",
                "query": query,
                "language": art.get("language"),
                "country": art.get("sourcecountry"),
            },
        )


class RSSNewsScraper(BaseScraper):
    """Generic RSS scraper pre-wired for major news outlets.

    The default feed list covers BBC, NPR, the Guardian, the New York
    Times, and Al Jazeera — between them we get broad coverage with
    stable feeds that don't change shape monthly. Per-feed metadata
    (outlet display name and canonical domain) is stamped on each item
    so downstream code can score credibility without re-parsing the
    URL.

    Reuters' public RSS endpoints were retired and now return zero
    entries; NYT World and Al Jazeera fill the same global-news slot.
    The AP top-news feed went dead in 2026 (zero entries on every
    fetch) and was dropped from the registry.
    """

    DEFAULT_FEEDS: dict[str, str] = {
        "bbc_news": "http://feeds.bbci.co.uk/news/rss.xml",
        "npr_news": "https://feeds.npr.org/1001/rss.xml",
        "guardian_world": "https://www.theguardian.com/world/rss",
        "nyt_world": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "aljazeera_all": "https://www.aljazeera.com/xml/rss/all.xml",
    }

    DEFAULT_OUTLETS: dict[str, tuple[str, str]] = {
        "bbc_news": ("BBC", "bbc.co.uk"),
        "npr_news": ("NPR", "npr.org"),
        "guardian_world": ("The Guardian", "theguardian.com"),
        "nyt_world": ("The New York Times", "nytimes.com"),
        "aljazeera_all": ("Al Jazeera", "aljazeera.com"),
    }

    def __init__(
        self,
        feeds: dict[str, str] | None = None,
        outlets: dict[str, tuple[str, str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.feeds = dict(feeds) if feeds is not None else dict(self.DEFAULT_FEEDS)
        self.outlets = (
            dict(outlets) if outlets is not None else dict(self.DEFAULT_OUTLETS)
        )

    async def scrape(
        self,
        feed_keys: list[str] | None = None,
        max_per_feed: int = 50,
        **_: Any,
    ) -> AsyncIterator[ScrapedItem]:
        if max_per_feed <= 0:
            return
        keys = feed_keys if feed_keys is not None else list(self.feeds.keys())

        # Two-pass design: fetch all feeds first, then interleave their
        # entries so the consumer's max_items budget is shared fairly.
        # The previous "feed-by-feed" iteration meant the first feed
        # alone could fill a small budget — concretely, BBC's 31
        # entries would crowd out NPR / Guardian / NYT / Al Jazeera
        # whenever ``max_items`` was below ~150.
        feed_payloads: list[tuple[str, str, str, str, list[Any]]] = []
        for key in keys:
            feed_url = self.feeds.get(key)
            if not feed_url:
                logger.warning(
                    "RSSNewsScraper: unknown feed key %r — skipping", key
                )
                continue

            outlet_name, source_domain = self.outlets.get(
                key, (key, urlparse(feed_url).netloc.lower())
            )

            body = await self.fetch(feed_url)
            if body is None:
                logger.warning(
                    "RSSNewsScraper: feed %s returned no body "
                    "(blocked or all retries failed)", key,
                )
                continue

            feed = feedparser.parse(body)
            if feed.bozo and not feed.entries:
                # Malformed AND no entries recovered: nothing to yield.
                # If feedparser recovered entries despite ``bozo``, we
                # still want them.
                logger.warning(
                    "RSSNewsScraper: feed %s parse failed: %s",
                    key, getattr(feed, "bozo_exception", "unknown"),
                )
                continue

            feed_payloads.append(
                (key, outlet_name, source_domain, feed_url, list(feed.entries[:max_per_feed]))
            )

        if not feed_payloads:
            return

        max_len = max(len(p[4]) for p in feed_payloads)
        for idx in range(max_len):
            for key, outlet_name, source_domain, feed_url, entries in feed_payloads:
                if idx >= len(entries):
                    continue
                item = await self._entry_to_item(
                    entries[idx],
                    feed_key=key,
                    outlet_name=outlet_name,
                    source_domain=source_domain,
                    feed_url=feed_url,
                )
                if item is None:
                    continue
                self.stats.items_yielded += 1
                yield item

    async def _entry_to_item(
        self,
        entry: Any,
        *,
        feed_key: str,
        outlet_name: str,
        source_domain: str,
        feed_url: str,
    ) -> ScrapedItem | None:
        url = (getattr(entry, "link", "") or "").strip()
        title = (getattr(entry, "title", "") or "").strip()
        if not url and not title:
            return None
        summary_raw = (
            getattr(entry, "summary", "")
            or getattr(entry, "description", "")
            or ""
        )
        image_url, image_strategy = await _extract_image_url(
            entry, fetcher=self.fetch
        )
        logger.info(
            "RSSNewsScraper: feed=%s strategy=%s image=%s",
            feed_key,
            image_strategy,
            image_url or "<none>",
        )
        return ScrapedItem(
            source_url=url,
            image_url=image_url,
            title=title,
            text=_strip_html(summary_raw),
            timestamp=_extract_timestamp(entry),
            source_domain=source_domain,
            metadata={
                "label": "authentic",
                "source_type": "rss",
                "outlet": outlet_name,
                "feed_key": feed_key,
                "feed_url": feed_url,
                "image_strategy": image_strategy,
            },
        )


class NewsScraperRegistry:
    """Combines the news scrapers behind a single iterator.

    RSS first (cheap and reliable), then GDELT for any user-supplied
    queries. Failures in any one source are logged and skipped rather
    than aborting — losing one outlet's slice is better than losing the
    whole batch.
    """

    def __init__(
        self,
        rss_scraper: RSSNewsScraper | None = None,
        gdelt_scraper: GDELTScraper | None = None,
    ) -> None:
        self.rss_scraper = (
            rss_scraper if rss_scraper is not None else RSSNewsScraper()
        )
        self.gdelt_scraper = (
            gdelt_scraper if gdelt_scraper is not None else GDELTScraper()
        )

    async def scrape_all(
        self,
        gdelt_queries: list[str] | None = None,
        max_items: int = 1000,
        gdelt_timespan: str = "1d",
        gdelt_sourcecountry: str | None = None,
        gdelt_sourcelang: str | None = "english",
    ) -> AsyncIterator[ScrapedItem]:
        if max_items <= 0:
            return

        emitted = 0
        try:
            async for item in self.rss_scraper.scrape():
                yield item
                emitted += 1
                if emitted >= max_items:
                    return
        except Exception as exc:  # noqa: BLE001 — registry must be tolerant
            logger.exception("RSSNewsScraper failed mid-run: %s", exc)

        if not gdelt_queries:
            return

        # Spread the remaining budget across queries so a noisy topic
        # can't starve the others. Cross-query URL dedup happens inside
        # scrape_topics().
        remaining = max_items - emitted
        per_query = max(1, remaining // len(gdelt_queries))
        try:
            async for item in self.gdelt_scraper.scrape_topics(
                queries=list(gdelt_queries),
                max_per_query=per_query,
                timespan=gdelt_timespan,
                sourcecountry=gdelt_sourcecountry,
                sourcelang=gdelt_sourcelang,
            ):
                yield item
                emitted += 1
                if emitted >= max_items:
                    return
        except Exception as exc:  # noqa: BLE001
            logger.exception("GDELTScraper.scrape_topics failed: %s", exc)

    async def close(self) -> None:
        await self.rss_scraper.close()
        await self.gdelt_scraper.close()

    async def __aenter__(self) -> "NewsScraperRegistry":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()
