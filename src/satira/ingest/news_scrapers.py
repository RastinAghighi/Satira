"""Authentic-news scrapers for the ingest pipeline.

These are the credibility counterpart to ``satire_scrapers``: same
:class:`ScrapedItem` shape, but every record is stamped with
``metadata["label"] = "authentic"`` so downstream classifiers can
contrast them against the satire feeds.

Two flavours live here. :class:`GDELTScraper` queries the public
GDELT 2.0 DOC API — free, machine-readable, and uniform per record,
which means we never have to scrape article HTML. :class:`RSSNewsScraper`
is a generic ``feedparser`` wrapper pre-configured for major outlets
(Reuters, BBC, AP, NPR, Guardian); it mirrors the satire RSS scrapers
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
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode, urlparse

import feedparser

from satira.ingest.base_scraper import BaseScraper, ScrapedItem


logger = logging.getLogger(__name__)


_GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

_IMG_TAG_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _extract_image_url(entry: Any) -> str | None:
    """Pull a featured image URL from an RSS entry, trying common locations.

    Same precedence as the satire scrapers: ``media:thumbnail`` →
    ``media:content`` → ``enclosure`` → embedded ``<img>`` in the body.
    """
    media_thumb = getattr(entry, "media_thumbnail", None)
    if media_thumb:
        url = media_thumb[0].get("url")
        if url:
            return url

    media_content = getattr(entry, "media_content", None)
    if media_content:
        url = media_content[0].get("url")
        if url:
            return url

    enclosures = getattr(entry, "enclosures", None) or []
    for enc in enclosures:
        etype = (enc.get("type") or "").lower()
        href = enc.get("href") or enc.get("url")
        if href and (etype.startswith("image/") or not etype):
            return href

    content_blocks = getattr(entry, "content", None) or []
    for block in content_blocks:
        match = _IMG_TAG_RE.search(block.get("value", "") or "")
        if match:
            return match.group(1)

    summary = getattr(entry, "summary", "") or ""
    match = _IMG_TAG_RE.search(summary)
    if match:
        return match.group(1)

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

    The DOC endpoint returns up to :attr:`MAX_RECORDS_PER_CALL` matching
    articles per request. To honour callers asking for more we paginate
    by walking the time window backwards using each batch's oldest
    ``seendate``, so a single ``scrape()`` call still produces a single
    async iterator regardless of how many underlying HTTP calls we made.

    API reference: https://api.gdeltproject.org/api/v2/doc/doc
    """

    API_URL = _GDELT_DOC_API
    MAX_RECORDS_PER_CALL = 250  # GDELT's hard cap.

    async def scrape(
        self,
        query: str,
        max_items: int = 500,
        start_date: datetime | None = None,
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
            url = self._build_url(query, page_size, end_dt, start_date)

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

    def _build_url(
        self,
        query: str,
        page_size: int,
        end_dt: datetime,
        start_date: datetime | None,
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

    The default feed list covers Reuters, BBC, AP, NPR, and the
    Guardian — between them we get broad coverage with stable feeds
    that don't change shape monthly. Per-feed metadata (outlet display
    name and canonical domain) is stamped on each item so downstream
    code can score credibility without re-parsing the URL.
    """

    DEFAULT_FEEDS: dict[str, str] = {
        "reuters_world": "https://feeds.reuters.com/reuters/worldNews",
        "bbc_news": "http://feeds.bbci.co.uk/news/rss.xml",
        "ap_top": "https://feeds.apnews.com/rss/apf-topnews",
        "npr_news": "https://feeds.npr.org/1001/rss.xml",
        "guardian_world": "https://www.theguardian.com/world/rss",
    }

    DEFAULT_OUTLETS: dict[str, tuple[str, str]] = {
        "reuters_world": ("Reuters", "reuters.com"),
        "bbc_news": ("BBC", "bbc.co.uk"),
        "ap_top": ("Associated Press", "apnews.com"),
        "npr_news": ("NPR", "npr.org"),
        "guardian_world": ("The Guardian", "theguardian.com"),
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

            for entry in feed.entries[:max_per_feed]:
                item = self._entry_to_item(
                    entry,
                    feed_key=key,
                    outlet_name=outlet_name,
                    source_domain=source_domain,
                    feed_url=feed_url,
                )
                if item is None:
                    continue
                self.stats.items_yielded += 1
                yield item

    def _entry_to_item(
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
        return ScrapedItem(
            source_url=url,
            image_url=_extract_image_url(entry),
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

        # Spread the remaining budget across queries so a noisy query
        # can't starve the others.
        remaining = max_items - emitted
        per_query = max(1, remaining // len(gdelt_queries))
        for query in gdelt_queries:
            try:
                async for item in self.gdelt_scraper.scrape(
                    query=query, max_items=per_query
                ):
                    yield item
                    emitted += 1
                    if emitted >= max_items:
                        return
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "GDELTScraper failed for query %r: %s", query, exc
                )

    async def close(self) -> None:
        await self.rss_scraper.close()
        await self.gdelt_scraper.close()

    async def __aenter__(self) -> "NewsScraperRegistry":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()
