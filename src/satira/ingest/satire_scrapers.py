"""Concrete RSS scrapers for known satire outlets.

Satire outlets publish full RSS feeds, so we get titles, summaries,
links, dates, and (usually) a featured image without ever touching the
article HTML. That's both polite — fewer requests to each origin — and
much more reliable than scraping article pages whose markup changes
without notice.

All three scrapers share the same shape: fetch a feed URL, hand the
body to ``feedparser``, and yield one :class:`ScrapedItem` per entry
with ``label="satire"`` stamped into ``metadata``. The shared logic
lives on :class:`_RSSSatireScraper`; concrete classes only declare the
feed URL, outlet name, and source domain.

The base scraper handles rate limiting, retries, and robots.txt for us
— if a feed is disallowed, ``fetch`` returns ``None`` and we yield
nothing for that source rather than raising.
"""
from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import feedparser

from satira.ingest.base_scraper import BaseScraper, ScrapedItem


logger = logging.getLogger(__name__)


_IMG_TAG_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _extract_image_url(entry: Any) -> str | None:
    """Pull a featured image URL from an RSS entry, trying common locations.

    RSS feeds carry images in several places depending on the publisher:
    ``<media:thumbnail>``, ``<media:content>``, ``<enclosure>``, or
    embedded ``<img>`` tags inside the description. We try them in
    rough order of how likely they are to point at the canonical hero
    image.
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
    """Best-effort timestamp from an RSS entry, defaulting to "now" in UTC.

    feedparser normalises ``pubDate`` / ``updated`` into UTC struct_time
    tuples in ``*_parsed`` attributes — we use those when present and
    fall back to a current-time UTC datetime so downstream code never
    has to deal with a missing timestamp.
    """
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


class _RSSSatireScraper(BaseScraper):
    """Shared RSS-feed scraping logic for satire outlets.

    Subclasses set ``feed_url``, ``outlet_name``, and
    ``source_domain``. Everything else — fetching, parsing, image and
    timestamp extraction, error handling — lives here so each concrete
    scraper stays a tiny declaration.
    """

    feed_url: str = ""
    outlet_name: str = ""
    source_domain: str = ""

    async def scrape(self, max_items: int = 100, **_: Any) -> AsyncIterator[ScrapedItem]:
        if not self.feed_url:
            raise ValueError(
                f"{type(self).__name__} must set a non-empty feed_url"
            )

        body = await self.fetch(self.feed_url)
        if body is None:
            logger.warning(
                "%s: feed fetch returned no body (blocked or all retries failed)",
                self.outlet_name or type(self).__name__,
            )
            return

        feed = feedparser.parse(body)
        if feed.bozo and not feed.entries:
            # Malformed feed AND no entries recovered — nothing to yield.
            # If feedparser recovered entries despite ``bozo``, we still
            # want them, so don't bail in that case.
            logger.warning(
                "%s: feed parse failed: %s",
                self.outlet_name or type(self).__name__,
                getattr(feed, "bozo_exception", "unknown"),
            )
            return

        for entry in feed.entries[:max_items]:
            item = self._entry_to_item(entry)
            if item is None:
                continue
            self.stats.items_yielded += 1
            yield item

    def _entry_to_item(self, entry: Any) -> ScrapedItem | None:
        url = getattr(entry, "link", "") or ""
        title = (getattr(entry, "title", "") or "").strip()

        # An entry with neither URL nor title is unusable — there is
        # nothing to dedupe against and no content for the model.
        if not url and not title:
            return None

        summary_raw = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
        text = _strip_html(summary_raw)

        return ScrapedItem(
            source_url=url,
            image_url=_extract_image_url(entry),
            title=title,
            text=text,
            timestamp=_extract_timestamp(entry),
            source_domain=self.source_domain,
            metadata={
                "label": "satire",
                "outlet": self.outlet_name,
                "feed_url": self.feed_url,
            },
        )


class TheOnionScraper(_RSSSatireScraper):
    """Scrapes The Onion's RSS feed (legal, public).

    Feed URL: https://www.theonion.com/rss
    """

    feed_url = "https://www.theonion.com/rss"
    outlet_name = "The Onion"
    source_domain = "theonion.com"


class BabylonBeeScraper(_RSSSatireScraper):
    """Scrapes Babylon Bee's RSS feed.

    Feed URL: https://babylonbee.com/feed
    """

    feed_url = "https://babylonbee.com/feed"
    outlet_name = "Babylon Bee"
    source_domain = "babylonbee.com"


class ReductressScraper(_RSSSatireScraper):
    """Scrapes Reductress RSS feed.

    Feed URL: https://reductress.com/feed/
    """

    feed_url = "https://reductress.com/feed/"
    outlet_name = "Reductress"
    source_domain = "reductress.com"


class SatireScraperRegistry:
    """Combines all known satire scrapers behind a single iterator.

    A registry beats a hand-coded ``async for`` over a list at call
    sites because (a) it owns the close logic, so callers don't leak
    HTTP clients when one scraper raises, and (b) it gives us a single
    place to add or feature-flag new outlets later without touching
    every caller.
    """

    def __init__(self, scrapers: list[BaseScraper] | None = None) -> None:
        if scrapers is None:
            scrapers = [
                TheOnionScraper(),
                BabylonBeeScraper(),
                ReductressScraper(),
            ]
        self.scrapers: list[BaseScraper] = scrapers

    async def scrape_all(
        self, max_items_per_source: int = 100
    ) -> AsyncIterator[ScrapedItem]:
        """Yield items from every registered satire source.

        A failure inside one scraper (network, parse, etc.) is logged
        and skipped rather than aborting the whole run — losing one
        outlet's batch is much better than losing all three.
        """
        for scraper in self.scrapers:
            try:
                async for item in scraper.scrape(max_items=max_items_per_source):
                    yield item
            except Exception as exc:  # noqa: BLE001 — registry must be tolerant
                logger.exception(
                    "scraper %s failed mid-run: %s", type(scraper).__name__, exc
                )

    async def close(self) -> None:
        for scraper in self.scrapers:
            await scraper.close()

    async def __aenter__(self) -> "SatireScraperRegistry":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()
