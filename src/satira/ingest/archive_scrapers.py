"""Paginated *archive* scrapers — deep history beyond the RSS recency window.

RSS feeds only expose the most recent ~10-25 items per outlet, which
caps how much satire the Tier 1 builder can collect in a pass. An
outlet's *archive* (sitemap or paginated listing) holds thousands of
historical articles. This module walks those archives, page by page,
with the same politeness guarantees as the RSS scrapers plus three
things archives specifically need: resumability (a multi-thousand
article walk will be interrupted), per-page image extraction, and an
**AI-scraping opt-out guard**.

Design: source-agnostic strategies, not publisher-bound classes
--------------------------------------------------------------------
The three concrete scrapers here are keyed to an archive *shape*, not
to a particular publisher:

* :class:`SitemapIndexArchiveScraper` — entry point is a
  ``<sitemapindex>`` of sub-sitemaps; each sub-sitemap is one "page".
* :class:`FlatSitemapArchiveScraper` — entry point is a single
  ``<urlset>``; pages are fixed-size slices of its URLs.
* :class:`PaginatedArchiveScraper` — no sitemap; walk HTML listing
  pages at a ``/page/{n}/``-style template and scrape their links.

Each is configured at construction with the source's URL, domain, and
label, so the same code serves any outlet you have permission to
archive. :class:`ArchiveScraperRegistry` runs a configured set
concurrently; because each scraper instance owns its own rate limiter
and HTTP client, their request pacing is isolated per domain.

Why no Onion / Babylon Bee / Reductress classes
------------------------------------------------
Reconnaissance against the three outlets the archive task originally
named found that scraping them is both partly infeasible and against
their expressed wishes, so none is wired in by default:

* **The Onion** — ``robots.txt`` ``Disallow: /`` for every AI/LLM
  crawler *and* generic scraping frameworks (``GPTBot``, ``ClaudeBot``,
  ``CCBot``, ``anthropic-ai``, ``Scrapy``, ``news-please``, …). Its real
  sitemap is ``/sitemap_index.xml`` (``/sitemap.xml`` 404s).
* **Reductress** — ``robots.txt`` carries ``Content-Signal:
  ai-train=no`` and its sitemaps are Cloudflare-403'd.
* **Babylon Bee** — serves no ``robots.txt`` and no sitemap at any
  standard path (custom JS app).

:func:`detect_ai_optout` encodes that policy: with ``respect_optout``
left at its default ``True``, an :class:`ArchiveScraper` fetches the
source's ``robots.txt`` before doing any real work and *refuses to
scrape* a source that blocks AI crawlers or declares ``ai-train=no``.
The guard is what lets this framework exist without re-introducing the
behaviour the project decided against; override it only with explicit
authorization for a source.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from abc import abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from satira.ingest.base_scraper import BaseScraper, ScrapedItem
from satira.ingest.domain_utils import normalize_domain


logger = logging.getLogger(__name__)


# --- opt-out detection -------------------------------------------------------
# Crawler user-agents whose presence in a ``Disallow: /`` rule we read as
# "this site does not want automated AI-training / bulk-ingest crawling."
# The list mixes LLM crawlers (GPTBot, ClaudeBot, …) with generic scraping
# frameworks (Scrapy, news-please) because a site blocking those is making
# the same statement. Lower-cased for case-insensitive matching.
_AI_CRAWLER_AGENTS: frozenset[str] = frozenset(
    {
        "gptbot",
        "chatgpt-user",
        "oai-searchbot",
        "ccbot",
        "anthropic-ai",
        "claudebot",
        "claude-web",
        "google-extended",
        "googleother",
        "applebot-extended",
        "perplexitybot",
        "perplexity-ai",
        "bytespider",
        "amazonbot",
        "meta-externalagent",
        "meta-externalfetcher",
        "facebookbot",
        "diffbot",
        "cohere-ai",
        "omgili",
        "omgilibot",
        "imagesiftbot",
        "dataforseobot",
        "scrapy",
        "news-please",
        "magpie-crawler",
        "ai2bot",
        "timpibot",
        "webzio-extended",
    }
)


@dataclass
class AIOptOutVerdict:
    """Result of scanning a ``robots.txt`` for AI-scraping opt-out signals."""

    opted_out: bool
    reason: str = ""
    blocked_agents: list[str] = field(default_factory=list)
    content_signal_optout: bool = False


def detect_ai_optout(robots_text: str) -> AIOptOutVerdict:
    """Scan ``robots.txt`` text for signals that AI/bulk scraping is unwanted.

    Two signals are recognised:

    * Any user-agent in :data:`_AI_CRAWLER_AGENTS` carrying a
      ``Disallow: /`` rule — an explicit block of that crawler.
    * A Cloudflare ``Content-Signal`` line declaring ``ai-train=no``.

    Returns an :class:`AIOptOutVerdict`; ``opted_out`` is ``True`` if
    either signal is present. Parsing follows the robots grouping rule
    (consecutive ``User-agent`` lines share the rules that follow) but is
    deliberately lenient — we only need to know whether a blocking rule
    exists, not to evaluate fetch permission for a specific path.
    """
    blocked: list[str] = []
    content_signal_optout = False

    cur_agents: list[str] = []
    cur_has_disallow_root = False
    seen_rule = False

    def _flush() -> None:
        if cur_has_disallow_root:
            for agent in cur_agents:
                if agent in _AI_CRAWLER_AGENTS:
                    blocked.append(agent)

    for raw in robots_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip().lower()
        value = value.strip()

        if key == "user-agent":
            # A user-agent line after a rule line starts a fresh group.
            if seen_rule:
                _flush()
                cur_agents = []
                cur_has_disallow_root = False
                seen_rule = False
            cur_agents.append(value.lower())
        elif key == "disallow":
            seen_rule = True
            if value == "/":
                cur_has_disallow_root = True
        elif key == "allow":
            seen_rule = True
        elif key == "content-signal":
            if "ai-train=no" in value.lower().replace(" ", ""):
                content_signal_optout = True
    _flush()

    blocked = sorted(set(blocked))
    reasons: list[str] = []
    if blocked:
        reasons.append("robots.txt blocks AI crawler(s): " + ", ".join(blocked))
    if content_signal_optout:
        reasons.append("robots.txt Content-Signal declares ai-train=no")

    return AIOptOutVerdict(
        opted_out=bool(blocked) or content_signal_optout,
        reason="; ".join(reasons),
        blocked_agents=blocked,
        content_signal_optout=content_signal_optout,
    )


# --- HTML / sitemap parsing helpers ------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TIME_RE = re.compile(
    r"<time\b[^>]*?\bdatetime\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE
)
_IMG_SRC_RE = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE
)
_HREF_RE = re.compile(
    r"<a\b[^>]*?\bhref\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE
)


def _meta_pattern_pair(key: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Build the (property-first, content-first) ``<meta>`` regex pair for ``key``.

    ``og:`` / ``twitter:`` meta tags put ``property``/``name`` and
    ``content`` in either attribute order; matching both orderings beats
    one permissive pattern that would also span unrelated attributes.
    """
    k = re.escape(key)
    prop_first = re.compile(
        r"<meta\b[^>]*?\b(?:property|name)\s*=\s*[\"']" + k + r"[\"']"
        r"[^>]*?\bcontent\s*=\s*[\"']([^\"']*)[\"']",
        re.IGNORECASE,
    )
    content_first = re.compile(
        r"<meta\b[^>]*?\bcontent\s*=\s*[\"']([^\"']*)[\"']"
        r"[^>]*?\b(?:property|name)\s*=\s*[\"']" + k + r"[\"']",
        re.IGNORECASE,
    )
    return prop_first, content_first


_META_KEYS: tuple[str, ...] = (
    "og:image",
    "og:image:url",
    "og:image:secure_url",
    "twitter:image",
    "twitter:image:src",
    "og:title",
    "twitter:title",
    "og:description",
    "twitter:description",
    "description",
    "article:published_time",
    "article:modified_time",
    "og:updated_time",
)
_META_PATTERNS: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    key: _meta_pattern_pair(key) for key in _META_KEYS
}


def _meta(html: str, *keys: str) -> str | None:
    """Return the first non-empty ``<meta>`` content among ``keys``, HTML-unescaped."""
    for key in keys:
        pair = _META_PATTERNS.get(key)
        if pair is None:
            continue
        for pattern in pair:
            match = pattern.search(html)
            if match:
                value = unescape(match.group(1)).strip()
                if value:
                    return value
    return None


def _strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text or "").strip()


def _parse_iso_datetime(value: str) -> datetime | None:
    """Best-effort parse of an ISO-8601-ish timestamp to an aware UTC datetime."""
    raw = value.strip()
    if not raw:
        return None
    candidate = raw
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(raw[: len(fmt) + 2], fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _local_name(tag: str) -> str:
    """Strip the ``{namespace}`` prefix ElementTree prepends to tag names."""
    return tag.rsplit("}", 1)[-1]


def parse_sitemap_xml(data: bytes) -> tuple[str, list[str]]:
    """Parse sitemap XML bytes into ``(kind, locs)``.

    ``kind`` is ``"index"`` for a ``<sitemapindex>``, ``"urlset"`` for a
    ``<urlset>``, or ``"unknown"`` if neither root was recognised.
    ``locs`` is every ``<loc>`` value found, in document order. Parsing
    from bytes (not text) lets ElementTree honour the XML encoding
    declaration. A malformed document yields ``("unknown", [])`` rather
    than raising — one bad sub-sitemap shouldn't abort a whole archive.
    """
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        logger.warning("sitemap parse failed: %s", exc)
        return "unknown", []

    locs: list[str] = []
    for element in root.iter():
        if _local_name(element.tag).lower() == "loc" and element.text:
            loc = element.text.strip()
            if loc:
                locs.append(loc)

    root_name = _local_name(root.tag).lower()
    if root_name == "sitemapindex":
        return "index", locs
    if root_name == "urlset":
        return "urlset", locs
    return "unknown", locs


# --- resumable state ---------------------------------------------------------
@dataclass
class ArchiveScraperState:
    """Persisted progress for a single archive scraper.

    ``last_page`` is the highest *fully consumed* page; resumption starts
    at ``last_page + 1``. ``scraped_urls`` carries every article URL
    already seen so a partially-consumed page (interrupted by the
    ``max_articles`` budget) is recovered without re-yielding duplicates.
    """

    last_page: int = 0
    scraped_urls: list[str] = field(default_factory=list)
    last_run: str | None = None


_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


def _slug(name: str) -> str:
    return _SLUG_RE.sub("_", name.strip().lower()).strip("_") or "archive"


# --- base archive scraper ----------------------------------------------------
class ArchiveScraper(BaseScraper):
    """Base class for paginated archive scrapers.

    Subclasses implement one required method:

    * :meth:`_get_article_urls` — return the article URLs for a 1-based
      page number, or ``[]`` when the archive is exhausted.

    and may override :meth:`_parse_article` (a generic ``og:``/``<title>``
    parser is provided) and :meth:`_entry_url` (the URL whose
    ``robots.txt`` permission gates the run; defaults to the archive
    root).

    The base class drives everything else: the AI-opt-out guard, the
    page loop (until empty, ``max_pages``, or ``max_articles``), per-page
    rate-limited article fetches, image extraction with the
    og→twitter→first-``<img>`` priority order, the *skip-if-no-image*
    rule, in-run URL dedup, resumable state, and structured per-page
    logging.
    """

    def __init__(
        self,
        *,
        name: str,
        archive_root: str,
        source_domain: str,
        label: str = "satire",
        requests_per_second: float = 1.0,
        max_pages: int | None = None,
        state_dir: Path | str = Path("./data/scraper_state"),
        respect_optout: bool = True,
        **base_kwargs: Any,
    ) -> None:
        if requests_per_second <= 0:
            raise ValueError(
                f"requests_per_second must be positive, got {requests_per_second}"
            )
        # Archives are big and slow on purpose: 1 req/sec by default. Map
        # the friendlier req/sec knob onto the base class's per-minute one.
        base_kwargs.pop("rate_limit_per_minute", None)
        super().__init__(
            rate_limit_per_minute=max(1, round(requests_per_second * 60)),
            **base_kwargs,
        )
        self.name = name
        self.archive_root = archive_root.rstrip("/")
        self.source_domain = source_domain
        self.label = label
        self.max_pages = max_pages
        self.respect_optout = respect_optout
        self.state_dir = Path(state_dir)
        self._state_path = self.state_dir / f"{_slug(name)}.json"

    # --- subclass hooks -------------------------------------------------
    @abstractmethod
    async def _get_article_urls(self, page: int) -> list[str]:
        """Return article URLs for 1-based ``page``; ``[]`` means exhausted."""
        raise NotImplementedError

    def _entry_url(self) -> str:
        """URL whose robots.txt permission gates the run. Override if needed."""
        return self.archive_root

    async def _parse_article(self, html: str, url: str) -> ScrapedItem | None:
        """Generic article parser: ``og:`` meta with ``<title>`` fallback.

        Returns ``None`` when no usable title is found. The image is left
        unset here — :meth:`_build_item` fills it via :meth:`_extract_image`
        so the priority order and skip-if-missing rule live in one place.
        """
        title = _meta(html, "og:title", "twitter:title")
        if not title:
            match = _TITLE_RE.search(html)
            if match:
                title = unescape(_strip_tags(match.group(1)))
        if not title:
            return None

        text = _meta(html, "og:description", "twitter:description", "description") or ""
        return ScrapedItem(
            source_url=url,
            image_url=None,
            title=title,
            text=text,
            timestamp=self._parse_pub_date(html),
            source_domain=self.source_domain,
            metadata={
                "label": self.label,
                "outlet": self.name,
                "source_type": "archive",
            },
        )

    # --- image + date extraction ---------------------------------------
    def _extract_image(self, html: str, base_url: str) -> str | None:
        """Extract a hero image URL in priority order, or ``None``.

        Order: ``og:image`` → ``twitter:image`` → first ``<img>`` in the
        page body. Relative URLs are resolved against ``base_url``. Data
        URIs, SVGs, and obvious sprite/tracking pixels are skipped so the
        last-resort ``<img>`` fallback doesn't latch onto chrome.
        """
        og = _meta(html, "og:image", "og:image:url", "og:image:secure_url")
        if og:
            return urljoin(base_url, og)
        twitter = _meta(html, "twitter:image", "twitter:image:src")
        if twitter:
            return urljoin(base_url, twitter)
        for match in _IMG_SRC_RE.finditer(html):
            src = match.group(1).strip()
            if not src or src.startswith("data:"):
                continue
            low = src.lower()
            if low.endswith(".svg") or any(
                token in low for token in ("sprite", "pixel", "1x1", "blank", "spacer")
            ):
                continue
            return urljoin(base_url, src)
        return None

    def _parse_pub_date(self, html: str) -> datetime:
        """Best-effort publish timestamp, defaulting to now (UTC)."""
        raw = _meta(
            html, "article:published_time", "og:updated_time", "article:modified_time"
        )
        if not raw:
            match = _TIME_RE.search(html)
            if match:
                raw = match.group(1)
        if raw:
            parsed = _parse_iso_datetime(raw)
            if parsed is not None:
                return parsed
        return datetime.now(timezone.utc)

    async def _build_item(self, html: str, url: str) -> ScrapedItem | None:
        """Parse an article and guarantee an image, or return ``None`` to skip.

        Archive items are only valuable to Tier 1 when they carry an
        image (RSS already fills the text-only quota), so an article with
        no extractable image is dropped here.
        """
        item = await self._parse_article(html, url)
        if item is None:
            return None
        if not item.image_url:
            item.image_url = self._extract_image(html, url)
        if not item.image_url:
            logger.debug("%s: no image for %s — skipping", self.name, url)
            return None
        return item

    # --- opt-out guard --------------------------------------------------
    async def _fetch_robots_text(self) -> str | None:
        """Fetch the source's ``robots.txt`` for the opt-out scan.

        Bypasses the per-request robots gate (fetching robots.txt itself
        is always permitted) and goes straight through the rate-limited
        client so the opt-out check is still polite. Returns ``None`` if
        robots.txt is missing or unreachable.
        """
        parsed = urlparse(self.archive_root)
        if not parsed.scheme or not parsed.netloc:
            return None
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        client = self._get_client()
        await self._enforce_rate_limit()
        try:
            response = await client.get(robots_url)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            logger.debug("%s: could not fetch robots.txt (%s)", self.name, exc)
            return None
        if response.status_code >= 400:
            return None
        return response.text

    async def _is_opted_out(self) -> AIOptOutVerdict | None:
        """Return an opt-out verdict if the source declined AI scraping, else ``None``."""
        text = await self._fetch_robots_text()
        if not text:
            return None
        verdict = detect_ai_optout(text)
        return verdict if verdict.opted_out else None

    # --- state ----------------------------------------------------------
    def _load_state(self) -> ArchiveScraperState:
        if not self._state_path.exists():
            return ArchiveScraperState()
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "%s: could not read state %s (%s) — starting fresh",
                self.name, self._state_path, exc,
            )
            return ArchiveScraperState()
        return ArchiveScraperState(
            last_page=int(data.get("last_page", 0)),
            scraped_urls=list(data.get("scraped_urls", []) or []),
            last_run=data.get("last_run"),
        )

    def _save_state(self, state: ArchiveScraperState) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_page": state.last_page,
            "scraped_urls": state.scraped_urls,
            "last_run": state.last_run,
            "name": self.name,
        }
        # Write-then-rename so an interruption mid-write can't corrupt the
        # state file and lose the whole resume point.
        tmp = self._state_path.with_name(self._state_path.name + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self._state_path)

    # --- main loop ------------------------------------------------------
    async def scrape(
        self,
        *,
        max_articles: int = 2000,
        resume: bool = True,
        **_: Any,
    ) -> AsyncIterator[ScrapedItem]:
        if max_articles <= 0:
            return

        entry = self._entry_url()
        if self.respect_robots and entry and not await self._check_robots_txt(entry):
            logger.error(
                "%s: robots.txt disallows %s — skipping source", self.name, entry
            )
            return

        if self.respect_optout:
            verdict = await self._is_opted_out()
            if verdict is not None:
                logger.error(
                    "%s: source has opted out of AI-training scraping (%s) — "
                    "skipping. Construct with respect_optout=False only with "
                    "explicit authorization for this source.",
                    self.name, verdict.reason,
                )
                return

        state = self._load_state() if resume else ArchiveScraperState()
        seen: set[str] = set(state.scraped_urls)
        page = state.last_page + 1
        emitted = 0
        total_images = 0

        logger.info(
            "%s: starting archive scrape (max_articles=%d, resume_from_page=%d, "
            "already_seen=%d)",
            self.name, max_articles, page, len(seen),
        )

        while emitted < max_articles:
            if self.max_pages is not None and page > self.max_pages:
                logger.info("%s: reached max_pages=%d — stopping", self.name, self.max_pages)
                break

            started = time.monotonic()
            try:
                urls = await self._get_article_urls(page)
            except Exception as exc:  # noqa: BLE001 — one bad page mustn't kill the walk
                logger.exception("%s: failed to list page %d: %s", self.name, page, exc)
                break

            if not urls:
                logger.info("%s: page %d returned no URLs — archive drained", self.name, page)
                break

            page_new = 0
            page_images = 0
            fully_consumed = True
            for url in urls:
                if url in seen:
                    continue
                if emitted >= max_articles:
                    fully_consumed = False
                    break
                seen.add(url)
                page_new += 1
                html = await self.fetch(url)
                if not html:
                    continue
                item = await self._build_item(html, url)
                if item is None:
                    continue
                emitted += 1
                page_images += 1
                total_images += 1
                self.stats.items_yielded += 1
                yield item
                if emitted % 100 == 0:
                    logger.info(
                        "%s: running totals — emitted=%d images=%d (on page %d)",
                        self.name, emitted, total_images, page,
                    )

            duration = time.monotonic() - started
            logger.info(
                "%s: page=%d articles_found=%d new=%d images_extracted=%d duration=%.1fs",
                self.name, page, len(urls), page_new, page_images, duration,
            )

            if fully_consumed:
                state.last_page = page
            state.scraped_urls = sorted(seen)
            state.last_run = datetime.now(timezone.utc).isoformat()
            self._save_state(state)

            if not fully_consumed:
                # Stopped mid-page on the article budget: leave last_page
                # unadvanced so the resume re-lists this page and the seen
                # set skips the URLs already done.
                break
            page += 1

        logger.info(
            "%s: archive scrape finished — emitted=%d images=%d",
            self.name, emitted, total_images,
        )


# --- sitemap strategies ------------------------------------------------------
class _SitemapArchiveScraperBase(ArchiveScraper):
    """Shared sitemap fetch/parse logic for the sitemap-driven strategies."""

    def __init__(self, *, sitemap_url: str, **kwargs: Any) -> None:
        parsed = urlparse(sitemap_url)
        archive_root = f"{parsed.scheme}://{parsed.netloc}"
        kwargs.setdefault("archive_root", archive_root)
        super().__init__(**kwargs)
        self.sitemap_url = sitemap_url

    def _entry_url(self) -> str:
        return self.sitemap_url

    async def _fetch_sitemap(self, url: str) -> bytes | None:
        """Fetch a sitemap, transparently gunzipping ``.xml.gz`` payloads."""
        data = await self.fetch_bytes(url)
        if data is None:
            return None
        if url.lower().endswith(".gz") or data[:2] == b"\x1f\x8b":
            try:
                data = gzip.decompress(data)
            except (OSError, EOFError) as exc:
                logger.warning("%s: could not gunzip %s (%s)", self.name, url, exc)
                return None
        return data

    def _filter_article_urls(self, locs: list[str]) -> list[str]:
        """Keep same-origin, non-nested-sitemap URLs from a ``<loc>`` list."""
        out: list[str] = []
        for loc in locs:
            low = loc.lower()
            if not low.startswith(("http://", "https://")):
                continue
            if low.rstrip("/").endswith(".xml"):
                continue
            if self.source_domain:
                domain = normalize_domain(loc)
                if domain and domain != self.source_domain:
                    continue
            out.append(loc)
        return out


class SitemapIndexArchiveScraper(_SitemapArchiveScraperBase):
    """Archive driven by a ``<sitemapindex>`` of sub-sitemaps.

    The entry point lists sub-sitemaps; each sub-sitemap becomes one
    page. This is the common WordPress/Yoast shape (``sitemap_index.xml``
    → ``post-sitemap.xml``, ``post-sitemap2.xml``, …). Falls back to
    treating the entry as a flat ``<urlset>`` (chunked by ``page_size``)
    if it turns out not to be an index, so a mis-typed config degrades
    gracefully rather than yielding nothing.
    """

    def __init__(self, *, page_size: int = 500, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.page_size = page_size
        self._loaded = False
        self._subsitemaps: list[str] = []
        self._flat_urls: list[str] = []

    async def _ensure_index(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        data = await self._fetch_sitemap(self.sitemap_url)
        if data is None:
            logger.warning("%s: sitemap index %s unreachable", self.name, self.sitemap_url)
            return
        kind, locs = parse_sitemap_xml(data)
        if kind == "index":
            self._subsitemaps = locs
            logger.info("%s: sitemap index has %d sub-sitemaps", self.name, len(locs))
        else:
            self._flat_urls = self._filter_article_urls(locs)
            logger.info(
                "%s: entry was a flat urlset (%d urls) — chunking by %d",
                self.name, len(self._flat_urls), self.page_size,
            )

    async def _get_article_urls(self, page: int) -> list[str]:
        await self._ensure_index()
        if self._subsitemaps:
            index = page - 1
            if index < 0 or index >= len(self._subsitemaps):
                return []
            data = await self._fetch_sitemap(self._subsitemaps[index])
            if data is None:
                return []
            _, locs = parse_sitemap_xml(data)
            return self._filter_article_urls(locs)
        # Flat-urlset fallback.
        start = (page - 1) * self.page_size
        return self._flat_urls[start : start + self.page_size]


class FlatSitemapArchiveScraper(_SitemapArchiveScraperBase):
    """Archive driven by a single flat ``<urlset>`` sitemap.

    The whole sitemap is fetched once, filtered to same-origin article
    URLs, and sliced into fixed-size pages of ``page_size`` URLs each so
    the resumable page loop and ``max_articles`` budget still apply.
    """

    def __init__(self, *, page_size: int = 500, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.page_size = page_size
        self._loaded = False
        self._urls: list[str] = []

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        data = await self._fetch_sitemap(self.sitemap_url)
        if data is None:
            logger.warning("%s: sitemap %s unreachable", self.name, self.sitemap_url)
            return
        _, locs = parse_sitemap_xml(data)
        self._urls = self._filter_article_urls(locs)
        logger.info("%s: flat sitemap has %d article urls", self.name, len(self._urls))

    async def _get_article_urls(self, page: int) -> list[str]:
        await self._ensure_loaded()
        start = (page - 1) * self.page_size
        return self._urls[start : start + self.page_size]


# --- pagination strategy -----------------------------------------------------
class PaginatedArchiveScraper(ArchiveScraper):
    """Archive driven by HTML listing pages (``/page/{page}/`` style).

    For sources without a usable sitemap. ``listing_url_template`` must
    contain a ``{page}`` placeholder; each page's HTML is fetched and its
    same-origin anchors are collected as article URLs. An optional
    ``article_url_regex`` narrows those anchors to real article paths
    (e.g. a date or ``/article/`` segment) so navigation/category links
    don't slip in.
    """

    def __init__(
        self,
        *,
        listing_url_template: str,
        article_url_regex: str | None = None,
        **kwargs: Any,
    ) -> None:
        if "{page}" not in listing_url_template:
            raise ValueError(
                "listing_url_template must contain a '{page}' placeholder"
            )
        parsed = urlparse(listing_url_template)
        kwargs.setdefault("archive_root", f"{parsed.scheme}://{parsed.netloc}")
        super().__init__(**kwargs)
        self.listing_url_template = listing_url_template
        self.article_url_regex = (
            re.compile(article_url_regex) if article_url_regex else None
        )

    def _entry_url(self) -> str:
        return self.listing_url_template.format(page=1)

    async def _get_article_urls(self, page: int) -> list[str]:
        url = self.listing_url_template.format(page=page)
        html = await self.fetch(url)
        if not html:
            return []
        return self._extract_listing_links(html, url)

    def _extract_listing_links(self, html: str, base_url: str) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for match in _HREF_RE.finditer(html):
            href = match.group(1).strip()
            if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
                continue
            absolute = urljoin(base_url, href).split("#", 1)[0]
            low = absolute.lower()
            if not low.startswith(("http://", "https://")):
                continue
            if self.source_domain:
                domain = normalize_domain(absolute)
                if domain and domain != self.source_domain:
                    continue
            if self.article_url_regex and not self.article_url_regex.search(absolute):
                continue
            if absolute in seen:
                continue
            seen.add(absolute)
            out.append(absolute)
        return out


# --- registry ----------------------------------------------------------------
_SCRAPER_TYPES = {
    "sitemap_index": SitemapIndexArchiveScraper,
    "flat_sitemap": FlatSitemapArchiveScraper,
    "paginated": PaginatedArchiveScraper,
}


def build_scraper_from_config(config: dict[str, Any]) -> ArchiveScraper | None:
    """Construct one archive scraper from a config dict, or ``None`` if invalid.

    Recognised ``type`` values: ``sitemap_index``, ``flat_sitemap``,
    ``paginated``. Common keys: ``name`` (required), ``source_domain``,
    ``label`` (default ``"satire"``), ``requests_per_second`` (default
    1.0), ``max_pages``, ``respect_optout`` (default ``True``). Type
    keys: ``sitemap_url`` (+optional ``page_size``) for the sitemap
    strategies; ``listing_url_template`` (+optional ``article_url_regex``)
    for ``paginated``.
    """
    scraper_type = config.get("type")
    cls = _SCRAPER_TYPES.get(scraper_type or "")
    if cls is None:
        logger.warning(
            "archive config: unknown or missing type %r — skipping entry %r",
            scraper_type, config.get("name"),
        )
        return None
    if "name" not in config:
        logger.warning("archive config: entry missing 'name' — skipping: %r", config)
        return None

    common: dict[str, Any] = {
        "name": config["name"],
        "source_domain": normalize_domain(config.get("source_domain", "")) or "",
        "label": config.get("label", "satire"),
        "requests_per_second": config.get("requests_per_second", 1.0),
        "max_pages": config.get("max_pages"),
        "respect_optout": config.get("respect_optout", True),
    }
    if "state_dir" in config:
        common["state_dir"] = config["state_dir"]

    try:
        if scraper_type in ("sitemap_index", "flat_sitemap"):
            return cls(
                sitemap_url=config["sitemap_url"],
                page_size=config.get("page_size", 500),
                **common,
            )
        # paginated
        return cls(
            listing_url_template=config["listing_url_template"],
            article_url_regex=config.get("article_url_regex"),
            **common,
        )
    except (KeyError, ValueError) as exc:
        logger.warning(
            "archive config: could not build %r (%s) — skipping", config.get("name"), exc
        )
        return None


class ArchiveScraperRegistry:
    """Runs a configured set of archive scrapers concurrently.

    Each scraper owns its own rate limiter and HTTP client, so running
    them under one :func:`asyncio.gather`-style fan-out keeps request
    pacing isolated *per domain* — a slow or rate-limited source can't
    stall the others, and none exceeds its own 1-req/sec budget.

    There is no built-in source list: per the project's opt-out policy
    (see the module docstring) nothing is scraped unless explicitly
    configured. Build one with :meth:`from_config` / :func:`build_scraper_from_config`.
    """

    def __init__(self, scrapers: list[ArchiveScraper] | None = None) -> None:
        self.scrapers: list[ArchiveScraper] = list(scrapers) if scrapers else []

    @classmethod
    def from_config(cls, configs: list[dict[str, Any]]) -> "ArchiveScraperRegistry":
        built = [build_scraper_from_config(c) for c in configs]
        return cls([s for s in built if s is not None])

    async def scrape_all(
        self,
        *,
        max_articles_per_source: int = 2000,
        resume: bool = True,
    ) -> AsyncIterator[ScrapedItem]:
        """Yield items from every configured scraper, running them concurrently."""
        if not self.scrapers:
            logger.warning(
                "ArchiveScraperRegistry: no sources configured — nothing to "
                "scrape. Supply permitted sources via --archive-config or "
                "ArchiveScraperRegistry(scrapers=[...]). The originally-planned "
                "outlets (The Onion / Babylon Bee / Reductress) were excluded "
                "for AI-scraping opt-out and feasibility reasons."
            )
            return

        queue: asyncio.Queue[ScrapedItem | object] = asyncio.Queue()
        sentinel = object()

        async def _drain(scraper: ArchiveScraper) -> None:
            try:
                async for item in scraper.scrape(
                    max_articles=max_articles_per_source, resume=resume
                ):
                    await queue.put(item)
            except Exception as exc:  # noqa: BLE001 — registry tolerates source failures
                logger.exception(
                    "archive scraper %s failed mid-run: %s",
                    getattr(scraper, "name", type(scraper).__name__), exc,
                )
            finally:
                await queue.put(sentinel)

        tasks = [asyncio.create_task(_drain(s)) for s in self.scrapers]
        finished = 0
        try:
            while finished < len(self.scrapers):
                item = await queue.get()
                if item is sentinel:
                    finished += 1
                    continue
                yield item  # type: ignore[misc]
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        for scraper in self.scrapers:
            await scraper.close()

    async def __aenter__(self) -> "ArchiveScraperRegistry":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()
