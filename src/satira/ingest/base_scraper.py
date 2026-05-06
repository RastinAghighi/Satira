"""Foundation for all Satira data scrapers.

Source-specific scrapers (news sites, social platforms, image boards) all
need the same boring infrastructure: a polite request rate, retry on
transient network errors, robots.txt compliance, and a uniform output
record so downstream ingest stages don't have to special-case each
source. ``BaseScraper`` collects that infrastructure in one place so each
concrete scraper can focus on parsing.

Rate limiting uses a simple monotonic-clock spacing algorithm rather
than a token bucket: scrapers usually run as a single async task per
source, so we just need to guarantee a minimum gap between requests.
The async lock around the spacing check makes that gap correct under
concurrency too (e.g. ``asyncio.gather`` over a list of URLs).

robots.txt results are cached per-host: a typical scrape session hits
the same domain hundreds or thousands of times, and refetching the
robots file every request would be both impolite and slow.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx


logger = logging.getLogger(__name__)


_MAX_RETRIES = 3
_BACKOFF_BASE = 0.5  # seconds; doubled each retry
_BACKOFF_CAP = 8.0   # cap so a long retry chain doesn't stall a scrape
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


@dataclass
class ScrapedItem:
    """Uniform output record produced by every scraper.

    ``metadata`` is the escape hatch for source-specific fields
    (subreddit, like count, author handle, …) so the common shape
    stays narrow.
    """

    source_url: str
    image_url: str | None
    title: str
    text: str
    timestamp: datetime
    source_domain: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScraperStats:
    requests_sent: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    retries: int = 0
    bytes_downloaded: int = 0
    robots_blocked: int = 0
    items_yielded: int = 0


class BaseScraper(ABC):
    """Abstract base for all data scrapers.

    Provides rate limiting, retry logic, robots.txt compliance, and
    structured output. Subclasses implement :meth:`scrape`.
    """

    def __init__(
        self,
        rate_limit_per_minute: int = 30,
        user_agent: str = "Satira-Ingest/1.0",
        respect_robots: bool = True,
        timeout: int = 30,
    ) -> None:
        if rate_limit_per_minute <= 0:
            raise ValueError(
                f"rate_limit_per_minute must be positive, got {rate_limit_per_minute}"
            )
        self.rate_limit_per_minute = rate_limit_per_minute
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self.timeout = timeout

        # Minimum spacing between requests, in seconds.
        self._min_interval = 60.0 / rate_limit_per_minute
        self._last_request_at: float = 0.0
        self._rate_lock = asyncio.Lock()

        self._robots_cache: dict[str, RobotFileParser | None] = {}
        self._robots_lock = asyncio.Lock()

        self.stats = ScraperStats()

        # Lazily created so subclasses constructed in sync code don't
        # need a running event loop.
        self._client: httpx.AsyncClient | None = None

    # --- public API -----------------------------------------------------
    async def fetch(self, url: str) -> str | None:
        """Fetch ``url`` as text with retry and rate limiting.

        Returns the response body on success, ``None`` if the request
        was blocked by robots.txt or if every retry failed.
        """
        response = await self._request(url)
        if response is None:
            return None
        return response.text

    async def fetch_image(self, url: str) -> bytes | None:
        """Fetch ``url`` as raw bytes, validating the content type.

        Returns ``None`` if the content-type is not ``image/*`` — saves
        a downstream image decoder from being handed an HTML error page
        dressed up with an ``.jpg`` extension.
        """
        response = await self._request(url)
        if response is None:
            return None
        ctype = response.headers.get("content-type", "").lower()
        if not ctype.startswith("image/"):
            logger.warning(
                "fetch_image got non-image content-type %r for %s", ctype, url
            )
            return None
        return response.content

    @abstractmethod
    async def scrape(self, **kwargs: Any) -> AsyncIterator[ScrapedItem]:
        """Yield :class:`ScrapedItem` records. Implemented by each scraper."""
        # The body here is unreachable — concrete subclasses override this
        # with an ``async def`` containing ``yield`` — but we still need a
        # ``yield`` in this stub so the function is recognised as an
        # async generator at the type level.
        if False:
            yield  # pragma: no cover

    async def close(self) -> None:
        """Release the underlying HTTP client. Safe to call repeatedly."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "BaseScraper":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    # --- request pipeline ----------------------------------------------
    async def _request(self, url: str) -> httpx.Response | None:
        if self.respect_robots and not await self._check_robots_txt(url):
            self.stats.robots_blocked += 1
            logger.info("robots.txt disallows %s — skipping", url)
            return None

        client = self._get_client()
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            await self._enforce_rate_limit()
            self.stats.requests_sent += 1
            try:
                response = await client.get(url)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                self.stats.requests_failed += 1
                logger.warning(
                    "request error on %s (attempt %d/%d): %s",
                    url, attempt + 1, _MAX_RETRIES + 1, exc,
                )
            else:
                if response.status_code < 400:
                    self.stats.requests_succeeded += 1
                    self.stats.bytes_downloaded += len(response.content)
                    return response
                if response.status_code in _RETRYABLE_STATUS:
                    self.stats.requests_failed += 1
                    logger.warning(
                        "retryable status %d on %s (attempt %d/%d)",
                        response.status_code, url, attempt + 1, _MAX_RETRIES + 1,
                    )
                else:
                    # 4xx that won't get better on retry — bail out.
                    self.stats.requests_failed += 1
                    logger.warning(
                        "non-retryable status %d on %s — giving up",
                        response.status_code, url,
                    )
                    return None

            if attempt < _MAX_RETRIES:
                self.stats.retries += 1
                await asyncio.sleep(self._backoff_delay(attempt))

        if last_exc is not None:
            logger.error("exhausted retries for %s: %s", url, last_exc)
        else:
            logger.error("exhausted retries for %s", url)
        return None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        # Exponential backoff with a small jitter so two scrapers
        # restarted at the same time don't fall into lockstep retries.
        base = min(_BACKOFF_CAP, _BACKOFF_BASE * (2 ** attempt))
        return base + random.uniform(0, base * 0.25)

    async def _enforce_rate_limit(self) -> None:
        async with self._rate_lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_request_at)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_at = time.monotonic()

    # --- robots.txt -----------------------------------------------------
    async def _check_robots_txt(self, url: str) -> bool:
        """Return whether ``url`` is allowed by its host's robots.txt.

        On any error fetching robots.txt we *allow* the request: a
        broken or missing robots file is treated as "no rules", which
        matches the behaviour of most well-behaved crawlers.
        """
        parsed = urlparse(url)
        if not parsed.netloc:
            return True
        host_key = f"{parsed.scheme}://{parsed.netloc}"

        async with self._robots_lock:
            if host_key not in self._robots_cache:
                self._robots_cache[host_key] = await self._load_robots(host_key)
            parser = self._robots_cache[host_key]

        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)

    async def _load_robots(self, host_key: str) -> RobotFileParser | None:
        robots_url = f"{host_key}/robots.txt"
        client = self._get_client()
        try:
            response = await client.get(robots_url)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            logger.info("could not fetch %s (%s) — assuming allow-all", robots_url, exc)
            return None
        if response.status_code >= 400:
            logger.info(
                "robots.txt at %s returned %d — assuming allow-all",
                robots_url, response.status_code,
            )
            return None
        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
        return parser
