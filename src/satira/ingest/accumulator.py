"""Daily-accumulating, deduplicated corpus built from the RSS scrapers.

RSS only ever exposes a feed's most recent ~10-25 items, so a single
scrape captures a thin slice of any outlet. The way to build *volume*
from feeds is to scrape every day and keep what's new — the recency
window slides forward, and over weeks the accumulated tail grows far
past anything one fetch could return.

:class:`CorpusAccumulator` is that "scrape daily, keep the new ones"
loop made persistent and idempotent:

* **Append-only.** Each run appends genuinely-new items to
  ``manifest.jsonl``; nothing already in the corpus is rewritten or
  reordered. The manifest is the corpus — losing the in-memory state
  between daily runs is fine because every run rebuilds its dedup
  indexes from the manifest on disk.
* **Robust dedup.** A scraped item is a duplicate if it collides on
  *any* of three axes against the existing corpus (and against other
  items in the same run): exact URL, a fuzzy title match (rapidfuzz
  ratio >= 95), or a near-identical image (perceptual-hash Hamming
  distance < 4). URL/title dedup is cheap and runs *before* image
  download so we don't refetch the hero image of an article we already
  have; phash dedup runs after, to catch the same photo syndicated
  under a fresh URL and headline.
* **Permanent images.** Images land in ``images/`` under the corpus
  directory via :class:`ImageDownloader`'s content-addressable store
  (filename = SHA-256 of the bytes), so an image is written once and
  never re-downloaded, even across runs.
* **Idempotent.** Running twice in one day adds nothing the second
  time — every item the second run scrapes already matches a manifest
  entry on URL, so dedup rejects all of them.

Items whose image download fails are still kept, as *text-only*
records (``image_path=None``). That is deliberate: a labelled headline
is useful training signal on its own, and — more importantly for an
accumulator — it means the item enters the corpus once and stops being
re-scraped and re-downloaded on every subsequent daily run. A transient
fetch failure shouldn't doom an item to be retried forever.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import imagehash

try:
    from rapidfuzz import fuzz
except ImportError as exc:  # pragma: no cover — surfaced at import
    raise ImportError(
        "rapidfuzz is required for the corpus accumulator "
        "(install via `poetry add rapidfuzz`)"
    ) from exc

from satira.ingest.base_scraper import BaseScraper, ScrapedItem
from satira.ingest.image_pipeline import ImageDownloader, ProcessedItem
from satira.ingest.news_scrapers import NewsScraperRegistry
from satira.ingest.satire_scrapers import SatireScraperRegistry


logger = logging.getLogger(__name__)


# Dedup thresholds. These mirror the Tier 1 builder so an item judged a
# duplicate here would also be judged one there, keeping the corpus and
# the curated splits consistent.
_TITLE_FUZZ_THRESHOLD = 95  # rapidfuzz ratio (0-100); >= this is a dup.
_PHASH_HAMMING_THRESHOLD = 4  # Hamming distance strictly < this is a dup.

# Image download concurrency. The semaphore bounds in-flight fetches;
# there's no outer batching because the daily candidate set is small
# (RSS exposes only a couple dozen items per feed).
_MAX_CONCURRENT_DOWNLOADS = 10

_MANIFEST_NAME = "manifest.jsonl"
_IMAGES_DIRNAME = "images"

# How far back get_corpus_stats() reports the per-day addition counts.
_STATS_WINDOW_DAYS = 30


# --- normalization helpers --------------------------------------------------
# Kept local (and identical to scripts/build_tier1_dataset.py) so the
# library doesn't depend on a script module; the logic is small enough
# that a shared copy is cheaper than a shared import.
def _norm_url(url: str | None) -> str:
    if not url:
        return ""
    return url.strip().lower().rstrip("/")


def _norm_title(title: str | None) -> str:
    if not title:
        return ""
    return re.sub(r"\s+", " ", title.strip().lower())


def _derive_license(item: ScrapedItem) -> str:
    """Record the *usage basis* under which an item was collected.

    This is provenance, not a verified SPDX license: it states how each
    scraped record may be used, given that we only ever store a public
    feed's title, a short summary excerpt, the article link, and a
    featured-image reference.

    * ``gdelt-doc-metadata`` — GDELT DOC API records (title + link +
      social image only; the article body is never stored).
    * ``rss-feed-excerpt`` — anything pulled from a public RSS feed
      (every satire outlet, plus the RSS news feeds): a headline and a
      summary excerpt syndicated by the publisher for exactly this kind
      of consumption.
    * ``unknown`` — provenance we can't attribute from metadata.
    """
    meta = item.metadata or {}
    source_type = meta.get("source_type")
    if source_type == "gdelt":
        return "gdelt-doc-metadata"
    if source_type == "rss" or meta.get("feed_url"):
        return "rss-feed-excerpt"
    return "unknown"


def _as_text_only(item: ScrapedItem) -> ProcessedItem:
    """Wrap a scraped item as a text-only :class:`ProcessedItem`.

    Used both for items the scraper found no image for and for items
    whose image download failed — in both cases ``image_path`` stays
    ``None`` and downstream stages branch on that.
    """
    return ProcessedItem(
        source_url=item.source_url,
        image_url=item.image_url,
        title=item.title,
        text=item.text,
        timestamp=item.timestamp,
        source_domain=item.source_domain,
        metadata=dict(item.metadata),
    )


def _parse_dt(value: Any) -> datetime | None:
    """Best-effort ISO-8601 parse, returning ``None`` on anything unusable."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


# --- report -----------------------------------------------------------------
@dataclass
class CollectionReport:
    """Outcome of a single :meth:`CorpusAccumulator.run_daily_collection`.

    The headline invariant is ``items_scraped == items_new +
    items_duplicate``: every item the scrapers yielded was either added
    to the corpus or rejected as a duplicate of something already in it
    (or of another item in the same run). ``by_source`` / ``by_label``
    count only the genuinely-new items.
    """

    run_date: datetime
    items_scraped: int
    items_new: int
    items_duplicate: int
    images_downloaded: int
    by_source: dict[str, int] = field(default_factory=dict)
    by_label: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view, used for the per-run collection log."""
        return {
            "run_date": self.run_date.isoformat(),
            "items_scraped": self.items_scraped,
            "items_new": self.items_new,
            "items_duplicate": self.items_duplicate,
            "images_downloaded": self.images_downloaded,
            "by_source": dict(self.by_source),
            "by_label": dict(self.by_label),
            "errors": list(self.errors),
        }


# --- accumulator ------------------------------------------------------------
class CorpusAccumulator:
    """Maintains a persistent, deduplicated corpus that grows each day.

    Construct with the corpus directory; call
    :meth:`run_daily_collection` once per day (e.g. from Windows Task
    Scheduler via ``scripts/daily_collect.py``).

    Scrapers and the image downloader can be injected for testing — pass
    pre-wired registries / an ``image_scraper`` backed by an httpx
    ``MockTransport`` to exercise the whole pipeline with no network.
    Injected registries are owned by the caller and are *not* closed
    here; the defaults are built and closed per run.
    """

    def __init__(
        self,
        corpus_dir: str = "./data/corpus",
        *,
        satire_registry: SatireScraperRegistry | None = None,
        news_registry: NewsScraperRegistry | None = None,
        image_scraper: BaseScraper | None = None,
        gdelt_queries: Sequence[str] | None = None,
        max_satire_per_source: int = 100,
        news_max_items: int = 1000,
        gdelt_timespan: str = "1d",
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.images_dir = self.corpus_dir / _IMAGES_DIRNAME
        self.manifest_path = self.corpus_dir / _MANIFEST_NAME
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._satire_registry = satire_registry
        self._news_registry = news_registry
        self._image_scraper = image_scraper
        # ``None`` means RSS-only news (NewsScraperRegistry treats an
        # empty query list as "skip GDELT"); GDELT is opt-in because its
        # API is comparatively flaky and the accumulator's whole premise
        # is reliable daily RSS growth.
        self.gdelt_queries = list(gdelt_queries) if gdelt_queries else None
        self.max_satire_per_source = max_satire_per_source
        self.news_max_items = news_max_items
        self.gdelt_timespan = gdelt_timespan

    # --- public API -----------------------------------------------------
    async def run_daily_collection(self) -> CollectionReport:
        """Scrape every feed, dedup against the corpus, append the new items.

        Steps: scrape satire + news → cheap URL/title dedup → download
        images for survivors → perceptual-hash dedup → append the
        genuinely-new items to the manifest → return a
        :class:`CollectionReport`.
        """
        run_date = datetime.now(timezone.utc)
        errors: list[str] = []

        scraped, scrape_errors = await self._scrape_all()
        errors.extend(scrape_errors)
        items_scraped = len(scraped)

        # Build dedup indexes from the corpus on disk. Doing this fresh
        # each run is what makes the accumulator stateless between runs.
        hashes, urls, titles = self._build_indexes(self.load_corpus())

        # Phase 1 — URL + title dedup, before any download. Reserve each
        # survivor's URL/title so two scraped items that collide *within
        # this run* also dedup against each other. Items with neither a
        # URL nor a title can't be deduped or used, so they're dropped.
        candidates: list[ScrapedItem] = []
        duplicates = 0
        for raw in scraped:
            probe = _as_text_only(raw)  # no image yet -> phash check is a no-op
            norm_url = _norm_url(raw.source_url)
            norm_title = _norm_title(raw.title)
            if not norm_url and not norm_title:
                duplicates += 1  # unusable; counts against scraped total
                continue
            if self._is_duplicate(probe, hashes, urls, titles):
                duplicates += 1
                continue
            self._register_url_title(norm_url, norm_title, urls, titles)
            candidates.append(raw)

        # Phase 2 — download images for the survivors.
        processed, dl_errors = await self._download(candidates)
        errors.extend(dl_errors)

        # Phase 3 — perceptual-hash dedup. URLs/titles are already
        # reserved, so only a phash collision can newly flag an item:
        # the same photo republished under a different URL and headline.
        new_items: list[ProcessedItem] = []
        for item in processed:
            if self._phash_duplicate(item, hashes):
                duplicates += 1
                continue
            if item.perceptual_hash:
                try:
                    hashes.append(imagehash.hex_to_hash(item.perceptual_hash))
                except ValueError:
                    pass
            new_items.append(item)

        self._append_records(new_items, run_date)

        # Count images that actually entered the corpus this run. A fetch
        # may have succeeded for an item later dropped by phash dedup, so
        # this is "new items stored with an image", not "fetches made".
        images_downloaded = sum(1 for p in new_items if p.image_path)
        by_source = Counter(p.source_domain or "<unknown>" for p in new_items)
        by_label = Counter(
            (p.metadata or {}).get("label", "<none>") for p in new_items
        )
        report = CollectionReport(
            run_date=run_date,
            items_scraped=items_scraped,
            items_new=len(new_items),
            items_duplicate=duplicates,
            images_downloaded=images_downloaded,
            by_source=dict(by_source),
            by_label=dict(by_label),
            errors=errors,
        )
        logger.info(
            "daily collection: scraped=%d new=%d duplicate=%d images=%d errors=%d",
            report.items_scraped,
            report.items_new,
            report.items_duplicate,
            report.images_downloaded,
            len(report.errors),
        )
        return report

    def load_corpus(self) -> list[dict[str, Any]]:
        """Load every accumulated item from the manifest.

        Returns an empty list if the corpus hasn't been created yet.
        Malformed manifest lines are skipped with a warning rather than
        aborting the load — one bad line shouldn't hide the whole
        corpus.
        """
        if not self.manifest_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "skipping malformed manifest line %d: %s", lineno, exc
                    )
        return records

    def get_corpus_stats(self) -> dict[str, Any]:
        """Summarize the corpus: totals, label/source mix, dates, recency.

        Includes total items, counts by label and by source, the article
        date range, the collection date range, items added per day over
        the last :data:`_STATS_WINDOW_DAYS` days, and the text-only
        ratio. All counts are computed from the manifest on disk.
        """
        corpus = self.load_corpus()
        total = len(corpus)
        if total == 0:
            return {
                "total_items": 0,
                "by_label": {},
                "by_source": {},
                "unique_sources": 0,
                "date_range": None,
                "collection_range": None,
                "added_per_day": {},
                "text_only": 0,
                "text_only_ratio": 0.0,
            }

        by_label = Counter(rec.get("label") or "<none>" for rec in corpus)
        by_source = Counter(rec.get("source") or "<unknown>" for rec in corpus)

        article_dts = [d for d in (_parse_dt(r.get("timestamp")) for r in corpus) if d]
        collection_dts = [
            d for d in (_parse_dt(r.get("collection_date")) for r in corpus) if d
        ]

        text_only = sum(1 for rec in corpus if not rec.get("image_path"))

        # Items added per day over the recency window, keyed by ISO date,
        # most-recent first. Days with no additions are omitted.
        cutoff = datetime.now(timezone.utc) - timedelta(days=_STATS_WINDOW_DAYS)
        per_day: Counter[str] = Counter()
        for d in collection_dts:
            dd = d if d.tzinfo else d.replace(tzinfo=timezone.utc)
            if dd >= cutoff:
                per_day[dd.date().isoformat()] += 1
        added_per_day = dict(sorted(per_day.items(), reverse=True))

        return {
            "total_items": total,
            "by_label": dict(by_label.most_common()),
            "by_source": dict(by_source.most_common()),
            "unique_sources": len(by_source),
            "date_range": self._date_range(article_dts),
            "collection_range": self._date_range(collection_dts),
            "added_per_day": added_per_day,
            "text_only": text_only,
            "text_only_ratio": text_only / total,
        }

    # --- dedup ----------------------------------------------------------
    def _is_duplicate(
        self,
        item: ProcessedItem,
        existing_hashes: list[imagehash.ImageHash],
        existing_urls: set[str],
        existing_titles: dict[str, list[str]],
    ) -> bool:
        """Whether ``item`` already exists, by URL, title, or image.

        Any one axis matching is enough. ``existing_titles`` is bucketed
        by the first two characters of the normalized title (the same
        cheap blocking the Tier 1 builder uses) so a fuzzy comparison
        only runs against plausibly-similar titles, not the whole
        corpus. The phash check is skipped when ``item`` has no
        perceptual hash (e.g. a not-yet-downloaded or text-only item).
        """
        url = _norm_url(item.source_url)
        if url and url in existing_urls:
            return True

        norm = _norm_title(item.title)
        if norm:
            for prev in existing_titles.get(norm[:2], ()):
                if fuzz.ratio(norm, prev) >= _TITLE_FUZZ_THRESHOLD:
                    return True

        return self._phash_duplicate(item, existing_hashes)

    @staticmethod
    def _phash_duplicate(
        item: ProcessedItem, existing_hashes: list[imagehash.ImageHash]
    ) -> bool:
        """Whether ``item``'s image is within the Hamming threshold of any kept one."""
        if not item.perceptual_hash:
            return False
        try:
            ih = imagehash.hex_to_hash(item.perceptual_hash)
        except ValueError:
            logger.warning(
                "invalid perceptual_hash %r — treating as non-duplicate",
                item.perceptual_hash,
            )
            return False
        return any((ih - prev) < _PHASH_HAMMING_THRESHOLD for prev in existing_hashes)

    @staticmethod
    def _register_url_title(
        norm_url: str,
        norm_title: str,
        existing_urls: set[str],
        existing_titles: dict[str, list[str]],
    ) -> None:
        """Add an accepted item's URL/title keys to the live dedup indexes."""
        if norm_url:
            existing_urls.add(norm_url)
        if norm_title:
            existing_titles[norm_title[:2]].append(norm_title)

    def _build_indexes(
        self, corpus: Iterable[dict[str, Any]]
    ) -> tuple[list[imagehash.ImageHash], set[str], dict[str, list[str]]]:
        """Build (phashes, urls, bucketed titles) indexes from manifest records."""
        hashes: list[imagehash.ImageHash] = []
        urls: set[str] = set()
        titles: dict[str, list[str]] = defaultdict(list)
        for rec in corpus:
            url = _norm_url(rec.get("source_url"))
            if url:
                urls.add(url)
            title = _norm_title(rec.get("title") or rec.get("text"))
            if title:
                titles[title[:2]].append(title)
            phash = rec.get("perceptual_hash")
            if phash:
                try:
                    hashes.append(imagehash.hex_to_hash(phash))
                except ValueError:
                    logger.warning("manifest has invalid perceptual_hash %r", phash)
        return hashes, urls, titles

    # --- scraping -------------------------------------------------------
    async def _scrape_all(self) -> tuple[list[ScrapedItem], list[str]]:
        """Run both registries, returning all scraped items plus any errors.

        Each registry already tolerates a single feed failing (it logs
        and continues); we additionally surface, in the returned error
        list, any sub-scraper that ended a run having only failed
        requests — those are the feeds worth flagging in the report.
        """
        items: list[ScrapedItem] = []
        errors: list[str] = []

        satire = self._satire_registry or SatireScraperRegistry()
        try:
            async for item in satire.scrape_all(
                max_items_per_source=self.max_satire_per_source
            ):
                items.append(item)
        except Exception as exc:  # noqa: BLE001 — one registry can't kill the run
            errors.append(f"satire registry failed: {exc!r}")
            logger.exception("satire registry failed mid-run")
        finally:
            errors.extend(self._scraper_stat_errors(satire.scrapers))
            if self._satire_registry is None:
                await satire.close()

        news = self._news_registry or NewsScraperRegistry()
        try:
            async for item in news.scrape_all(
                gdelt_queries=list(self.gdelt_queries) if self.gdelt_queries else None,
                max_items=self.news_max_items,
                gdelt_timespan=self.gdelt_timespan,
            ):
                items.append(item)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"news registry failed: {exc!r}")
            logger.exception("news registry failed mid-run")
        finally:
            errors.extend(
                self._scraper_stat_errors([news.rss_scraper, news.gdelt_scraper])
            )
            if self._news_registry is None:
                await news.close()

        return items, errors

    @staticmethod
    def _scraper_stat_errors(scrapers: Iterable[BaseScraper]) -> list[str]:
        """Flag scrapers that finished a run with failures and zero items.

        Best-effort, from each scraper's :class:`ScraperStats`. Note that
        ``RSSNewsScraper`` shares one stats object across all its feeds,
        so its message is an aggregate rather than per-feed.
        """
        errs: list[str] = []
        for scraper in scrapers:
            stats = getattr(scraper, "stats", None)
            if stats is None or stats.items_yielded > 0:
                continue
            name = getattr(scraper, "outlet_name", "") or type(scraper).__name__
            if stats.requests_failed:
                errs.append(
                    f"{name}: {stats.requests_failed} request failure(s), 0 items"
                )
            elif stats.robots_blocked:
                errs.append(f"{name}: blocked by robots.txt, 0 items")
        return errs

    # --- download -------------------------------------------------------
    async def _download(
        self, items: list[ScrapedItem]
    ) -> tuple[list[ProcessedItem], list[str]]:
        """Download images for ``items``, retaining failures as text-only.

        Returns ``(processed, errors)``. Every input produces exactly one
        :class:`ProcessedItem`: items with no image URL, and items whose
        download/validation failed, come back text-only
        (``image_path=None``); only successful downloads carry an image
        path and perceptual hash. Keeping failures means an item enters
        the corpus once instead of being retried every day.
        """
        with_image = [it for it in items if it.image_url]
        processed: list[ProcessedItem] = [
            _as_text_only(it) for it in items if not it.image_url
        ]
        errors: list[str] = []
        if not with_image:
            return processed, errors

        downloader = ImageDownloader(
            storage_path=str(self.images_dir), scraper=self._image_scraper
        )
        sem = asyncio.Semaphore(_MAX_CONCURRENT_DOWNLOADS)

        async def _one(it: ScrapedItem) -> tuple[ScrapedItem, ProcessedItem | None]:
            async with sem:
                try:
                    return it, await downloader.download(it)
                except Exception as exc:  # noqa: BLE001 — keep the batch alive
                    logger.exception("image download failed for %s: %s", it.image_url, exc)
                    return it, None

        try:
            pairs = await asyncio.gather(*(_one(it) for it in with_image))
        finally:
            # Only closes the HTTP client if we own it; an injected
            # image_scraper is the caller's to close.
            await downloader.close()

        for original, result in pairs:
            if result is not None:
                processed.append(result)
            else:
                # Validation/fetch failure: keep the headline, no image.
                processed.append(_as_text_only(original))
        return processed, errors

    # --- persistence ----------------------------------------------------
    def _append_records(
        self, items: list[ProcessedItem], collection_date: datetime
    ) -> None:
        """Append new items to the manifest as JSONL (append-only)."""
        if not items:
            return
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8") as fh:
            for item in items:
                record = self._to_record(item, collection_date)
                fh.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def _to_record(item: ProcessedItem, collection_date: datetime) -> dict[str, Any]:
        """Serialize a processed item into a manifest record."""
        return {
            "source_url": item.source_url,
            "image_url": item.image_url,
            "image_path": item.image_path,
            "title": item.title,
            "text": item.text,
            "label": (item.metadata or {}).get("label"),
            "source": item.source_domain,
            "license": _derive_license(item),
            "perceptual_hash": item.perceptual_hash,
            "image_dimensions": list(item.image_dimensions)
            if item.image_dimensions
            else None,
            "file_size_bytes": item.file_size_bytes,
            "timestamp": item.timestamp.isoformat() if item.timestamp else None,
            "collection_date": collection_date.isoformat(),
            "metadata": dict(item.metadata or {}),
        }

    # --- helpers --------------------------------------------------------
    @staticmethod
    def _date_range(dts: list[datetime]) -> dict[str, str] | None:
        if not dts:
            return None
        return {
            "earliest": min(dts).isoformat(),
            "latest": max(dts).isoformat(),
        }
