"""Build the Tier 1 (easy-baseline) training dataset.

Tier 1 is the cleanest, most obvious slice of the curriculum:

* ~5000 authentic news items — image + caption from major outlets,
  pulled via :class:`NewsScraperRegistry` (RSS + GDELT restricted to
  curated domains).
* ~3000 obvious satire items — image + headline from known satire
  outlets, pulled via :class:`SatireScraperRegistry`.

The pipeline runs in this order:

1. Scrape the two registries up to their targets.
2. Split items by whether the scraper found an image URL: ones with a
   URL go through the download path; ones without are kept as
   text-only :class:`ProcessedItem` records (``image_path=None``) so a
   feed entry with no hero image still contributes a labelled headline
   to the dataset.
3. Download images via :class:`ImageDownloader` (validation + hashing).
4. Quality filters — drop items with text shorter than
   :data:`_MIN_TEXT_LEN`, truncate items longer than
   :data:`_MAX_TEXT_LEN`, drop non-English items detected by
   ``langdetect``, and drop items matching the NSFW keyword list.
5. Deduplicate. URL exact match → fuzzy title (>95% similar via
   ``rapidfuzz``) → perceptual hash (Hamming distance < 4) →
   cross-tier check against any pre-existing Tier 2 / Tier 3 splits so
   a Tier 1 item never collides with a harder-tier item.
6. Verify labels with :class:`SourceCredibilityClassifier` — drop only
   items where the classifier directly contradicts the asserted label;
   leniency is fine here because curated allowlists already keep the
   tier "obvious".
7. Cap text-only items per label at ``_MAX_TEXT_ONLY_FRAC``.
8. Source balance: cap any single source at
   :data:`_MAX_SOURCE_FRAC` of its label total. Stratified random
   sampling per (label, source) keeps the surviving subset diverse.
9. Stratified 80/10/10 split into train/val/test.
10. Write JSONL splits to ``--output-dir`` and print rich summary stats.

Run with ``--dry-run`` to see the targets and queries without making
any network calls.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:  # langdetect's detect() is non-deterministic by default
    from langdetect import DetectorFactory, LangDetectException, detect
    DetectorFactory.seed = 0
except ImportError as exc:  # pragma: no cover — surfaced at script entry
    raise ImportError(
        "langdetect is required for the Tier 1 build "
        "(install via `poetry add langdetect`)"
    ) from exc

try:
    from rapidfuzz import fuzz
except ImportError as exc:  # pragma: no cover — surfaced at script entry
    raise ImportError(
        "rapidfuzz is required for the Tier 1 build "
        "(install via `poetry add rapidfuzz`)"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satira.ingest import (  # noqa: E402
    ArchiveScraperRegistry,
    GDELTScraper,
    HFDatasetLoader,
    ImageDownloader,
    NewsScraperRegistry,
    ProcessedItem,
    SatireScraperRegistry,
    ScrapedItem,
    SourceCredibilityClassifier,
)
from satira.ingest.huggingface_loader import KNOWN_SATIRE_DATASETS  # noqa: E402
from satira.ingest.source_credibility import NEWS, SATIRE  # noqa: E402


logger = logging.getLogger("satira.build_tier1")


# Topic queries used when ``--use-gdelt`` is set. Twenty-five broad
# topics × 500 records per topic ≈ 12500 raw articles per pass before
# cross-query dedup. With the dual-pass timespan strategy
# (``1d`` + a wider window such as ``7d``) this comfortably exceeds
# any realistic Tier 1 news target once RSS contributions are added in.
DEFAULT_GDELT_QUERIES: tuple[str, ...] = GDELTScraper.DEFAULT_QUERIES

# Records to request per GDELT query per pass. GDELT caps a single API
# call at 250, so 500 forces the paginating cursor in
# :meth:`GDELTScraper.scrape` to make two calls per query — twice the
# unique articles per topic without raising the topic count further.
_GDELT_MAX_PER_QUERY = 500

# Cap on the user-supplied ``--gdelt-timespan`` value. GDELT will
# happily accept much wider windows but a 30-day cap keeps Tier 1's
# temporal mix recent enough that the model isn't training on stale
# news cycles, and bounds the worst-case API call count.
_GDELT_MAX_TIMESPAN_DAYS = 30
_GDELT_TIMESPAN_RE = re.compile(r"^\s*(\d+)\s*([hdw])\s*$", re.IGNORECASE)
_GDELT_TIMESPAN_HOURS: dict[str, int] = {"h": 1, "d": 24, "w": 24 * 7}

# Some feeds (NPR, Al Jazeera) don't carry images in their RSS, so the
# items they contribute are necessarily text-only. We keep them — a
# labelled headline is still training signal — but cap the share so
# they don't crowd out the multimodal items the model is actually being
# trained on.
_MAX_TEXT_ONLY_FRAC = 0.20

# Source diversity: cap any single source at this fraction of its label
# total so the dataset isn't dominated by whichever feed happens to be
# the most prolific that day (often one HuggingFace corpus on the
# satire side, one wire service on the news side).
_MAX_SOURCE_FRAC = 0.25

# Quality filters.
_MIN_TEXT_LEN = 50
_MAX_TEXT_LEN = 5000

# Title-fuzzy threshold: rapidfuzz returns 0–100, so 95 means ≥95%
# similar. Catches near-identical headlines syndicated across outlets
# (a wire-service story republished verbatim) without merging
# headlines that just share a topic.
_TITLE_FUZZ_THRESHOLD = 95

# Perceptual-hash threshold passed through to ImageDownloader.
_PHASH_HAMMING_THRESHOLD = 4

# Conservative NSFW keyword filter — token boundaries enforced via
# regex below. Intentionally narrow (clear-cut adult-content terms
# only); a model-based filter is the long-term plan but a keyword
# pass is enough at the Tier 1 "obvious" level.
_NSFW_KEYWORDS: tuple[str, ...] = (
    "porn", "porno", "xxx", "nsfw", "nude", "nudes", "naked",
    "boobs", "tits", "pussy", "blowjob", "anal", "cum", "cumshot",
    "hentai", "milf", "onlyfans", "camgirl", "deepthroat",
)
_NSFW_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _NSFW_KEYWORDS) + r")\b",
    flags=re.IGNORECASE,
)

LABEL_AUTHENTIC = 0
LABEL_SATIRE = 1
_STR_TO_LABEL = {"authentic": LABEL_AUTHENTIC, "satire": LABEL_SATIRE}
_LABEL_NAMES = {LABEL_AUTHENTIC: "authentic", LABEL_SATIRE: "satire"}

_DOWNLOAD_BATCH = 50
_MAX_CONCURRENT_DOWNLOADS = 10


def _gdelt_timespan(value: str) -> str:
    """argparse type-validator for ``--gdelt-timespan``.

    Accepts GDELT-style ``<N><unit>`` strings where unit is ``h``/``d``/``w``
    and the resulting window is ≤ :data:`_GDELT_MAX_TIMESPAN_DAYS` days.
    Returns the canonicalized lowercase string.
    """
    match = _GDELT_TIMESPAN_RE.match(value or "")
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid GDELT timespan {value!r}; expected forms like '1d', '7d', '12h', '2w'"
        )
    n = int(match.group(1))
    unit = match.group(2).lower()
    if n <= 0:
        raise argparse.ArgumentTypeError(
            f"GDELT timespan must be positive (got {value!r})"
        )
    hours = n * _GDELT_TIMESPAN_HOURS[unit]
    if hours > _GDELT_MAX_TIMESPAN_DAYS * 24:
        raise argparse.ArgumentTypeError(
            f"GDELT timespan {value!r} exceeds max of "
            f"{_GDELT_MAX_TIMESPAN_DAYS}d"
        )
    return f"{n}{unit}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Tier 1 (easy baseline) Satira dataset."
    )
    parser.add_argument(
        "--target-news",
        type=int,
        default=5000,
        help="Target number of authentic news items to collect (default: 5000).",
    )
    parser.add_argument(
        "--target-satire",
        type=int,
        default=3000,
        help="Target number of satire items to collect (default: 3000).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/tier1"),
        help="Directory to write {train,val,test}.jsonl into (default: ./data/tier1).",
    )
    parser.add_argument(
        "--image-storage",
        type=Path,
        default=Path("./data/images"),
        help="Directory to store downloaded images (default: ./data/images).",
    )
    parser.add_argument(
        "--use-gdelt",
        action="store_true",
        help=(
            "Augment the RSS feeds with GDELT topic-based scraping "
            "(25 topics, 500 records each, run for both timespan=1d "
            "and --gdelt-timespan)."
        ),
    )
    parser.add_argument(
        "--gdelt-timespan",
        type=_gdelt_timespan,
        default="7d",
        help=(
            "Wider GDELT window run alongside the default 1d pass to "
            "expand temporal coverage. Accepts GDELT-style strings "
            "like '7d', '12h', '2w'. Capped at 30d (default: 7d)."
        ),
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help=(
            "Currently a no-op: all curated HuggingFace satire datasets "
            "have been disabled because the only available corpus "
            "(Onion_News) is single-publisher and text-only, which "
            "skews Tier 1's V-L training. The flag is preserved so "
            "callers don't break; passing it logs a warning."
        ),
    )
    parser.add_argument(
        "--hf-max-per-dataset",
        type=int,
        default=3000,
        help="Max rows to pull from each HuggingFace dataset (default: 3000).",
    )
    parser.add_argument(
        "--use-archive",
        action="store_true",
        help=(
            "Enable the paginated archive scrapers (ArchiveScraperRegistry) "
            "for deep-history satire beyond the RSS recency window. Requires "
            "--archive-config naming *permitted* sources; archive scrapers "
            "refuse any source that opts out of AI scraping (robots.txt "
            "AI-crawler Disallow or Content-Signal: ai-train=no). See "
            "src/satira/ingest/archive_scrapers.py."
        ),
    )
    parser.add_argument(
        "--archive-config",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file listing permitted archive sources: a list "
            "of scraper config dicts (or {\"sources\": [...]}). Each needs "
            "'type' (sitemap_index|flat_sitemap|paginated), 'name', "
            "'source_domain', and a type-specific URL. Required for "
            "--use-archive to scrape anything."
        ),
    )
    parser.add_argument(
        "--archive-max-articles",
        type=int,
        default=2000,
        help="Max articles to pull per archive source (default: 2000).",
    )
    parser.add_argument(
        "--archive-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Resume each archive source from its saved state in "
            "./data/scraper_state (default: True). Use --no-archive-resume "
            "or --archive-fresh to start over."
        ),
    )
    parser.add_argument(
        "--archive-fresh",
        action="store_true",
        help=(
            "Ignore saved archive state and restart each source from page 1 "
            "(default: False). Overrides --archive-resume."
        ),
    )
    parser.add_argument(
        "--archive-dry-run",
        action="store_true",
        help=(
            "Archive smoke test: scrape only 50 articles per source to verify "
            "the scrapers work before committing to a full multi-hour run."
        ),
    )
    parser.add_argument(
        "--tier2-dir",
        type=Path,
        default=Path("./data/tier2"),
        help=(
            "Directory containing existing Tier 2 splits to exclude from "
            "Tier 1 (cross-tier dedup). Missing directory = no exclusion."
        ),
    )
    parser.add_argument(
        "--tier3-dir",
        type=Path,
        default=Path("./data/tier3"),
        help=(
            "Directory containing existing Tier 3 splits to exclude from "
            "Tier 1 (cross-tier dedup). Missing directory = no exclusion."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be scraped without making any network calls.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the train/val/test shuffle (default: 42).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# --- scraping ----------------------------------------------------------------
async def scrape_news(
    target: int,
    *,
    use_gdelt: bool,
    gdelt_timespan: str,
    dry_run: bool,
) -> list[ScrapedItem]:
    if dry_run:
        print(f"[dry-run] news: would scrape up to {target} items")
        print("[dry-run] news: RSS feeds (registry default):")
        for key in NewsScraperRegistry().rss_scraper.feeds:
            print(f"             - {key}")
        if use_gdelt:
            print("[dry-run] news: GDELT topic queries:")
            for q in DEFAULT_GDELT_QUERIES:
                print(f"             - {q}")
            timespans = ["1d"]
            if gdelt_timespan != "1d":
                timespans.append(gdelt_timespan)
            print(
                f"[dry-run] news: GDELT passes: {timespans}, "
                f"{_GDELT_MAX_PER_QUERY} records/query"
            )
        else:
            print("[dry-run] news: GDELT disabled (pass --use-gdelt to enable)")
        return []

    items: list[ScrapedItem] = []
    seen_urls: set[str] = set()

    def _accept(item: ScrapedItem) -> bool:
        key = (item.source_url or "").strip().lower().rstrip("/")
        if key:
            if key in seen_urls:
                return False
            seen_urls.add(key)
        return True

    async with NewsScraperRegistry() as registry:
        bar = tqdm(total=target, desc="news scrape", unit="item")

        # 1. RSS first — cheap, reliable, and image-bearing.
        try:
            async for item in registry.rss_scraper.scrape():
                if not _accept(item):
                    continue
                items.append(item)
                bar.update(1)
                if len(items) >= target:
                    bar.close()
                    return items
        except Exception as exc:  # noqa: BLE001 — one source can't kill the run
            logger.exception("RSSNewsScraper failed mid-run: %s", exc)

        if not use_gdelt:
            bar.close()
            return items

        # 2. GDELT — run twice for breadth: a fresh "today" pass plus a
        #    wider window (default 7d) so the dataset isn't dominated by
        #    whatever single news cycle happened to be running. Skip the
        #    second pass when the user pinned the wide window to "1d".
        gdelt_passes = ["1d"]
        if gdelt_timespan != "1d":
            gdelt_passes.append(gdelt_timespan)

        for timespan in gdelt_passes:
            if len(items) >= target:
                break
            logger.info(
                "GDELT pass: timespan=%s queries=%d max_per_query=%d",
                timespan,
                len(DEFAULT_GDELT_QUERIES),
                _GDELT_MAX_PER_QUERY,
            )
            try:
                async for item in registry.gdelt_scraper.scrape_topics(
                    queries=list(DEFAULT_GDELT_QUERIES),
                    max_per_query=_GDELT_MAX_PER_QUERY,
                    timespan=timespan,
                ):
                    if not _accept(item):
                        continue
                    items.append(item)
                    bar.update(1)
                    if len(items) >= target:
                        break
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "GDELT scrape_topics (timespan=%s) failed: %s", timespan, exc
                )
        bar.close()
    return items


async def scrape_satire(target: int, dry_run: bool) -> list[ScrapedItem]:
    if dry_run:
        registry = SatireScraperRegistry()
        print(f"[dry-run] satire: would scrape up to {target} items from:")
        for s in registry.scrapers:
            outlet = getattr(s, "outlet_name", type(s).__name__)
            url = getattr(s, "feed_url", "")
            print(f"             - {outlet} ({url})")
        return []

    items: list[ScrapedItem] = []
    async with SatireScraperRegistry() as registry:
        per_source = max(1, target // max(1, len(registry.scrapers)))
        bar = tqdm(total=target, desc="satire scrape", unit="item")
        async for item in registry.scrape_all(max_items_per_source=per_source):
            items.append(item)
            bar.update(1)
            if len(items) >= target:
                break
        bar.close()
    return items


# --- archive scraping --------------------------------------------------------
def _load_archive_configs(path: Path | None) -> list[dict[str, Any]]:
    """Read the archive-source config JSON into a list of scraper dicts.

    Accepts either a bare JSON list or a ``{"sources": [...]}`` wrapper.
    Returns ``[]`` (with a logged error) on any read/parse problem so a
    bad config file degrades to "no archive sources" rather than aborting
    the whole Tier 1 build.
    """
    if path is None:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("could not read archive config %s: %s", path, exc)
        return []
    if isinstance(data, dict):
        data = data.get("sources", [])
    if not isinstance(data, list):
        logger.error(
            "archive config %s must be a JSON list (or {\"sources\": [...]})", path
        )
        return []
    return data


async def scrape_archive(
    *,
    config_path: Path | None,
    max_articles: int,
    resume: bool,
    dry_run: bool,
) -> list[ScrapedItem]:
    """Scrape configured archive sources for deep-history imaged satire.

    Returns an empty list when no permitted sources are configured (the
    default) — the registry self-reports that case. Archive items already
    carry an ``image_url`` (the scrapers skip image-less articles), so
    they flow through the same download/filter/dedup pipeline as the RSS
    and GDELT items.
    """
    configs = _load_archive_configs(config_path)

    if dry_run:
        print(
            f"[dry-run] archive: would scrape up to {max_articles} articles/source "
            f"(resume={resume})"
        )
        if not configs:
            print(
                "[dry-run] archive: no sources configured. Pass --archive-config "
                "with permitted sources. Archive scrapers REFUSE sources that opt "
                "out of AI scraping (robots AI-crawler Disallow / ai-train=no)."
            )
        else:
            print("[dry-run] archive: configured sources:")
            for c in configs:
                entry = c.get("sitemap_url") or c.get("listing_url_template") or ""
                print(
                    f"             - {c.get('name', '?')} "
                    f"[{c.get('type', '?')}] {entry}"
                )
        return []

    registry = ArchiveScraperRegistry.from_config(configs)
    if not registry.scrapers:
        logger.warning(
            "--use-archive was set but no valid archive sources are configured "
            "(see --archive-config); contributing 0 archive items."
        )
        return []

    items: list[ScrapedItem] = []
    async with registry:
        bar = tqdm(
            total=max_articles * len(registry.scrapers),
            desc="archive scrape",
            unit="item",
        )
        async for item in registry.scrape_all(
            max_articles_per_source=max_articles, resume=resume
        ):
            items.append(item)
            bar.update(1)
        bar.close()
    return items


# --- huggingface -------------------------------------------------------------
async def load_huggingface(
    max_per_dataset: int, dry_run: bool
) -> list[ScrapedItem]:
    """Load curated HuggingFace satire/news datasets — currently disabled.

    All entries in :data:`KNOWN_SATIRE_DATASETS` are commented out
    because the only available HF satire corpus
    (``Biddls/Onion_News``) is a single-publisher text-only corpus
    that skews V-L training. The function and ``--use-huggingface``
    flag are kept in place so callers/scripts don't break; instead
    a loud warning is logged and an empty list is returned.
    """
    if not KNOWN_SATIRE_DATASETS:
        logger.warning(
            "--use-huggingface was passed but all HuggingFace satire "
            "datasets are currently disabled (single-publisher / "
            "text-only monoculture). Returning 0 items. See "
            "src/satira/ingest/huggingface_loader.py for context."
        )
        if dry_run:
            print(
                "[dry-run] huggingface: disabled — no datasets configured"
            )
        return []

    if dry_run:
        print(
            f"[dry-run] huggingface: would load up to {max_per_dataset} rows "
            "from each of:"
        )
        for spec in KNOWN_SATIRE_DATASETS:
            print(f"             - {spec.dataset_id}")
        return []

    loader = HFDatasetLoader()
    return await loader.load_all(max_per_dataset=max_per_dataset)


# --- image download ----------------------------------------------------------
def _as_text_only(item: ScrapedItem) -> ProcessedItem:
    """Wrap a no-image scraped item as a text-only :class:`ProcessedItem`."""
    return ProcessedItem(
        source_url=item.source_url,
        image_url=item.image_url,
        title=item.title,
        text=item.text,
        timestamp=item.timestamp,
        source_domain=item.source_domain,
        metadata=dict(item.metadata),
    )


async def download_images(
    items: list[ScrapedItem], storage_path: Path
) -> tuple[list[ProcessedItem], int]:
    """Download images for ``items`` and wrap text-only entries as
    :class:`ProcessedItem` records.

    Returns ``(processed, download_failures)`` where ``processed``
    contains both successfully-downloaded items (``image_path`` set) and
    text-only items (``image_path=None``), and ``download_failures`` is
    the count of items that had an ``image_url`` but failed validation
    or fetch.
    """
    with_image = [it for it in items if it.image_url]
    text_only = [it for it in items if not it.image_url]

    processed: list[ProcessedItem] = [_as_text_only(it) for it in text_only]
    downloader = ImageDownloader(storage_path=str(storage_path))
    try:
        bar = tqdm(total=len(with_image), desc="image download", unit="item")
        for i in range(0, len(with_image), _DOWNLOAD_BATCH):
            batch = with_image[i : i + _DOWNLOAD_BATCH]
            results = await downloader.download_batch(
                batch, max_concurrent=_MAX_CONCURRENT_DOWNLOADS
            )
            processed.extend(results)
            bar.update(len(batch))
        bar.close()
    finally:
        await downloader.close()
    download_failures = len(with_image) - (len(processed) - len(text_only))
    return processed, download_failures


# --- quality filters ---------------------------------------------------------
def _item_text(item: ProcessedItem) -> str:
    """Best textual surface for filtering: title preferred, body as fallback."""
    return (item.title or item.text or "").strip()


def _looks_english(text: str) -> bool:
    """Heuristic English check using ``langdetect``.

    langdetect can throw on very short or numeric strings — a thrown
    detection is treated as "skip the language gate" rather than a
    hard drop, since the min-length filter already removes the worst
    short-text cases.
    """
    if not text:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return True


def quality_filter(
    items: list[ProcessedItem],
) -> tuple[list[ProcessedItem], Counter]:
    """Apply text-length, language, and NSFW filters.

    Items with text longer than :data:`_MAX_TEXT_LEN` are *truncated*
    in place (mutating ``title``/``text``) rather than dropped — the
    headline-style content this tier targets is rarely overlong, but
    a verbose RSS body shouldn't be discarded for it.

    Returns ``(kept, drops_counter)`` where the counter records each
    drop reason.
    """
    kept: list[ProcessedItem] = []
    drops: Counter = Counter()
    for item in tqdm(items, desc="quality filter", unit="item"):
        text = _item_text(item)
        if len(text) < _MIN_TEXT_LEN:
            drops["too_short"] += 1
            continue
        # Truncate, prefer the title field as that's what to_record
        # serializes; fall back to text if there's no title.
        if item.title and len(item.title) > _MAX_TEXT_LEN:
            item.title = item.title[:_MAX_TEXT_LEN]
            drops["truncated"] += 1
        elif item.text and len(item.text) > _MAX_TEXT_LEN:
            item.text = item.text[:_MAX_TEXT_LEN]
            drops["truncated"] += 1
        if _NSFW_RE.search(text):
            drops["nsfw"] += 1
            continue
        if not _looks_english(text):
            drops["non_english"] += 1
            continue
        kept.append(item)
    return kept, drops


# --- deduplication -----------------------------------------------------------
def _norm_url(url: str | None) -> str:
    if not url:
        return ""
    return url.strip().lower().rstrip("/")


def _norm_title(title: str | None) -> str:
    if not title:
        return ""
    return re.sub(r"\s+", " ", title.strip().lower())


def dedup_by_url(items: list[ProcessedItem]) -> tuple[list[ProcessedItem], int]:
    """Drop later items that share an exact source URL with an earlier one."""
    seen: set[str] = set()
    kept: list[ProcessedItem] = []
    dropped = 0
    for item in items:
        key = _norm_url(item.source_url)
        if key and key in seen:
            dropped += 1
            continue
        if key:
            seen.add(key)
        kept.append(item)
    return kept, dropped


def dedup_by_title(
    items: list[ProcessedItem], threshold: int = _TITLE_FUZZ_THRESHOLD
) -> tuple[list[ProcessedItem], int]:
    """Drop items whose normalized title is ≥``threshold``% similar to a kept one.

    Comparison is bucketed by the first two characters of the
    normalized title to keep the worst case bounded; in practice this
    is the cheapest correctness/speed trade-off for tens of thousands
    of headlines, and false negatives at the bucket boundary are rare
    enough not to matter for Tier 1.
    """
    kept: list[ProcessedItem] = []
    buckets: dict[str, list[str]] = defaultdict(list)
    dropped = 0
    for item in items:
        norm = _norm_title(item.title)
        if not norm:
            kept.append(item)
            continue
        bucket_key = norm[:2]
        is_dup = False
        for prev in buckets[bucket_key]:
            if fuzz.ratio(norm, prev) >= threshold:
                is_dup = True
                break
        if is_dup:
            dropped += 1
            continue
        buckets[bucket_key].append(norm)
        kept.append(item)
    return kept, dropped


def _load_cross_tier_keys(*tier_dirs: Path) -> tuple[set[str], set[str]]:
    """Collect URL and normalized-title keys from existing Tier 2/3 splits.

    Returns ``(urls, titles)``. Missing directories or unreadable files
    are silently skipped — the cross-tier check is best-effort. Reads
    every ``*.jsonl`` under each directory.
    """
    urls: set[str] = set()
    titles: set[str] = set()
    for tier_dir in tier_dirs:
        if not tier_dir.exists():
            continue
        for path in sorted(tier_dir.glob("*.jsonl")):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        url = _norm_url(
                            (row.get("metadata") or {}).get("source_url")
                            or row.get("source_url")
                        )
                        if url:
                            urls.add(url)
                        title = _norm_title(row.get("text") or row.get("title"))
                        if title:
                            titles.add(title)
            except OSError as exc:
                logger.warning("could not read %s for cross-tier dedup: %s", path, exc)
    return urls, titles


def drop_cross_tier(
    items: list[ProcessedItem], tier2_dir: Path, tier3_dir: Path
) -> tuple[list[ProcessedItem], int]:
    """Remove items already present in Tier 2 / Tier 3 splits."""
    urls, titles = _load_cross_tier_keys(tier2_dir, tier3_dir)
    if not urls and not titles:
        return items, 0
    kept: list[ProcessedItem] = []
    dropped = 0
    for item in items:
        url = _norm_url(item.source_url)
        title = _norm_title(item.title)
        if (url and url in urls) or (title and title in titles):
            dropped += 1
            continue
        kept.append(item)
    return kept, dropped


# --- label verification ------------------------------------------------------
def cap_text_only_per_label(
    verified: list[tuple[ProcessedItem, int]],
    max_frac: float = _MAX_TEXT_ONLY_FRAC,
) -> tuple[list[tuple[ProcessedItem, int]], int]:
    """Cap text-only items at ``max_frac`` of each label's total.

    NPR and Al Jazeera publish RSS without images, so their entries
    arrive here as text-only items. Without a cap they can dominate the
    news side of the dataset (especially when image-bearing feeds lose
    items to download failures), tilting Tier 1 toward a text-only
    mix. We cap *per label* so satire and news bins are balanced
    independently.

    The cap applies uniformly regardless of source. An earlier version
    exempted curated HuggingFace rows on the grounds that they were
    intentionally text-only and high-volume, but in practice the only
    available HF satire corpus (Onion_News, 33k Onion-only rows)
    skewed the dataset hard toward a single-publisher text-only
    monoculture. If HF datasets are ever re-enabled they should be
    subject to the same cap as RSS/GDELT.

    Returns ``(kept, dropped_count)``. Items are dropped from the tail
    of the text-only list so the cap is deterministic for a given input
    order.
    """
    by_label: dict[int, list[tuple[ProcessedItem, int]]] = defaultdict(list)
    for entry in verified:
        by_label[entry[1]].append(entry)

    kept: list[tuple[ProcessedItem, int]] = []
    dropped = 0
    for label, group in by_label.items():
        with_image = [e for e in group if e[0].image_path is not None]
        text_only = [e for e in group if e[0].image_path is None]
        if not text_only:
            kept.extend(with_image)
            continue
        # max_frac = text_kept / (with_image + text_kept)
        # → text_kept = max_frac * with_image / (1 - max_frac)
        if max_frac >= 1.0:
            cap = len(text_only)
        elif max_frac <= 0.0 or not with_image:
            cap = 0
        else:
            cap = int(max_frac * len(with_image) / (1 - max_frac))
        kept_text = text_only[:cap]
        kept.extend(with_image)
        kept.extend(kept_text)
        dropped += len(text_only) - len(kept_text)
    return kept, dropped


def verify_labels(
    items: list[ProcessedItem],
) -> tuple[list[tuple[ProcessedItem, int]], Counter]:
    """Drop items where the source credibility verdict directly contradicts
    the asserted label. Items with UNKNOWN/MIXED verdicts are kept."""
    classifier = SourceCredibilityClassifier()
    kept: list[tuple[ProcessedItem, int]] = []
    drops: Counter = Counter()

    for item in tqdm(items, desc="verify labels", unit="item"):
        asserted_str = (item.metadata or {}).get("label", "")
        asserted = _STR_TO_LABEL.get(asserted_str)
        if asserted is None:
            drops["no_label"] += 1
            continue
        verdict = classifier.classify(item.source_domain)
        if asserted == LABEL_AUTHENTIC and verdict.category == SATIRE:
            drops["news_classified_satire"] += 1
            continue
        if asserted == LABEL_SATIRE and verdict.category == NEWS:
            drops["satire_classified_news"] += 1
            continue
        kept.append((item, asserted))
    return kept, drops


# --- source balance ----------------------------------------------------------
def _solve_source_caps(
    counts: dict[str, int], max_frac: float, max_iter: int = 100
) -> dict[str, int]:
    """Find a per-source cap such that every source is ``<= max_frac`` of total.

    Fixed-point iteration: cap every source at ``floor(max_frac * total)``,
    then recompute total and repeat. When the system converges, every
    surviving source's share is at most ``max_frac``.

    With fewer than ``ceil(1/max_frac)`` sources the constraint is
    infeasible (you can't have all 3 of 3 sources under 25% — they'd
    sum to <75%). In that case we equalize to the smallest source's
    count and bail out, so the caller still gets a balanced subset
    rather than zero items.
    """
    if not counts:
        return {}
    if len(counts) * max_frac < 1:
        target = min(counts.values())
        return {s: target for s in counts}
    cur = dict(counts)
    for _ in range(max_iter):
        total = sum(cur.values())
        cap = int(max_frac * total)
        if cap <= 0:
            return {s: 0 for s in cur}
        new = {s: min(c, cap) for s, c in cur.items()}
        if new == cur:
            return cur
        cur = new
    return cur


def enforce_source_balance(
    verified: list[tuple[ProcessedItem, int]],
    *,
    max_frac: float = _MAX_SOURCE_FRAC,
    seed: int,
) -> tuple[list[tuple[ProcessedItem, int]], Counter]:
    """Cap each source at ``max_frac`` of its label total via random subsample.

    Per-label so satire and news bins are balanced independently.
    Within a label the cap is solved by :func:`_solve_source_caps`,
    then each source is randomly downsampled (seeded) to its cap.
    Random sampling is the stratification step: we don't know which
    of an outlet's items are best, so picking uniformly is the
    least-biased way to keep diversity within a source too.

    Returns ``(kept, drops_per_source)`` keyed by source domain so the
    summary stage can show where the cap kicked in.
    """
    rng = random.Random(seed)
    by_label: dict[int, list[tuple[ProcessedItem, int]]] = defaultdict(list)
    for entry in verified:
        by_label[entry[1]].append(entry)

    kept: list[tuple[ProcessedItem, int]] = []
    drops: Counter = Counter()
    if max_frac <= 0 or max_frac >= 1:
        for entries in by_label.values():
            kept.extend(entries)
        return kept, drops

    for label, group in by_label.items():
        if not group:
            continue
        by_source: dict[str, list[tuple[ProcessedItem, int]]] = defaultdict(list)
        for entry in group:
            by_source[entry[0].source_domain or "<unknown>"].append(entry)
        original_counts = {s: len(es) for s, es in by_source.items()}
        target_counts = _solve_source_caps(original_counts, max_frac)
        for source, entries in by_source.items():
            target = target_counts.get(source, len(entries))
            if target >= len(entries):
                kept.extend(entries)
                continue
            shuffled = list(entries)
            rng.shuffle(shuffled)
            kept.extend(shuffled[:target])
            dropped = len(shuffled) - target
            drops[source] += dropped
            logger.warning(
                "source %s exceeds %.0f%% of label %s "
                "(%d -> %d, dropped %d)",
                source,
                max_frac * 100,
                _LABEL_NAMES[label],
                len(entries),
                target,
                dropped,
            )
    return kept, drops


# --- record + split ----------------------------------------------------------
def to_record(item: ProcessedItem, label: int) -> dict[str, Any]:
    return {
        "image_path": item.image_path,
        "text": item.title or item.text,
        "label": label,
        "source": item.source_domain,
        "timestamp": item.timestamp.isoformat(),
        "perceptual_hash": item.perceptual_hash,
        "metadata": dict(item.metadata or {}),
    }


def stratified_split(
    records: list[dict[str, Any]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for label in sorted(by_label):
        group = by_label[label]
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=str) + "\n")


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _print_split_stats(name: str, split: list[dict[str, Any]]) -> None:
    """Detailed per-split summary: labels, sources, text-only ratio, dates."""
    n = len(split)
    print(f"\n  --- {name} (total={n}) ---")
    if n == 0:
        return

    labels = Counter(r["label"] for r in split)
    label_breakdown = {_LABEL_NAMES[k]: v for k, v in labels.items()}
    print(f"    labels: {label_breakdown}")

    sources = Counter(r["source"] for r in split)
    print(f"    sources ({len(sources)} unique):")
    for source, count in sources.most_common(10):
        pct = (count / n) * 100
        marker = "  <-- exceeds cap" if pct > _MAX_SOURCE_FRAC * 100 else ""
        print(f"      {source:40s} {count:5d}  {pct:5.1f}%{marker}")
    if len(sources) > 10:
        remainder = sum(c for _, c in sources.most_common()[10:])
        print(f"      ({len(sources) - 10} more sources, total {remainder})")

    text_only = sum(1 for r in split if not r.get("image_path"))
    text_only_frac = (text_only / n) if n else 0.0
    cap_marker = "" if text_only_frac <= _MAX_TEXT_ONLY_FRAC else "  <-- EXCEEDS CAP"
    print(
        f"    text-only: {text_only}/{n} = {text_only_frac:.1%}"
        f"{cap_marker}  (max={_MAX_TEXT_ONLY_FRAC:.0%})"
    )

    text_lengths = [len(r.get("text") or "") for r in split]
    avg_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0.0
    print(
        f"    text length: avg={avg_len:5.0f} "
        f"min={min(text_lengths) if text_lengths else 0} "
        f"max={max(text_lengths) if text_lengths else 0}"
    )

    timestamps = [t for t in (_parse_timestamp(r.get("timestamp")) for r in split) if t]
    if timestamps:
        ts_min = min(timestamps)
        ts_max = max(timestamps)
        span_days = (ts_max - ts_min).days
        # Plain ASCII arrow — the Windows console uses cp1252 by default
        # and chokes on unicode arrows.
        print(
            f"    date range: {ts_min.date()} -> {ts_max.date()}  "
            f"({span_days} day span, {len(timestamps)}/{n} dated)"
        )
    else:
        print("    date range: (no parseable timestamps)")


def print_stats(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> None:
    print("\n=== Tier 1 dataset summary ===")
    for name, split in (("train", train), ("val", val), ("test", test)):
        _print_split_stats(name, split)


# --- driver ------------------------------------------------------------------
async def run(args: argparse.Namespace) -> int:
    print("=== Tier 1 dataset build ===")
    print(f"  target news    : {args.target_news}")
    print(f"  target satire  : {args.target_satire}")
    print(f"  output dir     : {args.output_dir}")
    print(f"  image storage  : {args.image_storage}")
    print(f"  use gdelt      : {args.use_gdelt}")
    print(f"  gdelt timespan : {args.gdelt_timespan}")
    print(f"  use huggingface: {args.use_huggingface}")
    print(f"  use archive    : {args.use_archive}")
    if args.use_archive:
        archive_resume = args.archive_resume and not args.archive_fresh
        print(f"  archive config : {args.archive_config}")
        print(
            f"  archive limits : max_articles={args.archive_max_articles} "
            f"resume={archive_resume} dry_run={args.archive_dry_run}"
        )
    print(f"  tier2 dir      : {args.tier2_dir} (exists={args.tier2_dir.exists()})")
    print(f"  tier3 dir      : {args.tier3_dir} (exists={args.tier3_dir.exists()})")
    print(f"  dry run        : {args.dry_run}")
    print(f"  seed           : {args.seed}")

    news_items = await scrape_news(
        args.target_news,
        use_gdelt=args.use_gdelt,
        gdelt_timespan=args.gdelt_timespan,
        dry_run=args.dry_run,
    )
    satire_items = await scrape_satire(args.target_satire, args.dry_run)
    hf_items: list[ScrapedItem] = []
    if args.use_huggingface:
        hf_items = await load_huggingface(args.hf_max_per_dataset, args.dry_run)

    archive_items: list[ScrapedItem] = []
    if args.use_archive:
        archive_items = await scrape_archive(
            config_path=args.archive_config,
            max_articles=(50 if args.archive_dry_run else args.archive_max_articles),
            resume=(args.archive_resume and not args.archive_fresh),
            dry_run=args.dry_run,
        )

    if args.dry_run:
        print("\n[dry-run] no items written.")
        return 0

    print(f"\n[scrape] news scraped  : {len(news_items)}")
    print(f"[scrape] satire scraped: {len(satire_items)}")
    if args.use_archive:
        print(f"[scrape] archive scraped: {len(archive_items)}")
    if args.use_huggingface:
        hf_satire = sum(1 for it in hf_items if it.metadata.get("label") == "satire")
        hf_news = len(hf_items) - hf_satire
        print(
            f"[hf]     loaded={len(hf_items)} "
            f"(satire={hf_satire}, authentic={hf_news})"
        )

    all_items = news_items + satire_items + hf_items + archive_items
    processed, download_failures = await download_images(all_items, args.image_storage)
    text_only_count = sum(1 for p in processed if p.image_path is None)
    print(
        f"[download] processed={len(processed)} "
        f"text_only={text_only_count} "
        f"download_failures={download_failures}"
    )

    filtered, quality_drops = quality_filter(processed)
    print(f"[quality] kept={len(filtered)} drops={dict(quality_drops)}")

    url_deduped, url_dropped = dedup_by_url(filtered)
    print(f"[dedup-url] kept={len(url_deduped)} dropped={url_dropped}")

    title_deduped, title_dropped = dedup_by_title(url_deduped)
    print(f"[dedup-title] kept={len(title_deduped)} dropped={title_dropped}")

    deduper = ImageDownloader(storage_path=str(args.image_storage))
    try:
        deduped = deduper.deduplicate_by_phash(
            title_deduped, hamming_threshold=_PHASH_HAMMING_THRESHOLD
        )
    finally:
        await deduper.close()
    print(
        f"[dedup-phash] kept={len(deduped)} "
        f"dropped={len(title_deduped) - len(deduped)}"
    )

    cross_kept, cross_dropped = drop_cross_tier(
        deduped, args.tier2_dir, args.tier3_dir
    )
    print(f"[dedup-cross-tier] kept={len(cross_kept)} dropped={cross_dropped}")

    verified, drops = verify_labels(cross_kept)
    print(f"[verify] kept={len(verified)} drops={dict(drops)}")

    capped, text_only_dropped = cap_text_only_per_label(verified)
    text_only_kept = sum(1 for it, _ in capped if it.image_path is None)
    total_capped = len(capped)
    text_only_frac = (text_only_kept / total_capped) if total_capped else 0.0
    print(
        f"[cap-text-only] text_only_kept={text_only_kept} "
        f"text_only_dropped={text_only_dropped} "
        f"text_only_frac={text_only_frac:.1%} "
        f"(max={_MAX_TEXT_ONLY_FRAC:.0%})"
    )

    balanced, source_drops = enforce_source_balance(capped, seed=args.seed)
    print(
        f"[balance] kept={len(balanced)} "
        f"dropped={sum(source_drops.values())} "
        f"capped_sources={dict(source_drops) or '{}'}"
    )

    records = [to_record(item, label) for item, label in balanced]
    train, val, test = stratified_split(records, 0.8, 0.1, args.seed)
    print(f"[split] train={len(train)} val={len(val)} test={len(test)}")

    write_jsonl(train, args.output_dir / "train.jsonl")
    write_jsonl(val, args.output_dir / "val.jsonl")
    write_jsonl(test, args.output_dir / "test.jsonl")
    print(f"[write] wrote splits to {args.output_dir}")

    print_stats(train, val, test)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    try:
        return asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001 — surface every failure at the entry point
        logger.error("build failed: %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
