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
4. Deduplicate by perceptual hash. Text-only items pass through the
   dedupe step untouched — they have no phash to compare on.
5. Verify labels with :class:`SourceCredibilityClassifier` — drop only
   items where the classifier directly contradicts the asserted label;
   leniency is fine here because curated allowlists already keep the
   tier "obvious".
6. Stratified 80/10/10 split into train/val/test.
7. Write JSONL splits to ``--output-dir`` and print summary stats.

Run with ``--dry-run`` to see the targets and queries without making
any network calls.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satira.ingest import (  # noqa: E402
    GDELTScraper,
    ImageDownloader,
    NewsScraperRegistry,
    ProcessedItem,
    SatireScraperRegistry,
    ScrapedItem,
    SourceCredibilityClassifier,
)
from satira.ingest.source_credibility import NEWS, SATIRE  # noqa: E402


logger = logging.getLogger("satira.build_tier1")


# Topic queries used when ``--use-gdelt`` is set. Ten broad topics ×
# 250 records per topic ≈ 2500 raw articles per run before
# cross-query dedup; comfortably enough headroom for a 5000-item
# news target once RSS contributions are added in.
DEFAULT_GDELT_QUERIES: tuple[str, ...] = GDELTScraper.DEFAULT_QUERIES

# Some feeds (NPR, Al Jazeera) don't carry images in their RSS, so the
# items they contribute are necessarily text-only. We keep them — a
# labelled headline is still training signal — but cap the share so
# they don't crowd out the multimodal items the model is actually being
# trained on.
_MAX_TEXT_ONLY_FRAC = 0.20

LABEL_AUTHENTIC = 0
LABEL_SATIRE = 1
_STR_TO_LABEL = {"authentic": LABEL_AUTHENTIC, "satire": LABEL_SATIRE}
_LABEL_NAMES = {LABEL_AUTHENTIC: "authentic", LABEL_SATIRE: "satire"}

_DOWNLOAD_BATCH = 50
_MAX_CONCURRENT_DOWNLOADS = 10


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
            "(adds ~2500 candidate articles across 10 broad topics)."
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
    target: int, *, use_gdelt: bool, dry_run: bool
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
        else:
            print("[dry-run] news: GDELT disabled (pass --use-gdelt to enable)")
        return []

    gdelt_queries = list(DEFAULT_GDELT_QUERIES) if use_gdelt else None

    items: list[ScrapedItem] = []
    async with NewsScraperRegistry() as registry:
        bar = tqdm(total=target, desc="news scrape", unit="item")
        async for item in registry.scrape_all(
            gdelt_queries=gdelt_queries,
            max_items=target,
        ):
            items.append(item)
            bar.update(1)
            if len(items) >= target:
                break
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


def print_stats(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> None:
    print("\n=== Tier 1 dataset summary ===")
    for name, split in (("train", train), ("val", val), ("test", test)):
        labels = Counter(r["label"] for r in split)
        sources = Counter(r["source"] for r in split)
        label_breakdown = {_LABEL_NAMES[k]: v for k, v in labels.items()}
        top_sources = ", ".join(f"{s}={c}" for s, c in sources.most_common(5))
        print(
            f"  {name:5s}: total={len(split):5d} labels={label_breakdown} "
            f"top_sources=[{top_sources}]"
        )


# --- driver ------------------------------------------------------------------
async def run(args: argparse.Namespace) -> int:
    print("=== Tier 1 dataset build ===")
    print(f"  target news    : {args.target_news}")
    print(f"  target satire  : {args.target_satire}")
    print(f"  output dir     : {args.output_dir}")
    print(f"  image storage  : {args.image_storage}")
    print(f"  use gdelt      : {args.use_gdelt}")
    print(f"  dry run        : {args.dry_run}")
    print(f"  seed           : {args.seed}")

    news_items = await scrape_news(
        args.target_news, use_gdelt=args.use_gdelt, dry_run=args.dry_run
    )
    satire_items = await scrape_satire(args.target_satire, args.dry_run)

    if args.dry_run:
        print("\n[dry-run] no items written.")
        return 0

    print(f"\n[scrape] news scraped  : {len(news_items)}")
    print(f"[scrape] satire scraped: {len(satire_items)}")

    all_items = news_items + satire_items
    processed, download_failures = await download_images(all_items, args.image_storage)
    text_only_count = sum(1 for p in processed if p.image_path is None)
    print(
        f"[download] processed={len(processed)} "
        f"text_only={text_only_count} "
        f"download_failures={download_failures}"
    )

    deduper = ImageDownloader(storage_path=str(args.image_storage))
    try:
        deduped = deduper.deduplicate_by_phash(processed)
    finally:
        await deduper.close()
    print(f"[dedupe] kept={len(deduped)} dropped={len(processed) - len(deduped)}")

    verified, drops = verify_labels(deduped)
    print(f"[verify] kept={len(verified)} drops={dict(drops)}")

    capped, text_only_dropped = cap_text_only_per_label(verified)
    text_only_kept = sum(1 for it, _ in capped if it.image_path is None)
    total_capped = len(capped)
    text_only_frac = (text_only_kept / total_capped) if total_capped else 0.0
    print(
        f"[cap] text_only_kept={text_only_kept} "
        f"text_only_dropped={text_only_dropped} "
        f"text_only_frac={text_only_frac:.1%} "
        f"(max={_MAX_TEXT_ONLY_FRAC:.0%})"
    )

    records = [to_record(item, label) for item, label in capped]
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
