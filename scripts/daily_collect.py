"""Daily RSS corpus collection entry point.

Runs one pass of :class:`CorpusAccumulator.run_daily_collection`: scrape
every satire and news feed, dedup against the persistent corpus, append
the genuinely-new items, and log the result. RSS only exposes a feed's
most recent items, so this is meant to run *every day* — invoked by
Windows Task Scheduler — to accumulate volume over time.

The script is non-interactive and exits cleanly with a status code so a
scheduler can detect failures: ``0`` on a completed run (even one that
added nothing), ``1`` if the collection itself raised.

Usage::

    # one collection pass (the scheduler's daily invocation)
    python scripts/daily_collect.py

    # same thing, stated explicitly (handy when testing by hand)
    python scripts/daily_collect.py --once

    # print corpus stats without collecting anything
    python scripts/daily_collect.py --status

    # include GDELT topic queries alongside the RSS feeds
    python scripts/daily_collect.py --gdelt

Each run's :class:`CollectionReport` is written to
``./logs/collection/{YYYY-MM-DD}.json``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satira.ingest.accumulator import CollectionReport, CorpusAccumulator  # noqa: E402
from satira.ingest.news_scrapers import GDELTScraper  # noqa: E402


logger = logging.getLogger("satira.daily_collect")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one daily RSS corpus collection pass."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("./data/corpus"),
        help="Persistent corpus directory (default: ./data/corpus).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./logs/collection"),
        help="Where to write per-run collection logs (default: ./logs/collection).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Run a single collection pass and exit. This is already the "
            "default behaviour; the flag makes the intent explicit when "
            "running by hand."
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print corpus statistics and exit without collecting.",
    )
    parser.add_argument(
        "--gdelt",
        action="store_true",
        help=(
            "Augment the RSS news feeds with GDELT topic queries. Off by "
            "default: the daily run is RSS-first for reliability, and GDELT "
            "is comparatively flaky."
        ),
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


def write_report_log(report: CollectionReport, log_dir: Path) -> Path:
    """Write the report JSON to ``{log_dir}/{run-date}.json`` and return the path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{report.run_date.strftime('%Y-%m-%d')}.json"
    path.write_text(
        json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    return path


def print_report(report: CollectionReport, log_path: Path) -> None:
    """Human-readable summary of a collection run (ASCII-only for cp1252)."""
    print("\n=== daily collection report ===")
    print(f"  run date        : {report.run_date.isoformat()}")
    print(f"  items scraped   : {report.items_scraped}")
    print(f"  items new       : {report.items_new}")
    print(f"  items duplicate : {report.items_duplicate}")
    print(f"  images saved    : {report.images_downloaded}")

    if report.by_label:
        print("  new by label    :")
        for label, count in sorted(report.by_label.items()):
            print(f"      {label:12s} {count}")
    if report.by_source:
        print("  new by source   :")
        for source, count in sorted(
            report.by_source.items(), key=lambda kv: kv[1], reverse=True
        ):
            print(f"      {source:32s} {count}")
    if report.errors:
        print(f"  errors ({len(report.errors)}):")
        for err in report.errors:
            print(f"      - {err}")
    else:
        print("  errors          : none")
    print(f"  log written     : {log_path}")


def print_stats(stats: dict[str, Any]) -> None:
    """Human-readable corpus statistics (the --status view)."""
    print("\n=== corpus statistics ===")
    print(f"  total items     : {stats['total_items']}")
    if stats["total_items"] == 0:
        print("  (corpus is empty - run a collection first)")
        return

    print(f"  unique sources  : {stats['unique_sources']}")
    print(
        f"  text-only       : {stats['text_only']} "
        f"({stats['text_only_ratio']:.1%})"
    )

    print("  by label        :")
    for label, count in stats["by_label"].items():
        print(f"      {label:12s} {count}")

    print("  by source (top) :")
    for source, count in list(stats["by_source"].items())[:10]:
        print(f"      {source:32s} {count}")
    extra = len(stats["by_source"]) - 10
    if extra > 0:
        print(f"      (+{extra} more sources)")

    date_range = stats.get("date_range")
    if date_range:
        print(
            f"  article dates   : {date_range['earliest']} -> {date_range['latest']}"
        )
    coll_range = stats.get("collection_range")
    if coll_range:
        print(
            f"  collected       : {coll_range['earliest']} -> {coll_range['latest']}"
        )

    added = stats.get("added_per_day") or {}
    if added:
        print("  added per day (last 30d):")
        for day, count in list(added.items())[:30]:
            print(f"      {day}  {count}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    gdelt_queries = list(GDELTScraper.DEFAULT_QUERIES) if args.gdelt else None
    accumulator = CorpusAccumulator(
        corpus_dir=str(args.corpus_dir),
        gdelt_queries=gdelt_queries,
    )

    if args.status:
        print_stats(accumulator.get_corpus_stats())
        return 0

    print("=== daily corpus collection ===")
    print(f"  corpus dir : {args.corpus_dir}")
    print(f"  log dir    : {args.log_dir}")
    print(f"  gdelt      : {'enabled' if args.gdelt else 'disabled (RSS only)'}")

    try:
        report = asyncio.run(accumulator.run_daily_collection())
    except Exception as exc:  # noqa: BLE001 — surface any failure at the entry point
        logger.error("collection failed: %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        return 1

    log_path = write_report_log(report, args.log_dir)
    print_report(report, log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
