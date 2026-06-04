"""Load and report on the research-licensed HuggingFace fact-checking datasets.

Loads each spec in :data:`KNOWN_FACTCHECK_DATASETS` (LIAR2, …) through the
same :class:`HFDatasetLoader` the build uses, then prints, per dataset:

* total items produced by the adapter,
* the 5-class label distribution (and the original source labels),
* the text-only ratio (these corpora are text-only),
* license + citation attribution for the model card.

This is a *dataset-centric* report: it does not push rows through the
binary Tier 1 filter (which would drop the ``fabricated`` /
``misleading_context`` rows), so the numbers reflect what the loader
actually yields for the multi-class tiers.

Usage:
    py scripts/report_hf_datasets.py --max-per-dataset 20000
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satira.config import settings  # noqa: E402
from satira.ingest import KNOWN_FACTCHECK_DATASETS, HFDatasetLoader  # noqa: E402
from satira.ingest.base_scraper import ScrapedItem  # noqa: E402


logger = logging.getLogger("satira.report_hf")


def _print_class_distribution(labels: Counter, total: int, indent: str = "    ") -> None:
    """Print every canonical 5-class label, *including zeros*.

    Showing the full taxonomy (not just the labels that occurred) makes
    an empty class explicit — e.g. LIAR2 deliberately yields 0 satire,
    and that should be visible in the report, not silently absent.
    """
    for cls in settings.CLASS_NAMES:
        count = labels.get(cls, 0)
        pct = (count / total) if total else 0.0
        print(f"{indent}{cls:20s} {count:6d}  ({pct:5.1%})")
    extras = {k: v for k, v in labels.items() if k not in settings.CLASS_NAMES}
    for label, count in sorted(extras.items(), key=lambda kv: -kv[1]):
        print(f"{indent}{label:20s} {count:6d}  ({count / total:5.1%})  [non-canonical]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=20000,
        help="Max rows to pull per dataset (default: 20000, ~all of LIAR2 train).",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable HF streaming (downloads the full parquet to cache first).",
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args(argv)


def _report_one(dataset_id: str, license_: str, citation: str, items: list[ScrapedItem]) -> None:
    n = len(items)
    print("\n" + "=" * 70)
    print(f"{dataset_id}   (n={n})")
    print("=" * 70)
    if n == 0:
        print("  (no items produced)")
        return

    labels = Counter(it.metadata.get("label", "<none>") for it in items)
    originals = Counter(it.metadata.get("original_label", "<none>") for it in items)
    text_only = sum(1 for it in items if not it.image_url)

    print("  5-class label distribution (all classes shown):")
    _print_class_distribution(labels, n)
    print("  original source labels:")
    for label, count in originals.most_common():
        print(f"    {label:20s} {count:6d}  ({count / n:5.1%})")
    print(f"  text-only: {text_only}/{n} = {text_only / n:.1%}")
    print(f"  license  : {license_}")
    print(f"  citation : {citation}")


async def run(args: argparse.Namespace) -> int:
    print("=== HuggingFace fact-checking dataset report ===")
    print(f"  datasets       : {[s.dataset_id for s in KNOWN_FACTCHECK_DATASETS]}")
    print(f"  max-per-dataset: {args.max_per_dataset}")
    print(f"  streaming      : {not args.no_streaming}")
    if not KNOWN_FACTCHECK_DATASETS:
        print("  (no fact-checking datasets configured)")
        return 0

    loader = HFDatasetLoader(
        specs=KNOWN_FACTCHECK_DATASETS, streaming=not args.no_streaming
    )

    grand_total = 0
    grand_labels: Counter = Counter()
    grand_text_only = 0
    for spec in KNOWN_FACTCHECK_DATASETS:
        try:
            items = await loader.load_dataset(
                spec.dataset_id,
                split=spec.default_split,
                max_items=args.max_per_dataset,
            )
        except Exception as exc:  # noqa: BLE001 — report what loaded, surface the rest
            print(f"\n!! {spec.dataset_id} failed to load: {type(exc).__name__}: {exc}")
            continue
        _report_one(spec.dataset_id, spec.license, spec.citation, items)
        grand_total += len(items)
        grand_labels.update(it.metadata.get("label", "<none>") for it in items)
        grand_text_only += sum(1 for it in items if not it.image_url)

    print("\n" + "#" * 70)
    print(f"TOTAL items added: {grand_total}")
    if grand_total:
        print("Combined 5-class distribution (all classes shown):")
        _print_class_distribution(grand_labels, grand_total)
        print(f"Combined text-only: {grand_text_only}/{grand_total} = "
              f"{grand_text_only / grand_total:.1%}")
        print(
            "\nNote: text-only ratio is ~100%. If these feed a build with a "
            "text-only cap (Tier 1's is 20%), the cap must be raised for "
            "source_type='huggingface' rows or they will be largely dropped."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
