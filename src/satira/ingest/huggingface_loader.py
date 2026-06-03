"""Adapter for pre-labeled satire/news datasets on the HuggingFace Hub.

Scraping RSS feeds is slow, polite, and image-bound — fine for fresh
content but a poor source of bulk training data. The Hub already hosts
several curated satire/sarcasm classification datasets that are
publicly downloadable and well-labeled, so :class:`HFDatasetLoader`
plugs them into the same :class:`ScrapedItem` shape the scrapers
produce. Downstream stages (label verify, dedupe, split) treat HF rows
identically to RSS rows.

Only text-only datasets are wired in: HF rarely ships images alongside
classification corpora, and Tier 1 is permitted to carry text-only
items with ``image_url=None``. The build script's text-only cap is
relaxed for items stamped with ``metadata['source_type'] = 'huggingface'``
so a curated dataset of 30k clean labels isn't crowded out by a
20%-of-image-items ceiling intended for RSS feeds with missing
thumbnails.

Each row is mapped to a :class:`ScrapedItem` with:

* ``source_url`` — the original article URL when the dataset records it,
  else an empty string.
* ``image_url = None`` — text-only.
* ``source_domain`` — extracted from ``source_url`` when present, else
  the publisher hardcoded for the dataset (e.g. ``theonion.com`` for the
  Onion-only corpora). This lets :class:`SourceCredibilityClassifier`
  hit its known-source allowlists for these rows.
* ``metadata['label']`` — Satira's string label (``"satire"`` or
  ``"authentic"``), matching what the RSS scrapers stamp.
* ``metadata['source_type'] = 'huggingface'`` and
  ``metadata['hf_dataset']`` — provenance markers used by the cap step
  and by anyone debugging dataset composition later.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from satira.ingest.base_scraper import ScrapedItem
from satira.ingest.domain_utils import normalize_domain


logger = logging.getLogger(__name__)


_LABEL_AUTHENTIC = "authentic"
_LABEL_SATIRE = "satire"

# Header/body separator used by the Onion_News corpus.
_ONION_SEP = "#~#"


@dataclass(frozen=True)
class HFDatasetSpec:
    """How to load and adapt one HuggingFace dataset.

    ``adapter`` takes one row (a ``dict[str, Any]``) and returns a
    :class:`ScrapedItem` or ``None`` if the row should be skipped (e.g.
    a ``fake_or_satire`` row that's labelled "fake", not "satire").

    ``default_split`` and ``config`` give per-dataset overrides for the
    HF ``load_dataset`` call so callers don't need to remember each
    dataset's quirks.
    """

    dataset_id: str
    adapter: Callable[[dict[str, Any]], ScrapedItem | None]
    default_split: str = "train"
    config: str | None = None


# --- adapters ---------------------------------------------------------------
def _now_utc() -> datetime:
    # HF rows usually don't carry timestamps. We stamp the load time so
    # the temporal index has *something* monotonic to sort on; downstream
    # code that needs publication dates should look at the source URL,
    # not this field.
    return datetime.now(timezone.utc)


def _adapt_sarcasm_news_headline(row: dict[str, Any]) -> ScrapedItem | None:
    """raquiba/Sarcasm_News_Headline: 55k headlines, Onion vs HuffPost.

    Columns: ``is_sarcastic`` (0/1), ``headline``, ``article_link``.
    Label maps directly: 1 → satire, 0 → authentic.
    """
    headline = (row.get("headline") or "").strip()
    if not headline:
        return None
    is_sarcastic = row.get("is_sarcastic")
    if is_sarcastic not in (0, 1):
        return None
    label = _LABEL_SATIRE if is_sarcastic == 1 else _LABEL_AUTHENTIC
    url = (row.get("article_link") or "").strip()
    domain = normalize_domain(url) or (
        "theonion.com" if label == _LABEL_SATIRE else "huffpost.com"
    )
    return ScrapedItem(
        source_url=url,
        image_url=None,
        title=headline,
        text="",
        timestamp=_now_utc(),
        source_domain=domain,
        metadata={
            "label": label,
            "source_type": "huggingface",
            "hf_dataset": "raquiba/Sarcasm_News_Headline",
        },
    )


def _adapt_biddls_onion(row: dict[str, Any]) -> ScrapedItem | None:
    """Biddls/Onion_News: 33k Onion articles, all satire.

    Single ``text`` column with header and body joined by ``#~#``. Lines
    that are *only* the separator mark a missing field and we skip
    them — the dataset's own README calls this out.
    """
    raw = (row.get("text") or "").strip()
    if not raw or raw == _ONION_SEP:
        return None
    if _ONION_SEP in raw:
        header, _, body = raw.partition(_ONION_SEP)
        title = header.strip()
        body = body.strip()
    else:
        title = raw
        body = ""
    if not title:
        return None
    return ScrapedItem(
        source_url="",
        image_url=None,
        title=title,
        text=body,
        timestamp=_now_utc(),
        source_domain="theonion.com",
        metadata={
            "label": _LABEL_SATIRE,
            "source_type": "huggingface",
            "hf_dataset": "Biddls/Onion_News",
        },
    )


# All known HuggingFace satire datasets are intentionally disabled for
# now. Every publicly-available HF satire corpus we evaluated is too
# monocultural for V-L (vision-language) training:
#
#   * ``Biddls/Onion_News`` — 33k articles, 100% from The Onion, all
#     text-only. Loading this corpus collapses the satire side of the
#     dataset onto a single voice and a single visual prior (zero
#     images), which directly contradicts what Tier 1 is meant to
#     teach the model.
#   * ``raquiba/Sarcasm_News_Headline`` — headlines-only (avg ~73
#     chars), Onion vs HuffPost. Same single-publisher / text-only
#     concern, plus malformed ``article_link`` values that produced
#     the corrupt source ``huffingtonpost.comhttp:`` in past runs.
#
# The loader infrastructure (HFDatasetLoader, HFDatasetSpec, adapters)
# is left intact so a better-distributed multimodal satire corpus can
# be wired in by appending a spec here once one is found.
KNOWN_SATIRE_DATASETS: tuple[HFDatasetSpec, ...] = (
    # HFDatasetSpec(
    #     dataset_id="raquiba/Sarcasm_News_Headline",
    #     adapter=_adapt_sarcasm_news_headline,
    # ),
    # HFDatasetSpec(
    #     dataset_id="Biddls/Onion_News",
    #     adapter=_adapt_biddls_onion,
    # ),
)


# --- loader -----------------------------------------------------------------
class HFDatasetLoader:
    """Loads curated HuggingFace satire/news datasets into Satira's format.

    Streaming mode is on by default so loading 3k rows from a 55k-row
    parquet doesn't pull the whole file. Pass ``streaming=False`` only
    when reproducibility of row order matters and the dataset is small.

    The HF ``datasets`` library is imported lazily because it pulls in
    pyarrow + a multi-hundred-MB cache directory on first use; callers
    that don't pass ``--use-huggingface`` shouldn't pay that cost.
    """

    def __init__(
        self,
        specs: Iterable[HFDatasetSpec] = KNOWN_SATIRE_DATASETS,
        streaming: bool = True,
    ) -> None:
        self.specs: tuple[HFDatasetSpec, ...] = tuple(specs)
        self.streaming = streaming

    async def load_dataset(
        self,
        dataset_id: str,
        split: str = "train",
        max_items: int = 5000,
    ) -> list[ScrapedItem]:
        """Load up to ``max_items`` rows from one dataset.

        ``dataset_id`` must match one of the configured specs — this
        keeps adapter coverage explicit rather than guessing schemas at
        runtime.
        """
        spec = self._find_spec(dataset_id)
        if spec is None:
            raise ValueError(
                f"unknown dataset {dataset_id!r}; configured: "
                f"{[s.dataset_id for s in self.specs]}"
            )
        return await asyncio.to_thread(
            self._load_blocking, spec, split, max_items
        )

    async def load_all(
        self, max_per_dataset: int = 3000
    ) -> list[ScrapedItem]:
        """Load up to ``max_per_dataset`` rows from every configured spec.

        Datasets are loaded sequentially (HF caching is filesystem-bound
        and there's no win in parallelising the downloads). One failing
        dataset is logged and skipped so a transient Hub outage doesn't
        sink the whole build.
        """
        items: list[ScrapedItem] = []
        for spec in self.specs:
            try:
                loaded = await self.load_dataset(
                    spec.dataset_id,
                    split=spec.default_split,
                    max_items=max_per_dataset,
                )
            except Exception as exc:  # noqa: BLE001 — one bad dataset can't block the rest
                logger.warning(
                    "huggingface dataset %s failed to load: %s: %s",
                    spec.dataset_id, type(exc).__name__, exc,
                )
                continue
            logger.info(
                "huggingface dataset %s: loaded %d items", spec.dataset_id, len(loaded)
            )
            items.extend(loaded)
        return items

    # --- internals ---------------------------------------------------------
    def _find_spec(self, dataset_id: str) -> HFDatasetSpec | None:
        for spec in self.specs:
            if spec.dataset_id == dataset_id:
                return spec
        return None

    def _load_blocking(
        self, spec: HFDatasetSpec, split: str, max_items: int
    ) -> list[ScrapedItem]:
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "the `datasets` package is required for HuggingFace ingest "
                "(install via `poetry install` after adding the dependency)"
            ) from exc

        ds = load_dataset(
            spec.dataset_id,
            name=spec.config,
            split=split,
            streaming=self.streaming,
        )
        items: list[ScrapedItem] = []
        for row in ds:
            if len(items) >= max_items:
                break
            try:
                item = spec.adapter(row)
            except Exception as exc:  # noqa: BLE001 — bad row shouldn't kill the load
                logger.debug(
                    "adapter for %s rejected a row: %s", spec.dataset_id, exc
                )
                continue
            if item is not None:
                items.append(item)
        return items
