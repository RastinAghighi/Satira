"""Image download and processing pipeline.

Scrapers yield :class:`ScrapedItem` records pointing at *image URLs* —
this module turns those URLs into bytes on disk we can feed to vision
models. Three things have to happen between "URL" and "training-ready
image":

1. Download — done through a :class:`BaseScraper` so we get rate
   limiting, retries, and robots.txt compliance for free.
2. Validate — reject anything that isn't PNG/JPEG/WebP, anything bigger
   than ``max_size_mb`` (likely junk or an attack), and anything below
   ``min_dimensions`` (too small to be useful to a vision encoder).
3. Hash — every saved item carries a perceptual hash so a later dedupe
   pass can collapse near-duplicates (the same hero photo recompressed
   or resized across syndicated outlets) without re-reading every file.

Storage is content-addressable: the on-disk filename is the SHA-256 of
the bytes, so two scrapers picking up the same image collapse to a
single file on disk regardless of where they came from.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image, UnidentifiedImageError

from satira.ingest.base_scraper import BaseScraper, ScrapedItem


logger = logging.getLogger(__name__)


_ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
_FORMAT_EXT = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}


@dataclass(kw_only=True)
class ProcessedItem(ScrapedItem):
    """A :class:`ScrapedItem` enriched with the on-disk image and its hashes.

    ``kw_only=True`` lets us add required fields after the parent's
    optional ``metadata`` field without violating dataclass ordering.
    """

    image_path: str
    image_dimensions: tuple[int, int]
    perceptual_hash: str
    file_size_bytes: int


class _DefaultImageFetcher(BaseScraper):
    """Concrete :class:`BaseScraper` used purely for ``fetch_image``.

    :class:`BaseScraper` is abstract because ``scrape`` is abstract; the
    image pipeline never iterates a scrape stream — it just needs the
    fetch infrastructure — so we plug in a no-op implementation rather
    than forcing every caller to wire up a full scraper.
    """

    async def scrape(self, **_: Any) -> AsyncIterator[ScrapedItem]:
        if False:  # pragma: no cover — never iterated
            yield


class ImageDownloader:
    """Downloads, validates, hashes, and stores images for scraped items."""

    def __init__(
        self,
        storage_path: str = "./data/images",
        max_size_mb: int = 10,
        min_dimensions: tuple[int, int] = (200, 200),
        scraper: BaseScraper | None = None,
    ) -> None:
        if max_size_mb <= 0:
            raise ValueError(f"max_size_mb must be positive, got {max_size_mb}")
        if min_dimensions[0] <= 0 or min_dimensions[1] <= 0:
            raise ValueError(
                f"min_dimensions must be positive, got {min_dimensions}"
            )

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.min_dimensions = min_dimensions

        # An owned default scraper means callers don't have to wire one
        # up just to download an image. Pass one in if you want to share
        # rate limits or HTTP clients across hosts.
        self._scraper = scraper if scraper is not None else _DefaultImageFetcher()
        self._owns_scraper = scraper is None

    async def download(self, item: ScrapedItem) -> ProcessedItem | None:
        """Download, validate, hash, and store the image referenced by ``item``.

        Returns a :class:`ProcessedItem` on success or ``None`` if the
        item had no image, the fetch failed, or any validation step
        rejected the image.
        """
        if not item.image_url:
            return None

        data = await self._scraper.fetch_image(item.image_url)
        if data is None:
            return None
        if len(data) > self.max_size_bytes:
            logger.info(
                "image at %s exceeds max size (%d > %d bytes) — skipping",
                item.image_url, len(data), self.max_size_bytes,
            )
            return None

        decoded = self._decode_and_validate(data, item.image_url)
        if decoded is None:
            return None
        fmt, dimensions, phash = decoded

        # SHA-256 of the bytes is content-addressable: two scrapers
        # pulling the same hero image collapse to a single file on disk.
        digest = hashlib.sha256(data).hexdigest()
        path = self.storage_path / f"{digest}.{_FORMAT_EXT[fmt]}"
        if not path.exists():
            path.write_bytes(data)

        return ProcessedItem(
            source_url=item.source_url,
            image_url=item.image_url,
            title=item.title,
            text=item.text,
            timestamp=item.timestamp,
            source_domain=item.source_domain,
            metadata=dict(item.metadata),
            image_path=str(path),
            image_dimensions=dimensions,
            perceptual_hash=phash,
            file_size_bytes=len(data),
        )

    async def download_batch(
        self,
        items: list[ScrapedItem],
        max_concurrent: int = 10,
    ) -> list[ProcessedItem]:
        """Download many items concurrently, capped at ``max_concurrent`` in flight.

        Per-item failures (network errors, validation rejections, decode
        errors) are logged and skipped — losing one image must not abort
        a batch.
        """
        if max_concurrent <= 0:
            raise ValueError(
                f"max_concurrent must be positive, got {max_concurrent}"
            )

        sem = asyncio.Semaphore(max_concurrent)

        async def _bounded(it: ScrapedItem) -> ProcessedItem | None:
            async with sem:
                try:
                    return await self.download(it)
                except Exception as exc:  # noqa: BLE001 — keep batch alive
                    logger.exception(
                        "image download failed for %s: %s", it.image_url, exc
                    )
                    return None

        results = await asyncio.gather(*(_bounded(it) for it in items))
        return [r for r in results if r is not None]

    def deduplicate_by_phash(
        self,
        items: list[ProcessedItem],
        hamming_threshold: int = 4,
    ) -> list[ProcessedItem]:
        """Collapse near-duplicates by perceptual-hash distance.

        Two items are duplicates if their phashes differ by at most
        ``hamming_threshold`` bits — at the default of 4 (out of 64)
        that catches resizes, mild recompression, and most logo
        overlays without merging visually distinct images.

        First occurrence wins; ordering is otherwise preserved. The
        comparison is O(n²); fine at ingest scales (~thousands per
        batch). For larger sets, swap in a BK-tree later.
        """
        if hamming_threshold < 0:
            raise ValueError(
                f"hamming_threshold must be non-negative, got {hamming_threshold}"
            )

        kept: list[ProcessedItem] = []
        kept_hashes: list[imagehash.ImageHash] = []
        for item in items:
            try:
                ih = imagehash.hex_to_hash(item.perceptual_hash)
            except ValueError:
                logger.warning(
                    "skipping item with invalid perceptual_hash %r",
                    item.perceptual_hash,
                )
                continue
            if any((ih - prev) <= hamming_threshold for prev in kept_hashes):
                continue
            kept.append(item)
            kept_hashes.append(ih)
        return kept

    async def close(self) -> None:
        """Release the owned scraper's HTTP client, if we created it."""
        if self._owns_scraper:
            await self._scraper.close()

    async def __aenter__(self) -> "ImageDownloader":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    # --- helpers --------------------------------------------------------
    def _decode_and_validate(
        self, data: bytes, source: str
    ) -> tuple[str, tuple[int, int], str] | None:
        """Decode ``data`` with PIL and run format/dimension checks.

        Returns ``(format, (width, height), phash_hex)`` on success or
        ``None`` if the image fails any validation. The perceptual hash
        is computed here, not later, to avoid re-decoding the bytes.
        """
        try:
            with Image.open(BytesIO(data)) as img:
                img.load()
                fmt = (img.format or "").upper()
                if fmt not in _ALLOWED_FORMATS:
                    logger.info(
                        "image at %s has unsupported format %r — skipping",
                        source, fmt or "unknown",
                    )
                    return None

                width, height = img.size
                min_w, min_h = self.min_dimensions
                if width < min_w or height < min_h:
                    logger.info(
                        "image at %s below min dimensions (%dx%d < %dx%d) — skipping",
                        source, width, height, min_w, min_h,
                    )
                    return None

                phash = str(imagehash.phash(img))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            logger.warning("could not decode image at %s: %s", source, exc)
            return None

        return fmt, (width, height), phash
