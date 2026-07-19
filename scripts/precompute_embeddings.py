"""Precompute CLIP/RoBERTa embeddings for the image-bearing corpus.

For every image-bearing manifest item this writes one cache file
``data/embeddings/clipL14_robertabase/{item_id}.pt`` containing::

    {"vision": FloatTensor(257, 1024), "text": FloatTensor(T, 768), "text_len": int}

stored fp16 (T is the real RoBERTa token length, capped at 256, kept for
masking). Encoding is incremental: an item whose cache file already exists is
skipped, so nightly corpus additions only pay for the new items.

* Vision: ``CLIPVisionModel`` "openai/clip-vit-large-patch14" — the
  ``last_hidden_state`` patch sequence (257 tokens x 1024).
* Text: ``RobertaModel`` "roberta-base" — the ``last_hidden_state`` (T x 768).

CUDA is used when available; otherwise the work is batched for CPU and simply
takes the time it takes. Reports items encoded, wall time, and cache size.

Run ``python scripts/precompute_embeddings.py --help`` for options.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satira.data.corpus_dataset import (  # noqa: E402
    DEFAULT_EMBEDDING_DIR,
    DEFAULT_MANIFEST,
    MAX_TEXT_TOKENS,
    CorpusItem,
    load_corpus_items,
)

logger = logging.getLogger("satira.precompute")

VISION_MODEL_ID = "openai/clip-vit-large-patch14"
TEXT_MODEL_ID = "roberta-base"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute CLIP/RoBERTa embeddings.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Items per encode batch (default: 8; raise on GPU).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device for the encoders. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Encode at most this many (uncached) items — for smoke tests.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


def pick_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoders:
    """Lazily-loaded CLIP vision + RoBERTa text encoders."""

    def __init__(self, device: torch.device) -> None:
        from transformers import (  # imported here so --help stays fast
            AutoTokenizer,
            CLIPImageProcessor,
            CLIPVisionModel,
            RobertaModel,
        )

        self.device = device
        logger.info("loading vision encoder %s", VISION_MODEL_ID)
        self.image_processor = CLIPImageProcessor.from_pretrained(VISION_MODEL_ID)
        self.vision_model = CLIPVisionModel.from_pretrained(VISION_MODEL_ID).to(device).eval()

        logger.info("loading text encoder %s", TEXT_MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        self.text_model = RobertaModel.from_pretrained(TEXT_MODEL_ID).to(device).eval()

    @torch.no_grad()
    def encode_vision(self, images: list[Image.Image]) -> torch.Tensor:
        """(B, 257, 1024) patch sequence on CPU."""
        inputs = self.image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        out = self.vision_model(pixel_values=pixel_values)
        return out.last_hidden_state.detach().to("cpu")

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ((B, T, 768) last_hidden_state, (B,) real token lengths) on CPU."""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_TOKENS,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        lengths = attention_mask.sum(dim=1).detach().to("cpu")
        return out.last_hidden_state.detach().to("cpu"), lengths


def _load_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _cache_size_bytes(embedding_dir: Path) -> int:
    return sum(p.stat().st_size for p in embedding_dir.glob("*.pt"))


def _fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def encode_corpus(
    items: list[CorpusItem],
    embedding_dir: Path,
    encoders: Encoders,
    *,
    batch_size: int,
) -> tuple[int, int]:
    """Encode all ``items`` lacking a cache file. Returns (encoded, failed)."""
    embedding_dir.mkdir(parents=True, exist_ok=True)
    pending = [it for it in items if not (embedding_dir / f"{it.item_id}.pt").is_file()]
    logger.info(
        "%d items total, %d already cached, %d to encode",
        len(items),
        len(items) - len(pending),
        len(pending),
    )

    encoded = 0
    failed = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]

        loaded: list[tuple[CorpusItem, Image.Image]] = []
        for item in batch:
            try:
                loaded.append((item, _load_image(item.image_path)))
            except Exception as exc:  # noqa: BLE001 — one bad image shouldn't stop the run
                logger.warning("failed to load image for %s: %s", item.item_id, exc)
                failed += 1
        if not loaded:
            continue

        batch_items = [it for it, _ in loaded]
        images = [img for _, img in loaded]
        texts = [it.text for it in batch_items]

        vision = encoders.encode_vision(images)
        text_hidden, text_lengths = encoders.encode_text(texts)

        for i, item in enumerate(batch_items):
            length = int(text_lengths[i].item())
            payload = {
                "vision": vision[i].to(torch.float16).contiguous(),
                "text": text_hidden[i, :length].to(torch.float16).contiguous(),
                "text_len": length,
            }
            torch.save(payload, embedding_dir / f"{item.item_id}.pt")
            encoded += 1

        logger.info("encoded %d/%d", encoded, len(pending))

    return encoded, failed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = pick_device(args.device)
    logger.info("device=%s torch_threads=%d", device, torch.get_num_threads())

    items = load_corpus_items(manifest_path=args.manifest)
    if args.limit is not None:
        # Limit uncached items encoded this pass without disturbing item identity.
        cached = {it.item_id for it in items if (args.embedding_dir / f"{it.item_id}.pt").is_file()}
        pending = [it for it in items if it.item_id not in cached][: args.limit]
        items = [it for it in items if it.item_id in cached or it in pending]

    encoders = Encoders(device)

    start = time.perf_counter()
    encoded, failed = encode_corpus(
        items, args.embedding_dir, encoders, batch_size=args.batch_size
    )
    wall = time.perf_counter() - start

    size = _cache_size_bytes(args.embedding_dir)
    total_cached = len(list(args.embedding_dir.glob("*.pt")))
    print("\n=== precompute summary ===")
    print(f"  items encoded this run : {encoded}")
    print(f"  items failed           : {failed}")
    print(f"  total cached files     : {total_cached}")
    print(f"  wall time              : {wall:.1f}s")
    if encoded:
        print(f"  per-item               : {wall / encoded:.2f}s")
    print(f"  cache size             : {_fmt_bytes(size)}")
    print(f"  cache dir              : {args.embedding_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
