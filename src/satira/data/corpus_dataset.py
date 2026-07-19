"""Real-corpus dataset: manifest -> cached CLIP/RoBERTa embeddings.

Reads ``data/corpus/manifest.jsonl``, keeps only items whose image file exists
on disk, maps labels through the canonical taxonomy (:mod:`satira.labels`), and
serves the precomputed embedding tensors written by
``scripts/precompute_embeddings.py``.

A stratified train/val/test split is persisted to JSON keyed by item id so
membership is stable across runs — existing items never move, and nightly
additions are assigned into the split without disturbing what came before.

The per-item cache file (``{item_id}.pt``) is a dict::

    {"vision": FloatTensor(257, 1024) | None, "text": FloatTensor(T, 768), "text_len": int}

stored fp16; tensors are cast to the requested dtype on load.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

from satira import labels

# This file is src/satira/data/corpus_dataset.py; the repo root is three
# parents up (data -> satira -> src -> <root>... actually src is parents[2]).
REPO_ROOT = Path(__file__).resolve().parents[3]

# Cache namespace: the vision + text encoder pair defines the embedding format,
# so a different encoder writes to a different directory / split file.
EMBEDDING_VERSION = "clipL14_robertabase"

DEFAULT_MANIFEST = REPO_ROOT / "data" / "corpus" / "manifest.jsonl"
DEFAULT_EMBEDDING_DIR = REPO_ROOT / "data" / "embeddings" / EMBEDDING_VERSION
DEFAULT_SPLIT_PATH = REPO_ROOT / "data" / "corpus" / f"splits_{EMBEDDING_VERSION}.json"

# Encoder output shapes, matching config.vision_dim / config.text_dim.
VISION_TOKENS = 257
VISION_DIM = 1024
TEXT_DIM = 768
MAX_TEXT_TOKENS = 256

_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class CorpusItem:
    """One image-bearing manifest row, resolved and label-mapped."""

    item_id: str
    label: int
    label_str: str
    source: str
    image_path: str
    text: str


def item_id_for_row(row: dict) -> str:
    """Stable, filesystem-safe id for a manifest row.

    Derived from ``source_url`` — the article's canonical identity and the most
    stable field across re-scrapes — falling back to ``image_path`` if a row
    somehow lacks a URL. 16 hex chars (64 bits) is collision-safe for a corpus
    of this size.
    """
    key = (row.get("source_url") or "").strip() or (row.get("image_path") or "").strip()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def resolve_image_path(row: dict, repo_root: Path = REPO_ROOT) -> Path | None:
    """Return the on-disk image path for a row, or None if it doesn't exist.

    The manifest stores Windows-style relative paths (``data\\corpus\\images\\..``);
    they are normalized and tried both as-is and relative to the repo root.
    """
    raw = row.get("image_path")
    if not raw:
        return None
    norm = str(raw).replace("\\", "/")
    for candidate in (Path(norm), repo_root / norm):
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def item_text(row: dict) -> str:
    """Best text surface for encoding: headline and body joined when both exist.

    The body sometimes repeats the headline; the redundancy is harmless because
    the text encoder truncates to a fixed length anyway.
    """
    title = (row.get("title") or "").strip()
    body = (row.get("text") or "").strip()
    if title and body:
        return f"{title}\n\n{body}"
    return title or body


def load_corpus_items(
    manifest_path: Path = DEFAULT_MANIFEST,
    repo_root: Path = REPO_ROOT,
    *,
    allowed_labels: Iterable[int] | None = None,
) -> list[CorpusItem]:
    """Load image-bearing, label-mappable items from the manifest.

    Filters out rows with no on-disk image, an unmappable/missing label, empty
    text, or (when ``allowed_labels`` is given) a label outside that set. Items
    are de-duplicated by id; input order is preserved otherwise.
    """
    allowed = set(allowed_labels) if allowed_labels is not None else None
    items: list[CorpusItem] = []
    seen: set[str] = set()
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if resolve_image_path(row, repo_root) is None:
                continue
            label_str = row.get("label")
            if not isinstance(label_str, str):
                continue
            try:
                label = labels.str_to_int(label_str)
            except ValueError:
                continue
            if allowed is not None and label not in allowed:
                continue
            text = item_text(row)
            if not text:
                continue
            item_id = item_id_for_row(row)
            if item_id in seen:
                continue
            seen.add(item_id)
            items.append(
                CorpusItem(
                    item_id=item_id,
                    label=label,
                    label_str=labels.int_to_str(label),
                    source=row.get("source") or "",
                    image_path=str(resolve_image_path(row, repo_root)),
                    text=text,
                )
            )
    return items


def assign_splits(
    items: Sequence[CorpusItem],
    *,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    split_path: Path = DEFAULT_SPLIT_PATH,
) -> dict[str, str]:
    """Stratified train/val/test assignment, persisted to JSON for stability.

    Any id already recorded in ``split_path`` keeps its assignment; only items
    new to the split are placed, stratified by label so each split preserves the
    class ratio. The (union) assignment map is written back so it stays stable as
    the corpus grows. Returns ``{item_id: split}`` for the current items.
    """
    if not 0 < train_frac < 1 or not 0 <= val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(
            f"invalid split fractions: train={train_frac}, val={val_frac} "
            "(need 0<train<1, 0<=val<1, train+val<1)"
        )

    stored: dict[str, str] = {}
    if split_path.exists():
        stored = json.loads(split_path.read_text(encoding="utf-8"))

    assignments = dict(stored)
    new_by_label: dict[int, list[str]] = defaultdict(list)
    for item in items:
        if item.item_id not in assignments:
            new_by_label[item.label].append(item.item_id)

    rng = random.Random(seed)
    for label in sorted(new_by_label):
        ids = sorted(new_by_label[label])  # deterministic base order before shuffle
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        for i, item_id in enumerate(ids):
            if i < n_train:
                assignments[item_id] = "train"
            elif i < n_train + n_val:
                assignments[item_id] = "val"
            else:
                assignments[item_id] = "test"

    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(assignments, indent=2, sort_keys=True), encoding="utf-8")

    return {item.item_id: assignments[item.item_id] for item in items}


def split_items(
    items: Sequence[CorpusItem], assignments: dict[str, str]
) -> dict[str, list[CorpusItem]]:
    """Partition items into ``{"train": [...], "val": [...], "test": [...]}``."""
    out: dict[str, list[CorpusItem]] = {name: [] for name in _SPLITS}
    for item in items:
        out[assignments[item.item_id]].append(item)
    return out


def split_label_counts(
    items: Sequence[CorpusItem], assignments: dict[str, str]
) -> dict[str, Counter]:
    """Per-split, per-class-name counts."""
    counts = {name: Counter() for name in _SPLITS}
    for item in items:
        counts[assignments[item.item_id]][item.label_str] += 1
    return counts


def print_split_summary(items: Sequence[CorpusItem], assignments: dict[str, str]) -> None:
    """Print resulting per-class counts per split (plain ASCII for cp1252)."""
    counts = split_label_counts(items, assignments)
    all_labels = sorted({item.label_str for item in items})
    print(f"=== corpus split ({len(items)} image-bearing items) ===")
    header = "  ".join(f"{name:>18s}" for name in all_labels)
    print(f"  {'split':6s} {'n':>5s}  {header}")
    for name in _SPLITS:
        row = counts[name]
        total = sum(row.values())
        cells = "  ".join(f"{row.get(lbl, 0):>18d}" for lbl in all_labels)
        print(f"  {name:6s} {total:>5d}  {cells}")


class CorpusEmbeddingDataset(Dataset):
    """Serves cached CLIP/RoBERTa embeddings for a list of :class:`CorpusItem`.

    ``__getitem__`` returns the per-item tensors (not a padded batch); padding to
    the batch maximum and the key-padding mask are built by
    :func:`collate_embeddings`.
    """

    def __init__(
        self,
        items: Sequence[CorpusItem],
        embedding_dir: Path = DEFAULT_EMBEDDING_DIR,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.items = list(items)
        self.embedding_dir = Path(embedding_dir)
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.items)

    def cache_path(self, item_id: str) -> Path:
        return self.embedding_dir / f"{item_id}.pt"

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        path = self.cache_path(item.item_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"missing embedding cache for item {item.item_id} at {path}; "
                "run scripts/precompute_embeddings.py first"
            )
        cached = torch.load(path, map_location="cpu", weights_only=True)
        vision = cached.get("vision")
        if vision is not None:
            vision = vision.to(self.dtype)
        text = cached["text"].to(self.dtype)
        return {
            "item_id": item.item_id,
            "vision": vision,
            "text": text,
            "text_len": int(cached["text_len"]),
            "label": torch.tensor(item.label, dtype=torch.long),
            # Every item in this corpus is temporal/graph cold-start.
            "has_temporal": False,
            "has_graph": False,
            "metadata": {
                "source": item.source,
                "label_str": item.label_str,
                "image_path": item.image_path,
            },
        }


def collate_embeddings(samples: Sequence[dict]) -> dict:
    """Collate per-item embedding dicts into a padded batch.

    Vision is fixed-length so it stacks directly. Text is zero-padded to the
    batch's longest sequence, with ``text_key_padding_mask`` (True == padding)
    marking the pad positions. ``temporal_present`` / ``graph_present`` carry the
    per-item cold-start flags through to the engine.
    """
    if not samples:
        raise ValueError("collate_embeddings received an empty sample list")

    if any(s["vision"] is None for s in samples):
        raise ValueError(
            "collate_embeddings requires vision embeddings for every item "
            "(this run is image-bearing only)"
        )

    batch_size = len(samples)
    vision = torch.stack([s["vision"] for s in samples])

    text_lens = [int(s["text_len"]) for s in samples]
    t_max = max(text_lens)
    text_dim = samples[0]["text"].size(-1)
    dtype = samples[0]["text"].dtype

    text = torch.zeros(batch_size, t_max, text_dim, dtype=dtype)
    mask = torch.ones(batch_size, t_max, dtype=torch.bool)  # True == pad
    for i, sample in enumerate(samples):
        length = int(sample["text_len"])
        text[i, :length] = sample["text"][:length]
        mask[i, :length] = False

    return {
        "vision": vision,
        "text": text,
        "text_key_padding_mask": mask,
        "temporal_present": torch.tensor([bool(s["has_temporal"]) for s in samples]),
        "graph_present": torch.tensor([bool(s["has_graph"]) for s in samples]),
        "label": torch.stack([s["label"] for s in samples]),
        "item_id": [s["item_id"] for s in samples],
        "metadata": [s["metadata"] for s in samples],
        "text_len": text_lens,
    }
