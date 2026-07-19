import json
from pathlib import Path

import pytest
import torch

from satira.data import corpus_dataset as cd


# --- helpers ----------------------------------------------------------------
def _write_manifest(root: Path, rows: list[dict]) -> Path:
    (root / "data" / "corpus" / "images").mkdir(parents=True, exist_ok=True)
    manifest = root / "data" / "corpus" / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return manifest


def _touch_image(root: Path, name: str) -> str:
    rel = f"data\\corpus\\images\\{name}"  # Windows-style, as the real manifest stores
    img_dir = root / "data" / "corpus" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / name).write_bytes(b"not-a-real-image")
    return rel


def _row(root: Path, *, url: str, label: str, image: str | None, title="T", text="body text here") -> dict:
    return {
        "source_url": url,
        "image_path": _touch_image(root, image) if image else None,
        "title": title,
        "text": text,
        "label": label,
        "source": "example.com",
    }


def _write_cache(cache_dir: Path, item_id: str, text_len: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "vision": torch.randn(cd.VISION_TOKENS, cd.VISION_DIM, dtype=torch.float16),
        "text": torch.randn(text_len, cd.TEXT_DIM, dtype=torch.float16),
        "text_len": text_len,
    }
    torch.save(payload, cache_dir / f"{item_id}.pt")


# --- item id / resolution ---------------------------------------------------
def test_item_id_is_deterministic_and_url_derived() -> None:
    row = {"source_url": "https://x.test/a", "image_path": "p.jpg"}
    a = cd.item_id_for_row(row)
    assert a == cd.item_id_for_row(dict(row))
    assert len(a) == 16
    assert a != cd.item_id_for_row({"source_url": "https://x.test/b"})


def test_resolve_image_path_handles_backslashes_and_missing(tmp_path: Path) -> None:
    rel = _touch_image(tmp_path, "img.jpg")
    assert cd.resolve_image_path({"image_path": rel}, tmp_path) is not None
    assert cd.resolve_image_path({"image_path": "data\\corpus\\images\\nope.jpg"}, tmp_path) is None
    assert cd.resolve_image_path({"image_path": None}, tmp_path) is None


def test_item_text_prefers_body_and_combines() -> None:
    assert cd.item_text({"title": "H", "text": "B"}) == "H\n\nB"
    assert cd.item_text({"title": "H", "text": ""}) == "H"
    assert cd.item_text({"title": "", "text": "B"}) == "B"


# --- loading / filtering ----------------------------------------------------
def test_load_corpus_items_filters_and_maps_labels(tmp_path: Path) -> None:
    rows = [
        _row(tmp_path, url="https://x/1", label="authentic", image="a.jpg"),
        _row(tmp_path, url="https://x/2", label="satire", image="b.jpg"),
        _row(tmp_path, url="https://x/3", label="satire", image=None),  # no image -> dropped
        _row(tmp_path, url="https://x/4", label="nonsense", image="d.jpg"),  # bad label -> dropped
        _row(tmp_path, url="https://x/5", label="parody", image="e.jpg"),  # alias -> satire
    ]
    manifest = _write_manifest(tmp_path, rows)

    items = cd.load_corpus_items(manifest, tmp_path)
    labels_seen = sorted(it.label for it in items)
    assert labels_seen == [0, 1, 1]  # authentic + satire + parody(->satire)

    binary = cd.load_corpus_items(manifest, tmp_path, allowed_labels={0, 1})
    assert len(binary) == 3
    assert all(it.label in (0, 1) for it in binary)


# --- stratified split -------------------------------------------------------
def _corpus_items(n_auth: int, n_sat: int) -> list[cd.CorpusItem]:
    items = []
    for i in range(n_auth):
        items.append(cd.CorpusItem(f"auth{i:03d}", 0, "authentic", "s", "p", "t"))
    for i in range(n_sat):
        items.append(cd.CorpusItem(f"sat{i:03d}", 1, "satire", "s", "p", "t"))
    return items


def test_assign_splits_is_stratified_and_persistent(tmp_path: Path) -> None:
    split_path = tmp_path / "splits.json"
    items = _corpus_items(100, 40)

    assignments = cd.assign_splits(items, seed=42, split_path=split_path)
    assert split_path.is_file()

    counts = cd.split_label_counts(items, assignments)
    # 80/10/10 per class: authentic 80/10/10, satire 32/4/4.
    assert counts["train"]["authentic"] == 80
    assert counts["val"]["authentic"] == 10
    assert counts["test"]["authentic"] == 10
    assert counts["train"]["satire"] == 32
    assert sum(counts[s]["satire"] for s in ("train", "val", "test")) == 40


def test_assign_splits_keeps_existing_membership_stable(tmp_path: Path) -> None:
    split_path = tmp_path / "splits.json"
    items = _corpus_items(50, 20)

    first = cd.assign_splits(items, seed=1, split_path=split_path)
    # Re-run with a different seed and extra items; existing ids must not move.
    more = items + _corpus_items(10, 5)[-15:]
    second = cd.assign_splits(more, seed=999, split_path=split_path)
    for item in items:
        assert first[item.item_id] == second[item.item_id]


# --- cache round-trip + collate --------------------------------------------
def test_cache_round_trip_and_dtype(tmp_path: Path) -> None:
    cache = tmp_path / "emb"
    item = cd.CorpusItem("id0", 1, "satire", "src", "img.jpg", "text")
    _write_cache(cache, "id0", text_len=17)

    ds = cd.CorpusEmbeddingDataset([item], cache, dtype=torch.float32)
    sample = ds[0]

    assert sample["vision"].shape == (cd.VISION_TOKENS, cd.VISION_DIM)
    assert sample["text"].shape == (17, cd.TEXT_DIM)
    assert sample["text_len"] == 17
    assert sample["vision"].dtype == torch.float32  # cast up from fp16 on load
    assert sample["label"].item() == 1
    assert sample["has_temporal"] is False and sample["has_graph"] is False


def test_dataset_raises_on_missing_cache(tmp_path: Path) -> None:
    ds = cd.CorpusEmbeddingDataset(
        [cd.CorpusItem("ghost", 0, "authentic", "s", "p", "t")], tmp_path / "emb"
    )
    with pytest.raises(FileNotFoundError):
        _ = ds[0]


def test_collate_pads_text_and_builds_mask(tmp_path: Path) -> None:
    cache = tmp_path / "emb"
    items = [
        cd.CorpusItem("a", 0, "authentic", "s", "p", "t"),
        cd.CorpusItem("b", 1, "satire", "s", "p", "t"),
    ]
    _write_cache(cache, "a", text_len=5)
    _write_cache(cache, "b", text_len=12)
    ds = cd.CorpusEmbeddingDataset(items, cache)

    batch = cd.collate_embeddings([ds[0], ds[1]])

    assert batch["vision"].shape == (2, cd.VISION_TOKENS, cd.VISION_DIM)
    assert batch["text"].shape == (2, 12, cd.TEXT_DIM)  # padded to batch max
    mask = batch["text_key_padding_mask"]
    assert mask.shape == (2, 12)
    # Row 0 has 5 real tokens -> 7 padded; row 1 is full.
    assert mask[0, :5].sum().item() == 0 and mask[0, 5:].all()
    assert not mask[1].any()
    # Padded positions of the text tensor are zero-filled.
    assert torch.count_nonzero(batch["text"][0, 5:]) == 0
    assert not batch["temporal_present"].any() and not batch["graph_present"].any()
    assert batch["label"].tolist() == [0, 1]


def test_collate_rejects_missing_vision() -> None:
    sample = {
        "item_id": "x",
        "vision": None,
        "text": torch.randn(3, cd.TEXT_DIM),
        "text_len": 3,
        "label": torch.tensor(0),
        "has_temporal": False,
        "has_graph": False,
        "metadata": {},
    }
    with pytest.raises(ValueError):
        cd.collate_embeddings([sample])
