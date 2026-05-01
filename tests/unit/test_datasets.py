import torch

from satira.data.datasets import (
    CurriculumDataLoader,
    SatireDataset,
    create_mock_datasets,
)
from satira.training.curriculum import CurriculumScheduler


REQUIRED_KEYS = {"image", "text", "label", "metadata"}
TIER_KEYS = ("tier1_easy", "tier2_contradiction", "tier3_hard_negatives")


def test_satire_dataset_returns_required_keys() -> None:
    manifest = [{"image_path": "/x.jpg", "text": "headline", "label": 1}]
    ds = SatireDataset(manifest)

    item = ds[0]

    assert REQUIRED_KEYS.issubset(item.keys())
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == SatireDataset.IMAGE_SHAPE
    assert item["text"] == "headline"
    assert isinstance(item["label"], torch.Tensor)
    assert int(item["label"]) == 1
    assert isinstance(item["metadata"], dict)


def test_satire_dataset_length_matches_manifest() -> None:
    manifest = [
        {"image_path": f"/{i}.jpg", "text": f"t{i}", "label": 0} for i in range(7)
    ]
    ds = SatireDataset(manifest)
    assert len(ds) == 7


def test_satire_dataset_metadata_contains_non_consumed_fields() -> None:
    manifest = [
        {
            "image_path": "/p.jpg",
            "text": "hi",
            "label": 0,
            "difficulty": "hard",
            "source": "synthetic",
        }
    ]
    ds = SatireDataset(manifest)
    item = ds[0]
    assert item["metadata"]["image_path"] == "/p.jpg"
    assert item["metadata"]["difficulty"] == "hard"
    assert item["metadata"]["source"] == "synthetic"
    assert "text" not in item["metadata"]
    assert "label" not in item["metadata"]


def test_satire_dataset_applies_transform() -> None:
    manifest = [{"image_path": "/x.jpg", "text": "t", "label": 0}]
    transform_called = {"flag": False}

    def transform(img: torch.Tensor) -> torch.Tensor:
        transform_called["flag"] = True
        return torch.zeros_like(img)

    ds = SatireDataset(manifest, transform=transform)
    item = ds[0]
    assert transform_called["flag"]
    assert torch.equal(item["image"], torch.zeros(*SatireDataset.IMAGE_SHAPE))


def test_create_mock_datasets_have_expected_sizes() -> None:
    t1, t2, t3 = create_mock_datasets()
    assert len(t1) == 100
    assert len(t2) == 100
    assert len(t3) == 100


def test_create_mock_datasets_yield_correctly_shaped_items() -> None:
    t1, t2, t3 = create_mock_datasets()
    for ds in (t1, t2, t3):
        item = ds[0]
        assert REQUIRED_KEYS.issubset(item.keys())
        assert item["image"].shape == SatireDataset.IMAGE_SHAPE
        assert isinstance(item["text"], str)
        assert isinstance(item["label"], torch.Tensor)


def test_curriculum_loader_produces_full_batch() -> None:
    t1, t2, t3 = create_mock_datasets()
    scheduler = CurriculumScheduler(total_epochs=25)
    loader = CurriculumDataLoader(t1, t2, t3, scheduler, batch_size=64)

    batch = loader.get_batch(epoch=10)

    assert batch["image"].shape == (64, *SatireDataset.IMAGE_SHAPE)
    assert len(batch["text"]) == 64
    assert len(batch["metadata"]) == 64
    assert len(batch["tier"]) == 64


def test_curriculum_loader_tier_one_dominates_early_epoch() -> None:
    t1, t2, t3 = create_mock_datasets()
    scheduler = CurriculumScheduler(total_epochs=25)
    loader = CurriculumDataLoader(t1, t2, t3, scheduler, batch_size=64)

    batch = loader.get_batch(epoch=1)
    counts = {k: batch["tier"].count(k) for k in TIER_KEYS}

    assert counts["tier1_easy"] > counts["tier2_contradiction"]
    assert counts["tier1_easy"] > counts["tier3_hard_negatives"]


def test_curriculum_loader_tier_three_dominates_final_epoch() -> None:
    t1, t2, t3 = create_mock_datasets()
    scheduler = CurriculumScheduler(total_epochs=25)
    loader = CurriculumDataLoader(t1, t2, t3, scheduler, batch_size=64)

    batch = loader.get_batch(epoch=25)
    counts = {k: batch["tier"].count(k) for k in TIER_KEYS}

    assert counts["tier3_hard_negatives"] > counts["tier1_easy"]
    assert counts["tier3_hard_negatives"] > counts["tier2_contradiction"]


def test_curriculum_loader_counts_match_scheduler_weights() -> None:
    t1, t2, t3 = create_mock_datasets()
    scheduler = CurriculumScheduler(total_epochs=25)
    batch_size = 64
    loader = CurriculumDataLoader(t1, t2, t3, scheduler, batch_size=batch_size)

    for epoch in (1, 10, 25):
        weights = scheduler.get_tier_weights(epoch)
        batch = loader.get_batch(epoch=epoch)
        counts = {k: batch["tier"].count(k) for k in TIER_KEYS}

        assert sum(counts.values()) == batch_size
        for tier in TIER_KEYS:
            expected = weights[tier] * batch_size
            assert abs(counts[tier] - expected) <= 1.5, (
                f"epoch {epoch} tier {tier}: count {counts[tier]} too far from expected {expected:.2f}"
            )


def test_curriculum_loader_rejects_non_positive_batch_size() -> None:
    t1, t2, t3 = create_mock_datasets()
    scheduler = CurriculumScheduler(total_epochs=25)
    try:
        CurriculumDataLoader(t1, t2, t3, scheduler, batch_size=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for batch_size=0")
