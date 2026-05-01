from typing import Callable, Optional

import torch
from torch.utils.data import Dataset

from satira.training.curriculum import CurriculumScheduler


class SatireDataset(Dataset):
    """Image-text pairs with labels.

    Real implementation will load images from ``image_path`` and tokenize
    text. Until the data pipeline lands, ``__getitem__`` returns a random
    image tensor of the canonical shape so that downstream code (loaders,
    collators, training loops) can be exercised end-to-end against the
    final interface.
    """

    IMAGE_SHAPE = (3, 224, 224)

    def __init__(
        self,
        data_manifest: list[dict],
        transform: Optional[Callable] = None,
    ) -> None:
        self.data_manifest = data_manifest
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_manifest)

    def __getitem__(self, idx: int) -> dict:
        entry = self.data_manifest[idx]

        image = torch.randn(*self.IMAGE_SHAPE)
        if self.transform is not None:
            image = self.transform(image)

        text = entry.get("text", "")

        raw_label = entry.get("label", 0)
        if isinstance(raw_label, torch.Tensor):
            label = raw_label
        elif isinstance(raw_label, (int, bool)):
            label = torch.tensor(int(raw_label), dtype=torch.long)
        elif isinstance(raw_label, float):
            label = torch.tensor(raw_label, dtype=torch.float)
        else:
            label = raw_label

        metadata = {k: v for k, v in entry.items() if k not in ("text", "label")}

        return {
            "image": image,
            "text": text,
            "label": label,
            "metadata": metadata,
        }


class CurriculumDataLoader:
    """Builds mixed-tier batches according to the curriculum schedule.

    Each call to ``get_batch`` queries the scheduler for the epoch's tier
    weights and draws that proportion of samples from each tier dataset.
    Sampling is with replacement so an empty or small tier never blocks
    a batch.
    """

    TIER_KEYS = ("tier1_easy", "tier2_contradiction", "tier3_hard_negatives")

    def __init__(
        self,
        tier1: SatireDataset,
        tier2: SatireDataset,
        tier3: SatireDataset,
        scheduler: CurriculumScheduler,
        batch_size: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.tiers: dict[str, SatireDataset] = {
            "tier1_easy": tier1,
            "tier2_contradiction": tier2,
            "tier3_hard_negatives": tier3,
        }
        self.scheduler = scheduler
        self.batch_size = batch_size

    def _tier_counts(self, weights: dict[str, float]) -> dict[str, int]:
        counts = {k: int(round(weights[k] * self.batch_size)) for k in self.TIER_KEYS}
        diff = self.batch_size - sum(counts.values())
        if diff != 0:
            target = max(self.TIER_KEYS, key=lambda k: weights[k])
            counts[target] += diff
        return counts

    def get_batch(self, epoch: int) -> dict:
        weights = self.scheduler.get_tier_weights(epoch)
        counts = self._tier_counts(weights)

        samples: list[dict] = []
        tier_labels: list[str] = []
        for tier_name in self.TIER_KEYS:
            count = counts[tier_name]
            if count <= 0:
                continue
            dataset = self.tiers[tier_name]
            if len(dataset) == 0:
                raise ValueError(f"tier {tier_name} is empty but weight asks for {count} samples")
            indices = torch.randint(0, len(dataset), (count,)).tolist()
            for idx in indices:
                samples.append(dataset[idx])
                tier_labels.append(tier_name)

        images = torch.stack([s["image"] for s in samples])
        texts = [s["text"] for s in samples]
        metadata = [s["metadata"] for s in samples]

        labels_raw = [s["label"] for s in samples]
        if all(isinstance(lbl, torch.Tensor) for lbl in labels_raw):
            labels: object = torch.stack(labels_raw)
        else:
            labels = labels_raw

        return {
            "image": images,
            "text": texts,
            "label": labels,
            "metadata": metadata,
            "tier": tier_labels,
        }


def create_mock_datasets() -> tuple[SatireDataset, SatireDataset, SatireDataset]:
    """Three 100-sample datasets with random-but-shaped data, for tests."""
    tier1_manifest = [
        {
            "image_path": f"/mock/tier1/{i}.jpg",
            "text": f"easy authentic example {i}",
            "label": i % 2,
            "tier": "tier1_easy",
        }
        for i in range(100)
    ]
    tier2_manifest = [
        {
            "image_path": f"/mock/tier2/{i}.jpg",
            "text": f"topically coherent contradiction {i}",
            "original_text": f"original {i}",
            "label": 1,
            "difficulty": "hard",
            "tier": "tier2_contradiction",
        }
        for i in range(100)
    ]
    tier3_manifest = [
        {
            "image_path": f"/mock/tier3/{i}.jpg",
            "text": f"hard negative {i}",
            "label": i % 2,
            "tier": "tier3_hard_negatives",
        }
        for i in range(100)
    ]
    return (
        SatireDataset(tier1_manifest),
        SatireDataset(tier2_manifest),
        SatireDataset(tier3_manifest),
    )
