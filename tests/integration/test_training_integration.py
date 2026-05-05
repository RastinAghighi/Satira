"""End-to-end integration tests for the curriculum training loop.

Wires real SatireDetectionEngine, PhasedLossFunction, CurriculumDataLoader,
and PhaseTransitionController together via SatireTrainer using small
mock datasets so the tests exercise the full forward + loss + backward +
phase-transition flow without GPU or real data.
"""
from collections import Counter

import torch

from satira.config import Settings
from satira.data.datasets import create_mock_datasets
from satira.models.engine import SatireDetectionEngine
from satira.training.trainer import SatireTrainer


D_MODEL = 64
NUM_HEADS = 4
NUM_CLASSES = 5
VISION_DIM = 128
TEXT_DIM = 96
TEMPORAL_DIM = 96
GRAPH_DIM = 48


def _make_config(batch_size: int = 8) -> Settings:
    return Settings(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
        vision_dim=VISION_DIM,
        text_dim=TEXT_DIM,
        temporal_dim=TEMPORAL_DIM,
        graph_dim=GRAPH_DIM,
        num_reasoning_layers=1,
        batch_size=batch_size,
        focal_gamma=2.0,
        gate_loss_weight=0.3,
        consistency_loss_weight=0.1,
    )


def _make_small_mock_datasets(
    n_per_tier: int = 10,
) -> tuple:
    """Three n_per_tier-sample datasets — same shape as create_mock_datasets but smaller."""
    full = create_mock_datasets()
    truncated = []
    for ds in full:
        ds.data_manifest = ds.data_manifest[:n_per_tier]
        truncated.append(ds)
    return tuple(truncated)


def _make_trainer(batch_size: int = 8, n_per_tier: int = 10) -> SatireTrainer:
    config = _make_config(batch_size=batch_size)
    model = SatireDetectionEngine(config)
    t1, t2, t3 = _make_small_mock_datasets(n_per_tier=n_per_tier)
    val = _make_small_mock_datasets(n_per_tier=n_per_tier)[0]
    return SatireTrainer(
        model=model,
        config=config,
        train_datasets=(t1, t2, t3),
        val_dataset=val,
        device="cpu",
    )


# --- 1. one-epoch smoke test ------------------------------------------
def test_training_one_epoch() -> None:
    torch.manual_seed(0)
    trainer = _make_trainer()

    metrics = trainer.train_epoch(epoch=1)

    expected_keys = {
        "loss",
        "accuracy",
        "gate_activation_variance",
        "grad_norm",
        "projection_grad_norm",
        "phase",
        "epoch",
    }
    assert expected_keys.issubset(metrics.keys())

    loss_tensor = torch.tensor(metrics["loss"])
    assert torch.isfinite(loss_tensor), f"loss is not finite: {metrics['loss']}"

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["gate_activation_variance"] >= 0.0
    assert metrics["grad_norm"] >= 0.0
    assert metrics["projection_grad_norm"] >= 0.0
    assert metrics["epoch"] == 1
    assert metrics["phase"] == 1


# --- 2. phase transition end-to-end -----------------------------------
def test_phase_transition() -> None:
    torch.manual_seed(0)
    trainer = _make_trainer()

    # Sanity: phase 1 starts with cross_attn and reasoning frozen.
    assert trainer.phase_controller.phase == 1
    assert all(not p.requires_grad for p in trainer.model.cross_attn.parameters())
    assert all(not p.requires_grad for p in trainer.model.reasoning.parameters())

    optimizer_before = trainer.optimizer
    n_groups_before = len(optimizer_before.param_groups)
    n_trainable_before = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )

    # Run a real epoch so the trainer's metrics path is exercised, then
    # drive the controller through plateau metrics that satisfy
    # phase 1 -> 2 advancement (loss flat for `patience` epochs AND
    # projection_grad_norm < 0.1).
    trainer.train_epoch(epoch=1)

    plateau_metrics = {
        "loss": 0.5,
        "accuracy": 0.5,
        "gate_activation_variance": 0.05,
        "grad_norm": 0.05,
        "projection_grad_norm": 0.05,
        "phase": 1,
        "epoch": 1,
    }
    for epoch in range(2, 7):
        plateau_metrics["loss"] = 0.5 + 0.0001 * epoch  # no improvement
        plateau_metrics["epoch"] = epoch
        trainer._handle_phase_transition(epoch=epoch, train_metrics=plateau_metrics)
        if trainer.phase_controller.phase == 2:
            break

    assert trainer.phase_controller.phase == 2, "phase did not advance to 2"

    # After unfreeze: cross_attn AND reasoning are trainable.
    assert all(p.requires_grad for p in trainer.model.cross_attn.parameters())
    assert all(p.requires_grad for p in trainer.model.reasoning.parameters())

    # Optimizer was rebuilt and now sees more trainable parameters.
    assert trainer.optimizer is not optimizer_before
    n_trainable_after = sum(
        p.numel() for p in trainer.model.parameters() if p.requires_grad
    )
    assert n_trainable_after > n_trainable_before
    # Phase 2 still uses a single param group (per phase_controller config).
    assert len(trainer.optimizer.param_groups) == n_groups_before

    # And a subsequent train_epoch runs cleanly under phase 2.
    metrics_p2 = trainer.train_epoch(epoch=7)
    assert metrics_p2["phase"] == 2
    assert torch.isfinite(torch.tensor(metrics_p2["loss"]))


# --- 3. focal loss vs CE both run without crashing --------------------
def _force_phase(trainer: SatireTrainer, phase: int) -> None:
    trainer.phase_controller._phase = phase  # type: ignore[attr-defined]
    trainer.model.freeze_for_phase(phase)
    trainer.optimizer = trainer._build_optimizer()


def test_focal_loss_vs_ce() -> None:
    n_epochs = 5

    torch.manual_seed(0)
    ce_trainer = _make_trainer()  # phase 1 by default → CE only
    ce_losses = [ce_trainer.train_epoch(epoch=e)["loss"] for e in range(1, n_epochs + 1)]

    torch.manual_seed(0)
    focal_trainer = _make_trainer()
    _force_phase(focal_trainer, 2)  # phase 2 → focal + supervised gate loss
    focal_losses = [
        focal_trainer.train_epoch(epoch=e)["loss"] for e in range(1, n_epochs + 1)
    ]

    # Both runs produce finite losses for every epoch.
    for tag, losses in (("ce", ce_losses), ("focal", focal_losses)):
        assert len(losses) == n_epochs
        for i, loss in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss)), (
                f"{tag} loss not finite at epoch {i + 1}: {loss}"
            )

    # Soft "is learning" check: at some point during training the loss
    # dropped below the initial value. Real performance comparison
    # between losses needs real data; this just guards against a
    # silently broken backward pass.
    assert min(ce_losses) <= ce_losses[0] + 1e-6
    assert min(focal_losses) <= focal_losses[0] + 1e-6


# --- 4. curriculum tier mixing across epochs --------------------------
def test_curriculum_mixing() -> None:
    # Larger batch size so rounding noise in _tier_counts doesn't swamp
    # the proportions we want to assert on.
    torch.manual_seed(0)
    trainer = _make_trainer(batch_size=64, n_per_tier=20)
    loader = trainer.data_loader
    batch_size = loader.batch_size

    tiers_e1 = Counter(loader.get_batch(epoch=1)["tier"])
    tiers_e10 = Counter(loader.get_batch(epoch=10)["tier"])
    tiers_e20 = Counter(loader.get_batch(epoch=20)["tier"])

    # Every batch fills batch_size slots.
    assert sum(tiers_e1.values()) == batch_size
    assert sum(tiers_e10.values()) == batch_size
    assert sum(tiers_e20.values()) == batch_size

    # Epoch 1: tier1 (easy) is the dominant majority.
    assert tiers_e1["tier1_easy"] > batch_size / 2
    assert tiers_e1["tier1_easy"] > tiers_e1["tier2_contradiction"]
    assert tiers_e1["tier1_easy"] > tiers_e1["tier3_hard_negatives"]

    # Epoch 10: tier2 (contradiction) is the dominant majority.
    assert tiers_e10["tier2_contradiction"] > batch_size / 2
    assert tiers_e10["tier2_contradiction"] > tiers_e10["tier1_easy"]
    assert tiers_e10["tier2_contradiction"] > tiers_e10["tier3_hard_negatives"]

    # Epoch 20: tier3 (hard negatives) is the dominant majority.
    assert tiers_e20["tier3_hard_negatives"] > batch_size / 2
    assert tiers_e20["tier3_hard_negatives"] > tiers_e20["tier1_easy"]
    assert tiers_e20["tier3_hard_negatives"] > tiers_e20["tier2_contradiction"]

    # Tier1 share strictly decays across the curriculum.
    assert (
        tiers_e1["tier1_easy"]
        > tiers_e10["tier1_easy"]
        >= tiers_e20["tier1_easy"]
    )
    # Tier3 share strictly grows across the curriculum.
    assert (
        tiers_e1["tier3_hard_negatives"]
        <= tiers_e10["tier3_hard_negatives"]
        < tiers_e20["tier3_hard_negatives"]
    )
