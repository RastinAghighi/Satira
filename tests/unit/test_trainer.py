import torch

from satira.config import Settings
from satira.data.datasets import create_mock_datasets
from satira.models.engine import SatireDetectionEngine
from satira.training.trainer import SatireTrainer


def _small_config() -> Settings:
    return Settings(
        d_model=32,
        num_heads=4,
        num_classes=5,
        vision_dim=64,
        text_dim=48,
        temporal_dim=48,
        graph_dim=24,
        num_reasoning_layers=1,
        batch_size=8,
        focal_gamma=2.0,
        gate_loss_weight=0.3,
        consistency_loss_weight=0.1,
    )


def _make_trainer() -> SatireTrainer:
    config = _small_config()
    model = SatireDetectionEngine(config)
    t1, t2, t3 = create_mock_datasets()
    val = create_mock_datasets()[0]
    return SatireTrainer(
        model=model,
        config=config,
        train_datasets=(t1, t2, t3),
        val_dataset=val,
        device="cpu",
    )


def test_trainer_initializes_without_error() -> None:
    trainer = _make_trainer()
    assert trainer.phase_controller.phase == 1
    assert trainer.optimizer is not None
    assert trainer.data_loader.batch_size == 8
    assert trainer.device.type == "cpu"


def test_trainer_rejects_wrong_train_dataset_count() -> None:
    config = _small_config()
    model = SatireDetectionEngine(config)
    t1, t2, _ = create_mock_datasets()
    val = create_mock_datasets()[0]
    try:
        SatireTrainer(
            model=model,
            config=config,
            train_datasets=(t1, t2),  # type: ignore[arg-type]
            val_dataset=val,
            device="cpu",
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong tuple length")


def test_train_epoch_returns_expected_metric_keys() -> None:
    trainer = _make_trainer()
    metrics = trainer.train_epoch(epoch=1)
    expected = {
        "loss",
        "accuracy",
        "gate_activation_variance",
        "grad_norm",
        "projection_grad_norm",
        "phase",
        "epoch",
    }
    assert expected.issubset(metrics.keys())
    assert metrics["epoch"] == 1
    assert metrics["phase"] == 1
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_validate_returns_expected_keys() -> None:
    trainer = _make_trainer()
    metrics = trainer.validate()
    assert {"loss", "accuracy", "calibration_error", "f1_per_class"}.issubset(metrics.keys())
    assert len(metrics["f1_per_class"]) == trainer.config.num_classes
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["calibration_error"] <= 1.0


def test_train_epoch_updates_parameters() -> None:
    trainer = _make_trainer()
    before = {
        name: p.detach().clone()
        for name, p in trainer.model.named_parameters()
        if p.requires_grad
    }
    trainer.train_epoch(epoch=1)
    changed_any = False
    for name, p in trainer.model.named_parameters():
        if not p.requires_grad:
            continue
        if not torch.allclose(before[name], p.detach()):
            changed_any = True
            break
    assert changed_any, "no trainable parameter changed after one training step"


def test_phase_transition_triggers_freezing_and_optimizer_rebuild() -> None:
    trainer = _make_trainer()
    assert trainer.phase_controller.phase == 1
    optimizer_before = trainer.optimizer

    plateau_metrics = {
        "loss": 0.5,
        "accuracy": 0.5,
        "gate_activation_variance": 0.05,
        "grad_norm": 0.1,
        "projection_grad_norm": 0.05,
        "phase": 1,
        "epoch": 1,
    }

    trainer._handle_phase_transition(epoch=1, train_metrics=plateau_metrics)
    plateau_metrics["loss"] = 0.51
    trainer._handle_phase_transition(epoch=2, train_metrics=plateau_metrics)
    plateau_metrics["loss"] = 0.51
    trainer._handle_phase_transition(epoch=3, train_metrics=plateau_metrics)
    plateau_metrics["loss"] = 0.51
    trainer._handle_phase_transition(epoch=4, train_metrics=plateau_metrics)

    assert trainer.phase_controller.phase == 2
    assert trainer.optimizer is not optimizer_before
    assert all(p.requires_grad for p in trainer.model.cross_attn.parameters())
    assert all(p.requires_grad for p in trainer.model.reasoning.parameters())


def test_phase_one_blocks_advance_when_grad_norm_high() -> None:
    trainer = _make_trainer()
    plateau_metrics = {
        "loss": 0.5,
        "accuracy": 0.5,
        "gate_activation_variance": 0.05,
        "grad_norm": 5.0,
        "projection_grad_norm": 5.0,
        "phase": 1,
        "epoch": 1,
    }
    for epoch in range(1, 6):
        plateau_metrics["loss"] = 0.5 + 0.001 * epoch
        trainer._handle_phase_transition(epoch=epoch, train_metrics=plateau_metrics)
    assert trainer.phase_controller.phase == 1


def test_checkpoint_save_and_load_round_trip(tmp_path) -> None:
    trainer = _make_trainer()
    trainer.train_epoch(epoch=1)

    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path)
    assert path.exists()

    config = _small_config()
    fresh_model = SatireDetectionEngine(config)
    t1, t2, t3 = create_mock_datasets()
    val = create_mock_datasets()[0]
    fresh_trainer = SatireTrainer(
        model=fresh_model,
        config=config,
        train_datasets=(t1, t2, t3),
        val_dataset=val,
        device="cpu",
    )

    for (n1, p1), (n2, p2) in zip(
        trainer.model.named_parameters(), fresh_trainer.model.named_parameters()
    ):
        assert n1 == n2
        if not torch.allclose(p1, p2):
            break
    else:
        raise AssertionError("fresh trainer unexpectedly identical to trained trainer")

    fresh_trainer.load_checkpoint(path)

    for (n1, p1), (n2, p2) in zip(
        trainer.model.named_parameters(), fresh_trainer.model.named_parameters()
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2, atol=1e-6), f"param {n1} did not round-trip"

    assert fresh_trainer.phase_controller.phase == trainer.phase_controller.phase


def test_run_loop_terminates_and_records_history(tmp_path) -> None:
    trainer = _make_trainer()
    result = trainer.run(max_epochs=2, checkpoint_dir=tmp_path)
    assert "history" in result
    assert len(result["history"]) <= 2
    assert "best_val_loss" in result
    assert result["final_phase"] in (1, 2, 3)
