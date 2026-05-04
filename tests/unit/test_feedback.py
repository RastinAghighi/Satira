import torch

from satira.deployment.drift_monitor import EmbeddingDriftMonitor
from satira.deployment.feedback import FeedbackController, RetrainingDecision


DIM = 16
N_BASELINE = 500
N_SAMPLES = 500


def _gaussian(n: int, dim: int, mean: float = 0.0, scale: float = 1.0, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g) * scale + mean


def _stats_from_samples(samples: torch.Tensor) -> dict:
    norms = samples.norm(dim=1)
    sv = torch.linalg.svdvals(samples)
    return {
        "count": samples.shape[0],
        "mean": samples.mean(dim=0),
        "std": samples.std(dim=0, unbiased=False),
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std(unbiased=False).item()),
        "top_singular_values": sv[: min(10, sv.shape[0])],
    }


def _healthy_monitor(seed: int = 1) -> EmbeddingDriftMonitor:
    train = _gaussian(N_BASELINE, DIM, seed=seed)
    monitor = EmbeddingDriftMonitor(_stats_from_samples(train))
    prod = _gaussian(N_SAMPLES, DIM, seed=seed + 1)
    for row in prod:
        monitor.record(row)
    return monitor


def _critical_monitor(seed: int = 21) -> EmbeddingDriftMonitor:
    train = _gaussian(N_BASELINE, DIM, seed=seed)
    monitor = EmbeddingDriftMonitor(_stats_from_samples(train))
    # Mean shift + scale change — every component lights up.
    prod = _gaussian(N_SAMPLES, DIM, mean=4.0, scale=2.0, seed=seed + 1)
    for row in prod:
        monitor.record(row)
    return monitor


# --- priority 1: GNN architecture change ----------------------------------
def test_gnn_architecture_change_triggers_full_retrain() -> None:
    # Use a critical drift monitor to confirm priority 1 wins over priority 2.
    controller = FeedbackController(_critical_monitor())

    decision = controller.evaluate(
        {
            "gnn_architecture_changed": True,
            "override_rate": 0.30,
            "model_age_days": 90,
        }
    )

    assert isinstance(decision, RetrainingDecision)
    assert decision.action == "full_retrain"
    assert decision.priority == "critical"
    assert "GNN architecture" in decision.reason
    assert decision.config is not None
    assert decision.config["phases"] == [1, 2, 3]


# --- priority 2: drift critical -------------------------------------------
def test_critical_drift_triggers_phase3_finetune() -> None:
    controller = FeedbackController(_critical_monitor())

    decision = controller.evaluate(
        {
            "gnn_architecture_changed": False,
            "override_rate": 0.30,
            "model_age_days": 90,
        }
    )

    assert decision.action == "phase3_finetune"
    assert decision.priority == "high"
    assert decision.config is not None
    assert decision.config["phases"] == [3]
    assert decision.config["drift_composite"] >= 0.75


# --- priority 3: override rate spike --------------------------------------
def test_high_override_rate_triggers_targeted_finetune() -> None:
    controller = FeedbackController(
        _healthy_monitor(),
        baseline_override_rate=0.05,
    )

    # 12% > 2 * 5% baseline.
    decision = controller.evaluate(
        {
            "gnn_architecture_changed": False,
            "override_rate": 0.12,
            "model_age_days": 90,
        }
    )

    assert decision.action == "targeted_finetune"
    assert decision.priority == "medium"
    assert decision.config is not None
    assert decision.config["data_source"] == "moderator_corrections"
    assert decision.config["override_rate"] == 0.12


# --- priority 4: model age cadence ----------------------------------------
def test_old_model_triggers_scheduled_retrain() -> None:
    controller = FeedbackController(
        _healthy_monitor(seed=41),
        baseline_override_rate=0.05,
    )

    decision = controller.evaluate(
        {
            "gnn_architecture_changed": False,
            "override_rate": 0.04,  # below 2x baseline
            "model_age_days": 30,
        }
    )

    assert decision.action == "scheduled_retrain"
    assert decision.priority == "low"
    assert decision.config is not None
    assert decision.config["scheduled"] is True
    assert decision.config["model_age_days"] == 30


# --- healthy fall-through --------------------------------------------------
def test_healthy_signals_return_none() -> None:
    controller = FeedbackController(
        _healthy_monitor(seed=51),
        baseline_override_rate=0.05,
    )

    decision = controller.evaluate(
        {
            "gnn_architecture_changed": False,
            "override_rate": 0.04,
            "model_age_days": 7,
        }
    )

    assert decision.action == "none"
    assert decision.config is None
    assert "tolerance" in decision.reason
