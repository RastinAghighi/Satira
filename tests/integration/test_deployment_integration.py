"""End-to-end integration tests for the deployment pipeline.

Wires together the four deployment components — drift monitor, staged
controller, compatibility matrix, and multi-track synchroniser — using
real implementations and on-disk persistence. The flows mirror what
production does end-to-end: detect distribution drift, gate a staged
rollout against live signals, refuse incompatible model/context pairs,
and orchestrate context tracks across GNN architecture upgrades.
"""
from datetime import datetime, timedelta, timezone

import pytest
import torch

from satira.deployment.compatibility import CompatibilityMatrix
from satira.deployment.controller import (
    DeploymentState,
    StagedModelDeployment,
)
from satira.deployment.drift_monitor import EmbeddingDriftMonitor
from satira.deployment.synchronizer import MultiTrackSynchronizer


DIM = 16
N_BASELINE = 500
N_SAMPLES = 500
WINDOW = ("2026-01-01", "2026-01-15")
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


# --- helpers ----------------------------------------------------------------
def _gaussian(
    n: int,
    dim: int,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: int = 0,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g) * scale + mean


def _stats(samples: torch.Tensor) -> dict:
    """Mirror what GraphEmbeddingCache.compute_distribution_stats produces."""
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


def _record_all(monitor: EmbeddingDriftMonitor, batch: torch.Tensor) -> None:
    for row in batch:
        monitor.record(row)


class _ManualClock:
    """Advancing clock so tests can satisfy MIN_DWELL deterministically."""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now = self.now + timedelta(**kwargs)


# --- 1. drift detection pipeline -------------------------------------------
def test_drift_detection_pipeline() -> None:
    """A single monitor sees status walk healthy → warning → critical
    as the production embedding distribution drifts farther from the
    training reference."""
    train = _gaussian(N_BASELINE, DIM, seed=1)
    train_stats = _stats(train)
    monitor = EmbeddingDriftMonitor(train_stats)

    # Phase 1: in-distribution samples — composite score below warning.
    healthy = _gaussian(N_SAMPLES, DIM, seed=2)
    _record_all(monitor, healthy)
    healthy_report = monitor.compute_drift_report()
    assert healthy_report.status == "healthy"
    assert healthy_report.composite_score < monitor.warning_threshold

    # Phase 2: 2x scale — norm and structural signals fire, marginal
    # stays near zero. Composite lands in the warning band.
    monitor.recalibrate(train_stats)
    moderate = _gaussian(N_SAMPLES, DIM, scale=2.0, seed=3)
    _record_all(monitor, moderate)
    warning_report = monitor.compute_drift_report()
    assert warning_report.status == "warning"
    assert (
        monitor.warning_threshold
        <= warning_report.composite_score
        < monitor.critical_threshold
    )

    # Phase 3: large mean shift + scale — every signal saturates.
    monitor.recalibrate(train_stats)
    severe = _gaussian(N_SAMPLES, DIM, mean=4.0, scale=2.0, seed=4)
    _record_all(monitor, severe)
    critical_report = monitor.compute_drift_report()
    assert critical_report.status == "critical"
    assert critical_report.composite_score >= monitor.critical_threshold

    # Composite score strictly grows across phases.
    assert (
        healthy_report.composite_score
        < warning_report.composite_score
        < critical_report.composite_score
    )


# --- 2. staged rollout lifecycle -------------------------------------------
def test_staged_rollout_lifecycle(tmp_path) -> None:
    """SHADOW → CANARY_5 → CANARY_25 with dwell enforced at each stop."""
    clock = _ManualClock()
    controller = StagedModelDeployment(
        registry_path=str(tmp_path / "registry.json"),
        clock=clock,
    )
    controller.create_deployment("cls-v2")
    assert controller.get_current_state("cls-v2") == DeploymentState.SHADOW

    good_metrics = {
        "drift_score": 0.05,
        "override_rate": 0.01,
        "calibration_divergence": 0.01,
    }

    # SHADOW: dwell is 6h. Before that, gate holds and explicit promotion
    # is rejected.
    clock.advance(hours=1)
    assert controller.run_gate_check("cls-v2", good_metrics) == "hold"
    with pytest.raises(ValueError, match="dwell"):
        controller.execute_promotion("cls-v2", DeploymentState.CANARY_5)

    # After 6h dwell elapses, gate promotes and the transition succeeds.
    clock.advance(hours=5, seconds=1)
    assert controller.run_gate_check("cls-v2", good_metrics) == "promote"
    controller.execute_promotion("cls-v2", DeploymentState.CANARY_5)
    assert controller.get_current_state("cls-v2") == DeploymentState.CANARY_5

    # CANARY_5: dwell is 24h. Halfway through the gate still holds.
    clock.advance(hours=12)
    assert controller.run_gate_check("cls-v2", good_metrics) == "hold"
    with pytest.raises(ValueError, match="dwell"):
        controller.execute_promotion("cls-v2", DeploymentState.CANARY_25)

    # After the full 24h, gate promotes and CANARY_25 is reached.
    clock.advance(hours=12, seconds=1)
    assert controller.run_gate_check("cls-v2", good_metrics) == "promote"
    controller.execute_promotion("cls-v2", DeploymentState.CANARY_25)
    assert controller.get_current_state("cls-v2") == DeploymentState.CANARY_25


# --- 3. rollback on drift ---------------------------------------------------
def test_rollback_on_drift(tmp_path) -> None:
    """A canary in flight sees critical drift; the gate signals rollback
    and the executed transition lands in ROLLED_BACK."""
    clock = _ManualClock()
    controller = StagedModelDeployment(
        registry_path=str(tmp_path / "registry.json"),
        clock=clock,
    )
    controller.create_deployment("cls-v2")

    # Drive to CANARY_5 under clean signals.
    clock.advance(hours=6, seconds=1)
    controller.execute_promotion("cls-v2", DeploymentState.CANARY_5)
    assert controller.get_current_state("cls-v2") == DeploymentState.CANARY_5

    # Drift composite jumps past 0.75 — gate must demand rollback even
    # if the other signals are clean.
    drift_metrics = {
        "drift_score": 0.9,
        "override_rate": 0.01,
        "calibration_divergence": 0.01,
    }
    assert controller.run_gate_check("cls-v2", drift_metrics) == "rollback"

    controller.execute_rollback("cls-v2", reason="drift composite > 0.75")
    assert controller.get_current_state("cls-v2") == DeploymentState.ROLLED_BACK

    # A rolled-back deployment is terminal: gate holds, no further moves.
    assert controller.run_gate_check("cls-v2", drift_metrics) == "hold"


# --- 4. compatibility enforcement ------------------------------------------
def test_compatibility_enforcement(tmp_path) -> None:
    """A model that only understands GNN v1 must reject a context produced
    by GNN v2 with critical severity, even when its embedding stats look
    in-distribution."""
    matrix = CompatibilityMatrix(
        registry_path=str(tmp_path / "registry.json"),
    )

    matrix.register_model(
        model_version="cls-v1",
        compatible_gnn_versions=["gnn-v1"],
        graph_snapshot_window=WINDOW,
        training_stats=_stats(_gaussian(N_BASELINE, DIM, seed=1)),
    )
    matrix.register_context(
        context_version="ctx-v2",
        gnn_version="gnn-v2",
        embedding_stats=_stats(_gaussian(N_SAMPLES, DIM, seed=2)),
    )

    result = matrix.check_compatibility("cls-v1", "ctx-v2")
    assert result.compatible is False
    assert result.severity == "critical"
    assert "GNN" in result.reason


# --- 5. multi-track sync ---------------------------------------------------
def test_multi_track_sync(tmp_path) -> None:
    """v1 model is live; a v2-architecture context arrives and is parked
    until a v2 model gets promoted, at which point the parked context
    becomes the live one."""
    matrix = CompatibilityMatrix(
        registry_path=str(tmp_path / "registry.json"),
    )
    sync = MultiTrackSynchronizer(matrix)

    # v1 model is the active production classifier with a healthy v1 ctx.
    matrix.register_model(
        "cls-v1",
        ["gnn-v1"],
        WINDOW,
        _stats(_gaussian(N_BASELINE, DIM, seed=1)),
    )
    matrix.register_context(
        "ctx-v1",
        "gnn-v1",
        _stats(_gaussian(N_SAMPLES, DIM, seed=2)),
    )
    sync.on_model_promoted("cls-v1", ["gnn-v1"])
    deploy_v1 = sync.on_new_context("ctx-v1", "gnn-v1")
    assert deploy_v1["deployed"] is True

    # v2 context arrives — different GNN architecture, so the active v1
    # model can't read it. Synchroniser parks it on the gnn-v2 track
    # rather than swapping it in.
    matrix.register_context(
        "ctx-v2",
        "gnn-v2",
        _stats(_gaussian(N_SAMPLES, DIM, seed=3)),
    )
    parked = sync.on_new_context("ctx-v2", "gnn-v2")
    assert parked["deployed"] is False
    assert parked["track"] == "gnn-v2"
    assert sync.get_active_tracks().get("gnn-v2") == "ctx-v2"

    # v2 model is promoted with gnn-v2 on its allow-list. The previously
    # parked ctx-v2 becomes the live context immediately, and the dead
    # gnn-v1 track is pruned.
    matrix.register_model(
        "cls-v2",
        ["gnn-v2"],
        WINDOW,
        _stats(_gaussian(N_BASELINE, DIM, seed=4)),
    )
    promotion = sync.on_model_promoted("cls-v2", ["gnn-v2"])
    assert promotion["context_deployed"] == "ctx-v2"
    assert promotion["old_tracks_pruned"] == ["gnn-v1"]
    assert sync.get_active_tracks() == {"gnn-v2": "ctx-v2"}
