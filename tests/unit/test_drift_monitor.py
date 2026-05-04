import math

import pytest
import torch

from satira.deployment.drift_monitor import (
    DriftReport,
    EmbeddingDriftMonitor,
)


DIM = 16
N_BASELINE = 500
N_SAMPLES = 500


def _gaussian(n: int, dim: int, mean: float = 0.0, scale: float = 1.0, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g) * scale + mean


def _stats_from_samples(samples: torch.Tensor) -> dict:
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


# --- healthy report ---------------------------------------------------------
def test_healthy_report_with_in_distribution_embeddings() -> None:
    train = _gaussian(N_BASELINE, DIM, seed=1)
    monitor = EmbeddingDriftMonitor(_stats_from_samples(train))

    prod = _gaussian(N_SAMPLES, DIM, seed=2)
    _record_all(monitor, prod)

    report = monitor.compute_drift_report()

    assert isinstance(report, DriftReport)
    assert report.status == "healthy"
    assert report.composite_score < 0.5
    assert report.sample_size == N_SAMPLES
    assert report.recommendation == "no action required"


def test_compute_drift_report_with_too_few_samples_is_healthy_but_marked_insufficient() -> None:
    train = _gaussian(N_BASELINE, DIM, seed=3)
    monitor = EmbeddingDriftMonitor(_stats_from_samples(train))

    _record_all(monitor, _gaussian(10, DIM, seed=4))

    report = monitor.compute_drift_report()
    assert report.sample_size == 10
    assert report.status == "healthy"
    assert "insufficient" in report.recommendation


# --- warning report ---------------------------------------------------------
def test_warning_report_with_moderate_drift() -> None:
    train = _gaussian(N_BASELINE, DIM, seed=11)
    monitor = EmbeddingDriftMonitor(_stats_from_samples(train))

    # Scale every dim by 2x: mean stays at 0 (no marginal drift) but the
    # mean L2 norm doubles and every singular value doubles too.
    prod = _gaussian(N_SAMPLES, DIM, scale=2.0, seed=12)
    _record_all(monitor, prod)

    report = monitor.compute_drift_report()
    assert report.status == "warning"
    assert monitor.warning_threshold <= report.composite_score < monitor.critical_threshold


# --- critical report --------------------------------------------------------
def test_critical_report_with_large_drift() -> None:
    train = _gaussian(N_BASELINE, DIM, seed=21)
    monitor = EmbeddingDriftMonitor(_stats_from_samples(train))

    # Scale and shift — every signal lights up.
    prod = _gaussian(N_SAMPLES, DIM, mean=4.0, scale=2.0, seed=22)
    _record_all(monitor, prod)

    report = monitor.compute_drift_report()
    assert report.status == "critical"
    assert report.composite_score >= monitor.critical_threshold
    # All three components are clearly drifted.
    assert report.max_dimension_drift > 2.0
    assert report.norm_drift > 0.5
    assert report.structural_drift > 0.5


# --- structural drift detection --------------------------------------------
def test_svd_structural_shift_detection() -> None:
    """Production data has a different SV spectrum from training."""

    # Training: roughly isotropic — every dim has similar variance, so
    # the singular spectrum is roughly flat.
    train = _gaussian(N_BASELINE, DIM, scale=1.0, seed=31)
    monitor = EmbeddingDriftMonitor(_stats_from_samples(train))

    # Production: rank-collapse — only the first 4 dims carry signal,
    # the rest are near-zero. Mean stays at 0, but the top SVs grow
    # while the tail collapses.
    g = torch.Generator().manual_seed(32)
    prod = torch.zeros(N_SAMPLES, DIM)
    prod[:, :4] = torch.randn(N_SAMPLES, 4, generator=g) * 3.0
    prod[:, 4:] = torch.randn(N_SAMPLES, DIM - 4, generator=g) * 0.05

    _record_all(monitor, prod)
    report = monitor.compute_drift_report()

    assert report.structural_drift > 0.5
    assert report.status in ("warning", "critical")
    # Marginal stays small — the test isolates structural detection.
    assert report.max_dimension_drift < 1.0


def test_structural_drift_zero_when_spectrum_matches() -> None:
    train = _gaussian(N_BASELINE, DIM, seed=33)
    monitor = EmbeddingDriftMonitor(_stats_from_samples(train))

    # Same generator distribution → SV spectra should be very close.
    prod = _gaussian(N_SAMPLES, DIM, seed=34)
    _record_all(monitor, prod)

    report = monitor.compute_drift_report()
    assert report.structural_drift < 0.2


# --- recalibrate ------------------------------------------------------------
def test_recalibrate_resets_the_baseline() -> None:
    train = _gaussian(N_BASELINE, DIM, seed=41)
    monitor = EmbeddingDriftMonitor(_stats_from_samples(train))

    drifted = _gaussian(N_SAMPLES, DIM, mean=4.0, scale=2.0, seed=42)
    _record_all(monitor, drifted)
    pre = monitor.compute_drift_report()
    assert pre.status == "critical"

    # Adopt the drifted distribution as the new baseline.
    monitor.recalibrate(_stats_from_samples(drifted))

    # Buffer is cleared — the next report has no samples to score.
    cleared = monitor.compute_drift_report()
    assert cleared.sample_size == 0
    assert cleared.status == "healthy"
    assert "insufficient" in cleared.recommendation

    # Feeding fresh samples from the new distribution should now be healthy.
    fresh = _gaussian(N_SAMPLES, DIM, mean=4.0, scale=2.0, seed=43)
    _record_all(monitor, fresh)
    post = monitor.compute_drift_report()
    assert post.status == "healthy"
    assert post.composite_score < 0.5


# --- input validation -------------------------------------------------------
def test_record_rejects_non_1d_tensors() -> None:
    train = _gaussian(N_BASELINE, DIM, seed=51)
    monitor = EmbeddingDriftMonitor(_stats_from_samples(train))
    with pytest.raises(ValueError):
        monitor.record(torch.zeros(2, DIM))


def test_init_rejects_inverted_thresholds() -> None:
    train = _gaussian(N_BASELINE, DIM, seed=61)
    with pytest.raises(ValueError):
        EmbeddingDriftMonitor(
            _stats_from_samples(train),
            warning_threshold=0.8,
            critical_threshold=0.5,
        )


def test_init_rejects_missing_stats_keys() -> None:
    with pytest.raises(ValueError):
        EmbeddingDriftMonitor({"mean": torch.zeros(DIM)})


# --- buffer behaviour -------------------------------------------------------
def test_buffer_does_not_grow_unbounded() -> None:
    train = _gaussian(N_BASELINE, DIM, seed=71)
    monitor = EmbeddingDriftMonitor(_stats_from_samples(train))

    # Push past the maxlen — the deque should silently evict old samples.
    over = 10_500
    for i in range(over):
        monitor.record(torch.randn(DIM))

    report = monitor.compute_drift_report()
    assert report.sample_size <= 10_000


# --- recommendation surfaces dominant component ----------------------------
def test_critical_recommendation_names_dominant_component() -> None:
    train = _gaussian(N_BASELINE, DIM, seed=81)
    monitor = EmbeddingDriftMonitor(_stats_from_samples(train))

    # Big scale + modest mean shift — every signal lights up, but with
    # the 0.45 weight, structural dominates the composite.
    prod = _gaussian(N_SAMPLES, DIM, mean=2.0, scale=4.0, seed=82)
    _record_all(monitor, prod)

    report = monitor.compute_drift_report()
    assert report.status == "critical"
    assert "structural" in report.recommendation
    assert not math.isnan(report.max_dimension_drift)
