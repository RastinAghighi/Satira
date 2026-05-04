"""Embedding distribution drift monitor for production graph lookups.

Tracks the gap between graph embeddings observed at inference time and
the training distribution. The graph stream is the early-warning signal
for silent model degradation: when the offline GNN re-trains and emits
embeddings with rotated principal axes, the inference model's linear
projection layer silently misaligns even though the new vectors are
still well-formed.

Designed to ride along with the existing context-resolver path on CPU:
``record()`` is a deque append, and ``compute_drift_report()`` runs on
a separate cadence (e.g. once a minute) so the hot path stays free of
SVD work.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


_BUFFER_MAXLEN = 10000
_MIN_SAMPLES = 100
_EPS = 1e-8


@dataclass
class DriftReport:
    status: str  # one of "healthy", "warning", "critical"
    composite_score: float
    max_dimension_drift: float
    norm_drift: float
    structural_drift: float
    sample_size: int
    recommendation: str


class EmbeddingDriftMonitor:
    """Continuous distribution-drift monitor for production graph embeddings.

    Three weighted metrics make up the composite drift score:
      - marginal (0.30): max per-dimension mean shift, normalized by
        training std (i.e. the worst-dim z-score).
      - norm     (0.25): relative change in average embedding magnitude.
      - structural (0.45): change in the top singular-value spectrum,
        which carries the most weight because rotated principal axes
        break the downstream linear projection layer even when the
        embeddings still look healthy on a per-dim basis.

    Each raw signal is squashed into ``[0, 1]`` via a saturation scale
    before being weighted, so the composite score is also in ``[0, 1]``
    and directly comparable to ``warning_threshold`` /
    ``critical_threshold``.
    """

    MARGINAL_W = 0.30
    NORM_W = 0.25
    STRUCT_W = 0.45

    # Saturation scales: convert raw drift signals into [0, 1].
    # 5 standard deviations on the worst dim is "fully drifted".
    _MARGINAL_SAT_Z = 5.0
    # 100% mean-norm change is "fully drifted".
    _NORM_SAT_FRAC = 1.0

    def __init__(
        self,
        training_stats: dict,
        warning_threshold: float = 0.5,
        critical_threshold: float = 0.75,
    ) -> None:
        if critical_threshold <= warning_threshold:
            raise ValueError(
                "critical_threshold must exceed warning_threshold "
                f"(got {critical_threshold} <= {warning_threshold})"
            )
        self._training_stats = _copy_stats(training_stats)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._buffer: deque[torch.Tensor] = deque(maxlen=_BUFFER_MAXLEN)

    # --- public API -----------------------------------------------------
    def record(self, embedding: torch.Tensor) -> None:
        """Append an inference-time embedding to the rolling buffer."""
        if embedding.dim() != 1:
            raise ValueError(
                f"embedding must be 1-D, got shape {tuple(embedding.shape)}"
            )
        self._buffer.append(embedding.detach().clone())

    def compute_drift_report(self) -> DriftReport:
        n = len(self._buffer)
        if n < _MIN_SAMPLES:
            return DriftReport(
                status="healthy",
                composite_score=0.0,
                max_dimension_drift=0.0,
                norm_drift=0.0,
                structural_drift=0.0,
                sample_size=n,
                recommendation=(
                    f"insufficient samples ({n} < {_MIN_SAMPLES}) — keep collecting"
                ),
            )

        stacked = torch.stack(list(self._buffer), dim=0).to(torch.float32)
        train_mean = self._training_stats["mean"].to(torch.float32)
        train_std = self._training_stats["std"].to(torch.float32)
        train_norm_mean = float(self._training_stats["norm_mean"])
        train_sv = self._training_stats["top_singular_values"].to(torch.float32)

        # 1) Marginal — per-dim absolute z-score against training std.
        current_mean = stacked.mean(dim=0)
        denom = torch.clamp(train_std, min=_EPS)
        per_dim_z = (current_mean - train_mean).abs() / denom
        max_dim_drift = float(per_dim_z.max().item())
        marginal_norm = min(1.0, max_dim_drift / self._MARGINAL_SAT_Z)

        # 2) Norm — relative change in mean L2 norm.
        current_norm_mean = float(stacked.norm(dim=1).mean().item())
        norm_drift = abs(current_norm_mean - train_norm_mean) / max(
            train_norm_mean, _EPS
        )
        norm_norm = min(1.0, norm_drift / self._NORM_SAT_FRAC)

        # 3) Structural — energy redistribution across principal axes.
        structural_drift = _structural_shift(stacked, train_sv)
        structural_norm = min(1.0, structural_drift)

        composite = (
            self.MARGINAL_W * marginal_norm
            + self.NORM_W * norm_norm
            + self.STRUCT_W * structural_norm
        )

        status, recommendation = self._classify(
            composite, marginal_norm, norm_norm, structural_norm
        )

        return DriftReport(
            status=status,
            composite_score=composite,
            max_dimension_drift=max_dim_drift,
            norm_drift=norm_drift,
            structural_drift=structural_drift,
            sample_size=n,
            recommendation=recommendation,
        )

    def recalibrate(self, new_training_stats: dict) -> None:
        """Reset the reference distribution after a model swap.

        Clears the rolling buffer too: stale samples were measured
        against the old baseline and would skew the next report.
        """
        self._training_stats = _copy_stats(new_training_stats)
        self._buffer.clear()

    # --- internals ------------------------------------------------------
    def _classify(
        self,
        composite: float,
        marginal: float,
        norm: float,
        structural: float,
    ) -> tuple[str, str]:
        if composite >= self.critical_threshold:
            dominant = _dominant_component(marginal, norm, structural)
            return (
                "critical",
                f"halt graph-context inference path; dominant signal: {dominant}. "
                "Roll back to last known-good GNN snapshot or recalibrate baseline.",
            )
        if composite >= self.warning_threshold:
            dominant = _dominant_component(marginal, norm, structural)
            return (
                "warning",
                f"investigate GNN re-training; dominant signal: {dominant}. "
                "Page on-call if score keeps rising.",
            )
        return "healthy", "no action required"


def _copy_stats(stats: dict) -> dict:
    required = ("mean", "std", "norm_mean", "top_singular_values")
    missing = [k for k in required if k not in stats]
    if missing:
        raise ValueError(f"training_stats missing required keys: {missing}")
    return {
        "mean": stats["mean"].detach().clone(),
        "std": stats["std"].detach().clone(),
        "norm_mean": float(stats["norm_mean"]),
        "norm_std": float(stats.get("norm_std", 0.0)),
        "top_singular_values": stats["top_singular_values"].detach().clone(),
    }


def _structural_shift(stacked: torch.Tensor, train_sv: torch.Tensor) -> float:
    """Relative L2 distance between the top training and current SV spectra.

    Returns 0 when the spectra match. The raw value is unbounded above —
    callers saturate for the composite score; the report surfaces the
    raw magnitude so operators can tell "barely drifted" from
    "completely different shape." Singular values alone are
    rotation-invariant, so this catches energy redistribution (rank
    collapse, scale shifts, low-rank drift) rather than pure rotations.
    """
    if train_sv.numel() == 0:
        return 0.0
    try:
        cur_sv = torch.linalg.svdvals(stacked)
    except RuntimeError:
        return 0.0

    k = min(train_sv.shape[0], cur_sv.shape[0])
    if k == 0:
        return 0.0
    a = train_sv[:k]
    b = cur_sv[:k]

    a_norm = float(a.norm().item())
    if a_norm <= _EPS:
        return 0.0 if float(b.norm().item()) <= _EPS else 1.0
    return float((a - b).norm().item() / a_norm)


def _dominant_component(marginal: float, norm: float, structural: float) -> str:
    # Compare each signal's contribution to the composite, not its raw
    # magnitude — a small structural shift can still dominate because
    # of its 0.45 weight.
    contributions = {
        "marginal": EmbeddingDriftMonitor.MARGINAL_W * marginal,
        "norm": EmbeddingDriftMonitor.NORM_W * norm,
        "structural": EmbeddingDriftMonitor.STRUCT_W * structural,
    }
    return max(contributions, key=contributions.get)
