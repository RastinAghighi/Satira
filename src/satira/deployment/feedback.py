"""Retraining feedback controller.

Reads the live drift monitor plus a bundle of operator-supplied signals
(architecture changes, moderator-override rate, deployed-model age) and
picks the cheapest retraining action that will close the gap.

Severity is matched to scope on purpose: a GNN architecture swap
invalidates the projection layer's coordinate system and *requires* a
full retrain, but rolling the whole pipeline every time the override
rate ticks up would burn weeks of compute on something a Phase-3
fine-tune can fix. Priorities are checked in order so the most
expensive remediation wins ties — a fresh GNN architecture also drifts
the embedding distribution, but the architecture mismatch is the real
problem and full retrain subsumes the fine-tune.

Priority 1 (critical): GNN architecture change → full_retrain
Priority 2 (high):     drift crossed critical    → phase3_finetune
Priority 3 (medium):   override rate ≥ 2× base   → targeted_finetune
Priority 4 (low):      model age > 21 days       → scheduled_retrain
"""
from __future__ import annotations

from dataclasses import dataclass

from satira.deployment.drift_monitor import EmbeddingDriftMonitor


_OVERRIDE_RATE_MULTIPLIER = 2.0
_MODEL_AGE_LIMIT_DAYS = 21


@dataclass
class RetrainingDecision:
    action: str  # one of "full_retrain", "phase3_finetune", "targeted_finetune", "scheduled_retrain", "none"
    priority: str  # one of "critical", "high", "medium", "low"
    reason: str
    config: dict | None


class FeedbackController:
    """Picks a retraining action by walking priorities in severity order.

    The drift monitor is consulted live (``compute_drift_report()``) so
    the controller always sees the latest production distribution, not
    a stale snapshot the caller passed in. Other signals — architecture
    change flag, moderator override rate, deployed model age — are
    supplied per call because they are owned by external systems
    (deployment registry, review queue, model registry) the controller
    has no business reaching into.
    """

    def __init__(
        self,
        drift_monitor: EmbeddingDriftMonitor,
        baseline_override_rate: float = 0.05,
    ) -> None:
        if baseline_override_rate < 0:
            raise ValueError(
                f"baseline_override_rate must be non-negative, got {baseline_override_rate}"
            )
        self.drift_monitor = drift_monitor
        self.baseline_override_rate = baseline_override_rate

    def evaluate(self, signals: dict) -> RetrainingDecision:
        if signals.get("gnn_architecture_changed", False):
            return RetrainingDecision(
                action="full_retrain",
                priority="critical",
                reason=(
                    "GNN architecture changed; projection layer's coordinate "
                    "system is invalid and cannot be fine-tuned back into shape"
                ),
                config={"phases": [1, 2, 3], "reason_code": "gnn_arch_change"},
            )

        report = self.drift_monitor.compute_drift_report()
        if report.status == "critical":
            return RetrainingDecision(
                action="phase3_finetune",
                priority="high",
                reason=(
                    f"embedding drift crossed critical threshold "
                    f"(composite={report.composite_score:.3f}); refit the "
                    f"classifier head against the current distribution"
                ),
                config={
                    "phases": [3],
                    "drift_composite": report.composite_score,
                    "reason_code": "drift_critical",
                },
            )

        override_rate = float(signals.get("override_rate", 0.0))
        threshold = _OVERRIDE_RATE_MULTIPLIER * self.baseline_override_rate
        if override_rate >= threshold and override_rate > 0:
            return RetrainingDecision(
                action="targeted_finetune",
                priority="medium",
                reason=(
                    f"moderator override rate {override_rate:.1%} ≥ "
                    f"{_OVERRIDE_RATE_MULTIPLIER:.0f}× baseline "
                    f"{self.baseline_override_rate:.1%}; fine-tune on the "
                    f"correction set"
                ),
                config={
                    "phases": [3],
                    "data_source": "moderator_corrections",
                    "override_rate": override_rate,
                    "reason_code": "override_spike",
                },
            )

        model_age_days = int(signals.get("model_age_days", 0))
        if model_age_days > _MODEL_AGE_LIMIT_DAYS:
            return RetrainingDecision(
                action="scheduled_retrain",
                priority="low",
                reason=(
                    f"deployed model is {model_age_days} days old, exceeds "
                    f"{_MODEL_AGE_LIMIT_DAYS}-day refresh cadence"
                ),
                config={
                    "phases": [1, 2, 3],
                    "scheduled": True,
                    "model_age_days": model_age_days,
                    "reason_code": "age_cadence",
                },
            )

        return RetrainingDecision(
            action="none",
            priority="low",
            reason="all signals within tolerance",
            config=None,
        )
