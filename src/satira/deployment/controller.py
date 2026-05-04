"""Staged model deployment controller.

Drives a model version through shadow scoring, three canary expansions,
and finally production. State is persisted to a JSON registry on every
transition; ``get_current_state`` always re-reads from disk so a crash
mid-rollout is recoverable rather than silently half-applied.

Each transition uses a write-ahead pattern (``begin_transition`` then
``complete_transition``): the registry first records a
``pending_transition`` field, then promotes it to the new state. If the
process dies between the two writes, the next reader sees the pending
field and can resolve it deterministically.

Three signals trigger an automatic rollback at any stage:

  - embedding drift composite score > 0.75 (matches
    ``EmbeddingDriftMonitor.critical_threshold``)
  - moderator override rate > 12%
  - calibration divergence > 15%

Promotion criteria per stage are documented in Table 16.1 of the
deployment blueprint and encoded here in
``DeploymentStateMachine.MIN_DWELL`` and
``StagedModelDeployment.WARNING_DRIFT``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable

from satira.deployment.drift_monitor import EmbeddingDriftMonitor


class DeploymentState(Enum):
    SHADOW = "shadow"
    CANARY_5 = "canary_5"
    CANARY_25 = "canary_25"
    CANARY_50 = "canary_50"
    PRODUCTION = "production"
    ROLLED_BACK = "rolled_back"


@dataclass
class GateDecision:
    action: str  # "promote" | "hold" | "rollback"
    reason: str


class DeploymentStateMachine:
    """Encodes valid state transitions and minimum dwell times.

    Promotion is a strict linear walk shadow → canary_5 → canary_25 →
    canary_50 → production. Skipping stages is rejected: a stage that
    has not been observed under load cannot signal regressions, so the
    blueprint requires every checkpoint to ride each step.

    Rollback is allowed from any state except ROLLED_BACK itself
    (rolling back a rolled-back deployment is a no-op).
    """

    PROMOTION_PATH: dict[DeploymentState, DeploymentState] = {
        DeploymentState.SHADOW: DeploymentState.CANARY_5,
        DeploymentState.CANARY_5: DeploymentState.CANARY_25,
        DeploymentState.CANARY_25: DeploymentState.CANARY_50,
        DeploymentState.CANARY_50: DeploymentState.PRODUCTION,
    }

    MIN_DWELL: dict[str, timedelta] = {
        "shadow": timedelta(hours=6),
        "canary_5": timedelta(hours=24),
        "canary_25": timedelta(hours=12),
        "canary_50": timedelta(hours=12),
    }

    def can_promote(self, current: DeploymentState) -> DeploymentState | None:
        """Return the next state in the promotion path, or None at a terminus."""
        return self.PROMOTION_PATH.get(current)

    def can_rollback(self, current: DeploymentState) -> bool:
        return current != DeploymentState.ROLLED_BACK

    def min_dwell(self, current: DeploymentState) -> timedelta | None:
        return self.MIN_DWELL.get(current.value)


class StagedModelDeployment:
    """Manages shadow scoring, canary rollout, and automatic rollback.

    State is ALWAYS read from the registry, never from local memory, so
    a crashed process re-reading the file recovers the same view that
    any other reader would see. Use the ``begin_transition`` /
    ``complete_transition`` pair for every state change so an
    interrupted move is detectable.

    Rollback triggers (any one is sufficient, evaluated in
    ``run_gate_check``):

      - drift composite score > 0.75
      - override rate > 12%
      - calibration divergence > 15%

    Per-stage promotion criteria are documented in Table 16.1 of the
    deployment blueprint.
    """

    DRIFT_ROLLBACK = 0.75
    OVERRIDE_ROLLBACK = 0.12
    CALIBRATION_ROLLBACK = 0.15

    # Drift in [WARNING_DRIFT, DRIFT_ROLLBACK) holds promotion but does
    # not roll back — it's a sign to wait for the signal to resolve, not
    # an emergency.
    WARNING_DRIFT = 0.5

    # Promotion-readiness thresholds for evaluate_promotion_readiness.
    PROMOTE_AGREEMENT = 0.95
    DRIFT_CORRELATED_BLOCK = 0.25
    HIGH_DRIFT_BUCKET = 0.5

    def __init__(
        self,
        registry_path: str,
        drift_monitor: EmbeddingDriftMonitor | None = None,
        state_machine: DeploymentStateMachine | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.registry_path = Path(registry_path)
        self.drift_monitor = drift_monitor
        self.state_machine = state_machine or DeploymentStateMachine()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._deployments: dict[str, dict] = {}
        self._load()

    # --- lifecycle ------------------------------------------------------
    def create_deployment(self, version_id: str) -> None:
        """Register a new model version, starting in SHADOW."""
        self._load()
        if version_id in self._deployments:
            raise ValueError(f"deployment {version_id!r} already exists")
        now = _iso(self._clock())
        self._deployments[version_id] = {
            "version_id": version_id,
            "state": DeploymentState.SHADOW.value,
            "entered_state_at": now,
            "history": [],
            "pending_transition": None,
            "rollback_reason": None,
        }
        self._save()

    def get_current_state(self, version_id: str) -> DeploymentState:
        """Return the on-disk state — never the cached value."""
        record = self._read(version_id)
        return DeploymentState(record["state"])

    # --- gate -----------------------------------------------------------
    def run_gate_check(self, version_id: str, metrics: dict) -> str:
        """Decide whether to ``promote``, ``hold``, or ``rollback``.

        ``metrics`` should carry ``drift_score``, ``override_rate``, and
        ``calibration_divergence``. Missing keys default to zero, except
        ``drift_score`` which falls back to the live drift monitor when
        one is attached.
        """
        record = self._read(version_id)
        state = DeploymentState(record["state"])

        if state == DeploymentState.ROLLED_BACK:
            return "hold"

        drift = self._drift(metrics)
        override_rate = float(metrics.get("override_rate", 0.0))
        calibration_divergence = float(metrics.get("calibration_divergence", 0.0))

        if drift > self.DRIFT_ROLLBACK:
            return "rollback"
        if override_rate > self.OVERRIDE_ROLLBACK:
            return "rollback"
        if calibration_divergence > self.CALIBRATION_ROLLBACK:
            return "rollback"

        if self.state_machine.can_promote(state) is None:
            # Already at terminus (production); nothing to promote to.
            return "hold"

        if not self._dwell_satisfied(record):
            return "hold"

        if drift >= self.WARNING_DRIFT:
            # Don't promote into a degrading signal even if it hasn't
            # crossed the rollback line yet.
            return "hold"

        return "promote"

    # --- transitions ----------------------------------------------------
    def execute_promotion(self, version_id: str, next_state: DeploymentState) -> None:
        record = self._read(version_id)
        current = DeploymentState(record["state"])
        target = self.state_machine.can_promote(current)
        if target is None:
            raise ValueError(
                f"cannot promote from terminal state {current.value!r}"
            )
        if target != next_state:
            raise ValueError(
                f"invalid promotion {current.value!r} → {next_state.value!r}; "
                f"only {target.value!r} is reachable"
            )
        if not self._dwell_satisfied(record):
            raise ValueError(
                f"min dwell of {self.state_machine.min_dwell(current)} not "
                f"satisfied for state {current.value!r}"
            )
        self._begin_transition(version_id, next_state, reason="promote")
        self._complete_transition(version_id)

    def execute_rollback(self, version_id: str, reason: str) -> None:
        record = self._read(version_id)
        current = DeploymentState(record["state"])
        if not self.state_machine.can_rollback(current):
            raise ValueError(
                f"cannot roll back from terminal state {current.value!r}"
            )
        self._begin_transition(
            version_id,
            DeploymentState.ROLLED_BACK,
            reason=f"rollback: {reason}",
        )
        self._complete_transition(version_id, rollback_reason=reason)

    # --- evaluation -----------------------------------------------------
    def evaluate_promotion_readiness(self, comparisons: list[dict]) -> dict:
        """Aggregate shadow-vs-production comparisons into a recommendation.

        Each comparison must include ``shadow_label`` /
        ``production_label`` (or ``shadow_score`` / ``production_score``)
        and ``drift_value`` — the drift composite observed at prediction
        time.

        The crucial signal is *where* disagreements happen, not just how
        many. Disagreements that cluster in high-drift samples mean the
        new model is failing exactly where the embedding distribution is
        shifting; that is a worse failure mode than a uniform error rate
        and blocks promotion even when overall agreement looks fine.
        """
        if not comparisons:
            return {
                "agreement_rate": 0.0,
                "drift_correlated_disagreements": 0.0,
                "sample_size": 0,
                "recommendation": "hold",
                "reason": "no shadow comparisons collected yet",
            }

        n = len(comparisons)
        agreements = sum(1 for c in comparisons if _agrees(c))
        agreement_rate = agreements / n

        high_drift = [
            c for c in comparisons
            if float(c.get("drift_value", 0.0)) >= self.HIGH_DRIFT_BUCKET
        ]
        if high_drift:
            high_drift_disagree = sum(1 for c in high_drift if not _agrees(c))
            drift_correlated = high_drift_disagree / len(high_drift)
        else:
            drift_correlated = 0.0

        if drift_correlated > self.DRIFT_CORRELATED_BLOCK:
            recommendation = "block"
            reason = (
                f"{drift_correlated:.1%} of high-drift samples disagree; "
                f"promotion would deploy a model that is failing exactly "
                f"where embeddings are shifting"
            )
        elif agreement_rate < self.PROMOTE_AGREEMENT:
            recommendation = "hold"
            reason = (
                f"agreement rate {agreement_rate:.1%} below "
                f"{self.PROMOTE_AGREEMENT:.0%} promotion threshold"
            )
        else:
            recommendation = "promote"
            reason = (
                f"{agreement_rate:.1%} agreement, "
                f"{drift_correlated:.1%} drift-correlated disagreement"
            )

        return {
            "agreement_rate": agreement_rate,
            "drift_correlated_disagreements": drift_correlated,
            "sample_size": n,
            "recommendation": recommendation,
            "reason": reason,
        }

    # --- internals ------------------------------------------------------
    def _drift(self, metrics: dict) -> float:
        if "drift_score" in metrics:
            return float(metrics["drift_score"])
        if self.drift_monitor is None:
            return 0.0
        return float(self.drift_monitor.compute_drift_report().composite_score)

    def _read(self, version_id: str) -> dict:
        # Always re-read from disk so that a sibling process's writes
        # are visible — the in-memory dict is a cache, not the truth.
        self._load()
        if version_id not in self._deployments:
            raise KeyError(f"unknown deployment {version_id!r}")
        return self._deployments[version_id]

    def _dwell_satisfied(self, record: dict) -> bool:
        state = DeploymentState(record["state"])
        dwell = self.state_machine.min_dwell(state)
        if dwell is None:
            return True
        entered = datetime.fromisoformat(record["entered_state_at"])
        return self._clock() - entered >= dwell

    def _begin_transition(
        self,
        version_id: str,
        target: DeploymentState,
        reason: str,
    ) -> None:
        record = self._deployments[version_id]
        record["pending_transition"] = {
            "from": record["state"],
            "to": target.value,
            "started_at": _iso(self._clock()),
            "reason": reason,
        }
        self._save()

    def _complete_transition(
        self,
        version_id: str,
        rollback_reason: str | None = None,
    ) -> None:
        record = self._deployments[version_id]
        pending = record["pending_transition"]
        if pending is None:
            return
        now = _iso(self._clock())
        record["history"].append(
            {
                "from": pending["from"],
                "to": pending["to"],
                "entered_at": record["entered_state_at"],
                "exited_at": now,
                "reason": pending["reason"],
            }
        )
        record["state"] = pending["to"]
        record["entered_state_at"] = now
        record["pending_transition"] = None
        if rollback_reason is not None:
            record["rollback_reason"] = rollback_reason
        self._save()

    # --- persistence ----------------------------------------------------
    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"deployments": self._deployments}
        with self.registry_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _load(self) -> None:
        if not self.registry_path.exists():
            self._deployments = {}
            return
        with self.registry_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self._deployments = payload.get("deployments", {})


def _iso(when: datetime) -> str:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.isoformat()


def _agrees(c: dict) -> bool:
    if "shadow_label" in c and "production_label" in c:
        return c["shadow_label"] == c["production_label"]
    if "shadow_score" in c and "production_score" in c:
        return abs(float(c["shadow_score"]) - float(c["production_score"])) < 0.1
    return True
