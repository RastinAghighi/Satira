from datetime import datetime, timedelta, timezone

import pytest

from satira.deployment.controller import (
    DeploymentState,
    DeploymentStateMachine,
    StagedModelDeployment,
)


VERSION = "cls-v2"
T0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


class _ManualClock:
    """A minimal advancing clock so tests can satisfy MIN_DWELL deterministically."""

    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now = self.now + timedelta(**kwargs)


def _controller(tmp_path, clock: _ManualClock | None = None) -> StagedModelDeployment:
    return StagedModelDeployment(
        registry_path=str(tmp_path / "registry.json"),
        clock=clock or _ManualClock(),
    )


def _drive_to(controller: StagedModelDeployment, clock: _ManualClock, target: DeploymentState) -> None:
    """Walk a deployment from SHADOW up to ``target`` honouring dwell."""
    sequence = [
        DeploymentState.CANARY_5,
        DeploymentState.CANARY_25,
        DeploymentState.CANARY_50,
        DeploymentState.PRODUCTION,
    ]
    for next_state in sequence:
        if controller.get_current_state(VERSION) == target:
            return
        dwell = controller.state_machine.min_dwell(controller.get_current_state(VERSION))
        if dwell is not None:
            clock.advance(seconds=int(dwell.total_seconds()) + 1)
        controller.execute_promotion(VERSION, next_state)


# --- promotion sequence -----------------------------------------------------
def test_promotes_through_full_sequence(tmp_path) -> None:
    clock = _ManualClock()
    controller = _controller(tmp_path, clock)
    controller.create_deployment(VERSION)

    expected = [
        DeploymentState.CANARY_5,
        DeploymentState.CANARY_25,
        DeploymentState.CANARY_50,
        DeploymentState.PRODUCTION,
    ]
    for next_state in expected:
        # Wait out the minimum dwell for the current state, then promote.
        dwell = controller.state_machine.min_dwell(controller.get_current_state(VERSION))
        clock.advance(seconds=int(dwell.total_seconds()) + 1)
        controller.execute_promotion(VERSION, next_state)
        assert controller.get_current_state(VERSION) == next_state

    # Promotion past the terminal state is rejected.
    with pytest.raises(ValueError):
        controller.execute_promotion(VERSION, DeploymentState.PRODUCTION)


def test_promotion_rejects_skipping_a_stage(tmp_path) -> None:
    clock = _ManualClock()
    controller = _controller(tmp_path, clock)
    controller.create_deployment(VERSION)
    clock.advance(hours=24)

    # SHADOW → CANARY_25 skips CANARY_5; must be rejected.
    with pytest.raises(ValueError):
        controller.execute_promotion(VERSION, DeploymentState.CANARY_25)


# --- rollback ---------------------------------------------------------------
@pytest.mark.parametrize(
    "stop_at",
    [
        DeploymentState.SHADOW,
        DeploymentState.CANARY_5,
        DeploymentState.CANARY_25,
        DeploymentState.CANARY_50,
        DeploymentState.PRODUCTION,
    ],
)
def test_rollback_allowed_from_any_state(tmp_path, stop_at) -> None:
    clock = _ManualClock()
    controller = _controller(tmp_path, clock)
    controller.create_deployment(VERSION)
    _drive_to(controller, clock, stop_at)
    assert controller.get_current_state(VERSION) == stop_at

    controller.execute_rollback(VERSION, reason="manual abort")
    assert controller.get_current_state(VERSION) == DeploymentState.ROLLED_BACK


def test_cannot_rollback_an_already_rolled_back_deployment(tmp_path) -> None:
    controller = _controller(tmp_path)
    controller.create_deployment(VERSION)
    controller.execute_rollback(VERSION, reason="first")
    with pytest.raises(ValueError):
        controller.execute_rollback(VERSION, reason="second")


# --- dwell time -------------------------------------------------------------
def test_promotion_blocked_before_minimum_dwell(tmp_path) -> None:
    clock = _ManualClock()
    controller = _controller(tmp_path, clock)
    controller.create_deployment(VERSION)

    # SHADOW dwell is 6h; 1h is not enough.
    clock.advance(hours=1)
    with pytest.raises(ValueError, match="dwell"):
        controller.execute_promotion(VERSION, DeploymentState.CANARY_5)

    # After the dwell elapses the same call succeeds.
    clock.advance(hours=6)
    controller.execute_promotion(VERSION, DeploymentState.CANARY_5)
    assert controller.get_current_state(VERSION) == DeploymentState.CANARY_5


def test_gate_check_holds_when_dwell_not_met(tmp_path) -> None:
    clock = _ManualClock()
    controller = _controller(tmp_path, clock)
    controller.create_deployment(VERSION)
    clock.advance(hours=1)

    decision = controller.run_gate_check(
        VERSION,
        metrics={"drift_score": 0.1, "override_rate": 0.01, "calibration_divergence": 0.01},
    )
    assert decision == "hold"


# --- gate-check rollback triggers ------------------------------------------
def test_gate_check_rolls_back_on_high_drift(tmp_path) -> None:
    controller = _controller(tmp_path)
    controller.create_deployment(VERSION)
    decision = controller.run_gate_check(
        VERSION,
        metrics={"drift_score": 0.9, "override_rate": 0.0, "calibration_divergence": 0.0},
    )
    assert decision == "rollback"


def test_gate_check_rolls_back_on_high_override_rate(tmp_path) -> None:
    controller = _controller(tmp_path)
    controller.create_deployment(VERSION)
    decision = controller.run_gate_check(
        VERSION,
        metrics={"drift_score": 0.1, "override_rate": 0.20, "calibration_divergence": 0.01},
    )
    assert decision == "rollback"


def test_gate_check_rolls_back_on_calibration_divergence(tmp_path) -> None:
    controller = _controller(tmp_path)
    controller.create_deployment(VERSION)
    decision = controller.run_gate_check(
        VERSION,
        metrics={"drift_score": 0.1, "override_rate": 0.01, "calibration_divergence": 0.30},
    )
    assert decision == "rollback"


def test_gate_check_promotes_when_clean_and_dwell_satisfied(tmp_path) -> None:
    clock = _ManualClock()
    controller = _controller(tmp_path, clock)
    controller.create_deployment(VERSION)
    clock.advance(hours=7)

    decision = controller.run_gate_check(
        VERSION,
        metrics={"drift_score": 0.1, "override_rate": 0.01, "calibration_divergence": 0.01},
    )
    assert decision == "promote"


def test_gate_check_holds_when_drift_in_warning_band(tmp_path) -> None:
    clock = _ManualClock()
    controller = _controller(tmp_path, clock)
    controller.create_deployment(VERSION)
    clock.advance(hours=7)

    # 0.6 is between WARNING_DRIFT (0.5) and DRIFT_ROLLBACK (0.75) — hold,
    # don't promote, but also don't roll back.
    decision = controller.run_gate_check(
        VERSION,
        metrics={"drift_score": 0.6, "override_rate": 0.01, "calibration_divergence": 0.01},
    )
    assert decision == "hold"


# --- promotion-readiness evaluation ----------------------------------------
def test_evaluate_promotion_readiness_promotes_high_agreement(tmp_path) -> None:
    controller = _controller(tmp_path)
    comparisons = [
        {"shadow_label": True, "production_label": True, "drift_value": 0.1}
        for _ in range(100)
    ]
    # One disagreement on a low-drift sample — overall agreement 99%.
    comparisons.append(
        {"shadow_label": False, "production_label": True, "drift_value": 0.1}
    )

    result = controller.evaluate_promotion_readiness(comparisons)
    assert result["recommendation"] == "promote"
    assert result["agreement_rate"] >= 0.95
    assert result["drift_correlated_disagreements"] == 0.0


def test_drift_correlated_disagreements_block_promotion(tmp_path) -> None:
    controller = _controller(tmp_path)

    # 100 comparisons, 95% overall agreement — would promote on raw rate
    # alone — but every disagreement is in the high-drift bucket, which
    # is the failure mode we explicitly want to catch.
    comparisons = []
    for _ in range(95):
        comparisons.append(
            {"shadow_label": True, "production_label": True, "drift_value": 0.1}
        )
    for _ in range(5):
        comparisons.append(
            {"shadow_label": False, "production_label": True, "drift_value": 0.8}
        )

    result = controller.evaluate_promotion_readiness(comparisons)
    assert result["agreement_rate"] == pytest.approx(0.95)
    # 5 / 5 high-drift samples disagree.
    assert result["drift_correlated_disagreements"] == pytest.approx(1.0)
    assert result["recommendation"] == "block"


def test_evaluate_promotion_readiness_handles_no_comparisons(tmp_path) -> None:
    controller = _controller(tmp_path)
    result = controller.evaluate_promotion_readiness([])
    assert result["recommendation"] == "hold"
    assert result["sample_size"] == 0


# --- crash recovery / persistence ------------------------------------------
def test_state_persists_across_controller_instances(tmp_path) -> None:
    clock = _ManualClock()
    path = tmp_path / "registry.json"

    a = StagedModelDeployment(registry_path=str(path), clock=clock)
    a.create_deployment(VERSION)
    clock.advance(hours=7)
    a.execute_promotion(VERSION, DeploymentState.CANARY_5)

    # A second controller — same registry, fresh memory — sees the
    # promoted state without any in-process handoff.
    b = StagedModelDeployment(registry_path=str(path), clock=clock)
    assert b.get_current_state(VERSION) == DeploymentState.CANARY_5


# --- state machine ---------------------------------------------------------
def test_state_machine_promotion_path_is_linear() -> None:
    sm = DeploymentStateMachine()
    assert sm.can_promote(DeploymentState.SHADOW) == DeploymentState.CANARY_5
    assert sm.can_promote(DeploymentState.CANARY_5) == DeploymentState.CANARY_25
    assert sm.can_promote(DeploymentState.CANARY_25) == DeploymentState.CANARY_50
    assert sm.can_promote(DeploymentState.CANARY_50) == DeploymentState.PRODUCTION
    assert sm.can_promote(DeploymentState.PRODUCTION) is None
    assert sm.can_promote(DeploymentState.ROLLED_BACK) is None


def test_state_machine_rollback_blocked_only_at_rolled_back() -> None:
    sm = DeploymentStateMachine()
    for state in DeploymentState:
        if state == DeploymentState.ROLLED_BACK:
            assert not sm.can_rollback(state)
        else:
            assert sm.can_rollback(state)
