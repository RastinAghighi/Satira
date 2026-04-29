import math

from satira.training.curriculum import CurriculumScheduler, PhaseTransitionController


TIER_KEYS = ("tier1_easy", "tier2_contradiction", "tier3_hard_negatives")


def test_tier_weights_sum_to_one_at_every_epoch() -> None:
    scheduler = CurriculumScheduler(total_epochs=25)
    for epoch in range(1, 26):
        weights = scheduler.get_tier_weights(epoch)
        total = sum(weights[k] for k in TIER_KEYS)
        assert math.isclose(total, 1.0, abs_tol=1e-6), (
            f"weights at epoch {epoch} sum to {total}: {weights}"
        )


def test_tier_one_dominant_at_first_epoch() -> None:
    scheduler = CurriculumScheduler(total_epochs=25)
    weights = scheduler.get_tier_weights(1)
    assert weights["tier1_easy"] > weights["tier2_contradiction"]
    assert weights["tier1_easy"] > weights["tier3_hard_negatives"]


def test_tier_three_dominant_at_final_epoch() -> None:
    scheduler = CurriculumScheduler(total_epochs=25)
    weights = scheduler.get_tier_weights(25)
    assert weights["tier3_hard_negatives"] > weights["tier1_easy"]
    assert weights["tier3_hard_negatives"] > weights["tier2_contradiction"]


def test_tier_weights_change_smoothly_between_consecutive_epochs() -> None:
    scheduler = CurriculumScheduler(total_epochs=25)
    prev = scheduler.get_tier_weights(1)
    for epoch in range(2, 26):
        cur = scheduler.get_tier_weights(epoch)
        for key in TIER_KEYS:
            jump = abs(cur[key] - prev[key])
            assert jump <= 0.15, (
                f"weight {key} jumped by {jump:.4f} between epoch {epoch - 1} and {epoch}"
            )
        prev = cur


def test_phase_controller_starts_in_phase_one() -> None:
    controller = PhaseTransitionController(patience=3)
    assert controller.phase == 1


def test_phase_controller_stays_in_phase_one_when_loss_decreases() -> None:
    controller = PhaseTransitionController(patience=3)
    losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    for i, loss in enumerate(losses, start=1):
        advanced = controller.should_advance_phase(
            epoch=i,
            metrics={"loss": loss, "projection_grad_norm": 0.05},
        )
        assert advanced is False, f"unexpectedly advanced at epoch {i} with decreasing loss"
    assert controller.phase == 1


def test_phase_controller_advances_on_plateau_and_low_grad_norm() -> None:
    controller = PhaseTransitionController(patience=3)

    controller.should_advance_phase(
        epoch=1, metrics={"loss": 0.5, "projection_grad_norm": 0.05}
    )

    history = []
    for epoch in range(2, 5):
        advanced = controller.should_advance_phase(
            epoch=epoch,
            metrics={"loss": 0.51, "projection_grad_norm": 0.05},
        )
        history.append(advanced)

    assert any(history), f"controller never advanced despite plateau + low grad norm: {history}"
    assert controller.phase == 2


def test_phase_one_does_not_advance_on_plateau_when_grad_norm_high() -> None:
    controller = PhaseTransitionController(patience=2)
    controller.should_advance_phase(
        epoch=1, metrics={"loss": 0.5, "projection_grad_norm": 5.0}
    )
    controller.should_advance_phase(
        epoch=2, metrics={"loss": 0.51, "projection_grad_norm": 5.0}
    )
    advanced = controller.should_advance_phase(
        epoch=3, metrics={"loss": 0.51, "projection_grad_norm": 5.0}
    )
    assert advanced is False
    assert controller.phase == 1


def test_phase_two_does_not_advance_when_gate_variance_low() -> None:
    controller = PhaseTransitionController(patience=2)

    controller.should_advance_phase(
        epoch=1, metrics={"loss": 0.5, "projection_grad_norm": 0.05}
    )
    controller.should_advance_phase(
        epoch=2, metrics={"loss": 0.51, "projection_grad_norm": 0.05}
    )
    advanced_to_two = controller.should_advance_phase(
        epoch=3, metrics={"loss": 0.51, "projection_grad_norm": 0.05}
    )
    assert advanced_to_two is True
    assert controller.phase == 2

    controller.should_advance_phase(
        epoch=4, metrics={"loss": 0.4, "gate_activation_variance": 0.05}
    )
    controller.should_advance_phase(
        epoch=5, metrics={"loss": 0.41, "gate_activation_variance": 0.05}
    )
    advanced_to_three = controller.should_advance_phase(
        epoch=6, metrics={"loss": 0.41, "gate_activation_variance": 0.05}
    )

    assert advanced_to_three is False
    assert controller.phase == 2


def test_phase_two_advances_on_plateau_and_high_gate_variance() -> None:
    controller = PhaseTransitionController(patience=2)

    controller.should_advance_phase(
        epoch=1, metrics={"loss": 0.5, "projection_grad_norm": 0.05}
    )
    controller.should_advance_phase(
        epoch=2, metrics={"loss": 0.51, "projection_grad_norm": 0.05}
    )
    controller.should_advance_phase(
        epoch=3, metrics={"loss": 0.51, "projection_grad_norm": 0.05}
    )
    assert controller.phase == 2

    controller.should_advance_phase(
        epoch=4, metrics={"loss": 0.4, "gate_activation_variance": 0.25}
    )
    controller.should_advance_phase(
        epoch=5, metrics={"loss": 0.41, "gate_activation_variance": 0.25}
    )
    advanced = controller.should_advance_phase(
        epoch=6, metrics={"loss": 0.41, "gate_activation_variance": 0.25}
    )

    assert advanced is True
    assert controller.phase == 3


def test_phase_three_never_advances() -> None:
    controller = PhaseTransitionController(patience=1)
    controller._phase = 3  # type: ignore[attr-defined]

    for epoch in range(1, 10):
        advanced = controller.should_advance_phase(
            epoch=epoch,
            metrics={"loss": 0.5, "gate_activation_variance": 0.5},
        )
        assert advanced is False
    assert controller.phase == 3


def test_optimizer_config_returns_phase_specific_modules() -> None:
    controller = PhaseTransitionController(patience=1)

    cfg1 = controller.get_optimizer_config()
    assert cfg1["phase"] == 1
    assert "projections" in cfg1["unfrozen_modules"]
    assert cfg1["learning_rate"] > 0

    controller._phase = 2  # type: ignore[attr-defined]
    cfg2 = controller.get_optimizer_config()
    assert cfg2["phase"] == 2
    assert len(cfg2["unfrozen_modules"]) > len(cfg1["unfrozen_modules"])
    assert cfg2["learning_rate"] < cfg1["learning_rate"]

    controller._phase = 3  # type: ignore[attr-defined]
    cfg3 = controller.get_optimizer_config()
    assert cfg3["phase"] == 3
    assert cfg3["learning_rate"] < cfg2["learning_rate"]
