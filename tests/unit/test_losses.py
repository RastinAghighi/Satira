import pytest
import torch
import torch.nn.functional as F

from satira.training.losses import (
    PhasedLossFunction,
    contradiction_gate_loss,
    focal_loss,
    temporal_consistency_loss,
)


CLASS_GATE_TARGETS = {0: 0.1, 1: 0.9, 2: 0.8, 3: 0.2, 4: 0.4}


def test_focal_loss_reduces_to_cross_entropy_when_gamma_zero() -> None:
    torch.manual_seed(0)
    logits = torch.randn(16, 5)
    targets = torch.randint(0, 5, (16,))

    focal = focal_loss(logits, targets, gamma=0.0)
    ce = F.cross_entropy(logits, targets)

    assert torch.allclose(focal, ce, atol=1e-6)


def test_focal_loss_downweights_high_confidence_predictions() -> None:
    targets = torch.tensor([0, 0, 0, 0])

    confident_logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    ce = F.cross_entropy(confident_logits, targets)
    focal = focal_loss(confident_logits, targets, gamma=2.0)

    assert focal.item() < ce.item()
    assert focal.item() < 1e-3, (
        f"focal loss should be near-zero for very confident correct predictions, got {focal.item()}"
    )


def test_focal_loss_does_not_downweight_low_confidence_predictions() -> None:
    targets = torch.tensor([0, 0, 0, 0])
    uniform_logits = torch.zeros(4, 5)

    ce = F.cross_entropy(uniform_logits, targets)
    focal = focal_loss(uniform_logits, targets, gamma=2.0)

    p_t = 1.0 / 5.0
    expected_ratio = (1.0 - p_t) ** 2
    assert torch.isclose(focal, ce * expected_ratio, atol=1e-5)


def test_focal_loss_class_weights_scale_per_class() -> None:
    torch.manual_seed(7)
    logits = torch.randn(32, 3)
    targets = torch.randint(0, 3, (32,))

    base = focal_loss(logits, targets, gamma=2.0)
    doubled = focal_loss(
        logits, targets, gamma=2.0, class_weights=torch.tensor([2.0, 2.0, 2.0])
    )
    assert torch.allclose(doubled, 2.0 * base, atol=1e-5)


def test_focal_loss_class_weights_applied_after_modulation() -> None:
    """Verify class_weights are NOT passed as F.cross_entropy weight."""
    targets = torch.tensor([0, 1])
    logits = torch.tensor(
        [
            [4.0, 0.0],
            [0.0, 4.0],
        ]
    )
    weights = torch.tensor([5.0, 1.0])

    expected_log_probs = F.log_softmax(logits, dim=-1)
    log_p_t = expected_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t = log_p_t.exp()
    focal_weight = (1.0 - p_t) ** 2
    per_sample = focal_weight * (-log_p_t) * weights[targets]
    expected = per_sample.mean()

    actual = focal_loss(logits, targets, gamma=2.0, class_weights=weights)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_focal_loss_returns_scalar() -> None:
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 5, (4,))
    loss = focal_loss(logits, targets)
    assert loss.dim() == 0


def _make_gates(values: torch.Tensor, seq: int = 4, d: int = 8) -> torch.Tensor:
    batch = values.shape[0]
    return values.view(batch, 1, 1).expand(batch, seq, d).contiguous()


def test_contradiction_gate_loss_low_when_matching_target() -> None:
    targets = torch.tensor([0, 1, 2, 3, 4])
    target_values = torch.tensor(
        [CLASS_GATE_TARGETS[int(t)] for t in targets], dtype=torch.float32
    )
    t_gate = _make_gates(target_values)
    v_gate = _make_gates(target_values)

    matching = contradiction_gate_loss(t_gate, v_gate, targets, CLASS_GATE_TARGETS)

    opposing_values = 1.0 - target_values
    t_gate_opp = _make_gates(opposing_values)
    v_gate_opp = _make_gates(opposing_values)
    opposing = contradiction_gate_loss(t_gate_opp, v_gate_opp, targets, CLASS_GATE_TARGETS)

    assert matching.item() < opposing.item()


def test_contradiction_gate_loss_high_when_opposing_target() -> None:
    targets = torch.tensor([1, 1, 1, 1])
    open_values = torch.full((4,), 0.9)
    closed_values = torch.full((4,), 0.1)

    aligned = contradiction_gate_loss(
        _make_gates(open_values), _make_gates(open_values), targets, CLASS_GATE_TARGETS
    )
    opposed = contradiction_gate_loss(
        _make_gates(closed_values),
        _make_gates(closed_values),
        targets,
        CLASS_GATE_TARGETS,
    )

    assert opposed.item() > aligned.item()
    assert opposed.item() > 1.0


def test_contradiction_gate_loss_authentic_class_prefers_closed_gates() -> None:
    targets = torch.tensor([0, 0, 0, 0])
    closed = contradiction_gate_loss(
        _make_gates(torch.full((4,), 0.1)),
        _make_gates(torch.full((4,), 0.1)),
        targets,
        CLASS_GATE_TARGETS,
    )
    open_ = contradiction_gate_loss(
        _make_gates(torch.full((4,), 0.9)),
        _make_gates(torch.full((4,), 0.9)),
        targets,
        CLASS_GATE_TARGETS,
    )
    assert closed.item() < open_.item()


def test_contradiction_gate_loss_returns_scalar() -> None:
    t_gate = torch.rand(8, 4, 16)
    v_gate = torch.rand(8, 4, 16)
    targets = torch.randint(0, 5, (8,))
    loss = contradiction_gate_loss(t_gate, v_gate, targets, CLASS_GATE_TARGETS)
    assert loss.dim() == 0


def test_temporal_consistency_loss_zero_when_logits_equal() -> None:
    logits = torch.randn(8, 5)
    loss = temporal_consistency_loss(logits, logits.clone(), lambda_consistency=0.1)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_temporal_consistency_loss_increases_with_divergence() -> None:
    base = torch.randn(8, 5)

    small_perturbation = base + 0.01 * torch.randn_like(base)
    large_perturbation = base + 1.0 * torch.randn_like(base)

    small_loss = temporal_consistency_loss(base, small_perturbation, lambda_consistency=1.0)
    large_loss = temporal_consistency_loss(base, large_perturbation, lambda_consistency=1.0)

    assert small_loss.item() < large_loss.item()
    assert small_loss.item() >= 0.0


def test_temporal_consistency_loss_is_symmetric() -> None:
    torch.manual_seed(3)
    a = torch.randn(8, 5)
    b = torch.randn(8, 5)

    forward = temporal_consistency_loss(a, b, lambda_consistency=1.0)
    backward = temporal_consistency_loss(b, a, lambda_consistency=1.0)

    assert torch.allclose(forward, backward, atol=1e-6)


def test_temporal_consistency_loss_scales_with_lambda() -> None:
    a = torch.randn(8, 5)
    b = torch.randn(8, 5)

    base = temporal_consistency_loss(a, b, lambda_consistency=0.1)
    scaled = temporal_consistency_loss(a, b, lambda_consistency=1.0)

    assert torch.allclose(scaled, base * 10.0, atol=1e-5)


def _make_phased_loss() -> PhasedLossFunction:
    return PhasedLossFunction(
        class_gate_targets=CLASS_GATE_TARGETS,
        class_weights=torch.tensor([1.0, 2.0, 1.5, 1.0, 1.2]),
        gamma=2.0,
        lambda_consistency=0.1,
    )


def test_phased_loss_phase_one_is_cross_entropy() -> None:
    loss_fn = _make_phased_loss()
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))

    actual = loss_fn.compute(phase=1, logits=logits, targets=targets)
    expected = F.cross_entropy(logits, targets, weight=loss_fn.class_weights)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_phased_loss_phase_two_combines_focal_and_gate() -> None:
    loss_fn = _make_phased_loss()
    torch.manual_seed(11)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    t_gate = torch.rand(8, 4, 16)
    v_gate = torch.rand(8, 4, 16)

    actual = loss_fn.compute(
        phase=2, logits=logits, targets=targets, t_gate=t_gate, v_gate=v_gate
    )
    expected_focal = focal_loss(
        logits, targets, gamma=2.0, class_weights=loss_fn.class_weights
    )
    expected_gate = contradiction_gate_loss(t_gate, v_gate, targets, CLASS_GATE_TARGETS)
    assert torch.allclose(actual, expected_focal + expected_gate, atol=1e-6)


def test_phased_loss_phase_three_adds_temporal_consistency() -> None:
    loss_fn = _make_phased_loss()
    torch.manual_seed(13)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    t_gate = torch.rand(8, 4, 16)
    v_gate = torch.rand(8, 4, 16)
    logits_alt = torch.randn(8, 5)

    actual = loss_fn.compute(
        phase=3,
        logits=logits,
        targets=targets,
        t_gate=t_gate,
        v_gate=v_gate,
        logits_alt_snapshot=logits_alt,
    )
    phase2 = loss_fn.compute(
        phase=2, logits=logits, targets=targets, t_gate=t_gate, v_gate=v_gate
    )
    consistency = temporal_consistency_loss(logits, logits_alt, lambda_consistency=0.1)

    assert torch.allclose(actual, phase2 + consistency, atol=1e-6)


def test_phased_loss_phase_three_requires_alt_snapshot() -> None:
    loss_fn = _make_phased_loss()
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 5, (4,))
    t_gate = torch.rand(4, 4, 16)
    v_gate = torch.rand(4, 4, 16)

    with pytest.raises(ValueError):
        loss_fn.compute(
            phase=3, logits=logits, targets=targets, t_gate=t_gate, v_gate=v_gate
        )


def test_phased_loss_phase_two_requires_gates() -> None:
    loss_fn = _make_phased_loss()
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 5, (4,))

    with pytest.raises(ValueError):
        loss_fn.compute(phase=2, logits=logits, targets=targets)


def test_phased_loss_rejects_unknown_phase() -> None:
    loss_fn = _make_phased_loss()
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 5, (4,))

    with pytest.raises(ValueError):
        loss_fn.compute(phase=4, logits=logits, targets=targets)


def test_phased_loss_returns_scalar_each_phase() -> None:
    loss_fn = _make_phased_loss()
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    t_gate = torch.rand(8, 4, 16)
    v_gate = torch.rand(8, 4, 16)
    logits_alt = torch.randn(8, 5)

    p1 = loss_fn.compute(phase=1, logits=logits, targets=targets)
    p2 = loss_fn.compute(
        phase=2, logits=logits, targets=targets, t_gate=t_gate, v_gate=v_gate
    )
    p3 = loss_fn.compute(
        phase=3,
        logits=logits,
        targets=targets,
        t_gate=t_gate,
        v_gate=v_gate,
        logits_alt_snapshot=logits_alt,
    )
    assert p1.dim() == 0 and p2.dim() == 0 and p3.dim() == 0
