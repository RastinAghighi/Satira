import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal Loss with class weighting applied AFTER focal modulation.

    Passing class_weights as the ``weight`` parameter to F.cross_entropy bakes
    the weighting into ce_loss before computing p_t, contaminating the focal
    modulation factor (1 - p_t)^gamma. We compute log_probs unweighted, gather
    p_t for the true class, then apply class_weights as a separate scalar per
    sample.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_p_t = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    p_t = log_p_t.exp()

    focal_weight = (1.0 - p_t).pow(gamma)
    ce_per_sample = -log_p_t

    loss = focal_weight * ce_per_sample

    if class_weights is not None:
        sample_weights = class_weights.to(loss.device, dtype=loss.dtype)[targets]
        loss = loss * sample_weights

    return loss.mean()


def _build_class_gate_target_tensor(
    class_gate_targets: dict[int, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    max_class = max(class_gate_targets.keys())
    return torch.tensor(
        [class_gate_targets[i] for i in range(max_class + 1)],
        device=device,
        dtype=dtype,
    )


def contradiction_gate_loss(
    t_gate: torch.Tensor,
    v_gate: torch.Tensor,
    targets: torch.Tensor,
    class_gate_targets: dict[int, float],
) -> torch.Tensor:
    """Supervised gate loss against class-specific gate targets.

    L1 sparsity pushes ALL gates toward zero regardless of whether contradiction
    exists. This supervised version trains gates against per-class targets so
    that satire/parody samples (which need cross-modal contradiction signal)
    keep their gates open while authentic samples close them.

    The mean activation per sample (averaged across all non-batch dims of t_gate
    and v_gate jointly) is compared via BCE to the target for that sample's
    class.
    """
    if t_gate.size(0) != v_gate.size(0):
        raise ValueError("t_gate and v_gate must share the batch dimension")

    batch_size = t_gate.size(0)
    t_mean = t_gate.reshape(batch_size, -1).mean(dim=-1)
    v_mean = v_gate.reshape(batch_size, -1).mean(dim=-1)
    gate_mean = 0.5 * (t_mean + v_mean)

    target_table = _build_class_gate_target_tensor(
        class_gate_targets, device=gate_mean.device, dtype=gate_mean.dtype
    )
    sample_targets = target_table[targets]

    eps = 1e-7
    gate_mean = gate_mean.clamp(min=eps, max=1.0 - eps)

    return F.binary_cross_entropy(gate_mean, sample_targets)


def temporal_consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    lambda_consistency: float = 0.1,
) -> torch.Tensor:
    """Symmetric KL between predictions from two graph snapshots.

    Penalizes over-reliance on a specific graph embedding version: a robust
    model should predict similarly when the same sample is scored against an
    older or newer snapshot of the entity graph.
    """
    log_p = F.log_softmax(logits_a, dim=-1)
    log_q = F.log_softmax(logits_b, dim=-1)
    p = log_p.exp()
    q = log_q.exp()

    kl_pq = F.kl_div(log_q, p, reduction="batchmean")
    kl_qp = F.kl_div(log_p, q, reduction="batchmean")

    return lambda_consistency * (kl_pq + kl_qp)


class PhasedLossFunction(nn.Module):
    """Composes the loss for each curriculum phase.

    Phase 1: Cross-entropy only (warm-up; no focal pressure, no gate guidance).
    Phase 2: Focal + supervised contradiction gate loss.
    Phase 3: Phase 2 + temporal consistency across graph snapshots.
    """

    def __init__(
        self,
        class_gate_targets: dict[int, float],
        class_weights: torch.Tensor,
        gamma: float = 2.0,
        lambda_consistency: float = 0.1,
        gate_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.class_gate_targets = dict(class_gate_targets)
        self.register_buffer("class_weights", class_weights.float())
        self.gamma = gamma
        self.lambda_consistency = lambda_consistency
        self.gate_loss_weight = gate_loss_weight

    def compute(
        self,
        phase: int,
        logits: torch.Tensor,
        targets: torch.Tensor,
        t_gate: torch.Tensor | None = None,
        v_gate: torch.Tensor | None = None,
        logits_alt_snapshot: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if phase == 1:
            return F.cross_entropy(logits, targets, weight=self.class_weights)

        if phase == 2:
            if t_gate is None or v_gate is None:
                raise ValueError("Phase 2 requires t_gate and v_gate")
            cls_loss = focal_loss(
                logits, targets, gamma=self.gamma, class_weights=self.class_weights
            )
            gate_loss = contradiction_gate_loss(
                t_gate, v_gate, targets, self.class_gate_targets
            )
            return cls_loss + self.gate_loss_weight * gate_loss

        if phase == 3:
            if t_gate is None or v_gate is None:
                raise ValueError("Phase 3 requires t_gate and v_gate")
            if logits_alt_snapshot is None:
                raise ValueError("Phase 3 requires logits_alt_snapshot")
            cls_loss = focal_loss(
                logits, targets, gamma=self.gamma, class_weights=self.class_weights
            )
            gate_loss = contradiction_gate_loss(
                t_gate, v_gate, targets, self.class_gate_targets
            )
            consistency = temporal_consistency_loss(
                logits, logits_alt_snapshot, lambda_consistency=self.lambda_consistency
            )
            return cls_loss + self.gate_loss_weight * gate_loss + consistency

        raise ValueError(f"Unknown phase {phase}; expected 1, 2, or 3")
