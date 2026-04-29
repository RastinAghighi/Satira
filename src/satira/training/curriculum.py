import math


class CurriculumScheduler:
    """Manages smooth transitions between data tiers using sigmoid scheduling.
    No hard cutoffs between phases — prevents distribution cliffs.

    At epoch 1: ~80% Tier 1, ~15% Tier 2, ~5% Tier 3
    At epoch 10: ~10% Tier 1, ~60% Tier 2, ~30% Tier 3
    At epoch 20: ~5% Tier 1, ~20% Tier 2, ~75% Tier 3
    """

    TIER1_DECAY_CENTER = 5
    TIER2_RISE_CENTER = 5
    TIER2_FALL_CENTER = 16
    TIER3_RISE_CENTER = 14
    SLOPE = 0.5

    def __init__(self, total_epochs: int = 25) -> None:
        self.total_epochs = total_epochs

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def get_tier_weights(self, epoch: int) -> dict[str, float]:
        tier1_raw = self._sigmoid(self.SLOPE * (self.TIER1_DECAY_CENTER - epoch))
        tier2_raw = self._sigmoid(
            self.SLOPE * (epoch - self.TIER2_RISE_CENTER)
        ) - self._sigmoid(self.SLOPE * (epoch - self.TIER2_FALL_CENTER))
        tier3_raw = self._sigmoid(self.SLOPE * (epoch - self.TIER3_RISE_CENTER))

        tier2_raw = max(tier2_raw, 0.0)

        total = tier1_raw + tier2_raw + tier3_raw
        return {
            "tier1_easy": tier1_raw / total,
            "tier2_contradiction": tier2_raw / total,
            "tier3_hard_negatives": tier3_raw / total,
        }


class PhaseTransitionController:
    """Determines when to advance training phases based on convergence metrics,
    NOT fixed epoch counts.

    Phase 1->2: loss plateau AND projection gradient norm < 0.1
    Phase 2->3: loss plateau AND gate activation variance > 0.15
    Phase 3: runs until early stopping
    """

    GRAD_NORM_THRESHOLD = 0.1
    GATE_VARIANCE_THRESHOLD = 0.15

    def __init__(self, patience: int = 3) -> None:
        self.patience = patience
        self._phase = 1
        self._best_loss = float("inf")
        self._epochs_since_improvement = 0

    @property
    def phase(self) -> int:
        return self._phase

    def _update_plateau(self, loss: float) -> bool:
        if loss < self._best_loss - 1e-6:
            self._best_loss = loss
            self._epochs_since_improvement = 0
        else:
            self._epochs_since_improvement += 1
        return self._epochs_since_improvement >= self.patience

    def _advance(self) -> None:
        self._phase += 1
        self._best_loss = float("inf")
        self._epochs_since_improvement = 0

    def should_advance_phase(self, epoch: int, metrics: dict) -> bool:
        loss = metrics["loss"]

        if self._phase == 3:
            self._update_plateau(loss)
            return False

        plateau = self._update_plateau(loss)
        if not plateau:
            return False

        if self._phase == 1:
            grad_norm = metrics.get("projection_grad_norm", float("inf"))
            if grad_norm < self.GRAD_NORM_THRESHOLD:
                self._advance()
                return True
            return False

        gate_variance = metrics.get("gate_activation_variance", 0.0)
        if gate_variance > self.GATE_VARIANCE_THRESHOLD:
            self._advance()
            return True
        return False

    def get_optimizer_config(self) -> dict:
        if self._phase == 1:
            return {
                "phase": 1,
                "unfrozen_modules": ["projections"],
                "learning_rate": 1e-3,
            }
        if self._phase == 2:
            return {
                "phase": 2,
                "unfrozen_modules": [
                    "projections",
                    "cross_attention",
                    "reasoning",
                    "modality_dropout",
                ],
                "learning_rate": 5e-4,
            }
        if self._phase == 3:
            return {
                "phase": 3,
                "unfrozen_modules": ["all"],
                "learning_rate": 1e-4,
            }
        raise ValueError(f"Unknown phase {self._phase}")
