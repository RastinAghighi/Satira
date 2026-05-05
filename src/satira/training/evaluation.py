import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from satira.config import Settings
from satira.models.engine import SatireDetectionEngine


@dataclass
class EvalReport:
    """Structured evaluation result with classification and diagnostic metrics."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_metrics: dict[str, dict]
    calibration_error: float
    gate_analysis: dict
    attention_entropy: dict
    total_samples: int

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class_metrics": self.per_class_metrics,
            "calibration_error": self.calibration_error,
            "gate_analysis": self.gate_analysis,
            "attention_entropy": self.attention_entropy,
            "total_samples": self.total_samples,
        }

    def summary(self) -> str:
        lines = [
            f"EvalReport (n={self.total_samples})",
            f"  accuracy:        {self.accuracy:.4f}",
            f"  macro F1:        {self.macro_f1:.4f}",
            f"  weighted F1:     {self.weighted_f1:.4f}",
            f"  calibration ECE: {self.calibration_error:.4f}",
            "  per-class:",
        ]
        for cls_name, m in self.per_class_metrics.items():
            lines.append(
                f"    {cls_name:>22s}: f1={m['f1']:.3f} "
                f"p={m['precision']:.3f} r={m['recall']:.3f} "
                f"auroc={m['auroc']:.3f}"
            )

        per_class_gates = self.gate_analysis.get("per_class", {})
        if per_class_gates:
            lines.append("  gate activations:")
            for cls_name, stats in per_class_gates.items():
                lines.append(
                    f"    {cls_name:>22s}: mean={stats['mean']:.3f} "
                    f"var={stats['variance']:.4f} n={stats['count']}"
                )

        if self.attention_entropy:
            lines.append("  attention entropy:")
            for stream_name, stats in self.attention_entropy.items():
                if isinstance(stats, dict):
                    lines.append(
                        f"    {stream_name:>22s}: mean={stats.get('mean', 0.0):.3f} "
                        f"normalized={stats.get('normalized_mean', 0.0):.3f} "
                        f"collapsed_frac={stats.get('low_entropy_fraction', 0.0):.3f}"
                    )

        return "\n".join(lines)


class ModelEvaluator:
    """Comprehensive evaluation beyond simple accuracy.

    Metrics:
      * Per-class F1, precision, recall, one-vs-rest AUROC.
      * Macro and support-weighted F1.
      * Expected Calibration Error (ECE) with 15 confidence bins — measures
        whether predicted probability matches empirical correctness.
      * Gate activation analysis: per-class mean and variance of contradiction
        gate activations, useful for spotting collapsed gates.
      * Attention entropy: distribution-level entropy of attention weights.
        Low entropy means attention has collapsed onto a few keys (often bad).
    """

    NUM_ECE_BINS = 15
    LOW_ENTROPY_THRESHOLD = 0.3
    EPS = 1e-12

    def __init__(
        self,
        model: SatireDetectionEngine,
        config: Settings,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.config = config
        use_cuda = torch.cuda.is_available() and device != "cpu"
        self.device = torch.device(device if use_cuda else "cpu")
        self.model.to(self.device)

    def _class_names(self) -> list[str]:
        names = list(getattr(self.config, "CLASS_NAMES", []) or [])
        if len(names) < self.config.num_classes:
            names = names + [
                f"class_{i}" for i in range(len(names), self.config.num_classes)
            ]
        return names[: self.config.num_classes]

    def evaluate(self, dataloader: Iterable[dict]) -> EvalReport:
        self.model.eval()

        all_logits: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_t_gates: list[torch.Tensor] = []
        all_v_gates: list[torch.Tensor] = []
        all_t2v: list[torch.Tensor] = []
        all_v2t: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in dataloader:
                v = batch["v_patches"].to(self.device)
                t = batch["t_tokens"].to(self.device)
                temp = batch["temporal_ctx"].to(self.device)
                graph = batch["graph_ctx"].to(self.device)
                targets = batch["label"].to(self.device, dtype=torch.long)

                logits, t2v_w, v2t_w, t_gate, v_gate = self.model(v, t, temp, graph)

                all_logits.append(logits.detach().cpu())
                all_targets.append(targets.detach().cpu())
                all_t_gates.append(t_gate.detach().cpu())
                all_v_gates.append(v_gate.detach().cpu())
                all_t2v.append(t2v_w.detach().cpu())
                all_v2t.append(v2t_w.detach().cpu())

        if not all_logits:
            return EvalReport(
                accuracy=0.0,
                macro_f1=0.0,
                weighted_f1=0.0,
                per_class_metrics={},
                calibration_error=0.0,
                gate_analysis={"per_class": {}, "overall_mean": 0.0, "overall_variance": 0.0},
                attention_entropy={},
                total_samples=0,
            )

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        t_gates = torch.cat(all_t_gates, dim=0)
        v_gates = torch.cat(all_v_gates, dim=0)
        t2v_weights = torch.cat(all_t2v, dim=0)
        v2t_weights = torch.cat(all_v2t, dim=0)

        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        accuracy = (preds == targets).float().mean().item()

        per_class, macro_f1, weighted_f1 = self._classification_metrics(
            probs=probs, preds=preds, targets=targets
        )
        ece = self.calibration_error(predictions=probs, targets=targets)

        combined_gates = torch.cat([t_gates, v_gates], dim=1)
        gates = self.gate_analysis(combined_gates, targets)

        attention = {
            "t2v": self.attention_entropy(t2v_weights),
            "v2t": self.attention_entropy(v2t_weights),
        }

        return EvalReport(
            accuracy=float(accuracy),
            macro_f1=float(macro_f1),
            weighted_f1=float(weighted_f1),
            per_class_metrics=per_class,
            calibration_error=float(ece),
            gate_analysis=gates,
            attention_entropy=attention,
            total_samples=int(targets.numel()),
        )

    def calibration_error(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Expected Calibration Error with 15 confidence bins."""
        if predictions.numel() == 0 or targets.numel() == 0:
            return 0.0

        confidence, preds = predictions.max(dim=-1)
        correct = (preds == targets).float()
        n = confidence.size(0)

        bin_edges = torch.linspace(0.0, 1.0, self.NUM_ECE_BINS + 1)
        ece = 0.0
        for i in range(self.NUM_ECE_BINS):
            lo = bin_edges[i].item()
            hi = bin_edges[i + 1].item()
            if i == self.NUM_ECE_BINS - 1:
                in_bin = (confidence >= lo) & (confidence <= hi)
            else:
                in_bin = (confidence >= lo) & (confidence < hi)
            bin_size = int(in_bin.sum().item())
            if bin_size == 0:
                continue
            avg_conf = confidence[in_bin].mean().item()
            avg_acc = correct[in_bin].mean().item()
            ece += (bin_size / n) * abs(avg_conf - avg_acc)

        return float(ece)

    def gate_analysis(
        self, gate_activations: torch.Tensor, targets: torch.Tensor
    ) -> dict:
        """Per-class mean and variance of contradiction-gate activations."""
        if gate_activations.numel() == 0 or targets.numel() == 0:
            return {"per_class": {}, "overall_mean": 0.0, "overall_variance": 0.0}

        flat = gate_activations.reshape(gate_activations.size(0), -1).float()
        sample_means = flat.mean(dim=-1)

        names = self._class_names()
        per_class: dict[str, dict] = {}
        for c in range(self.config.num_classes):
            mask = targets == c
            count = int(mask.sum().item())
            cls_name = names[c]
            if count == 0:
                per_class[cls_name] = {"mean": 0.0, "variance": 0.0, "count": 0}
                continue
            cls_values = flat[mask]
            per_class[cls_name] = {
                "mean": float(cls_values.mean().item()),
                "variance": float(cls_values.var(unbiased=False).item()),
                "count": count,
            }

        overall_var = (
            float(sample_means.var(unbiased=False).item())
            if sample_means.numel() > 1
            else 0.0
        )
        return {
            "per_class": per_class,
            "overall_mean": float(flat.mean().item()),
            "overall_variance": overall_var,
        }

    def attention_entropy(self, attention_weights: torch.Tensor) -> dict:
        """Entropy stats for an attention-weights tensor.

        Expects weights normalized over the last dim (key axis). Reports the
        mean entropy, normalized entropy (entropy / log(K)), and the fraction
        of query positions whose normalized entropy falls below
        ``LOW_ENTROPY_THRESHOLD`` — those positions have collapsed attention.
        """
        if attention_weights.numel() == 0:
            return {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "normalized_mean": 0.0,
                "low_entropy_fraction": 0.0,
            }

        probs = attention_weights.clamp_min(self.EPS).float()
        entropy = -(probs * probs.log()).sum(dim=-1)

        key_dim = attention_weights.size(-1)
        max_entropy = math.log(key_dim) if key_dim > 1 else 1.0
        normalized = entropy / max_entropy

        return {
            "mean": float(entropy.mean().item()),
            "min": float(entropy.min().item()),
            "max": float(entropy.max().item()),
            "normalized_mean": float(normalized.mean().item()),
            "low_entropy_fraction": float(
                (normalized < self.LOW_ENTROPY_THRESHOLD).float().mean().item()
            ),
        }

    def _classification_metrics(
        self,
        probs: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[dict[str, dict], float, float]:
        names = self._class_names()
        per_class: dict[str, dict] = {}
        f1_values: list[float] = []
        supports: list[int] = []
        total = int(targets.numel())

        for c in range(self.config.num_classes):
            tp = int(((preds == c) & (targets == c)).sum().item())
            fp = int(((preds == c) & (targets != c)).sum().item())
            fn = int(((preds != c) & (targets == c)).sum().item())
            support = tp + fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            auroc = self._auroc_one_vs_rest(probs[:, c], (targets == c).long())

            per_class[names[c]] = {
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "auroc": float(auroc),
                "support": support,
            }
            f1_values.append(f1)
            supports.append(support)

        macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0
        if total > 0:
            weighted_f1 = sum(f * s for f, s in zip(f1_values, supports)) / total
        else:
            weighted_f1 = 0.0

        return per_class, float(macro_f1), float(weighted_f1)

    def _auroc_one_vs_rest(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        n = scores.numel()
        if n == 0:
            return 0.5
        n_pos = int(labels.sum().item())
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5

        order = scores.argsort(descending=True)
        sorted_labels = labels[order].float()
        cum_pos = sorted_labels.cumsum(0)
        neg_at_each_rank = (1.0 - sorted_labels) * cum_pos
        return float(neg_at_each_rank.sum().item() / (n_pos * n_neg))
