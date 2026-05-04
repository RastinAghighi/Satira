"""Model + context version compatibility matrix.

The inference path pairs a Satira classifier checkpoint with a graph
embedding snapshot (the "context"). Two checkpoints can disagree about
how to read a single embedding vector — different GNN architecture
versions produce incompatible coordinate systems, and even the same
architecture drifts apart as the offline pipeline re-trains the GNN.

This registry tracks every (model, context) pair we've registered and
decides whether a given combination is safe to load. Two layers:

  1. Hard — the context's GNN architecture version must appear on the
     model's allow-list. A mismatch is unrecoverable: the projection
     layer has memorised the old coordinate system and cannot be
     "rotated back" at inference time.
  2. Soft — the embedding distribution shouldn't have drifted past
     tolerance. Same composite metric as ``EmbeddingDriftMonitor``
     (marginal mean shift + norm scale + top-SV spectrum), so the two
     monitors agree on what "drift" means when they look at the same
     data.

Persisted to JSON so deployment tooling can ask "what's the latest
context I can pair with model X?" without holding a process open.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


_EPS = 1e-8

# Composite drift weights mirror EmbeddingDriftMonitor — the registry
# and the live monitor must score the same data the same way, otherwise
# a context can pass the offline gate and still trip the production
# alarm (or vice-versa).
_MARGINAL_W = 0.30
_NORM_W = 0.25
_STRUCT_W = 0.45

_MARGINAL_SAT_Z = 5.0
_NORM_SAT_FRAC = 1.0


@dataclass
class CompatibilityResult:
    compatible: bool
    reason: str
    severity: str  # one of "healthy", "warning", "critical"


class CompatibilityMatrix:
    """Registry of model and context versions with hard/soft compatibility checks.

    Hard compatibility: a context's GNN architecture version must
    appear on the model's allow-list. Soft compatibility: the embedding
    distribution (mean / norm / SV spectrum) must not have drifted past
    ``CRITICAL_THRESHOLD``. A composite drift score in
    ``[WARNING_THRESHOLD, CRITICAL_THRESHOLD)`` is still loadable but
    surfaces a warning so on-call can decide whether to recalibrate.
    """

    WARNING_THRESHOLD = 0.5
    CRITICAL_THRESHOLD = 0.75

    def __init__(self, registry_path: str = "./data/compatibility_registry.json") -> None:
        self.registry_path = Path(registry_path)
        self._models: dict[str, dict] = {}
        self._contexts: dict[str, dict] = {}
        self._load()

    # --- registration ---------------------------------------------------
    def register_model(
        self,
        model_version: str,
        compatible_gnn_versions: list[str],
        graph_snapshot_window: tuple[str, str],
        training_stats: dict,
    ) -> None:
        """Record a classifier checkpoint and the graph state it was trained on."""
        self._models[model_version] = {
            "model_version": model_version,
            "compatible_gnn_versions": list(compatible_gnn_versions),
            "graph_snapshot_window": [graph_snapshot_window[0], graph_snapshot_window[1]],
            "training_stats": _stats_to_json(training_stats),
            "registered_at": _now_iso(),
        }
        self._save()

    def register_context(
        self,
        context_version: str,
        gnn_version: str,
        embedding_stats: dict,
    ) -> None:
        """Record a graph embedding snapshot and which GNN produced it.

        Re-registering an existing version bumps it to the most-recent
        position so ``find_best_compatible_context`` returns the latest
        upload first.
        """
        self._contexts.pop(context_version, None)
        self._contexts[context_version] = {
            "context_version": context_version,
            "gnn_version": gnn_version,
            "embedding_stats": _stats_to_json(embedding_stats),
            "registered_at": _now_iso(),
        }
        self._save()

    # --- queries --------------------------------------------------------
    def check_compatibility(
        self,
        model_version: str,
        context_version: str,
    ) -> CompatibilityResult:
        if model_version not in self._models:
            raise KeyError(f"unknown model version {model_version!r}")
        if context_version not in self._contexts:
            raise KeyError(f"unknown context version {context_version!r}")

        model = self._models[model_version]
        context = self._contexts[context_version]

        # Hard: architecture allow-list. Even a perfect-looking
        # distribution is unsafe if the GNN itself is a different model.
        if context["gnn_version"] not in model["compatible_gnn_versions"]:
            return CompatibilityResult(
                compatible=False,
                severity="critical",
                reason=(
                    f"GNN architecture mismatch: context uses "
                    f"{context['gnn_version']!r}, model accepts "
                    f"{model['compatible_gnn_versions']}"
                ),
            )

        # Soft: distributional drift between training-time stats and
        # current embedding stats.
        train_stats = _stats_from_json(model["training_stats"])
        embed_stats = _stats_from_json(context["embedding_stats"])
        score = _compute_drift_score(train_stats, embed_stats)

        if score >= self.CRITICAL_THRESHOLD:
            return CompatibilityResult(
                compatible=False,
                severity="critical",
                reason=(
                    f"embedding drift {score:.3f} exceeds critical threshold "
                    f"{self.CRITICAL_THRESHOLD}"
                ),
            )
        if score >= self.WARNING_THRESHOLD:
            return CompatibilityResult(
                compatible=True,
                severity="warning",
                reason=(
                    f"embedding drift {score:.3f} in warning range "
                    f"[{self.WARNING_THRESHOLD}, {self.CRITICAL_THRESHOLD})"
                ),
            )
        return CompatibilityResult(
            compatible=True,
            severity="healthy",
            reason=(
                f"GNN version {context['gnn_version']!r} on allow-list, "
                f"drift score {score:.3f} within tolerance"
            ),
        )

    def find_best_compatible_context(self, model_version: str) -> str | None:
        """Return the most recently registered context that pairs cleanly with the model."""
        if model_version not in self._models:
            raise KeyError(f"unknown model version {model_version!r}")

        # Insertion order — latest registration last — so reversing
        # gives us most-recent-first.
        for context_version in reversed(self._contexts):
            result = self.check_compatibility(model_version, context_version)
            if result.compatible:
                return context_version
        return None

    # --- persistence ----------------------------------------------------
    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"models": self._models, "contexts": self._contexts}
        with self.registry_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def _load(self) -> None:
        if not self.registry_path.exists():
            return
        with self.registry_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        self._models = payload.get("models", {})
        self._contexts = payload.get("contexts", {})


# --- helpers ------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stats_to_json(stats: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in stats.items():
        if isinstance(v, torch.Tensor):
            out[k] = {"__tensor__": True, "data": v.detach().cpu().tolist()}
        else:
            out[k] = v
    return out


def _stats_from_json(stats: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in stats.items():
        if isinstance(v, dict) and v.get("__tensor__"):
            out[k] = torch.tensor(v["data"], dtype=torch.float32)
        else:
            out[k] = v
    return out


def _compute_drift_score(train_stats: dict, current_stats: dict) -> float:
    """Composite drift score in ``[0, 1]``, weights matching EmbeddingDriftMonitor."""
    train_mean = train_stats["mean"]
    train_std = train_stats["std"]
    cur_mean = current_stats["mean"]

    denom = torch.clamp(train_std, min=_EPS)
    max_z = float(((cur_mean - train_mean).abs() / denom).max().item())
    marginal = min(1.0, max_z / _MARGINAL_SAT_Z)

    train_norm = float(train_stats["norm_mean"])
    cur_norm = float(current_stats["norm_mean"])
    norm_diff = abs(cur_norm - train_norm) / max(train_norm, _EPS)
    norm = min(1.0, norm_diff / _NORM_SAT_FRAC)

    train_sv = train_stats["top_singular_values"]
    cur_sv = current_stats["top_singular_values"]
    k = min(int(train_sv.shape[0]), int(cur_sv.shape[0]))
    if k == 0:
        structural = 0.0
    else:
        a = train_sv[:k]
        b = cur_sv[:k]
        a_norm = float(a.norm().item())
        if a_norm <= _EPS:
            structural = 0.0
        else:
            structural = min(1.0, float((a - b).norm().item() / a_norm))

    return _MARGINAL_W * marginal + _NORM_W * norm + _STRUCT_W * structural
