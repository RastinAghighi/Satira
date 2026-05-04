import pytest
import torch

from satira.deployment.compatibility import (
    CompatibilityMatrix,
    CompatibilityResult,
)


DIM = 16
N = 500
WINDOW = ("2026-01-01", "2026-01-15")


def _gaussian(n: int, dim: int, mean: float = 0.0, scale: float = 1.0, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g) * scale + mean


def _stats(samples: torch.Tensor) -> dict:
    """Mirror GraphEmbeddingCache.compute_distribution_stats."""
    norms = samples.norm(dim=1)
    sv = torch.linalg.svdvals(samples)
    return {
        "count": samples.shape[0],
        "mean": samples.mean(dim=0),
        "std": samples.std(dim=0, unbiased=False),
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std(unbiased=False).item()),
        "top_singular_values": sv[: min(10, sv.shape[0])],
    }


def _matrix(tmp_path) -> CompatibilityMatrix:
    return CompatibilityMatrix(registry_path=str(tmp_path / "registry.json"))


# --- hard compatibility -----------------------------------------------------
def test_gnn_version_mismatch_returns_critical_incompatibility(tmp_path) -> None:
    m = _matrix(tmp_path)
    train = _gaussian(N, DIM, seed=1)
    m.register_model(
        model_version="cls-v1",
        compatible_gnn_versions=["gnn-v1"],
        graph_snapshot_window=WINDOW,
        training_stats=_stats(train),
    )
    # Same distribution as training — soft check would pass — but the
    # GNN that produced it is a different architecture.
    m.register_context(
        context_version="ctx-1",
        gnn_version="gnn-v2",
        embedding_stats=_stats(_gaussian(N, DIM, seed=2)),
    )

    result = m.check_compatibility("cls-v1", "ctx-1")
    assert isinstance(result, CompatibilityResult)
    assert result.compatible is False
    assert result.severity == "critical"
    assert "GNN" in result.reason


# --- soft compatibility -----------------------------------------------------
def test_embedding_drift_within_tolerance_returns_healthy(tmp_path) -> None:
    m = _matrix(tmp_path)
    train = _gaussian(N, DIM, seed=11)
    m.register_model("cls-v1", ["gnn-v1"], WINDOW, _stats(train))

    # Fresh draw from the same distribution — drift should be minimal.
    m.register_context("ctx-1", "gnn-v1", _stats(_gaussian(N, DIM, seed=12)))

    result = m.check_compatibility("cls-v1", "ctx-1")
    assert result.compatible is True
    assert result.severity == "healthy"


def test_embedding_drift_in_warning_range_returns_warning(tmp_path) -> None:
    m = _matrix(tmp_path)
    train = _gaussian(N, DIM, seed=21)
    m.register_model("cls-v1", ["gnn-v1"], WINDOW, _stats(train))

    # Scale every dim by 2x: marginal stays small (mean ~0) but the
    # mean L2 norm doubles and the SV spectrum doubles too. That puts
    # the composite score in the warning band.
    m.register_context(
        "ctx-1",
        "gnn-v1",
        _stats(_gaussian(N, DIM, scale=2.0, seed=22)),
    )

    result = m.check_compatibility("cls-v1", "ctx-1")
    assert result.compatible is True
    assert result.severity == "warning"


# --- find_best_compatible_context ------------------------------------------
def test_find_best_compatible_context_returns_most_recent_compatible(tmp_path) -> None:
    m = _matrix(tmp_path)
    m.register_model(
        "cls-v1",
        ["gnn-v1"],
        WINDOW,
        _stats(_gaussian(N, DIM, seed=31)),
    )

    # Three contexts, registered oldest → newest:
    #   ctx-old:    healthy
    #   ctx-broken: critical drift (huge mean+scale shift)
    #   ctx-new:    healthy
    # Most recent compatible is ctx-new.
    m.register_context("ctx-old", "gnn-v1", _stats(_gaussian(N, DIM, seed=32)))
    m.register_context(
        "ctx-broken",
        "gnn-v1",
        _stats(_gaussian(N, DIM, mean=4.0, scale=2.0, seed=33)),
    )
    m.register_context("ctx-new", "gnn-v1", _stats(_gaussian(N, DIM, seed=34)))

    assert m.find_best_compatible_context("cls-v1") == "ctx-new"

    # A subsequent registration with a mismatched GNN is the newest by
    # timestamp but fails the hard check, so the result still points at
    # the previous compatible context.
    m.register_context(
        "ctx-wrong-arch",
        "gnn-v2",
        _stats(_gaussian(N, DIM, seed=35)),
    )
    assert m.find_best_compatible_context("cls-v1") == "ctx-new"


def test_find_best_compatible_context_returns_none_when_nothing_matches(tmp_path) -> None:
    m = _matrix(tmp_path)
    m.register_model("cls-v1", ["gnn-v1"], WINDOW, _stats(_gaussian(N, DIM, seed=41)))
    m.register_context("ctx-1", "gnn-v2", _stats(_gaussian(N, DIM, seed=42)))
    m.register_context(
        "ctx-2",
        "gnn-v1",
        _stats(_gaussian(N, DIM, mean=4.0, scale=2.0, seed=43)),
    )

    assert m.find_best_compatible_context("cls-v1") is None


# --- persistence ------------------------------------------------------------
def test_registry_round_trips_through_disk(tmp_path) -> None:
    path = tmp_path / "registry.json"
    m1 = CompatibilityMatrix(registry_path=str(path))
    m1.register_model(
        "cls-v1",
        ["gnn-v1"],
        WINDOW,
        _stats(_gaussian(N, DIM, seed=51)),
    )
    m1.register_context("ctx-1", "gnn-v1", _stats(_gaussian(N, DIM, seed=52)))

    m2 = CompatibilityMatrix(registry_path=str(path))
    result = m2.check_compatibility("cls-v1", "ctx-1")
    assert result.compatible is True
    assert result.severity == "healthy"


# --- error handling ---------------------------------------------------------
def test_check_compatibility_raises_for_unknown_versions(tmp_path) -> None:
    m = _matrix(tmp_path)
    m.register_model("cls-v1", ["gnn-v1"], WINDOW, _stats(_gaussian(N, DIM, seed=61)))
    m.register_context("ctx-1", "gnn-v1", _stats(_gaussian(N, DIM, seed=62)))

    with pytest.raises(KeyError):
        m.check_compatibility("cls-missing", "ctx-1")
    with pytest.raises(KeyError):
        m.check_compatibility("cls-v1", "ctx-missing")
