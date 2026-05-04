import torch

from satira.deployment.compatibility import CompatibilityMatrix
from satira.deployment.synchronizer import MultiTrackSynchronizer


DIM = 16
N = 500
WINDOW = ("2026-01-01", "2026-01-15")


def _gaussian(n: int, dim: int, mean: float = 0.0, scale: float = 1.0, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g) * scale + mean


def _stats(samples: torch.Tensor) -> dict:
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


def test_compatible_context_is_deployed_immediately(tmp_path) -> None:
    m = _matrix(tmp_path)
    m.register_model("cls-v1", ["gnn-v1"], WINDOW, _stats(_gaussian(N, DIM, seed=1)))
    m.register_context("ctx-1", "gnn-v1", _stats(_gaussian(N, DIM, seed=2)))

    sync = MultiTrackSynchronizer(m)
    sync.on_model_promoted("cls-v1", ["gnn-v1"])

    # A second compatible context arriving from the offline pipeline
    # must promote to live immediately.
    m.register_context("ctx-2", "gnn-v1", _stats(_gaussian(N, DIM, seed=3)))
    result = sync.on_new_context("ctx-2", "gnn-v1")

    assert result["deployed"] is True
    assert result["track"] == "gnn-v1"
    assert sync.get_active_tracks() == {"gnn-v1": "ctx-2"}


def test_incompatible_context_is_stored_but_not_deployed(tmp_path) -> None:
    m = _matrix(tmp_path)
    m.register_model("cls-v1", ["gnn-v1"], WINDOW, _stats(_gaussian(N, DIM, seed=11)))
    sync = MultiTrackSynchronizer(m)
    sync.on_model_promoted("cls-v1", ["gnn-v1"])

    # New GNN architecture rolls out before a model that can read it —
    # the snapshot must be parked, not deployed.
    m.register_context("ctx-v2", "gnn-v2", _stats(_gaussian(N, DIM, seed=12)))
    result = sync.on_new_context("ctx-v2", "gnn-v2")

    assert result["deployed"] is False
    assert result["track"] == "gnn-v2"
    assert "gnn-v2" in sync.get_active_tracks()
    assert sync.get_active_tracks()["gnn-v2"] == "ctx-v2"


def test_model_promotion_triggers_context_switch_to_compatible_track(tmp_path) -> None:
    m = _matrix(tmp_path)

    # v1 model is live; v1 context is on its track.
    m.register_model("cls-v1", ["gnn-v1"], WINDOW, _stats(_gaussian(N, DIM, seed=21)))
    m.register_context("ctx-v1", "gnn-v1", _stats(_gaussian(N, DIM, seed=22)))
    sync = MultiTrackSynchronizer(m)
    sync.on_model_promoted("cls-v1", ["gnn-v1"])
    assert sync.on_new_context("ctx-v1", "gnn-v1")["deployed"] is True

    # v2 context lands but cannot deploy yet — v1 model can't read it.
    m.register_context("ctx-v2", "gnn-v2", _stats(_gaussian(N, DIM, seed=23)))
    parked = sync.on_new_context("ctx-v2", "gnn-v2")
    assert parked["deployed"] is False

    # v2 model is promoted with gnn-v2 on its allow-list — the parked
    # ctx-v2 should now be the deployed context.
    m.register_model("cls-v2", ["gnn-v2"], WINDOW, _stats(_gaussian(N, DIM, seed=24)))
    promotion = sync.on_model_promoted("cls-v2", ["gnn-v2"])

    assert promotion["context_deployed"] == "ctx-v2"


def test_old_tracks_are_pruned_after_promotion(tmp_path) -> None:
    m = _matrix(tmp_path)

    # Two tracks exist while v1 is live: gnn-v1 deployed, gnn-v2 parked.
    m.register_model("cls-v1", ["gnn-v1"], WINDOW, _stats(_gaussian(N, DIM, seed=31)))
    m.register_context("ctx-v1", "gnn-v1", _stats(_gaussian(N, DIM, seed=32)))
    m.register_context("ctx-v2", "gnn-v2", _stats(_gaussian(N, DIM, seed=33)))
    sync = MultiTrackSynchronizer(m)
    sync.on_model_promoted("cls-v1", ["gnn-v1"])
    sync.on_new_context("ctx-v1", "gnn-v1")
    sync.on_new_context("ctx-v2", "gnn-v2")
    assert set(sync.get_active_tracks()) == {"gnn-v1", "gnn-v2"}

    # v2 model is promoted with only gnn-v2 on its allow-list — the
    # gnn-v1 track is now dead weight and must be dropped.
    m.register_model("cls-v2", ["gnn-v2"], WINDOW, _stats(_gaussian(N, DIM, seed=34)))
    promotion = sync.on_model_promoted("cls-v2", ["gnn-v2"])

    assert promotion["old_tracks_pruned"] == ["gnn-v1"]
    assert sync.get_active_tracks() == {"gnn-v2": "ctx-v2"}
