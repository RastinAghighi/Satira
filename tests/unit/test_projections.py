import pytest
import torch

from satira.models.projections import DriftRobustProjection, StandardProjection


def test_standard_projection_shape_3d() -> None:
    proj = StandardProjection(input_dim=768, output_dim=512)
    x = torch.randn(4, 16, 768)
    out = proj(x)
    assert out.shape == (4, 16, 512)


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_standard_projection_varied_batch_sizes(batch_size: int) -> None:
    proj = StandardProjection(input_dim=128, output_dim=64)
    x = torch.randn(batch_size, 10, 128)
    out = proj(x)
    assert out.shape == (batch_size, 10, 64)


def test_drift_robust_projection_shape() -> None:
    proj = DriftRobustProjection(input_dim=256, output_dim=512)
    x = torch.randn(8, 256)
    out = proj(x)
    assert out.shape == (8, 512)


def test_drift_robust_projection_updates_stats_during_eval() -> None:
    proj = DriftRobustProjection(input_dim=64, output_dim=32, momentum=0.1)
    proj.eval()

    initial_mean = proj.running_mean.clone()
    initial_var = proj.running_var.clone()

    x = torch.randn(16, 64) + 5.0
    with torch.no_grad():
        proj(x)

    assert not torch.allclose(proj.running_mean, initial_mean), (
        "running_mean should update during eval to track production drift"
    )
    assert not torch.allclose(proj.running_var, initial_var), (
        "running_var should update during eval to track production drift"
    )
