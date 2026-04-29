import torch

from satira.models.classifier import CalibratedClassifier


def test_output_shape() -> None:
    head = CalibratedClassifier(d_model=64, num_classes=5)
    head.eval()

    cls_embedding = torch.randn(8, 64)
    logits = head(cls_embedding)

    assert logits.shape == (8, 5)


def test_temperature_one_matches_no_calibration() -> None:
    head = CalibratedClassifier(d_model=64, num_classes=5)
    head.eval()

    cls_embedding = torch.randn(4, 64)

    with torch.no_grad():
        with_calibration = head(cls_embedding, calibrate=True)
        without_calibration = head(cls_embedding, calibrate=False)

    assert torch.allclose(with_calibration, without_calibration)


def test_high_temperature_softens_distribution() -> None:
    head = CalibratedClassifier(d_model=64, num_classes=5)
    head.eval()

    cls_embedding = torch.randn(4, 64)

    with torch.no_grad():
        baseline = torch.softmax(head(cls_embedding, calibrate=False), dim=-1)
        with torch.no_grad():
            head.temperature.fill_(2.0)
        softened = torch.softmax(head(cls_embedding, calibrate=True), dim=-1)

    baseline_max = baseline.max(dim=-1).values
    softened_max = softened.max(dim=-1).values
    assert torch.all(softened_max <= baseline_max + 1e-6)
    assert torch.any(softened_max < baseline_max)


def test_low_temperature_sharpens_distribution() -> None:
    head = CalibratedClassifier(d_model=64, num_classes=5)
    head.eval()

    cls_embedding = torch.randn(4, 64)

    with torch.no_grad():
        baseline = torch.softmax(head(cls_embedding, calibrate=False), dim=-1)
        with torch.no_grad():
            head.temperature.fill_(0.5)
        sharpened = torch.softmax(head(cls_embedding, calibrate=True), dim=-1)

    baseline_max = baseline.max(dim=-1).values
    sharpened_max = sharpened.max(dim=-1).values
    assert torch.all(sharpened_max >= baseline_max - 1e-6)
    assert torch.any(sharpened_max > baseline_max)


def test_bottleneck_dimensionality() -> None:
    d_model = 64
    num_classes = 5
    head = CalibratedClassifier(d_model=d_model, num_classes=num_classes)

    linear_layers = [m for m in head.head if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 2

    first, second = linear_layers
    assert first.in_features == d_model
    assert first.out_features == d_model // 4
    assert second.in_features == d_model // 4
    assert second.out_features == num_classes


def test_temperature_is_learnable_scalar() -> None:
    head = CalibratedClassifier(d_model=32, num_classes=3)

    assert isinstance(head.temperature, torch.nn.Parameter)
    assert head.temperature.requires_grad
    assert head.temperature.numel() == 1
