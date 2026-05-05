import torch

from satira.config import Settings
from satira.models.engine import SatireDetectionEngine
from satira.training.evaluation import EvalReport, ModelEvaluator


def _small_config() -> Settings:
    return Settings(
        d_model=32,
        num_heads=4,
        num_classes=5,
        vision_dim=64,
        text_dim=48,
        temporal_dim=48,
        graph_dim=24,
        num_reasoning_layers=1,
        batch_size=8,
    )


def _make_evaluator() -> ModelEvaluator:
    config = _small_config()
    model = SatireDetectionEngine(config)
    return ModelEvaluator(model=model, config=config, device="cpu")


def test_ece_is_zero_for_perfectly_calibrated_model() -> None:
    evaluator = _make_evaluator()
    num_classes = evaluator.config.num_classes
    n = 50

    targets = torch.randint(0, num_classes, (n,), dtype=torch.long)
    predictions = torch.zeros(n, num_classes)
    predictions[torch.arange(n), targets] = 1.0

    ece = evaluator.calibration_error(predictions, targets)
    assert ece == 0.0


def test_ece_is_one_for_maximally_miscalibrated_model() -> None:
    evaluator = _make_evaluator()
    num_classes = evaluator.config.num_classes
    n = 40

    targets = torch.zeros(n, dtype=torch.long)
    wrong_class = (targets + 1) % num_classes
    predictions = torch.zeros(n, num_classes)
    predictions[torch.arange(n), wrong_class] = 1.0

    ece = evaluator.calibration_error(predictions, targets)
    assert ece == 1.0


def test_gate_analysis_returns_per_class_statistics() -> None:
    evaluator = _make_evaluator()
    num_classes = evaluator.config.num_classes
    n = 30
    seq_len = 12
    d_model = evaluator.config.d_model

    gate_activations = torch.rand(n, seq_len, d_model)
    targets = torch.arange(n) % num_classes

    result = evaluator.gate_analysis(gate_activations, targets)

    assert "per_class" in result
    assert "overall_mean" in result
    assert "overall_variance" in result

    per_class = result["per_class"]
    assert len(per_class) == num_classes

    for stats in per_class.values():
        assert "mean" in stats
        assert "variance" in stats
        assert "count" in stats
        assert 0.0 <= stats["mean"] <= 1.0
        assert stats["variance"] >= 0.0
        assert stats["count"] > 0

    total_count = sum(s["count"] for s in per_class.values())
    assert total_count == n


def test_gate_analysis_handles_empty_class() -> None:
    evaluator = _make_evaluator()
    n = 10
    gate_activations = torch.rand(n, 4, evaluator.config.d_model)
    targets = torch.zeros(n, dtype=torch.long)

    result = evaluator.gate_analysis(gate_activations, targets)
    per_class = result["per_class"]

    zero_count_classes = [s for s in per_class.values() if s["count"] == 0]
    assert len(zero_count_classes) == evaluator.config.num_classes - 1
    for stats in zero_count_classes:
        assert stats["mean"] == 0.0
        assert stats["variance"] == 0.0


def test_attention_entropy_uniform_is_maximal() -> None:
    evaluator = _make_evaluator()
    batch, q_len, k_len = 4, 6, 8

    uniform = torch.full((batch, q_len, k_len), 1.0 / k_len)
    result = evaluator.attention_entropy(uniform)

    assert result["normalized_mean"] >= 1.0 - 1e-6
    assert result["low_entropy_fraction"] == 0.0


def test_attention_entropy_collapsed_is_low() -> None:
    evaluator = _make_evaluator()
    batch, q_len, k_len = 4, 6, 8

    collapsed = torch.zeros(batch, q_len, k_len)
    collapsed[..., 0] = 1.0
    result = evaluator.attention_entropy(collapsed)

    assert result["normalized_mean"] < 1e-3
    assert result["low_entropy_fraction"] == 1.0


def test_eval_report_summary_returns_valid_string() -> None:
    report = EvalReport(
        accuracy=0.85,
        macro_f1=0.80,
        weighted_f1=0.82,
        per_class_metrics={
            "authentic": {
                "f1": 0.90,
                "precision": 0.88,
                "recall": 0.92,
                "auroc": 0.95,
                "support": 50,
            },
            "satire": {
                "f1": 0.70,
                "precision": 0.72,
                "recall": 0.68,
                "auroc": 0.85,
                "support": 30,
            },
        },
        calibration_error=0.05,
        gate_analysis={
            "per_class": {
                "authentic": {"mean": 0.30, "variance": 0.05, "count": 50},
                "satire": {"mean": 0.80, "variance": 0.04, "count": 30},
            },
            "overall_mean": 0.5,
            "overall_variance": 0.07,
        },
        attention_entropy={
            "t2v": {
                "mean": 1.5,
                "min": 0.2,
                "max": 2.1,
                "normalized_mean": 0.7,
                "low_entropy_fraction": 0.1,
            },
            "v2t": {
                "mean": 1.6,
                "min": 0.4,
                "max": 2.2,
                "normalized_mean": 0.75,
                "low_entropy_fraction": 0.05,
            },
        },
        total_samples=80,
    )

    summary = report.summary()
    assert isinstance(summary, str)
    assert summary.strip() != ""
    assert "n=80" in summary
    assert "0.8500" in summary
    assert "authentic" in summary
    assert "satire" in summary
    assert "ECE" in summary or "calibration" in summary.lower()


def test_eval_report_to_dict_round_trips_fields() -> None:
    report = EvalReport(
        accuracy=0.5,
        macro_f1=0.4,
        weighted_f1=0.45,
        per_class_metrics={},
        calibration_error=0.1,
        gate_analysis={"per_class": {}, "overall_mean": 0.0, "overall_variance": 0.0},
        attention_entropy={},
        total_samples=10,
    )
    d = report.to_dict()
    assert d["accuracy"] == 0.5
    assert d["total_samples"] == 10
    assert "per_class_metrics" in d
    assert "gate_analysis" in d


def _synthetic_batch(config: Settings, batch_size: int = 4) -> dict:
    return {
        "v_patches": torch.randn(batch_size, 10, config.vision_dim),
        "t_tokens": torch.randn(batch_size, 12, config.text_dim),
        "temporal_ctx": torch.randn(batch_size, config.temporal_dim),
        "graph_ctx": torch.randn(batch_size, config.graph_dim),
        "label": torch.randint(0, config.num_classes, (batch_size,), dtype=torch.long),
    }


def test_evaluate_runs_end_to_end() -> None:
    evaluator = _make_evaluator()
    batches = [_synthetic_batch(evaluator.config, batch_size=4) for _ in range(3)]

    report = evaluator.evaluate(batches)

    assert isinstance(report, EvalReport)
    assert report.total_samples == 12
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.calibration_error <= 1.0
    assert len(report.per_class_metrics) == evaluator.config.num_classes
    assert "t2v" in report.attention_entropy
    assert "v2t" in report.attention_entropy


def test_evaluate_handles_empty_dataloader() -> None:
    evaluator = _make_evaluator()
    report = evaluator.evaluate([])
    assert report.total_samples == 0
    assert report.accuracy == 0.0
