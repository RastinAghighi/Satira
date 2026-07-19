import torch

from satira.config import Settings
from satira.models.engine import SatireDetectionEngine


def _make_engine() -> SatireDetectionEngine:
    config = Settings(
        d_model=64,
        num_heads=4,
        num_classes=5,
        vision_dim=128,
        text_dim=96,
        temporal_dim=96,
        graph_dim=48,
        num_reasoning_layers=2,
    )
    return SatireDetectionEngine(config)


def _make_inputs(batch: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    v_patches = torch.randn(batch, 10, 128)
    t_tokens = torch.randn(batch, 12, 96)
    temporal_ctx = torch.randn(batch, 96)
    graph_ctx = torch.randn(batch, 48)
    return v_patches, t_tokens, temporal_ctx, graph_ctx


def test_forward_pass_runs() -> None:
    engine = _make_engine()
    engine.eval()

    v, t, temp, graph = _make_inputs(batch=4)
    logits, t2v_weights, v2t_weights, t_gate, v_gate = engine(v, t, temp, graph)

    assert torch.isfinite(logits).all()
    assert torch.isfinite(t2v_weights).all()
    assert torch.isfinite(v2t_weights).all()
    assert torch.isfinite(t_gate).all()
    assert torch.isfinite(v_gate).all()


def test_output_shapes() -> None:
    engine = _make_engine()
    engine.eval()

    batch, text_len, vision_len = 4, 12, 10
    v_patches = torch.randn(batch, vision_len, 128)
    t_tokens = torch.randn(batch, text_len, 96)
    temporal_ctx = torch.randn(batch, 96)
    graph_ctx = torch.randn(batch, 48)

    logits, t2v_weights, v2t_weights, t_gate, v_gate = engine(
        v_patches, t_tokens, temporal_ctx, graph_ctx
    )

    assert logits.shape == (batch, 5)
    assert t2v_weights.shape == (batch, text_len, vision_len)
    assert v2t_weights.shape == (batch, vision_len, text_len)
    assert t_gate.shape == (batch, text_len, 64)
    assert v_gate.shape == (batch, vision_len, 64)
    assert torch.all((t_gate >= 0) & (t_gate <= 1))
    assert torch.all((v_gate >= 0) & (v_gate <= 1))


def test_text_padding_mask_makes_logits_invariant_to_padded_text() -> None:
    """End-to-end: scribbling over padded text positions must not change the
    logits when a matching text_key_padding_mask is supplied."""
    torch.manual_seed(0)
    engine = _make_engine().eval()
    batch, vision_len, text_max = 3, 10, 12
    v = torch.randn(batch, vision_len, 128)
    t = torch.randn(batch, text_max, 96)
    temp = torch.zeros(batch, 96)
    graph = torch.zeros(batch, 48)
    lengths = torch.tensor([4, 8, 12])
    mask = torch.arange(text_max)[None, :] >= lengths[:, None]
    absent = torch.zeros(batch, dtype=torch.bool)

    with torch.no_grad():
        logits1, *_ = engine(
            v, t, temp, graph,
            text_key_padding_mask=mask, temporal_present=absent, graph_present=absent,
        )
        t_scribbled = t.clone()
        for i, length in enumerate(lengths.tolist()):
            t_scribbled[i, length:] = torch.randn(text_max - length, 96) * 100
        logits2, *_ = engine(
            v, t_scribbled, temp, graph,
            text_key_padding_mask=mask, temporal_present=absent, graph_present=absent,
        )

    assert torch.allclose(logits1, logits2, atol=1e-5)


def test_absent_presence_flags_substitute_learned_fallback() -> None:
    """With presence flags all-False, the temporal/graph streams become the
    learned fallback in both train and eval mode."""
    engine = _make_engine()
    batch = 4
    _, _, temp, graph = _make_inputs(batch=batch)
    absent = torch.zeros(batch, dtype=torch.bool)

    for mode in ("eval", "train"):
        getattr(engine, mode)()
        proj_t = engine.temp_proj(temp)
        proj_g = engine.graph_proj(graph)
        out_t, out_g = engine.modality_dropout.substitute_absent(
            proj_t, proj_g, absent, absent
        )
        fb_t = engine.modality_dropout.temporal_fallback.view(1, -1).expand(batch, -1)
        fb_g = engine.modality_dropout.graph_fallback.view(1, -1).expand(batch, -1)
        assert torch.allclose(out_t, fb_t)
        assert torch.allclose(out_g, fb_g)


def test_freeze_for_phase_1_freezes_fusion() -> None:
    engine = _make_engine()
    engine.freeze_for_phase(1)

    assert all(not p.requires_grad for p in engine.cross_attn.parameters())
    assert all(not p.requires_grad for p in engine.reasoning.parameters())

    assert all(p.requires_grad for p in engine.v_proj.parameters())
    assert all(p.requires_grad for p in engine.t_proj.parameters())
    assert all(p.requires_grad for p in engine.temp_proj.parameters())
    assert all(p.requires_grad for p in engine.graph_proj.parameters())
    assert all(p.requires_grad for p in engine.classifier.parameters())


def test_freeze_for_phase_2_unfreezes_fusion() -> None:
    engine = _make_engine()
    engine.freeze_for_phase(1)
    engine.freeze_for_phase(2)

    assert all(p.requires_grad for p in engine.cross_attn.parameters())
    assert all(p.requires_grad for p in engine.reasoning.parameters())


def test_count_parameters_returns_reasonable_numbers() -> None:
    engine = _make_engine()
    counts = engine.count_parameters()

    expected_keys = {
        "v_proj",
        "t_proj",
        "temp_proj",
        "graph_proj",
        "cross_attn",
        "modality_dropout",
        "reasoning",
        "classifier",
        "total",
    }
    assert set(counts.keys()) == expected_keys

    component_total = sum(v for k, v in counts.items() if k != "total")
    assert counts["total"] == component_total

    for name, count in counts.items():
        assert count > 0, f"Component {name} has zero parameters"


def test_parameter_groups_for_phase_3() -> None:
    engine = _make_engine()
    groups = engine.get_parameter_groups()

    assert len(groups) == 3
    names = {g["name"] for g in groups}
    assert names == {"encoders", "fusion", "classifier"}

    lrs = {g["name"]: g["lr"] for g in groups}
    assert lrs["encoders"] == 1e-5
    assert lrs["fusion"] == 2e-4
    assert lrs["classifier"] == 1e-4

    grouped = sum(len(g["params"]) for g in groups)
    total = sum(1 for _ in engine.parameters())
    assert grouped == total


def test_state_dict_round_trip(tmp_path) -> None:
    engine = _make_engine()
    engine.eval()

    path = tmp_path / "engine.pt"
    torch.save(engine.state_dict(), path)

    restored = _make_engine()
    restored.load_state_dict(torch.load(path))
    restored.eval()

    v, t, temp, graph = _make_inputs(batch=2)
    with torch.no_grad():
        original_logits, *_ = engine(v, t, temp, graph)
        restored_logits, *_ = restored(v, t, temp, graph)

    assert torch.allclose(original_logits, restored_logits, atol=1e-6)
