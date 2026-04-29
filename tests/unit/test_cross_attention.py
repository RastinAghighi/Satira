import torch

from satira.models.cross_attention import ContrastiveCrossAttention


def test_output_shapes_match_inputs() -> None:
    module = ContrastiveCrossAttention(d_model=64, num_heads=8)
    v_emb = torch.randn(2, 10, 64)
    t_emb = torch.randn(2, 16, 64)

    t_out, v_out, t2v_w, v2t_w, t_gate, v_gate = module(v_emb, t_emb)

    assert t_out.shape == t_emb.shape
    assert v_out.shape == v_emb.shape


def test_attention_weights_sum_to_one_along_keys() -> None:
    module = ContrastiveCrossAttention(d_model=32, num_heads=4)
    v_emb = torch.randn(2, 7, 32)
    t_emb = torch.randn(2, 5, 32)

    _, _, t2v_w, v2t_w, _, _ = module(v_emb, t_emb)

    # t2v_w has shape (batch, query_seq=text, key_seq=vision)
    assert torch.allclose(t2v_w.sum(dim=-1), torch.ones_like(t2v_w.sum(dim=-1)), atol=1e-5)
    assert torch.allclose(v2t_w.sum(dim=-1), torch.ones_like(v2t_w.sum(dim=-1)), atol=1e-5)


def test_gate_activations_in_unit_interval() -> None:
    module = ContrastiveCrossAttention(d_model=32, num_heads=4)
    v_emb = torch.randn(3, 8, 32)
    t_emb = torch.randn(3, 12, 32)

    _, _, _, _, t_gate, v_gate = module(v_emb, t_emb)

    assert torch.all(t_gate >= 0.0) and torch.all(t_gate <= 1.0)
    assert torch.all(v_gate >= 0.0) and torch.all(v_gate <= 1.0)


def test_handles_different_seq_lengths() -> None:
    module = ContrastiveCrossAttention(d_model=48, num_heads=6)
    v_emb = torch.randn(2, 20, 48)
    t_emb = torch.randn(2, 4, 48)

    t_out, v_out, t2v_w, v2t_w, t_gate, v_gate = module(v_emb, t_emb)

    assert t_out.shape == (2, 4, 48)
    assert v_out.shape == (2, 20, 48)
    assert t2v_w.shape == (2, 4, 20)
    assert v2t_w.shape == (2, 20, 4)
    assert t_gate.shape == (2, 4, 48)
    assert v_gate.shape == (2, 20, 48)


def test_no_contradiction_yields_zero_delta() -> None:
    """When v_emb == t_emb and attention is identity, deltas vanish.

    We force the attention modules to identity (Q=K=V=I, output projection=I)
    so that with a single token, the grounded output equals the input. Under
    that condition t_delta = v_delta = 0 and the gate-weighted residual adds
    nothing — t_out and v_out exactly equal the input.
    """
    d_model = 16
    module = ContrastiveCrossAttention(d_model=d_model, num_heads=4)

    with torch.no_grad():
        for attn in (module.text_to_vision, module.vision_to_text):
            attn.in_proj_weight.copy_(torch.eye(d_model).repeat(3, 1))
            attn.in_proj_bias.zero_()
            attn.out_proj.weight.copy_(torch.eye(d_model))
            attn.out_proj.bias.zero_()

    x = torch.randn(2, 1, d_model)
    t_out, v_out, _, _, _, _ = module(x, x)

    assert torch.allclose(t_out, x, atol=1e-5)
    assert torch.allclose(v_out, x, atol=1e-5)
