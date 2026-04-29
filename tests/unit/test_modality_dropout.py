import torch

from satira.models.modality_dropout import StructuredModalityDropout


def test_eval_mode_passes_through() -> None:
    module = StructuredModalityDropout(
        d_model=32,
        temporal_drop_prob=0.5,
        graph_drop_prob=0.5,
        both_drop_prob=0.5,
    )
    module.eval()

    temp_emb = torch.randn(8, 32)
    graph_emb = torch.randn(8, 32)

    temp_out, graph_out = module(temp_emb, graph_emb)
    assert torch.equal(temp_out, temp_emb)
    assert torch.equal(graph_out, graph_emb)


def test_drop_prob_one_replaces_all_with_fallback() -> None:
    module = StructuredModalityDropout(
        d_model=16,
        temporal_drop_prob=1.0,
        graph_drop_prob=1.0,
        both_drop_prob=1.0,
    )
    module.train()

    temp_emb = torch.randn(8, 16)
    graph_emb = torch.randn(8, 16)

    temp_out, graph_out = module(temp_emb, graph_emb)

    expected_temp = module.temporal_fallback.view(1, -1).expand_as(temp_emb)
    expected_graph = module.graph_fallback.view(1, -1).expand_as(graph_emb)

    assert torch.allclose(temp_out, expected_temp)
    assert torch.allclose(graph_out, expected_graph)


def test_fallback_embeddings_are_learnable() -> None:
    module = StructuredModalityDropout(d_model=16)

    assert module.temporal_fallback.requires_grad
    assert module.graph_fallback.requires_grad
    assert isinstance(module.temporal_fallback, torch.nn.Parameter)
    assert isinstance(module.graph_fallback, torch.nn.Parameter)


def test_batch_independence() -> None:
    torch.manual_seed(42)
    module = StructuredModalityDropout(
        d_model=8,
        temporal_drop_prob=0.5,
        graph_drop_prob=0.5,
        both_drop_prob=0.0,
    )
    module.train()

    temp_emb = torch.randn(64, 8)
    graph_emb = torch.randn(64, 8)

    temp_out, graph_out = module(temp_emb, graph_emb)

    temp_fallback = module.temporal_fallback.view(1, -1).expand_as(temp_emb)
    graph_fallback = module.graph_fallback.view(1, -1).expand_as(graph_emb)

    temp_dropped = torch.all(torch.isclose(temp_out, temp_fallback), dim=-1)
    graph_dropped = torch.all(torch.isclose(graph_out, graph_fallback), dim=-1)

    assert temp_dropped.any() and (~temp_dropped).any(), (
        "Some batch elements should be dropped and some kept (temporal)"
    )
    assert graph_dropped.any() and (~graph_dropped).any(), (
        "Some batch elements should be dropped and some kept (graph)"
    )

    assert not torch.equal(temp_dropped, graph_dropped), (
        "Temporal and graph masks should be sampled independently"
    )
