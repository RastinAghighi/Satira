import pytest
import torch

from satira.models.reasoning import ContextualReasoningBlock


def test_output_shape() -> None:
    block = ContextualReasoningBlock(d_model=64, num_heads=8, num_layers=2)
    block.eval()

    grounded_text = torch.randn(4, 12, 64)
    grounded_vision = torch.randn(4, 10, 64)
    temp_emb = torch.randn(4, 64)
    graph_emb = torch.randn(4, 64)

    out = block(grounded_text, grounded_vision, temp_emb, graph_emb)
    assert out.shape == (4, 64)


@pytest.mark.parametrize("text_len,vision_len", [(1, 1), (5, 20), (16, 4), (32, 32)])
def test_varying_sequence_lengths(text_len: int, vision_len: int) -> None:
    block = ContextualReasoningBlock(d_model=32, num_heads=4, num_layers=2)
    block.eval()

    grounded_text = torch.randn(2, text_len, 32)
    grounded_vision = torch.randn(2, vision_len, 32)
    temp_emb = torch.randn(2, 32)
    graph_emb = torch.randn(2, 32)

    out = block(grounded_text, grounded_vision, temp_emb, graph_emb)
    assert out.shape == (2, 32)


@pytest.mark.parametrize("batch_size", [1, 32])
def test_varying_batch_sizes(batch_size: int) -> None:
    block = ContextualReasoningBlock(d_model=32, num_heads=4, num_layers=2)
    block.eval()

    grounded_text = torch.randn(batch_size, 8, 32)
    grounded_vision = torch.randn(batch_size, 6, 32)
    temp_emb = torch.randn(batch_size, 32)
    graph_emb = torch.randn(batch_size, 32)

    out = block(grounded_text, grounded_vision, temp_emb, graph_emb)
    assert out.shape == (batch_size, 32)


def test_type_embeddings_affect_output() -> None:
    torch.manual_seed(0)
    block = ContextualReasoningBlock(d_model=32, num_heads=4, num_layers=2)
    block.eval()

    grounded_text = torch.randn(2, 6, 32)
    grounded_vision = torch.randn(2, 6, 32)
    temp_emb = torch.randn(2, 32)
    graph_emb = torch.randn(2, 32)

    with torch.no_grad():
        out_with = block(grounded_text, grounded_vision, temp_emb, graph_emb)

        original_weights = block.type_embeddings.weight.clone()
        block.type_embeddings.weight.zero_()
        out_zeroed = block(grounded_text, grounded_vision, temp_emb, graph_emb)
        block.type_embeddings.weight.copy_(original_weights)

    assert not torch.allclose(out_with, out_zeroed, atol=1e-5), (
        "Zeroing type embeddings should change the output"
    )
