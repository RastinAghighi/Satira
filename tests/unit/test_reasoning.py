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


def test_text_padding_mask_makes_cls_invariant_to_padded_text() -> None:
    """With the mask, the CLS readout ignores padded text tokens; without it,
    the same scribble changes the output — so the mask is doing real work."""
    torch.manual_seed(0)
    block = ContextualReasoningBlock(d_model=32, num_heads=4, num_layers=2).eval()
    gt = torch.randn(2, 6, 32)
    gv = torch.randn(2, 5, 32)
    temp = torch.randn(2, 32)
    graph = torch.randn(2, 32)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 3:] = True

    gt_scribbled = gt.clone()
    gt_scribbled[:, 3:] = torch.randn(2, 3, 32) * 50

    with torch.no_grad():
        masked_a = block(gt, gv, temp, graph, text_key_padding_mask=mask)
        masked_b = block(gt_scribbled, gv, temp, graph, text_key_padding_mask=mask)
        unmasked_a = block(gt, gv, temp, graph)
        unmasked_b = block(gt_scribbled, gv, temp, graph)

    assert torch.allclose(masked_a, masked_b, atol=1e-5)
    assert not torch.allclose(unmasked_a, unmasked_b, atol=1e-4)


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
