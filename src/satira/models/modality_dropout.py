import torch
import torch.nn as nn


class StructuredModalityDropout(nn.Module):
    """Drops temporal and/or graph context during training.

    Uses LEARNED FALLBACK EMBEDDINGS (not zeros) because production cache
    misses return a learned default, not zero vectors. Zero-vector dropout
    creates a train-test mismatch.

    Supports correlated (joint) dropout: both streams failing simultaneously,
    which happens during breaking events when both the index and entity
    resolution lag.
    """

    def __init__(
        self,
        d_model: int,
        temporal_drop_prob: float = 0.15,
        graph_drop_prob: float = 0.15,
        both_drop_prob: float = 0.05,
    ) -> None:
        super().__init__()
        self.temporal_drop_prob = temporal_drop_prob
        self.graph_drop_prob = graph_drop_prob
        self.both_drop_prob = both_drop_prob

        self.temporal_fallback = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.graph_fallback = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(
        self,
        temp_emb: torch.Tensor,
        graph_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return temp_emb, graph_emb

        batch_size = temp_emb.size(0)
        device = temp_emb.device

        joint_r = torch.rand(batch_size, device=device)
        temp_r = torch.rand(batch_size, device=device)
        graph_r = torch.rand(batch_size, device=device)

        drop_both = joint_r < self.both_drop_prob
        drop_temp = drop_both | (temp_r < self.temporal_drop_prob)
        drop_graph = drop_both | (graph_r < self.graph_drop_prob)

        temp_mask = drop_temp.view(batch_size, *([1] * (temp_emb.dim() - 1)))
        graph_mask = drop_graph.view(batch_size, *([1] * (graph_emb.dim() - 1)))

        temp_fallback = self.temporal_fallback.view(
            *([1] * (temp_emb.dim() - 1)), -1
        )
        graph_fallback = self.graph_fallback.view(
            *([1] * (graph_emb.dim() - 1)), -1
        )

        temp_out = torch.where(
            temp_mask, temp_fallback.expand_as(temp_emb), temp_emb
        )
        graph_out = torch.where(
            graph_mask, graph_fallback.expand_as(graph_emb), graph_emb
        )

        return temp_out, graph_out
