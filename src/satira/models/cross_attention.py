import torch
import torch.nn as nn


class ContrastiveCrossAttention(nn.Module):
    """Bidirectional cross-attention with explicit contradiction gates.

    Unlike standard cross-attention which computes soft alignment (finding what
    matches), this module explicitly preserves the MISMATCH signal through
    gated residuals.

    t_delta = t_emb - t_grounded captures what text claims that vision doesn't
    support. v_delta = v_emb - v_grounded captures what vision shows that text
    doesn't mention. The contradiction_gate (Sigmoid) learns when the delta is
    meaningful vs noise.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.text_to_vision = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.vision_to_text = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.contradiction_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        v_emb: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        t_grounded, t2v_weights = self.text_to_vision(
            query=t_emb, key=v_emb, value=v_emb, need_weights=True
        )
        v_grounded, v2t_weights = self.vision_to_text(
            query=v_emb, key=t_emb, value=t_emb, need_weights=True
        )

        t_delta = t_emb - t_grounded
        v_delta = v_emb - v_grounded

        t_gate = self.contradiction_gate(torch.cat([t_emb, t_grounded], dim=-1))
        v_gate = self.contradiction_gate(torch.cat([v_emb, v_grounded], dim=-1))

        t_out = t_grounded + t_gate * t_delta
        v_out = v_grounded + v_gate * v_delta

        return t_out, v_out, t2v_weights, v2t_weights, t_gate, v_gate
