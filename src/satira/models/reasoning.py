import torch
import torch.nn as nn


class ContextualReasoningBlock(nn.Module):
    """Self-attention reasoning over all four modality streams.

    A transformer encoder integrates grounded text, grounded vision, a
    temporal embedding, and a graph embedding. Each token receives a
    *semantic* type embedding (not positional) so the attention mechanism
    knows which modality each token came from.

    Type IDs: 0=CLS, 1=temporal, 2=graph, 3=language/vision.

    Returns the CLS token readout after `num_layers` of self-attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.type_embeddings = nn.Embedding(4, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        grounded_text: torch.Tensor,
        grounded_vision: torch.Tensor,
        temp_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        text_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = grounded_text.size(0)

        if temp_emb.dim() == 2:
            temp_emb = temp_emb.unsqueeze(1)
        if graph_emb.dim() == 2:
            graph_emb = graph_emb.unsqueeze(1)

        cls = self.cls_token.expand(batch_size, -1, -1)

        text_len = grounded_text.size(1)
        vision_len = grounded_vision.size(1)

        seq = torch.cat([cls, temp_emb, graph_emb, grounded_text, grounded_vision], dim=1)

        type_ids = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=seq.device),
                torch.ones(1, dtype=torch.long, device=seq.device),
                torch.full((1,), 2, dtype=torch.long, device=seq.device),
                torch.full((text_len + vision_len,), 3, dtype=torch.long, device=seq.device),
            ]
        )

        type_emb = self.type_embeddings(type_ids).unsqueeze(0)
        seq = seq + type_emb

        src_key_padding_mask = self._build_src_mask(
            text_key_padding_mask,
            batch_size=batch_size,
            prefix_len=1 + temp_emb.size(1) + graph_emb.size(1),
            text_len=text_len,
            vision_len=vision_len,
            device=seq.device,
        )

        out = self.encoder(seq, src_key_padding_mask=src_key_padding_mask)
        return out[:, 0]

    @staticmethod
    def _build_src_mask(
        text_key_padding_mask: torch.Tensor | None,
        *,
        batch_size: int,
        prefix_len: int,
        text_len: int,
        vision_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Expand a text-only padding mask to cover the full fused sequence.

        The sequence is ``[cls, temporal, graph, text..., vision...]``. Only the
        text span can carry padding, so the CLS/temporal/graph prefix and the
        (fixed-length) vision span are always attendable; the supplied
        ``(batch, text_len)`` mask fills the text span. Returns ``None`` when no
        mask is given, which keeps the unmasked path byte-for-byte unchanged.
        """
        if text_key_padding_mask is None:
            return None
        mask = text_key_padding_mask.to(device=device, dtype=torch.bool)
        prefix = torch.zeros(batch_size, prefix_len, dtype=torch.bool, device=device)
        vision = torch.zeros(batch_size, vision_len, dtype=torch.bool, device=device)
        return torch.cat([prefix, mask, vision], dim=1)
