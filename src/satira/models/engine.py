import torch
import torch.nn as nn

from satira.config import Settings
from satira.models.classifier import CalibratedClassifier
from satira.models.cross_attention import ContrastiveCrossAttention
from satira.models.modality_dropout import StructuredModalityDropout
from satira.models.projections import DriftRobustProjection, StandardProjection
from satira.models.reasoning import ContextualReasoningBlock


class SatireDetectionEngine(nn.Module):
    """Complete four-stream hierarchical fusion model for satire detection.

    Architecture:
        1. Input projections (StandardProjection for V/T/temporal,
           DriftRobustProjection for graph).
        2. Contrastive cross-attention with contradiction gates (Stage 1).
        3. Structured modality dropout on context streams.
        4. Contextual reasoning with modality-type embeddings (Stage 2).
        5. Calibrated classification with bottleneck (Stage 3).

    Returns ``(logits, t2v_weights, v2t_weights, t_gate, v_gate)``.
    """

    def __init__(self, config: Settings) -> None:
        super().__init__()
        self.config = config

        self.v_proj = StandardProjection(config.vision_dim, config.d_model)
        self.t_proj = StandardProjection(config.text_dim, config.d_model)
        self.temp_proj = StandardProjection(config.temporal_dim, config.d_model)
        self.graph_proj = DriftRobustProjection(config.graph_dim, config.d_model)

        self.cross_attn = ContrastiveCrossAttention(config.d_model, config.num_heads)

        self.modality_dropout = StructuredModalityDropout(
            d_model=config.d_model,
            temporal_drop_prob=config.temporal_drop_prob,
            graph_drop_prob=config.graph_drop_prob,
            both_drop_prob=config.joint_drop_prob,
        )

        self.reasoning = ContextualReasoningBlock(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_reasoning_layers,
        )

        self.classifier = CalibratedClassifier(config.d_model, config.num_classes)

    def forward(
        self,
        v_patches: torch.Tensor,
        t_tokens: torch.Tensor,
        temporal_ctx: torch.Tensor,
        graph_ctx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        v_emb = self.v_proj(v_patches)
        t_emb = self.t_proj(t_tokens)
        temp_emb = self.temp_proj(temporal_ctx)
        graph_emb = self.graph_proj(graph_ctx)

        t_out, v_out, t2v_weights, v2t_weights, t_gate, v_gate = self.cross_attn(
            v_emb, t_emb
        )

        temp_emb, graph_emb = self.modality_dropout(temp_emb, graph_emb)

        cls_embedding = self.reasoning(t_out, v_out, temp_emb, graph_emb)

        logits = self.classifier(cls_embedding)

        return logits, t2v_weights, v2t_weights, t_gate, v_gate

    def get_parameter_groups(self) -> list[dict]:
        """Parameter groups with per-stage learning rates for Phase 3 fine-tuning.

        Encoders (input projections) train slowest because their pretrained
        statistics shouldn't be perturbed; fusion layers move fastest because
        they're learning task-specific composition; the classifier sits in
        between.
        """
        encoder_params = (
            list(self.v_proj.parameters())
            + list(self.t_proj.parameters())
            + list(self.temp_proj.parameters())
            + list(self.graph_proj.parameters())
        )
        fusion_params = (
            list(self.cross_attn.parameters())
            + list(self.modality_dropout.parameters())
            + list(self.reasoning.parameters())
        )
        classifier_params = list(self.classifier.parameters())

        return [
            {"params": encoder_params, "lr": 1e-5, "name": "encoders"},
            {"params": fusion_params, "lr": 2e-4, "name": "fusion"},
            {"params": classifier_params, "lr": 1e-4, "name": "classifier"},
        ]

    def freeze_for_phase(self, phase: int) -> None:
        """Freeze/unfreeze components for a given training phase.

        Phase 1: Train projections + classifier only (cross-attn and reasoning
            frozen) so the heads learn against stable fused features first.
        Phase 2: Unfreeze fusion (cross-attn + reasoning) — full end-to-end.
        Phase 3: Everything trainable; learning rates handle differentiation
            via ``get_parameter_groups``.
        """
        if phase == 1:
            for p in self.cross_attn.parameters():
                p.requires_grad = False
            for p in self.reasoning.parameters():
                p.requires_grad = False
            for p in self.v_proj.parameters():
                p.requires_grad = True
            for p in self.t_proj.parameters():
                p.requires_grad = True
            for p in self.temp_proj.parameters():
                p.requires_grad = True
            for p in self.graph_proj.parameters():
                p.requires_grad = True
            for p in self.classifier.parameters():
                p.requires_grad = True
        elif phase == 2:
            for p in self.cross_attn.parameters():
                p.requires_grad = True
            for p in self.reasoning.parameters():
                p.requires_grad = True
        elif phase == 3:
            for p in self.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown phase: {phase}. Expected 1, 2, or 3.")

    def count_parameters(self) -> dict:
        """Return total parameter counts per sub-module."""

        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        return {
            "v_proj": _count(self.v_proj),
            "t_proj": _count(self.t_proj),
            "temp_proj": _count(self.temp_proj),
            "graph_proj": _count(self.graph_proj),
            "cross_attn": _count(self.cross_attn),
            "modality_dropout": _count(self.modality_dropout),
            "reasoning": _count(self.reasoning),
            "classifier": _count(self.classifier),
            "total": _count(self),
        }
