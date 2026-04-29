import torch
import torch.nn as nn


class CalibratedClassifier(nn.Module):
    """Classification head with information bottleneck and learnable temperature.

    The bottleneck (d_model -> d_model // 4 -> num_classes) forces compression
    of the multimodal representation, improving generalization.

    The learnable temperature is trained post-hoc on a calibration set (freeze
    everything else, optimize temperature against NLL). When ``calibrate=True``,
    logits are divided by temperature before being returned.
    """

    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 4, num_classes),
        )
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, cls_embedding: torch.Tensor, calibrate: bool = True) -> torch.Tensor:
        logits = self.head(cls_embedding)
        if calibrate:
            logits = logits / self.temperature
        return logits
