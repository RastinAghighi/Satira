import torch
import torch.nn as nn


class StandardProjection(nn.Module):
    """Linear projection followed by LayerNorm and GELU.

    Used for vision, text, and temporal projections where input
    distributions are stable across train/inference.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.linear(x)))


class DriftRobustProjection(nn.Module):
    """Projection that continuously adapts its input normalization to drift.

    Unlike standard BatchNorm, the running statistics are updated during
    *both* training and inference. Graph embeddings drift between retraining
    cycles (new entities appear, edge distributions shift), and freezing
    normalization stats at training time would cause accumulating bias in
    production. By updating running stats at inference, the projection tracks
    the live distribution and keeps downstream features well-conditioned.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.momentum = momentum

        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))

        self.linear1 = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        with torch.no_grad():
            self.running_mean.mul_(1 - self.momentum).add_(
                batch_mean.detach(), alpha=self.momentum
            )
            self.running_var.mul_(1 - self.momentum).add_(
                batch_var.detach(), alpha=self.momentum
            )

        x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)

        h = self.linear1(x_norm)
        h = self.norm(h)
        h = self.act(h)
        return self.linear2(h)
