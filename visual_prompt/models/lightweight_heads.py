from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskVectorMLP(nn.Module):
    """Legacy vector heads or direct spatial feature-to-mask head (m6_spatial)."""

    def __init__(self, channels: int, hidden: int, initial_vector: torch.Tensor, variant: str):
        super().__init__()
        self.channels = channels
        self.variant = variant
        if variant == "m6_spatial":
            self.net = nn.Sequential(
                nn.Conv2d(channels, hidden, 1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(hidden, 1, 1),
            )
            return
        descriptor_dim = 2 * channels + 4
        self.net = nn.Sequential(
            nn.Linear(descriptor_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels + 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if variant == "m1_direct":
            with torch.no_grad():
                self.net[-1].bias[:channels].copy_(F.normalize(initial_vector.flatten(), dim=0))

    def forward(
        self,
        features: torch.Tensor,
        initial_vector: torch.Tensor,
        similarity: torch.Tensor,
        gaussian: torch.Tensor,
        contrast: torch.Tensor,
        residual_scale: float,
        temperature: float,
    ) -> torch.Tensor:
        if self.variant == "m6_spatial":
            # Direct feature-to-mask prediction; no vector or prompt fusion.
            return self.net(features).squeeze(1)
        batch, channels, height, width = features.shape
        center = features[:, :, height // 2, width // 2]
        mean = features.mean(dim=(-2, -1))
        sim_stats = torch.stack(
            [
                similarity[:, height // 2, width // 2],
                similarity.amax(dim=(-2, -1)),
                similarity.mean(dim=(-2, -1)),
                similarity.std(dim=(-2, -1)),
            ],
            dim=1,
        )
        output = self.net(torch.cat([center, mean, sim_stats], dim=1))
        vector_part = output[:, :channels]
        sim_weight, gaussian_weight, contrast_weight, bias = output[:, channels:].unbind(dim=1)
        if self.variant == "m1_direct":
            vector = F.normalize(vector_part, dim=1)
        else:
            vector = F.normalize(initial_vector[None] + residual_scale * vector_part, dim=1)
        logits = torch.einsum("bchw,bc->bhw", features, vector) / temperature + bias[:, None, None]
        if self.variant in {"m3_similarity", "m4_point", "m5_contrast"}:
            logits = logits + sim_weight[:, None, None] * similarity
        if self.variant in {"m4_point", "m5_contrast"}:
            logits = logits + gaussian_weight[:, None, None] * gaussian
        if self.variant == "m5_contrast":
            logits = logits + contrast_weight[:, None, None] * contrast
        return logits


class ObjectnessMLP(nn.Module):
    """Small binary classifier for pooled local candidate descriptors."""

    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        return self.net(descriptor).squeeze(-1)


def soft_iou_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(-2, -1))
    union = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) - intersection
    return (1.0 - (intersection + epsilon) / (union + epsilon)).mean()
