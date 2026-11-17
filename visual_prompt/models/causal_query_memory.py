"""SANSA-inspired causal prototype memory for the fixed radial Mask path.

This is not SAM2's native dense Memory Encoder/Attention. It isolates sequential
prediction write-back using learned foreground/background correspondence tokens.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.radial_augmented_heads import RadialMaskVectorMLP
from visual_prompt.pipelines.p2p_framework import mask_descriptor, mask_forward


class CausalQueryMask(nn.Module):
    def __init__(self, e0, width=32):
        super().__init__()
        self.base = RadialMaskVectorMLP(352, 64, e0)
        self.project = nn.Conv2d(352, width, 1, bias=False)
        self.readout = nn.Sequential(nn.Conv2d(2 * width + 2, 16, 1), nn.GELU(), nn.Conv2d(16, 1, 1, bias=False))
        nn.init.zeros_(self.readout[-1].weight)

    def embedding(self, feature):
        return F.normalize(self.project(feature[:, :352]), dim=1, eps=1e-6)

    def encode(self, feature, mask, valid):
        """One differentiable foreground/background pair per image/crop."""
        emb = self.embedding(feature)
        weights = torch.stack((mask, 1 - mask), 1) * valid[:, None]
        tokens = torch.einsum('bdhw,bkhw->bkd', emb, weights)
        mass = weights.sum((-2, -1))
        tokens = F.normalize(tokens / mass.clamp_min(1e-6)[:, :, None], dim=-1, eps=1e-6)
        available = mass > 1e-6
        return tokens, available

    def forward(self, feature, e0, memory=None):
        z = mask_forward(self.base, feature, mask_descriptor(feature, feature[:, :352], e0), e0)
        if memory is None:
            return z
        # Support remains the first slot and is never overwritten by predictions.
        tokens = torch.stack([entry[0] for entry in memory], 1)  # B,M,2,D
        available = torch.stack([entry[1] for entry in memory], 1)
        q = self.embedding(feature)
        scores = torch.einsum('bdhw,bmkd->bmkhw', q, tokens)
        scores = scores.masked_fill(~available[:, :, :, None, None], -1e4)
        weights = (scores * 10).softmax(1) * available[:, :, :, None, None]
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
        read = torch.einsum('bmkhw,bmkd->bkdhw', weights, tokens)
        products = q[:, None] * read
        similarity = products.sum(2)
        context = torch.cat((products.flatten(1, 2), similarity), 1)
        return z + self.readout(context)[:, 0]


def sequence_forward(model, queries, e0, support, support_mask, support_valid,
                     valid, mode):
    """queries B,T,C,H,W; all query labels are excluded from this interface."""
    if mode not in ('loss_only', 'support_only', 'propagation'):
        raise ValueError(mode)
    memory = [model.encode(support, support_mask, support_valid)] if mode != 'loss_only' else None
    outputs = []
    for step in range(queries.shape[1]):
        feature = queries[:, step]
        z = model(feature, e0, memory)
        outputs.append(z)
        if mode == 'propagation':
            # Soft prediction only. No query GT, no detached state, no future frame.
            memory.append(model.encode(feature, z.sigmoid(), valid[:, step]))
    return torch.stack(outputs, 1)


def bce_dice(logits, target, valid):
    weights = valid.to(logits.dtype)
    dims = (-2, -1)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * weights).sum(dims) / weights.sum(dims).clamp_min(1)
    p = logits.sigmoid() * weights
    t = target * weights
    dice = 1 - (2 * (p * t).sum(dims) + 1) / (p.sum(dims) + t.sum(dims) + 1)
    return (bce + dice).mean()
