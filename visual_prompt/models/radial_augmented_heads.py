"""Existing independent heads with lossless radial channels appended to their inputs."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.lightweight_heads import MaskVectorMLP, ObjectnessMLP
from visual_prompt.models.residual_objectness import ResidualObjectness


class F4RadialObjectness(ObjectnessMLP):
    """Train one complete classifier on F4 plus the radial point descriptor."""

    def __init__(self, f4_channels=1440, radial_channels=256, hidden=128):
        super().__init__(f4_channels + radial_channels, hidden)
        self.f4_channels, self.radial_channels = f4_channels, radial_channels

    def forward(self, f4, radial):
        if f4.ndim != 2 or radial.ndim != 2:
            raise ValueError('Expected batched F4 and radial vectors')
        if f4.shape != (radial.shape[0], self.f4_channels) or radial.shape[1] != self.radial_channels:
            raise ValueError('Unexpected F4/radial descriptor dimensions')
        return super().forward(torch.cat((f4, radial), dim=1))


class RadialMaskVectorMLP(MaskVectorMLP):
    """SAM and standardized radial maps jointly condition/read the mask vector.

    SAM channels are untouched, e0 is zero-padded, and similarity statistics are
    computed only from SAM. Initial SAM-input weights match the original MLP
    initialized with the same seed; added descriptor columns start at zero.
    This is a fresh initialization, not a continuation of trained Mask weights.
    """

    def __init__(self, sam_channels, radial_channels, e0, hidden=128):
        reference = MaskVectorMLP(sam_channels, hidden, e0, 'm2_residual')
        extended = F.pad(e0.flatten(), (0, radial_channels))
        super().__init__(sam_channels+radial_channels, hidden, extended, 'm2_residual')
        self.sam_channels, self.radial_channels = sam_channels, radial_channels
        with torch.no_grad():
            weight = self.net[0].weight
            weight.zero_()
            weight[:, :sam_channels].copy_(reference.net[0].weight[:, :sam_channels])
            weight[:, self.channels:self.channels+sam_channels].copy_(reference.net[0].weight[:, sam_channels:2*sam_channels])
            weight[:, -4:].copy_(reference.net[0].weight[:, -4:])
            self.net[0].bias.copy_(reference.net[0].bias)

    def forward(self, sam_features, radial_features, e0):
        if sam_features.shape[1] != self.sam_channels or radial_features.shape[1] != self.radial_channels:
            raise ValueError('Unexpected SAM/radial feature channels')
        if sam_features.shape[0] != radial_features.shape[0] or sam_features.shape[-2:] != radial_features.shape[-2:]:
            raise ValueError('SAM and radial feature maps must be spatially aligned')
        sim = torch.einsum('bchw,c->bhw', sam_features, e0)
        zeros = torch.zeros_like(sim)
        return super().forward(torch.cat((sam_features, radial_features), 1),
                               F.pad(e0, (0, self.radial_channels)), sim, zeros, zeros, .25, .1)


class RadialResidualObjectness(ResidualObjectness):
    """Keep frozen F4 + trainable CNN; append radial point features to branch MLP."""

    def __init__(self, base_state, radial_channels=256, bound=2.):
        super().__init__(base_state, bound)
        self.radial_channels = radial_channels
        original = self.branch.mlp[0]
        extended = nn.Linear(original.in_features+radial_channels, original.out_features)
        with torch.no_grad():
            extended.weight.zero_()
            extended.weight[:, :original.in_features].copy_(original.weight)
            extended.bias.copy_(original.bias)
        self.branch.mlp[0] = extended

    def forward(self, layers, raw, f4, radial, return_delta=False):
        with torch.no_grad():
            base = self.base(f4)
        spatial = self.branch.spatial_descriptor(layers, raw)
        if radial.shape != (spatial.shape[0], self.radial_channels):
            raise ValueError('Expected one radial descriptor per candidate point')
        correction = self.branch.mlp(torch.cat((spatial, radial), 1)).squeeze(-1)
        delta = self.bound*torch.tanh(correction/self.bound)
        logits = base+delta
        return (logits, delta) if return_delta else logits
