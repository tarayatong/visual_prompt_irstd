"""Local-encoding mask heads with optional spatial and inner-edge supervision."""
import torch
from torch import nn
from visual_prompt.models.lightweight_heads import MaskVectorMLP


class RegionMaskHead(nn.Module):
    def __init__(self, channels, initial_vector, variant='plain', hidden=128):
        super().__init__()
        self.variant = variant
        self.main = MaskVectorMLP(channels, hidden, initial_vector, 'm2_residual')
        if variant != 'plain':
            self.spatial = nn.Sequential(
                nn.Conv2d(channels, 32, 1), nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
                nn.Conv2d(32, 1, 1),
            )
            self.fuse = nn.Conv2d(2, 1, 1)
            with torch.no_grad():
                self.fuse.weight.zero_()
                self.fuse.weight[0, 0, 0, 0] = 1
                self.fuse.bias.zero_()

    def forward(self, features, e0):
        similarity = torch.einsum('bchw,c->bhw', features, e0)
        zeros = torch.zeros_like(similarity)
        main = self.main(features, e0, similarity, zeros, zeros, .25, .1)
        if self.variant == 'plain':
            return main, main, None
        spatial = self.spatial(features)[:, 0]
        final = self.fuse(torch.stack((main, spatial), dim=1))[:, 0]
        return final, main, spatial


def inner_edge(mask, thickness):
    # Explicit zero outside each crop; output is an inner band, always a subset of GT.
    import torch.nn.functional as F
    padded = F.pad(mask[:, None], (thickness,) * 4, value=0)
    eroded = -F.max_pool2d(-padded, 2 * thickness + 1, stride=1)
    return (mask - eroded[:, 0]).clamp(0, 1)
