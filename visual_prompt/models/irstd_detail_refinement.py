"""Conditioned native-pixel residual refinement for the fixed radial Mask head."""
import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.radial_augmented_heads import RadialMaskVectorMLP
from visual_prompt.pipelines.p2p_framework import mask_descriptor, mask_forward


class DetailMask(nn.Module):
    def __init__(self, e0, refinement=False, width=16, bound=2.0):
        super().__init__()
        self.base = RadialMaskVectorMLP(352, 64, e0)
        self.bound = bound
        self.refinement = refinement
        if refinement:
            self.shallow = nn.Sequential(nn.Conv2d(2, width, 3, padding=1), nn.GELU(),
                                         nn.Conv2d(width, width, 3, padding=1), nn.GELU())
            self.condition = nn.Linear(836, 2 * width)
            self.output = nn.Conv2d(width + 1, 1, 3, padding=1)
            nn.init.zeros_(self.condition.weight)
            nn.init.zeros_(self.condition.bias)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, features, gray, e0):
        descriptor = mask_descriptor(features, features[:, :352], e0)
        base = mask_forward(self.base, features, descriptor, e0)
        if not self.refinement:
            return base
        local_mean = F.avg_pool2d(gray, 5, 1, 2, count_include_pad=False)
        shallow = self.shallow(torch.cat((gray, gray - local_mean), 1))
        scale, shift = self.condition(descriptor).chunk(2, 1)
        shallow = shallow * (1 + scale.tanh()[:, :, None, None]) + shift[:, :, None, None]
        residual = self.output(torch.cat((shallow, base.sigmoid()[:, None]), 1))[:, 0]
        return base + self.bound * residual.tanh()


def boundary_band(target, valid):
    """One-pixel inner/outer band; do not supervise artificial invalid boundaries."""
    target = target[:, None].float()
    valid = valid[:, None].float()
    dilation = F.max_pool2d(target, 3, 1, 1)
    erosion = 1 - F.max_pool2d(1 - target, 3, 1, 1)
    safe_valid = 1 - F.max_pool2d(1 - valid, 3, 1, 1)
    return ((dilation - erosion) * safe_valid)[:, 0]


def band_bce(logits, target, valid):
    band = boundary_band(target, valid)
    count = band.sum((-2, -1))
    loss = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * band).sum((-2, -1))
    active = count > 0
    return (loss[active] / count[active]).mean() if active.any() else logits.sum() * 0
