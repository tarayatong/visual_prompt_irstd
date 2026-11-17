"""Matched-width radial ablations without any candidate-center SAM input."""
import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.center_sam_channel_mask import CenterSAMChannelMask


class NoCenterRadialMask(CenterSAMChannelMask):
    def __init__(self, variant, mean, std):
        assert variant in ('none', 'fixed', 'learned')
        super().__init__('sam' if variant == 'none' else 'radial', mean, std)
        self.radial_variant = variant
        if variant == 'learned':
            identity = self.mapping.clone()
            del self.mapping
            self.mapping = nn.Parameter(identity)

    def forward(self, sam, raw, valid):
        radial = torch.zeros_like(raw) if self.radial_variant == 'none' else self.base.encode(raw, valid)
        if self.radial_variant == 'learned':
            radial = F.conv2d(radial, self.mapping)
        # Zero slots match the center-present experiment's head dimensions.
        # No center location is read, broadcast, or used in the dot product.
        features = torch.cat((sam, radial, torch.zeros_like(sam)), 1)
        z = self.base.head(features)
        self.diagnostics = dict(
            SAM_norm=float(sam.detach().norm(dim=1).mean()),
            radial_norm=float(radial.detach().norm(dim=1).mean()),
            radial_norm_max=float(radial.detach().norm(dim=1).max()),
            center_norm=0.,
            mapping_deviation_from_identity=float((self.mapping.detach() - torch.eye(64, device=self.mapping.device).reshape(64, 64, 1, 1)).norm()),
            logit_min=float(z.detach().min()), logit_max=float(z.detach().max()))
        return z
