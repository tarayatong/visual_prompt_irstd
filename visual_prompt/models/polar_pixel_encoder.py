"""Pixelwise axial polar features; independent of crop-level Mask heads."""
import math
import torch
from torch import nn
from torch.nn import functional as F


RADII = (1., 2., 3., 4., 6., 8., 12., 16.)


def polar_derivatives(contrast, valid):
    """C, dP/dr, (1/r)dP/dtheta, opposite-even and opposite-odd fields.

    C=I(center)-P is enough to compute both derivatives: the center cancels.
    Derivatives precede channel statistics. Output: B,5,8,8,H,W, plus masks.
    Radial differences use actual intervals; angular differences are periodic.
    """
    b, channels, h, w = contrast.shape
    assert channels == 64 and valid.shape == contrast.shape
    c = contrast.reshape(b, 8, 8, h, w)
    v = valid.reshape_as(c)
    radii = contrast.new_tensor(RADII)
    dr = torch.diff(torch.cat((radii.new_zeros(1), radii))).reshape(1, 1, 8, 1, 1)
    previous = torch.cat((torch.zeros_like(c[:, :, :1]), c[:, :, :-1]), 2)
    previous_valid = torch.cat((torch.ones_like(v[:, :, :1]), v[:, :, :-1]), 2)
    gr_valid = v & previous_valid
    gr = ((previous-c)/dr).masked_fill(~gr_valid, 0)
    ga_valid = v & v.roll(1, 1) & v.roll(-1, 1)
    ga = ((c.roll(1, 1)-c.roll(-1, 1))/(2*radii.reshape(1,1,8,1,1)*(math.pi/4))).masked_fill(~ga_valid, 0)
    pair_valid = gr_valid & gr_valid.roll(4, 1)
    even = ((gr+gr.roll(4,1))/2).masked_fill(~pair_valid, 0)
    odd = ((gr-gr.roll(4,1))/2).masked_fill(~pair_valid, 0)
    fields = torch.stack((c.masked_fill(~v,0), gr, ga, even, odd), 1)
    masks = torch.stack((v, gr_valid, ga_valid, pair_valid, pair_valid), 1)
    return fields, masks


class PolarPixelEncoder(nn.Module):
    def __init__(self, mean, std, hidden=4):
        super().__init__()
        self.register_buffer('feature_mean', mean.detach().reshape(1,5,8,8,1,1).clone())
        self.register_buffer('feature_std', std.detach().reshape(1,5,8,8,1,1).clone())
        self.radial = nn.Conv2d(5, hidden, (1,3))
        self.angular = nn.Conv2d(5, hidden, (3,1))
        self.fuse = nn.Conv2d(2*hidden, 1, 1)

    def forward(self, raw, valid):
        fields, masks = polar_derivatives(raw, valid)
        q = ((fields-self.feature_mean)/self.feature_std).masked_fill(~masks,0)
        b, _, _, _, h, w = q.shape
        normalized = F.normalize(q.reshape(b,5,64,h,w), dim=2, eps=1e-6)*.25
        x = normalized.reshape(b,5,8,8,h,w).permute(0,4,5,1,2,3).reshape(b*h*w,5,8,8)
        radial = F.gelu(self.radial(F.pad(x,(1,1,0,0),mode='reflect')))
        angular = F.gelu(self.angular(F.pad(x,(0,0,1,1),mode='circular')))
        delta = self.fuse(torch.cat((radial,angular),1))
        delta = delta.reshape(b,h,w,64).permute(0,3,1,2).masked_fill(~valid,0)
        return delta
