"""Learned polar geometry and signed radial/angular derivatives on decoder U."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.tiny_radial_pre_h0 import PreH0RadialFusion, install_radial_pre_h0


class LearnedPolarDescriptor(nn.Module):
    def __init__(self, channels=2):
        super().__init__()
        self.channels = channels
        self.compress = nn.Sequential(nn.Conv2d(32, channels, 1, bias=False), nn.GroupNorm(1, channels))
        gaps = torch.tensor([1., 1., 1., 1., 2., 2., 4., 4.])
        self.raw_gaps = nn.Parameter(torch.log(torch.expm1(gaps-.1)))
        self.raw_span = nn.Parameter(torch.zeros(()))
        self.rotation = nn.Parameter(torch.zeros(()))
        self.pair_offsets = nn.Parameter(torch.zeros(4))
        self.radius_logits = nn.Parameter(torch.zeros(5, 8, 8))
        self.family_scales = nn.Parameter(torch.zeros(5))
        self.register_buffer('base_angles', torch.arange(8)*math.pi/4)
        self.base_grid_cache = {}

    def sampling_geometry(self):
        gaps = F.softplus(self.raw_gaps)+.1
        span = 16*torch.exp(.25*self.raw_span.tanh())
        radii = span*gaps.cumsum(0)/gaps.sum()
        pair = (math.pi/32)*self.pair_offsets.tanh()
        angles = self.base_angles+(math.pi/8)*self.rotation.tanh()+torch.cat([pair, pair])
        return radii, angles

    @torch.no_grad()
    def geometry_snapshot(self):
        radii, angles = self.sampling_geometry()
        return dict(radii_grid_pixels=radii.cpu().tolist(), angles_degrees=(angles*180/math.pi).cpu().tolist(),
                    radius_weights=self.radius_logits.softmax(-1).cpu().tolist(),
                    family_scales=torch.exp(self.family_scales.tanh()).cpu().tolist(),
                    families=['contrast', 'radial_first', 'radial_second', 'tangential_first', 'angular_second'])

    def forward(self, features):
        small = self.compress(features).float()
        batch, channels, height, width = small.shape
        assert batch == 1
        radii, angles = self.sampling_geometry()
        key = (height, width, str(small.device))
        if key not in self.base_grid_cache:
            y = (torch.arange(height, device=small.device)+.5)*2/height-1
            x = (torch.arange(width, device=small.device)+.5)*2/width-1
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            self.base_grid_cache[key] = torch.stack([xx, yy], -1)[None]
        dx = angles.cos()[:, None]*radii[None, :]
        dy = angles.sin()[:, None]*radii[None, :]
        offsets = torch.stack([dx*2/width, dy*2/height], -1).reshape(64, 1, 1, 2)
        grid = self.base_grid_cache[key]+offsets
        valid = ((grid[..., 0] >= -1+1/width) & (grid[..., 0] <= 1-1/width)
                 & (grid[..., 1] >= -1+1/height) & (grid[..., 1] <= 1-1/height)).reshape(8, 8, 1, height, width)
        rings = F.grid_sample(small.expand(64, -1, -1, -1), grid,
                              mode='bilinear', padding_mode='border', align_corners=False)
        rings = rings.reshape(8, 8, channels, height, width)
        center = small[0][None, None]
        previous = torch.cat([center.expand(8, 1, -1, -1, -1), rings[:, :-1]], 1)
        gaps = torch.diff(torch.cat([radii.new_zeros(1), radii]))
        radial_first = (rings-previous)/gaps[None, :, None, None, None]
        radial_valid = valid & torch.cat([torch.ones_like(valid[:, :1]), valid[:, :-1]], 1)
        radial_second = 2*(radial_first[:, 1:]-radial_first[:, :-1])/(gaps[1:]+gaps[:-1])[None, :, None, None, None]
        radial_second = torch.cat([radial_second, torch.zeros_like(radial_second[:, :1])], 1)
        second_valid = radial_valid[:, 1:] & radial_valid[:, :-1]
        second_valid = torch.cat([second_valid, torch.zeros_like(second_valid[:, :1])], 1)
        next_gap = torch.cat([angles[1:]-angles[:-1], angles[:1]+2*math.pi-angles[-1:]])
        previous_gap = next_gap.roll(1)
        hp = previous_gap[:, None, None, None, None]
        hn = next_gap[:, None, None, None, None]
        angular_first = (-hn/(hp*(hp+hn))*rings.roll(1, 0)
                         +(hn-hp)/(hp*hn)*rings+hp/(hn*(hp+hn))*rings.roll(-1, 0))
        angular_second = 2*((rings.roll(-1, 0)-rings)/hn-(rings-rings.roll(1, 0))/hp)/(hp+hn)
        angular_valid = valid & valid.roll(1, 0) & valid.roll(-1, 0)
        tangential = angular_first/radii[None, :, None, None, None]
        angular_curvature = angular_second/radii.square()[None, :, None, None, None]
        basis = [(center-rings, valid), (radial_first, radial_valid),
                 (radial_second, second_valid), (tangential, angular_valid),
                 (angular_curvature, angular_valid)]
        outputs = []
        for family, (response, mask) in enumerate(basis):
            weights = self.radius_logits[family].softmax(-1)[:, :, None, None, None]*mask
            pooled = (response*weights).sum(1)/weights.sum(1).clamp_min(1e-6)
            pooled = pooled*torch.exp(self.family_scales[family].tanh())
            outputs.append(pooled.reshape(1, 8*channels, height, width))
        return F.normalize(torch.cat(outputs, 1), dim=1, eps=1e-4)*.25


class LearnedPolarFusion(PreH0RadialFusion):
    def __init__(self, mean, std):
        super().__init__('feature_u', mean, std)
        self.feature_radial = LearnedPolarDescriptor(channels=2)
        self.stem[0] = nn.Conv2d(5*8*2, 32, 1, bias=False)


def install_learned_polar(model, mode, mean, std):
    assert mode == 'polar_learned'
    install_radial_pre_h0(model, 'feature_u', mean, std)
    model.radial_branch.fusion = LearnedPolarFusion(mean, std).to(model.device)
    return model
