"""Radial augmentation of dc2 output U BEFORE native H0 addition/activation."""
import math
import types

import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.tiny_prototype_radial import QueryRadialInjection


class FeatureRadialDescriptor(nn.Module):
    """Differentiable8-direction x8-radius contrast on compressed decoder U.

    Compress32->4 before spatial sampling to bound memory. Radius units are
    pixels of the256x256 U grid, unlike raw-image radii in the other arm.
    """
    def __init__(self, neighbor_values=False):
        super().__init__()
        self.neighbor_values = neighbor_values
        self.compress = nn.Sequential(nn.Conv2d(32, 4, 1, bias=False), nn.GroupNorm(1, 4))
        offsets = []
        for direction in range(8):
            angle = direction*math.pi/4
            for radius in (1., 2., 3., 4., 6., 8., 12., 16.):
                offsets.append((radius*math.cos(angle), radius*math.sin(angle)))
        self.register_buffer('offsets', torch.tensor(offsets))
        self.grid_cache = {}

    def forward(self, features):
        small = self.compress(features).float()
        contrast, _ = self.sample(small)
        descriptor = contrast.reshape(1, -1, *small.shape[-2:])
        return F.normalize(descriptor, dim=1, eps=1e-6)*.25

    def sample(self, small):
        batch, _, height, width = small.shape
        assert batch == 1  # Native SANSA decodes each episode independently.
        key = (height, width, str(small.device))
        if key not in self.grid_cache:
            y = (torch.arange(height, device=small.device)+.5)*2/height-1
            x = (torch.arange(width, device=small.device)+.5)*2/width-1
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            base = torch.stack([xx, yy], -1)[None]
            offset = self.offsets.to(small.device)*small.new_tensor([2/width, 2/height])
            grid = base+offset[:, None, None, :]
            valid = ((grid[..., 0] >= -1+1/width) & (grid[..., 0] <= 1-1/width)
                     & (grid[..., 1] >= -1+1/height) & (grid[..., 1] <= 1-1/height))
            self.grid_cache[key] = (grid, valid)
        grid, valid = self.grid_cache[key]
        sampled = F.grid_sample(small.expand(64, -1, -1, -1), grid,
                                mode='bilinear', padding_mode='border', align_corners=False)
        descriptor = sampled if self.neighbor_values else small-sampled
        return descriptor.masked_fill(~valid[:, None], 0), valid


class PreH0RadialFusion(nn.Module):
    def __init__(self, mode, mean, std):
        super().__init__()
        self.mode = mode
        self.register_buffer('radial_mean', torch.as_tensor(mean).float().reshape(1, 64, 1, 1).clone())
        self.register_buffer('radial_std', torch.as_tensor(std).float().reshape(1, 64, 1, 1).clone())
        self.feature_radial = FeatureRadialDescriptor() if mode == 'feature_u' else None
        channels = 256 if mode == 'feature_u' else 64
        self.stem = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False),
                                  nn.GroupNorm(1, 32), nn.GELU(),
                                  nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.GELU())
        self.context = nn.Sequential(nn.GroupNorm(1, 32), nn.Conv2d(32, 32, 1, bias=False))
        self.gate = nn.Conv2d(64, 32, 1)
        self.projection = nn.Conv2d(32, 32, 1, bias=False)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.projection.weight)

    def forward(self, upsampled, raw, valid):
        if self.mode == 'feature_u':
            radial = self.feature_radial(upsampled)
        else:
            radial = (raw.float()-self.radial_mean)/self.radial_std
            radial = F.normalize(radial.masked_fill(~valid, 0), dim=1, eps=1e-6)*.25
        radial = self.stem(radial)
        gate = self.gate(torch.cat([self.context(upsampled), radial], 1)).sigmoid()
        return upsampled+.1*self.projection(gate*radial)


def install_radial_pre_h0(model, mode, mean, std):
    assert mode in ('image_u', 'feature_u')
    branch = QueryRadialInjection('decoder', mean, std).to(model.device)
    branch.fusion = PreH0RadialFusion(mode, mean, std).to(model.device)
    model.add_module('radial_branch', branch)
    native_decode = model._compute_decoder_out_w_mem

    def decode(self, backbone, index, memory_index, memory_bank):
        assert branch.active is None and branch.inputs is not None
        assert branch.cursor < len(branch.inputs)
        branch.active = branch.inputs[branch.cursor]
        branch.cursor += 1
        try:
            return native_decode(backbone, index, memory_index, memory_bank)
        finally:
            branch.active = None

    model._compute_decoder_out_w_mem = types.MethodType(decode, model)

    def after_dc2(module, args, output):
        return branch.inject(output)

    # Native decoder next adds feat_s0 (=H0), then applies act2, unmodified.
    model.sam.sam_mask_decoder.output_upscaling[3].register_forward_hook(after_dc2)
    return model
