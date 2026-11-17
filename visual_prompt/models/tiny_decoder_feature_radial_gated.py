"""Two-scale gated radial residuals extracted from each decoder feature map."""
import math
import types

import torch
from torch import nn

from visual_prompt.models.tiny_decoder_radial_gated import (
    GatedRadialStage, install_decoder_radial_gated,
)
from visual_prompt.models.tiny_radial_pre_h0 import FeatureRadialDescriptor


class StageFeatureRadial(FeatureRadialDescriptor):
    def __init__(self, channels, resolution, neighbor_values):
        super().__init__(neighbor_values=neighbor_values)
        self.compress = nn.Sequential(nn.Conv2d(channels, 4, 1, bias=False), nn.GroupNorm(1, 4))
        # Same relative coverage: half the pixel radii on the128 grid.
        self.offsets = self.offsets*(resolution/256)


class TwoScaleFeatureRadialFusion(nn.Module):
    def __init__(self, active_stages=(0, 1), neighbor_values=False, fixed_gate=False):
        super().__init__()
        self.active_stages = active_stages
        self.descriptors = nn.ModuleList([
            StageFeatureRadial(64, 128, neighbor_values),
            StageFeatureRadial(32, 256, neighbor_values),
        ])
        self.stem = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False), nn.GroupNorm(1, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.GELU())
        self.stages = nn.ModuleList([GatedRadialStage(64), GatedRadialStage(32)])
        # Construct in canonical order so retained weights have the same initialization.
        for index, stage in enumerate(self.stages):
            if index not in active_stages:
                self.descriptors[index] = None
                self.stages[index] = None
            elif fixed_gate:
                stage.norm = stage.context = stage.gate = None

    def forward_stage(self, features, stage):
        radial = self.stem(self.descriptors[stage](features))
        return self.stages[stage](features, radial)


class FixedFeatureRadialFusion(nn.Module):
    """Add a signed center-surround residual with fixed channel groups and gain."""
    def __init__(self, active_stages, scale):
        super().__init__()
        self.active_stages = active_stages
        self.scale = scale
        if not math.isfinite(scale):
            raise ValueError('Fixed radial scale must be finite')
        self.descriptors = nn.ModuleList()
        for stage, resolution in enumerate((128, 256)):
            descriptor = None
            if stage in active_stages:
                descriptor = FeatureRadialDescriptor()
                descriptor.compress = nn.Identity()
                descriptor.offsets.mul_(resolution / 256)
            self.descriptors.append(descriptor)

    def forward_stage(self, features, stage):
        if self.scale == 0:
            return features
        _, channels, height, width = features.shape
        group_width = channels // 4
        pooled = features.float().reshape(1, 4, group_width, height, width).mean(dim=2)
        contrast, valid = self.descriptors[stage].sample(pooled)
        radial = contrast.sum(dim=0, keepdim=True) / valid.sum(dim=0).clamp_min(1)
        radial = radial.repeat_interleave(group_width, dim=1)
        return features + (self.scale * radial).to(features.dtype)


def install_decoder_feature_radial_gated(
    model, mode, mean, std, *, active_stages=(0, 1), neighbor_values=False, fixed_gate=False,
    fixed_scale=None,
):
    assert mode == 'feature_decoder_gated'
    # Keep the exact insertion points, query-only hooks, counters and residual
    # heads of the completed image-radial gate; replace its descriptor source.
    install_decoder_radial_gated(model, 'decoder_gated', mean, std)
    branch = model.radial_branch
    fusion = (
        TwoScaleFeatureRadialFusion(active_stages, neighbor_values, fixed_gate)
        if fixed_scale is None else FixedFeatureRadialFusion(active_stages, fixed_scale)
    )
    branch.fusion = fusion.to(model.device)
    branch.stages_per_query = len(active_stages)

    def inject_stage(self, features, stage):
        if self.active is None or stage not in self.fusion.active_stages:
            return features
        encoder_size = model.sam.image_size
        channels, stride = (64, 8) if stage == 0 else (32, 4)
        expected = (channels, encoder_size // stride, encoder_size // stride)
        assert tuple(features.shape[1:]) == expected
        self.applied += 1
        return self.fusion.forward_stage(features, stage)

    branch.inject_stage = types.MethodType(inject_stage, branch)
    return model
