"""Two-scale, decoder-conditioned radial side branch; native path preserved."""
import types

import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.tiny_prototype_radial import QueryRadialInjection


class GatedRadialStage(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.context = nn.Conv2d(channels, 32, 1, bias=False)
        self.gate = nn.Conv2d(64, 32, 1)
        self.output = nn.Conv2d(32, channels, 1, bias=False)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.output.weight)

    def forward(self, features, radial):
        if radial.shape[-2:] != features.shape[-2:]:
            radial = F.interpolate(radial, features.shape[-2:], mode='area')
        if self.gate is None:
            gate = .5
        else:
            context = self.context(self.norm(features))
            gate = self.gate(torch.cat([context, radial], 1)).sigmoid()
        return features + .1*self.output(gate*radial)


class MultiScaleRadialFusion(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('radial_mean', torch.as_tensor(mean).float().reshape(1, 64, 1, 1).clone())
        self.register_buffer('radial_std', torch.as_tensor(std).float().reshape(1, 64, 1, 1).clone())
        self.stem = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.GroupNorm(1, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.GELU())
        self.stages = nn.ModuleList([GatedRadialStage(64), GatedRadialStage(32)])

    def encode(self, raw, valid):
        radial = (raw.float()-self.radial_mean)/self.radial_std
        radial = F.normalize(radial.masked_fill(~valid, 0), dim=1, eps=1e-6)*.25
        return self.stem(radial)


class MultiScaleRadialInjection(QueryRadialInjection):
    def __init__(self, mean, std):
        super().__init__('decoder', mean, std)
        self.fusion = MultiScaleRadialFusion(mean, std)
        self.encoded = None
        self.stages_per_query = 2

    def inject_stage(self, features, stage):
        if self.active is None:
            return features
        expected = (64, 128, 128) if stage == 0 else (32, 256, 256)
        assert tuple(features.shape[1:]) == expected
        if stage == 0:
            assert self.encoded is None
            self.encoded = self.fusion.encode(*self.active)
        else:
            assert self.encoded is not None
        self.applied += 1
        return self.fusion.stages[stage](features, self.encoded)

    def finish(self):
        assert self.active is None and self.encoded is None
        assert self.inputs is not None
        assert self.cursor == len(self.inputs) and self.applied == self.stages_per_query*len(self.inputs)
        self.inputs = None


def install_decoder_radial_gated(model, mode, mean, std):
    assert mode == 'decoder_gated'
    branch = MultiScaleRadialInjection(mean, std).to(model.device)
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
            branch.encoded = None

    model._compute_decoder_out_w_mem = types.MethodType(decode, model)
    decoder = model.sam.sam_mask_decoder

    def hook128(module, args, output):
        return branch.inject_stage(output, 0)

    def hook256(module, args, output):
        return branch.inject_stage(output, 1)

    decoder.output_upscaling[2].register_forward_hook(hook128)
    decoder.output_upscaling[4].register_forward_hook(hook256)
    return model
