"""RCR controls for angle--radius convolution and query-image fusion."""

from copy import deepcopy
import types

import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.tiny_prototype_feature_radial import install_prototype_feature_radial
from visual_prompt.models.tiny_radial_pre_h0 import FeatureRadialDescriptor


class PolarGridEncoder(nn.Module):
    def __init__(self, channels, use_convolution):
        super().__init__()
        self.use_convolution = use_convolution
        if use_convolution:
            self.radial = nn.Conv2d(channels, 1, (1, 3), bias=False)
            self.angular = nn.Conv2d(channels, 1, (3, 1), bias=False)
            self.joint = nn.Conv2d(channels, 1, (3, 3), bias=False)
            self.fuse = nn.Conv2d(3, 1, 1, bias=False)
        else:
            self.pointwise = nn.Conv2d(channels, 1, 1, bias=False)

    def encode_grids(self, grids):
        if not self.use_convolution:
            return self.pointwise(grids)
        # All four convolutions are linear. Fold their weights to avoid
        # storing three activations for every pixel, retaining every gradient.
        radial_weight, angular_weight, joint_weight = self.fuse.weight.flatten()
        kernel = (radial_weight * F.pad(self.radial.weight, (0, 0, 1, 1))
                  + angular_weight * F.pad(self.angular.weight, (1, 1, 0, 0))
                  + joint_weight * self.joint.weight)
        grids = F.pad(grids, (0, 0, 1, 1), mode='circular')
        grids = F.pad(grids, (1, 1, 0, 0), mode='replicate')
        return F.conv2d(grids, kernel)

    def forward(self, descriptor):
        _, _, height, width = descriptor.shape
        # The existing descriptor flattens angle, radius, then feature channel.
        grids = descriptor.reshape(8, 8, -1, height * width).permute(3, 2, 0, 1)
        encoded = self.encode_grids(grids)
        return encoded.reshape(height * width, 64).T.reshape(1, 64, height, width).contiguous()


class ImageRadialDescriptor(FeatureRadialDescriptor):
    def __init__(self):
        super().__init__()
        self.compress = nn.Identity()
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406]).reshape(1, 3, 1, 1))
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225]).reshape(1, 3, 1, 1))
        self.register_buffer('luminance', torch.tensor([.299, .587, .114]).reshape(1, 3, 1, 1))

    def forward(self, query_image, size):
        # Work on intensity from the input image before SAM2 normalization.
        rgb = query_image.float() * self.rgb_std + self.rgb_mean
        intensity = (rgb * self.luminance).sum(1, keepdim=True)
        aligned = F.interpolate(intensity, size=size, mode='bilinear', align_corners=False)
        return super().forward(aligned)


class PaperResidualGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.GroupNorm(1, 32)
        self.context = nn.Conv2d(32, 32, 1, bias=False)
        self.gate = nn.Conv2d(32 + 64, 64, 1)
        self.output = nn.Conv2d(64, 32, 1, bias=False)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.output.weight)

    def forward(self, features, radial):
        context = self.context(self.norm(features))
        gate = self.gate(torch.cat([context, radial], dim=1)).sigmoid()
        return features + .1 * self.output(gate * radial)


def image_fusion(channels):
    """Learn a 1x1 fusion from an initial identity on the decoder branch."""
    fusion = nn.Conv2d(2 * channels, channels, 1, bias=False)
    with torch.no_grad():
        fusion.weight.zero_()
        fusion.weight[:, :channels, 0, 0].copy_(torch.eye(channels))
    return fusion


class PolarGridRCR(nn.Module):
    def __init__(self, feature_descriptor, use_convolution, use_image):
        super().__init__()
        self.stages = nn.ModuleList([None, PaperResidualGate()])
        self.feature_descriptor = feature_descriptor
        self.feature_encoder = PolarGridEncoder(4, use_convolution)
        self.image_descriptor = ImageRadialDescriptor() if use_image else None
        if use_image:
            self.image_encoder = PolarGridEncoder(1, use_convolution)
            self.image_fusion = image_fusion(64)

    def forward_stage(self, features, query_image):
        radial = self.feature_encoder(self.feature_descriptor(features))
        if self.image_descriptor is not None:
            image_radial = self.image_encoder(self.image_descriptor(query_image, features.shape[-2:]))
            radial = self.image_fusion(torch.cat([radial, image_radial], dim=1))
        return self.stages[1](features, radial)


class DualLegacyRCR(nn.Module):
    def __init__(self, decoder_rcr):
        super().__init__()
        self.decoder_rcr = decoder_rcr
        self.image_descriptor = ImageRadialDescriptor()
        self.image_stem = deepcopy(decoder_rcr.stem)
        self.image_stem[0] = nn.Conv2d(64, 32, 1, bias=False)
        self.image_fusion = image_fusion(32)

    @property
    def stages(self):
        return self.decoder_rcr.stages

    def forward_stage(self, features, query_image):
        feature_radial = self.decoder_rcr.stem(self.decoder_rcr.descriptors[1](features))
        image_radial = self.image_stem(self.image_descriptor(query_image, features.shape[-2:]))
        radial = self.image_fusion(torch.cat([feature_radial, image_radial], dim=1))
        return self.stages[1](features, radial)


def install_polar_grid_rcr(model, *, encoder, use_image=False):
    if encoder not in {'polar_conv', 'pointwise', 'legacy'} or (encoder == 'legacy' and not use_image):
        raise ValueError(f'Unsupported RCR control: {encoder}, use_image={use_image}')
    install_prototype_feature_radial(model, active_stages=(1,))
    branch = model.radial_branch
    if encoder == 'legacy':
        replacement = DualLegacyRCR(branch.fusion)
    else:
        replacement = PolarGridRCR(branch.fusion.descriptors[1], encoder == 'polar_conv', use_image)
    branch.fusion = replacement.to(model.device)

    def inject_stage(self, features, stage):
        if self.active is None or stage != 1:
            return features
        assert tuple(features.shape[1:]) == (32, 256, 256)
        image, _ = self.active
        if use_image:
            assert image is not None and image.shape[:2] == (1, 3)
        self.applied += 1
        return self.fusion.forward_stage(features, image)

    branch.inject_stage = types.MethodType(inject_stage, branch)
    return model
