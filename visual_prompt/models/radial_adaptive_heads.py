"""Learn radial readouts while retaining feature concatenation and vector masks."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.lightweight_heads import ObjectnessMLP, soft_iou_loss
from visual_prompt.models.radial_augmented_heads import RadialMaskVectorMLP
from visual_prompt.models.region_mask_heads import inner_edge


def projection(channels, config, spatial=False):
    if config.get('projection', 'none') == 'none':
        return nn.Identity(), channels
    output = int(config.get('projected_channels', 32))
    linear = (lambda a, b: nn.Conv2d(a, b, 1, bias=False)) if spatial else (lambda a, b: nn.Linear(a, b, bias=False))
    activation = nn.Identity if config['projection'] == 'linear' else nn.GELU
    return nn.Sequential(linear(channels, output), activation(), linear(output, output)), output


class AdaptiveRadialObjectness(nn.Module):
    def __init__(self, radial_channels, config, f4_channels=1440):
        super().__init__(); self.config = dict(config)
        self.project, output = projection(radial_channels, config)
        self.main = ObjectnessMLP(f4_channels+output, 128)
        self.aux = nn.Linear(output, 1) if config.get('auxiliary') == 'radial_classification' else None

    def radial_representation(self, radial):
        projected = self.project(radial)
        if 'output_norm' in self.config:
            projected = F.normalize(projected, dim=1, eps=1e-6)*float(self.config['output_norm'])
        return projected

    def forward(self, f4, radial):
        self.last_radial = self.radial_representation(radial)
        return self.main(torch.cat((f4, self.last_radial), 1))

    def forward_radial_zero(self, f4, radial):
        with torch.no_grad():
            projected = self.radial_representation(radial)
        return self.main(torch.cat((f4, torch.zeros_like(projected)), 1))

    def auxiliary_loss(self, labels):
        if self.aux is None:
            return self.last_radial.sum()*0
        return float(self.config.get('aux_weight', .1))*F.binary_cross_entropy_with_logits(self.aux(self.last_radial).squeeze(1), labels)


class AdaptiveRadialMask(nn.Module):
    def __init__(self, radial_channels, e0, config, sam_channels=352):
        super().__init__(); self.config = dict(config); self.sam_channels = sam_channels
        self.conditioned = config.get('condition', 'none') in ('candidate', 'zero_control')
        inputs = 3*radial_channels+2 if self.conditioned else radial_channels
        self.project, output = projection(inputs, config, spatial=True)
        self.main = RadialMaskVectorMLP(sam_channels, output, e0)
        self.aux_kind = config.get('auxiliary', 'none')
        self.aux_vector = nn.Sequential(nn.Linear(2*output, 128), nn.GELU(), nn.Linear(128, output+1)) if self.aux_kind == 'radial_mask' else None
        self.edge = nn.Conv2d(output, 1, 1) if self.aux_kind == 'edge' else None

    def radial_representation(self, radial):
        if self.conditioned:
            b, _, h, w = radial.shape
            center = radial[:, :, h//2:h//2+1, w//2:w//2+1].expand_as(radial)
            yy, xx = torch.meshgrid(torch.arange(h, device=radial.device, dtype=radial.dtype),
                                    torch.arange(w, device=radial.device, dtype=radial.dtype), indexing='ij')
            positions = torch.stack(((xx-w//2)/32, (yy-h//2)/32), 0)[None].expand(b, -1, -1, -1)
            extra = torch.cat((center, radial-center, positions), 1)
            if self.config.get('condition') == 'zero_control':
                extra = torch.zeros_like(extra)
            radial = torch.cat((radial, extra), 1)
        projected = self.project(radial)
        if 'output_norm' in self.config:
            projected = F.normalize(projected, dim=1, eps=1e-6)*float(self.config['output_norm'])
        return projected

    def forward_features(self, features, e0, zero_radial=False):
        sam, radial = features[:, :self.sam_channels], features[:, self.sam_channels:]
        projected = self.radial_representation(radial)
        if zero_radial:
            projected = torch.zeros_like(projected)
        self.last_radial = projected
        return self.main(sam, projected, e0)

    def auxiliary_loss(self, logits, targets, positive_count):
        phi = self.last_radial; zero = logits.sum()*0
        weight = float(self.config.get('aux_weight', .1))
        if self.aux_kind == 'hard_negative':
            negative = logits[positive_count:].flatten(1)
            return weight*F.softplus(negative.topk(min(16, negative.shape[1]), dim=1).values).mean() if len(negative) else zero
        if self.aux_vector is not None:
            h, w = phi.shape[-2:]
            output = self.aux_vector(torch.cat((phi[:, :, h//2, w//2], phi.mean((-2, -1))), 1))
            vector = F.normalize(output[:, :-1], dim=1)
            aux = torch.einsum('bchw,bc->bhw', phi, vector)/.1+output[:, -1, None, None]
            value = soft_iou_loss(aux[:positive_count], targets[:positive_count])
            if len(aux) > positive_count:
                value = value+F.softplus(aux[positive_count:]).mean()
            return weight*value
        if self.edge is not None:
            boundary = inner_edge(targets, int(self.config.get('thickness', 1)))
            raw = F.binary_cross_entropy_with_logits(self.edge(phi)[:, 0], boundary, reduction='none')
            pos, neg = boundary > 0, boundary == 0
            value = (raw[pos].mean() if pos.any() else zero)+(raw[neg].mean() if neg.any() else zero)
            return weight*value
        return zero
