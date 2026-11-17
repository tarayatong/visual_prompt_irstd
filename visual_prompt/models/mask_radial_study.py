"""Fresh Mask-only radial encoders; no objectness or pretrained Mask weights."""
import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.lightweight_heads import MaskVectorMLP
from visual_prompt.models.radial_profile import DirectionalRadialOperator, RadialProfileConfig


def balanced_batch_iou(logits, target, epsilon=1e-6):
    """One foreground IoU over a balanced batch, including empty negative ROIs.

    Empty negative ROIs contribute predicted mass to the common union, so they
    receive a useful gradient without adding BCE or an auxiliary objective.
    """
    p = logits.sigmoid()
    intersection = (p * target).sum()
    union = p.sum() + target.sum() - intersection
    return 1 - (intersection + epsilon) / (union + epsilon)


class MaskRadialStudy(nn.Module):
    def __init__(self, e0, mean, std, config):
        super().__init__()
        initial_rng = torch.get_rng_state()
        self.config = dict(config)
        self.register_buffer('e0', e0.detach().clone())
        self.register_buffer('mean', mean.detach().reshape(1, 64, 1, 1).clone())
        self.register_buffer('std', std.detach().reshape(1, 64, 1, 1).clone())
        kind, channels = config['kind'], config['channels']
        self.log_gain = nn.Parameter(torch.zeros(1, 64, 1, 1), requires_grad=kind == 'gain')
        self.encoder = None
        groups = 8 if kind in ('direction', 'radius') else 1
        if kind in ('free', 'direction', 'radius', 'dynamic'):
            layers = []
            incoming = 64
            for level in range(config['depth']):
                outgoing = channels if level == config['depth'] - 1 else 32
                layers.append(nn.Conv2d(incoming, outgoing, 1, bias=False, groups=groups))
                if level < config['depth'] - 1:
                    layers.append(nn.GELU())
                incoming = outgoing
            self.encoder = nn.Sequential(*layers)
        self.dynamic = None
        if kind == 'dynamic':
            self.dynamic = nn.Sequential(nn.Conv2d(64, 16, 1), nn.GELU(), nn.Conv2d(16, channels * 64, 1))
            nn.init.zeros_(self.dynamic[-1].weight)
            nn.init.zeros_(self.dynamic[-1].bias)
        self.sam_projection = None
        sc = config['sam_channels']
        if sc != 352:
            self.sam_projection = nn.Conv2d(352, sc, 1, bias=False)
            nn.init.orthogonal_(self.sam_projection.weight[:, :, 0, 0])
        c = sc + channels
        # Common SAM-side initialization across radial widths at the same seed.
        # No teacher / old Mask state is used.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(initial_rng)
            reference = MaskVectorMLP(sc, 128, torch.zeros(sc), 'm2_residual')
        self.net = nn.Sequential(nn.Linear(2 * c + 4, 128), nn.GELU(), nn.Linear(128, c + 4))
        with torch.no_grad():
            self.net[0].weight.zero_()
            self.net[0].weight[:, :sc].copy_(reference.net[0].weight[:, :sc])
            self.net[0].weight[:, c:c + sc].copy_(reference.net[0].weight[:, sc:2 * sc])
            self.net[0].weight[:, -4:].copy_(reference.net[0].weight[:, -4:])
            self.net[0].bias.copy_(reference.net[0].bias)
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()
        self.operator = None
        geometry = config['geometry']
        if geometry != 'fixed':
            self.operator = DirectionalRadialOperator(1, RadialProfileConfig(
                components=('contrast',), fusion='concat', radii=(1., 2., 3., 4., 6., 8., 12., 16.),
                opposite_pairs=False, learnable_radii=geometry in ('radii', 'both'),
                learnable_rotation=geometry in ('rotation', 'both'), chunk_size=4096))

    def forward(self, sam, raw, valid, gray=None, points=None):
        if self.operator is not None:
            profiles = self.operator.sample_profiles(gray, points)
            b, _, h, w = raw.shape
            raw = profiles.values.flatten(2).transpose(1, 2).reshape(b, 64, h, w)
            valid = profiles.valid.flatten(2).transpose(1, 2).reshape(b, 64, h, w)
            raw = raw + (raw.half().float() - raw).detach()
        phi = F.normalize(((raw - self.mean) / self.std).masked_fill(~valid, 0), dim=1, eps=1e-6) * .25
        phi = phi * self.log_gain.clamp(-2, 2).exp()
        if self.config['kind'] == 'radius':
            b, _, h, w = phi.shape
            phi = phi.reshape(b, 8, 8, h, w).transpose(1, 2).reshape(b, 64, h, w)
        radial = self.encoder(phi) if self.encoder is not None else phi
        if self.config['kind'] == 'none':
            radial = phi[:, :0]
        if self.dynamic is not None:
            b, _, h, w = phi.shape
            weights = self.dynamic(phi).reshape(b, self.config['channels'], 64, h, w).tanh()
            radial = radial + (weights * phi[:, None]).sum(2) / 8
        if self.config['output_norm'] == 'l2':
            radial = F.normalize(radial, dim=1, eps=1e-6) * .25
        if self.sam_projection is None:
            s, e = sam, self.e0
        else:
            s = F.normalize(self.sam_projection(sam), dim=1, eps=1e-6)
            e = F.normalize(self.sam_projection.weight[:, :, 0, 0] @ self.e0, dim=0)
        features = torch.cat((s, radial), 1)
        sim = torch.einsum('bchw,c->bhw', sam, self.e0)
        h, w = sim.shape[-2:]
        stats = torch.stack((sim[:, h//2, w//2], sim.amax((-2, -1)), sim.mean((-2, -1)), sim.std((-2, -1))), 1)
        out = self.net(torch.cat((features[:, :, h//2, w//2], features.mean((-2, -1)), stats), 1))
        vector = F.normalize(F.pad(e, (0, radial.shape[1]))[None] + .25 * out[:, :features.shape[1]], dim=1)
        return torch.einsum('bchw,bc->bhw', features, vector) / .1 + out[:, -1, None, None]

    def optimizer_groups(self, training):
        groups = [dict(name='mask_mlp', params=list(self.net.parameters()), lr=training['head_lr'])]
        radial = [p for module in (self.encoder, self.dynamic) if module is not None for p in module.parameters()]
        if self.log_gain.requires_grad:
            radial.append(self.log_gain)
        if radial:
            groups.append(dict(name='radial', params=radial, lr=training['radial_lr']))
        if self.sam_projection is not None:
            groups.append(dict(name='mask_sam_projection', params=list(self.sam_projection.parameters()), lr=training['projection_lr']))
        if self.operator is not None:
            groups.append(dict(name='radial_geometry', params=[p for p in self.operator.parameters() if p.requires_grad], lr=training['geometry_lr']))
        return groups
