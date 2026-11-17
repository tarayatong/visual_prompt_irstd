"""Independent pixelwise Mask heads; no residual contrast bypass in polar mode."""
import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.polar_pixel_encoder import PolarPixelEncoder


def masked_batch_iou(logits, target, valid, epsilon=1e-6):
    # Select before sigmoid: ignored pixels have exactly zero loss gradient.
    p, y = logits[valid].sigmoid(), target[valid]
    intersection = (p*y).sum()
    return 1-(intersection+epsilon)/(p.sum()+y.sum()-intersection+epsilon)


class PixelwisePolarMask(nn.Module):
    """SAM + radial maps -> per-pixel logit or normalized dynamic vector.

    There is no crop descriptor, support-vector residual, or legacy fallback.
    """
    def __init__(self, mean, std, config, feature_mean, feature_std):
        super().__init__()
        if config['head'] not in ('logit','dynamic_normalized'):
            raise ValueError('Only pixelwise logit and dynamic_normalized heads are supported; crop heads were removed.')
        if config['representation'] not in ('gain','polar'):
            raise ValueError('Expected gain or polar radial representation')
        self.config = dict(config)
        self.register_buffer('mean',mean.detach().reshape(1,64,1,1).clone())
        self.register_buffer('std',std.detach().reshape(1,64,1,1).clone())
        self.log_gain = nn.Parameter(torch.zeros(1,64,1,1))
        self.encoder = None
        self.operator = None  # Fixed sampling; consumed by the shared data loop.
        self.capture_diagnostics = False
        self.diagnostics = {}
        if config['representation'] == 'polar':
            self.encoder = PolarPixelEncoder(feature_mean, feature_std, config['polar_hidden'])
        self.net = nn.Sequential(nn.Conv2d(416,128,1), nn.GELU(),
            nn.Conv2d(128,1 if config['head']=='logit' else 417,1))

    def encode(self, sam, raw, valid):
        if self.encoder is None:
            y = F.normalize(((raw-self.mean)/self.std).masked_fill(~valid,0), dim=1, eps=1e-6)*.25
        else:
            y = self.encoder(raw, valid)
        radial = (y*self.log_gain.clamp(-2,2).exp()).masked_fill(~valid,0)
        return torch.cat((sam,radial),1), radial

    def forward(self, sam, raw, valid, gray=None, points=None):
        features, radial = self.encode(sam,raw,valid)
        vector = None
        out = self.net(features)
        if self.config['head']=='logit':
            logits = out[:,0]  # No artificial temperature for direct logits.
            bias = self.net[-1].bias
        else:
            vector = F.normalize(out[:,:416],dim=1,eps=1e-6)
            bias = out[:,416]
            logits = (features*vector).sum(1)/.1+bias
        if self.capture_diagnostics:
            def summary(value):
                q = value.detach().float().flatten()
                return dict(mean=float(q.mean()), min=float(q.min()), max=float(q.max()),
                    quantiles=torch.quantile(q,q.new_tensor([.01,.5,.99])).cpu().tolist())
            self.diagnostics = dict(sam_norm=summary(sam.norm(dim=1)),radial_norm=summary(radial.norm(dim=1)),
                logits=summary(logits), bias=summary(bias), gain=summary(self.log_gain.clamp(-2,2).exp()))
            if vector is not None:self.diagnostics['vector_norm']=summary(vector.norm(dim=1))
        return logits

    def optimizer_groups(self, training):
        radial = [self.log_gain]
        if self.encoder is not None:radial.extend(self.encoder.parameters())
        return [dict(name='mask_mlp',params=list(self.net.parameters()),lr=training['head_lr']),
            dict(name='radial',params=radial,lr=training['radial_lr'])]
