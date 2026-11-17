"""Fixed radial probes: unchanged geometry, shorter geometry, or soft L2 scale."""
import torch
from visual_prompt.models.fixed_then_learned_radial import FixedThenLearnedRadial


class NUAAFixedRadial(FixedThenLearnedRadial):
    def __init__(self,variant,mean,std):
        assert variant in ('zero','fixed','short','amplitude')
        super().__init__('zero' if variant=='zero' else 'fixed',mean,std)
        self.variant=variant

    def forward(self,sam,raw,valid):
        if self.variant!='amplitude':return super().forward(sam,raw,valid)
        mean=self.base.feature_mean[:,0].reshape(1,64,1,1)
        std=self.base.feature_std[:,0].reshape(1,64,1,1)
        q=((raw-mean)/std).masked_fill(~valid,0)
        # sqrt(64) is the reference norm of 64 unit-variance channels. This
        # fixed denominator retains magnitude for small contrasts and bounds
        # large responses, unlike unit-normalizing every nonzero pixel.
        radial=.25*q/(q.square().sum(1,keepdim=True)+64).sqrt()
        z=self.base.head(torch.cat((sam,radial),1))
        self.diagnostics=dict(radial_norm=float(radial.detach().norm(dim=1).mean()),mapping_norm=float(self.mapping.norm()),
                              logit_min=float(z.detach().min()),logit_max=float(z.detach().max()))
        return z
