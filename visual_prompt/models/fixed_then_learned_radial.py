"""Equal-width zero, fixed identity, and trainable identity radial controls."""
import torch
from torch import nn
from torch.nn import functional as F
from visual_prompt.models.memory_radial_study import MemoryRadialMask


class FixedThenLearnedRadial(nn.Module):
    def __init__(self,variant,mean,std):
        super().__init__()
        assert variant in ('zero','fixed','learned')
        self.variant=variant
        self.base=MemoryRadialMask('R1',mean,std)
        identity=torch.eye(64).reshape(64,64,1,1)
        if variant=='learned':self.mapping=nn.Parameter(identity)
        else:self.register_buffer('mapping',identity)
        self.diagnostics={}

    def forward(self,sam,raw,valid):
        radial=torch.zeros_like(raw) if self.variant=='zero' else F.conv2d(self.base.encode(raw,valid),self.mapping)
        z=self.base.head(torch.cat((sam,radial),1))
        self.diagnostics=dict(radial_norm=float(radial.detach().norm(dim=1).mean()),mapping_norm=float(self.mapping.detach().norm()),
                              logit_min=float(z.detach().min()),logit_max=float(z.detach().max()))
        return z
