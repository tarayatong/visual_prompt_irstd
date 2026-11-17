"""Center-SAM channels retained; only add an identity-initialized learned radial map."""
import torch
from torch import nn
from torch.nn import functional as F
from visual_prompt.models.center_sam_channel_mask import CenterSAMChannelMask


class LearnedRadialCenterSAM(CenterSAMChannelMask):
    def __init__(self,mean,std):
        super().__init__('radial_center',mean,std)
        identity=self.mapping.clone()
        del self.mapping
        self.mapping=nn.Parameter(identity)

    def forward(self,sam,raw,valid):
        radial=F.conv2d(self.base.encode(raw,valid),self.mapping)
        h,w=sam.shape[-2:]
        center=sam[:,:,h//2:h//2+1,w//2:w//2+1].expand(-1,-1,h,w)
        features=torch.cat((sam,radial,center),1)
        out=self.base.head.net(features)
        vector=F.normalize(out[:,:768],dim=1,eps=1e-6)
        bias=out[:,768]
        z=(features*vector).sum(1)/.1+bias
        identity=torch.eye(64,device=self.mapping.device).reshape(64,64,1,1)
        self.diagnostics=dict(SAM_norm=float(sam.detach().norm(dim=1).mean()),radial_norm=float(radial.detach().norm(dim=1).mean()),
            radial_norm_max=float(radial.detach().norm(dim=1).max()),center_norm=float(center.detach().norm(dim=1).mean()),
            mapping_deviation_from_identity=float((self.mapping.detach()-identity).norm()),mapping_norm=float(self.mapping.detach().norm()),
            logit_min=float(z.detach().min()),logit_max=float(z.detach().max()),bias_mean=float(bias.detach().mean()),bias_std=float(bias.detach().std()))
        return z
