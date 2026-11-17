"""Equal-width 2x2 center-SAM/radial channel controls, no crop statistics/e0."""
import torch
from visual_prompt.models.memory_radial_study import MemoryRadialMask
from visual_prompt.models.pixelwise_sam_mask import PixelwiseSAMMask


class CenterSAMChannelMask(torch.nn.Module):
    def __init__(self,variant,mean,std):
        super().__init__()
        assert variant in ('sam','radial','center','radial_center')
        self.variant=variant
        self.base=MemoryRadialMask('R1',mean,std)
        self.base.head=PixelwiseSAMMask(768,128)
        self.register_buffer('mapping',torch.eye(64).reshape(64,64,1,1))
        self.diagnostics={}

    def forward(self,sam,raw,valid):
        radial=self.base.encode(raw,valid) if self.variant in ('radial','radial_center') else torch.zeros_like(raw)
        h,w=sam.shape[-2:]
        center=sam[:,:,h//2:h//2+1,w//2:w//2+1].expand(-1,-1,h,w) if self.variant in ('center','radial_center') else torch.zeros_like(sam)
        features=torch.cat((sam,radial,center),1)
        z=self.base.head(features)
        self.diagnostics=dict(radial_norm=float(radial.detach().norm(dim=1).mean()),center_norm=float(center.detach().norm(dim=1).mean()),
                              feature_norm=float(features.detach().norm(dim=1).mean()),logit_min=float(z.detach().min()),logit_max=float(z.detach().max()))
        return z
