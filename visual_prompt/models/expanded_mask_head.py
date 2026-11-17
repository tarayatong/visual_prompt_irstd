"""Capacity controls on a frozen, full 352-channel SAM feature encoder."""
import torch
from torch import nn
from visual_prompt.models.backbone_radial_mask import BackboneRadialMask
from visual_prompt.models.pixelwise_sam_mask import PixelwiseSAMMask


class ExpandedMaskHead(BackboneRadialMask):
    def __init__(self,initial,kind):
        super().__init__(352,initial['radial_mean'],initial['radial_std'])
        self.kind=kind
        if kind=='wide512':
            self.head=PixelwiseSAMMask(768,512)
            state=self.state_dict()
            for key,value in initial.items():
                if key=='head.net.0.weight':state[key][:128].copy_(value)
                elif key=='head.net.0.bias':state[key][:128].copy_(value)
                elif key=='head.net.2.weight':
                    state[key].zero_();state[key][:,:128].copy_(value)
                else:state[key].copy_(value)
            self.load_state_dict(state)
        else:
            self.load_state_dict(initial)
            if kind=='spatial512':
                self.adapter=nn.Sequential(nn.Conv2d(352,512,1),nn.GELU(),
                    nn.Conv2d(512,512,3,padding=1,groups=512),nn.GELU(),nn.Conv2d(512,352,1))
                nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
            elif kind!='control':raise ValueError(kind)

    def forward(self,sam,raw,valid):
        if self.kind=='spatial512':sam=sam+self.adapter(sam)
        return super().forward(sam,raw,valid)
