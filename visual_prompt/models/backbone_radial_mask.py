"""Channel-generic version of the retained center-SAM + learned-radial head."""
import torch
from torch import nn
from torch.nn import functional as F
from visual_prompt.models.pixelwise_sam_mask import PixelwiseSAMMask


class BackboneRadialMask(nn.Module):
    def __init__(self, channels, mean, std):
        super().__init__()
        self.channels = channels
        self.register_buffer('radial_mean', mean.detach().reshape(1,64,1,1).clone())
        self.register_buffer('radial_std', std.detach().reshape(1,64,1,1).clone())
        self.head = PixelwiseSAMMask(2*channels+64,128)
        self.mapping = nn.Parameter(torch.eye(64).reshape(64,64,1,1))

    def forward(self, sam, raw, valid):
        radial=F.normalize(((raw-self.radial_mean)/self.radial_std).masked_fill(~valid,0),dim=1,eps=1e-6)*.25
        radial=F.conv2d(radial,self.mapping)
        h,w=sam.shape[-2:]
        center=sam[:,:,h//2:h//2+1,w//2:w//2+1].expand_as(sam)
        return self.head(torch.cat((sam,radial,center),1))
