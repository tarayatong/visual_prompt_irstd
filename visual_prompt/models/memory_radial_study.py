"""Eight pixelwise radial designs on one frozen SAM/memory/Obj base."""
import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.pixelwise_sam_mask import PixelwiseSAMMask
from visual_prompt.models.polar_pixel_encoder import polar_derivatives


VARIANTS = {
    'R1': dict(name='fixed_contrast64',channels=64),
    'R2': dict(name='contrast64_gain',channels=64),
    'R3': dict(name='direction_linear8',channels=8),
    'R4': dict(name='free_linear8',channels=8),
    'R5': dict(name='free_linear64',channels=64),
    'R6': dict(name='axial_C64',channels=64,fields=1),
    'R7': dict(name='axial_C_Gr_Gtheta64',channels=64,fields=3),
    'R8': dict(name='axial_all5_64',channels=64,fields=5),
}


class MemoryRadialMask(nn.Module):
    def __init__(self,variant,mean,std):
        super().__init__()
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant;self.spec=VARIANTS[variant]
        self.register_buffer('feature_mean',mean.detach().reshape(1,5,8,8,1,1).clone())
        self.register_buffer('feature_std',std.detach().reshape(1,5,8,8,1,1).clone())
        # Construct the head first: all variants with the same width have
        # exactly the same initial head at a given seed, regardless of encoder.
        self.head=PixelwiseSAMMask(352+self.spec['channels'],128)
        self.project=None;self.radial=None;self.angular=None;self.fuse=None
        self.register_parameter('log_gain',None)
        if variant in ('R2','R6','R7','R8'):
            self.log_gain=nn.Parameter(torch.zeros(1,64,1,1))
        if variant in ('R3','R4','R5'):
            self.project=nn.Conv2d(64,self.spec['channels'],1,bias=False,groups=8 if variant=='R3' else 1)
        if variant in ('R6','R7','R8'):
            self.radial=nn.Conv2d(self.spec['fields'],4,(1,3))
            self.angular=nn.Conv2d(self.spec['fields'],4,(3,1))
            self.fuse=nn.Conv2d(8,1,1)  # Ordinary nonzero initialization, no bypass.
        self.capture_diagnostics=False
        self.diagnostics={}

    def encode(self,raw,valid):
        if self.radial is None:
            mean=self.feature_mean[:,0].reshape(1,64,1,1)
            std=self.feature_std[:,0].reshape(1,64,1,1)
            y=F.normalize(((raw-mean)/std).masked_fill(~valid,0),dim=1,eps=1e-6)*.25
            if self.project is not None:y=self.project(y)
        else:
            fields,masks=polar_derivatives(raw,valid)
            count=self.spec['fields']
            q=((fields[:,:count]-self.feature_mean[:,:count])/self.feature_std[:,:count]).masked_fill(~masks[:,:count],0)
            b,_,_,_,h,w=q.shape
            q=F.normalize(q.reshape(b,count,64,h,w),dim=2,eps=1e-6)*.25
            q=q.reshape(b,count,8,8,h,w).permute(0,4,5,1,2,3).reshape(b*h*w,count,8,8)
            radial=F.gelu(self.radial(F.pad(q,(1,1,0,0),mode='reflect')))
            angular=F.gelu(self.angular(F.pad(q,(0,0,1,1),mode='circular')))
            y=self.fuse(torch.cat((radial,angular),1)).reshape(b,h,w,64).permute(0,3,1,2).masked_fill(~valid,0)
        if self.log_gain is not None:y=y*self.log_gain.clamp(-2,2).exp()
        return y

    def forward(self,sam,raw,valid):
        radial=self.encode(raw,valid)
        logits=self.head(torch.cat((sam,radial),1))
        if self.capture_diagnostics:
            self.diagnostics=dict(SAM_norm_mean=float(sam.detach().norm(dim=1).mean()),radial_norm_mean=float(radial.detach().norm(dim=1).mean()),
                radial_norm_max=float(radial.detach().norm(dim=1).max()),logit_min=float(logits.detach().min()),
                logit_max=float(logits.detach().max()),logit_mean=float(logits.detach().mean()))
        return logits
