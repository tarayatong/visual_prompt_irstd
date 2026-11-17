"""Zero-initialized radial residual for native SAM feature interfaces."""
import torch
from torch import nn
from torch.nn import functional as F


class RadialSAMFusion(nn.Module):
    """Project existing64-channel contrast descriptors into a SAM feature map.

    The caller supplies spatially aligned raw radial contrast and validity maps,
    computed from the image without labels. Statistics must use training data
    only. This module changes neither memory construction nor decoder outputs
    directly; its result replaces the selected decoder input feature map.
    """

    def __init__(self, sam_channels, radial_mean, radial_std):
        super().__init__()
        mean=torch.as_tensor(radial_mean,dtype=torch.float32).detach().reshape(1,64,1,1).clone()
        std=torch.as_tensor(radial_std,dtype=torch.float32).detach().reshape(1,64,1,1).clone()
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std>0).all():
            raise ValueError('Require finite train-only radial statistics and positive std')
        self.register_buffer('radial_mean',mean)
        self.register_buffer('radial_std',std)
        self.projection=nn.Conv2d(64,sam_channels,1,bias=False)
        nn.init.zeros_(self.projection.weight)

    def forward(self, sam_features, radial_contrast, radial_valid):
        if sam_features.ndim!=4 or sam_features.shape[1]!=self.projection.out_channels:
            raise ValueError('Unexpected SAM feature shape')
        if radial_contrast.ndim!=4 or radial_contrast.shape[:2]!=(sam_features.shape[0],64):
            raise ValueError('Expected B,64,H,W radial contrast')
        if radial_valid.shape!=radial_contrast.shape or radial_valid.dtype!=torch.bool:
            raise ValueError('Expected a matching boolean radial validity map')
        if radial_contrast.device!=sam_features.device:
            raise ValueError('SAM and radial features must share a device')
        # Same normalized contrast representation as the existing radial head.
        radial=(radial_contrast.float()-self.radial_mean)/self.radial_std
        radial=F.normalize(radial.masked_fill(~radial_valid,0),dim=1,eps=1e-6)*.25
        if radial.shape[-2:]!=sam_features.shape[-2:]:
            radial=F.interpolate(radial,size=sam_features.shape[-2:],mode='bilinear',align_corners=False)
        delta=self.projection(radial.to(dtype=self.projection.weight.dtype))
        return sam_features+delta.to(dtype=sam_features.dtype)
