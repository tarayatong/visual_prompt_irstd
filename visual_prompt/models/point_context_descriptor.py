"""Frozen point-centered, layer-preserving neighborhood descriptors."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def context_offsets(device,dtype):
    core=torch.tensor([(x,y) for y in (-1.,0.,1.) for x in (-1.,0.,1.)],device=device,dtype=dtype)
    theta=torch.arange(8,device=device,dtype=dtype)*(math.pi/4)
    rings=[torch.stack((r*theta.cos(),r*theta.sin()),1) for r in (4.,8.,16.)]
    return torch.cat((torch.zeros(1,2,device=device,dtype=dtype),core,*rings),0)


def point_context_descriptor(layers,points_xy,image_size,source_points_xy=None,source_size=512):
    """Sample exact center/core/rings in original-pixel units, no spatial CNN.

    Input maps have batch 1 (shared image) or N (separately encoded crops).
    Each layer is normalized at each sampled position before center minus ring.
    Output order: for each layer, center, core-minus-ring4/8/16.
    Missing source pixels are excluded from ring means, never semantic inputs.
    """
    if len(layers)!=3 or [x.shape[1] for x in layers]!=[32,64,256]:
        raise ValueError('Expected native SAM layers with 32/64/256 channels')
    n=len(points_xy);offsets=context_offsets(points_xy.device,points_xy.dtype)
    xy=points_xy[:,None]+offsets[None]
    source=points_xy if source_points_xy is None else source_points_xy
    absolute=source[:,None]+offsets[None]
    good=((absolute>=0)&(absolute<=source_size-1)).all(-1)
    grid=(2*xy/(image_size-1)-1)[:,None]
    blocks=[]
    for layer in layers:
        if layer.shape[0] not in (1,n):raise ValueError('Feature batch must be 1 or N')
        sampled=F.grid_sample(layer.expand(n,-1,-1,-1).float(),grid,align_corners=True,padding_mode='zeros')[:,:,0]
        sampled=F.normalize(sampled,dim=1,eps=1e-6)
        def average(start,end):
            weight=good[:,start:end].to(sampled.dtype)
            return (sampled[:,:,start:end]*weight[:,None]).sum(-1)/weight.sum(-1).clamp_min(1)[:,None]
        center=sampled[:,:,0];core=average(1,10)
        blocks.append(torch.cat((center,core-average(10,18),core-average(18,26),core-average(26,34)),1))
    return torch.cat(blocks,1)
