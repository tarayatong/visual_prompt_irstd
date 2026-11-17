"""Small spatial objectness heads for aligned SAM ROIs and raw image crops."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_channels: int, out_channels: int, kernel_size: int = 3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False),
        nn.GroupNorm(4, out_channels), nn.GELU(),
    )


class SpatialObjectness(nn.Module):
    """Adapt input branches, fuse spatially, and classify the query point.

    ``layers`` contain aligned 32x32 full-image SAM ROIs with 32/64/256
    channels. ``raw`` is the original 64x64 gray crop in [0, 1]. No local
    SAM re-encoding is used. Standalone modes do not consume similarity or
    hand statistics. Optional ``f4_dim`` appends the entire original F4 vector
    ahead of the spatial descriptor. Candidates remain support-sim Top5.
    """
    def __init__(self, mode: str, f4_dim: int = 0):
        super().__init__()
        if mode not in ('sam', 'raw', 'fusion'):
            raise ValueError(mode)
        self.mode = mode
        self.f4_dim = f4_dim
        self.sam_adapters = nn.ModuleList(
            [conv_block(c, 16, 1) for c in (32, 64, 256)] if mode!='raw' else []
        )
        self.raw_adapter = nn.Sequential(conv_block(1,16), conv_block(16,16)) if mode!='sam' else None
        channels = (48 if mode!='raw' else 0) + (16 if mode!='sam' else 0)
        self.fusion = nn.Sequential(conv_block(channels+1,32), conv_block(32,32))
        self.mlp = nn.Sequential(nn.Linear(96+f4_dim,128), nn.GELU(), nn.Linear(128,1))
        # Physical extent [-32, +31] aligns image pixels and every SAM layer.
        offsets = torch.linspace(-32,31,32)
        yy,xx = torch.meshgrid(offsets,offsets,indexing='ij')
        self.register_buffer('point_map',torch.exp(-(xx.square()+yy.square())/(2*2.0**2))[None,None])
        center = torch.tensor([[[[1/63,1/63]]]],dtype=torch.float32)
        neighbors = torch.tensor([-2.,0.,2.])
        ny,nx = torch.meshgrid(neighbors,neighbors,indexing='ij')
        local_grid = torch.stack((2*(nx+32)/63-1,2*(ny+32)/63-1),dim=-1)[None]
        self.register_buffer('center_grid',center)
        self.register_buffer('neighbor_grid',local_grid)

    def spatial_descriptor(self, layers: list[torch.Tensor], raw: torch.Tensor):
        parts = []
        if self.mode!='raw':
            if len(layers)!=3:
                raise ValueError('Expected three aligned SAM feature layers')
            parts.extend(adapter(layer) for adapter,layer in zip(self.sam_adapters,layers))
        if self.raw_adapter is not None:
            image = self.raw_adapter((raw-.5)/.25)
            parts.append(F.interpolate(image,size=(32,32),mode='bilinear',align_corners=True))
        n = parts[0].shape[0]
        feature = self.fusion(torch.cat(parts+[self.point_map.expand(n,-1,-1,-1)],dim=1))
        center = F.grid_sample(feature,self.center_grid.expand(n,-1,-1,-1),align_corners=True).flatten(1)
        local = F.grid_sample(feature,self.neighbor_grid.expand(n,-1,-1,-1),align_corners=True).mean((-2,-1))
        whole = feature.mean((-2,-1))
        return torch.cat((center,local,whole),dim=1)

    def forward(self, layers: list[torch.Tensor], raw: torch.Tensor, f4: torch.Tensor | None = None):
        spatial = self.spatial_descriptor(layers,raw)
        if self.f4_dim:
            if f4 is None or f4.shape != (spatial.shape[0],self.f4_dim):
                raise ValueError(f'Expected F4 input of shape (N, {self.f4_dim})')
            spatial = torch.cat((f4,spatial),dim=1)
        return self.mlp(spatial).squeeze(1)

    def parameter_counts(self):
        count=lambda module:sum(p.numel() for p in module.parameters() if p.requires_grad)
        sam=count(self.sam_adapters)
        raw=count(self.raw_adapter) if self.raw_adapter is not None else 0
        return dict(sam_adapters=sam,raw_adapter=raw,fusion_cnn=count(self.fusion),
                    objectness_mlp=count(self.mlp),total=count(self))
