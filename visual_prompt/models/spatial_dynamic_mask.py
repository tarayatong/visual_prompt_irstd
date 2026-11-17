"""Identity-initialized spatial mixing before the existing dynamic-vector readout."""
import torch
from torch import nn
from torch.nn import functional as F


class SpatialDynamicHead(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.channels=original.channels
        self.net=original.net
        hidden=self.net[0].out_channels
        self.spatial=nn.Conv2d(hidden,hidden,3,padding=1,bias=True)
        nn.init.dirac_(self.spatial.weight)
        nn.init.zeros_(self.spatial.bias)

    def forward(self,features):
        hidden=self.net[1](self.net[0](features))
        out=self.net[2](self.spatial(hidden))
        vector=F.normalize(out[:,:self.channels],dim=1,eps=1e-6)
        return (features*vector).sum(1)/.1+out[:,self.channels]
