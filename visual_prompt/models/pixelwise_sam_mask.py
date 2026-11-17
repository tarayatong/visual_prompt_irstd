"""SAM-only control for pixelwise radial-feature studies on a frozen memory base."""
import torch
from torch import nn
from torch.nn import functional as F


class PixelwiseSAMMask(nn.Module):
    def __init__(self,channels=352,hidden=128):
        super().__init__()
        self.channels=channels
        self.net=nn.Sequential(nn.Conv2d(channels,hidden,1),nn.GELU(),nn.Conv2d(hidden,channels+1,1))

    def forward(self,features):
        if features.ndim!=4 or features.shape[1]!=self.channels:
            raise ValueError(f'Expected B,{self.channels},H,W SAM feature maps')
        out=self.net(features)
        vector=F.normalize(out[:,:self.channels],dim=1,eps=1e-6)
        return (features*vector).sum(1)/.1+out[:,self.channels]
