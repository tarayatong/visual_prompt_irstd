"""Whole-ROI objectness with learned spatial processing and average/max pooling."""
import torch
from torch import nn


class RegionalObjectness(nn.Module):
    def __init__(self,channels=352,width=64,hidden=32):
        super().__init__()
        self.spatial=nn.Sequential(nn.Conv2d(channels,width,1),nn.GELU(),
            nn.Conv2d(width,width,3,padding=1,groups=width),nn.GELU())
        self.classifier=nn.Sequential(nn.Linear(2*width,hidden),nn.GELU(),nn.Linear(hidden,1))

    def forward(self,features):
        spatial=self.spatial(features)
        pooled=torch.cat([spatial.mean((-2,-1)),spatial.amax((-2,-1))],1)
        return self.classifier(pooled).squeeze(-1)
