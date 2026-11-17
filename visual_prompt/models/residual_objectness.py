"""A bounded CNN correction around an unchanged F4 objectness classifier."""
import torch
from torch import nn

from visual_prompt.models.lightweight_heads import ObjectnessMLP
from visual_prompt.models.spatial_objectness import SpatialObjectness


class ResidualObjectness(nn.Module):
    def __init__(self,base_state: dict,bound: float=2.):
        super().__init__()
        self.base=ObjectnessMLP(1440,128)
        self.base.load_state_dict(base_state)
        self.base.requires_grad_(False)
        self.base.eval()
        self.branch=SpatialObjectness('fusion')
        nn.init.zeros_(self.branch.mlp[-1].weight)
        nn.init.zeros_(self.branch.mlp[-1].bias)
        self.bound=bound

    def train(self,mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self,layers,raw,f4,return_delta=False):
        with torch.no_grad():base=self.base(f4)
        correction=self.branch(layers,raw)
        delta=self.bound*torch.tanh(correction/self.bound)
        logits=base+delta
        return (logits,delta) if return_delta else logits

    def parameter_counts(self):
        return dict(trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    frozen_F4=sum(p.numel() for p in self.base.parameters()),
                    total=sum(p.numel() for p in self.parameters()))
