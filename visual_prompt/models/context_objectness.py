"""Function-preserving expansion of the original fully trainable F4 classifier."""
import torch
from torch import nn
from visual_prompt.models.lightweight_heads import ObjectnessMLP


class ContextObjectness(nn.Module):
    def __init__(self,source_state,extra_channels):
        super().__init__();self.extra_channels=extra_channels
        self.main=ObjectnessMLP(1440+extra_channels,128)
        with torch.no_grad():
            self.main.net[0].weight.zero_()
            self.main.net[0].weight[:,:1440].copy_(source_state['net.0.weight'])
            self.main.net[0].bias.copy_(source_state['net.0.bias'])
            self.main.net[2].weight.copy_(source_state['net.2.weight'])
            self.main.net[2].bias.copy_(source_state['net.2.bias'])

    def forward(self,f4,extra):
        if extra.shape!=(len(f4),self.extra_channels):raise ValueError('Extra descriptor dimension mismatch')
        return self.main(torch.cat((f4,extra),1))
