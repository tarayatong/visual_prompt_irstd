"""Axial polar residuals added to the retained P2P Mask feature path."""
import torch
from torch import nn
from visual_prompt.models.polar_gradient_mask import PolarGradientEncoder
from visual_prompt.models.radial_augmented_heads import RadialMaskVectorMLP
from visual_prompt.pipelines.p2p_framework import mask_forward


class P2PPolarMask(RadialMaskVectorMLP):
    def __init__(self,e0,mode,mean=None,std=None):
        super().__init__(352,64,e0)
        if mode not in ('fixed','gain','polar'):raise ValueError(mode)
        self.mode=mode
        self.log_gain=nn.Parameter(torch.zeros(1,64,1,1),requires_grad=mode!='fixed')
        self.encoder=PolarGradientEncoder(mean,std,4) if mode=='polar' else None

    def forward(self,base,descriptor,e0,raw=None,valid=None):
        if self.mode=='fixed':return mask_forward(self,base,descriptor,e0)
        original=base[:,352:]
        radial=original*self.log_gain.clamp(-2,2).exp()
        if self.encoder is not None:
            _,delta=self.encoder(raw,valid);radial=radial+delta
        delta=radial-original
        # Preserve the existing fp16->fp32 feature path and CPU descriptor.
        # A zero residual starts at exactly the fixed64 function.
        updated=descriptor.clone()
        updated[:,352:416]=updated[:,352:416]+delta[:,:,32,32]
        updated[:,768:832]=updated[:,768:832]+delta.mean((-2,-1))
        features=torch.cat((base[:,:352],radial),1)
        return mask_forward(self,features,updated,e0)
