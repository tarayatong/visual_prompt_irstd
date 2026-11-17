"""Identity-start learned radial kernels with the original 64D feature contract."""
import torch
from torch import nn
from torch.nn import functional as F


class LearnableRadialFeatures(nn.Module):
    def __init__(self,mean,std,mode='kernel'):
        super().__init__()
        if mode not in ('fixed','gain','kernel','kernel_free'):raise ValueError(mode)
        self.mode=mode
        self.register_buffer('mean',mean.detach().float().reshape(1,64,1,1).clone())
        self.register_buffer('std',std.detach().float().reshape(1,64,1,1).clone())
        if mode in ('kernel','kernel_free'):self.delta=nn.Parameter(torch.zeros(64,64,1,1))
        else:self.register_parameter('delta',None)
        if mode=='kernel_free':self.center_delta=nn.Parameter(torch.zeros(1,64,1,1))
        else:self.register_parameter('center_delta',None)
        if mode!='fixed':self.log_gain=nn.Parameter(torch.zeros(1,64,1,1))
        else:self.register_parameter('log_gain',None)

    def forward(self,contrast,valid,center=None):
        raw=contrast.masked_fill(~valid,0)
        if self.delta is not None:raw=raw+F.conv2d(raw,self.delta)
        if self.center_delta is not None:
            if center is None:raise ValueError('Unrestricted kernels require the center intensity')
            raw=raw+self.center_delta*center
        normalized=((raw-self.mean)/self.std).masked_fill(~valid,0)
        output=F.normalize(normalized,dim=1,eps=1e-6)*.25
        if self.log_gain is not None:output=output*self.log_gain.clamp(-2,2).exp()
        return output

    def regularization(self):
        value=self.mean.sum()*0
        if self.delta is not None:value=value+self.delta.square().mean()
        if self.log_gain is not None:value=value+self.log_gain.square().mean()
        if self.center_delta is not None:value=value+self.center_delta.square().mean()
        return value

    def effective_sampling_kernel(self):
        # contrast_j = center - radial_sample_j. Thus arbitrary learned mixing
        # is a 65-tap radial kernel with sum zero; signed weights allow edges.
        mixing=torch.eye(64,device=self.mean.device)
        if self.delta is not None:mixing=mixing+self.delta[:,:,0,0]
        center=mixing.sum(1,keepdim=True)
        if self.center_delta is not None:center=center+self.center_delta.reshape(64,1)
        return torch.cat((center,-mixing),1)
