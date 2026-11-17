"""Learned candidate-position-guided attention reference for pixelwise Mask."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.pixelwise_sam_mask import PixelwiseSAMMask


class CandidateReferenceMask(nn.Module):
    def __init__(self,channels=352,hidden=128,reference_channels=64):
        super().__init__()
        self.channels=channels;self.reference_channels=reference_channels
        # Fresh common SAM initialization, not a trained checkpoint. The new
        # context columns start at zero, so initial predictions match SAM-only.
        initial_rng=torch.get_rng_state()
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(initial_rng)
            control=PixelwiseSAMMask(channels,hidden)
        self.net=nn.Sequential(nn.Conv2d(channels+reference_channels,hidden,1),nn.GELU(),nn.Conv2d(hidden,channels+1,1))
        with torch.no_grad():
            self.net[0].weight[:,:channels].copy_(control.net[0].weight)
            self.net[0].weight[:,channels:].zero_()
            self.net[0].bias.copy_(control.net[0].bias)
            self.net[2].load_state_dict(control.net[2].state_dict())
        self.project=nn.Conv2d(channels,reference_channels,1)
        self.key=nn.Conv2d(reference_channels,reference_channels,1)
        self.position=nn.Conv2d(2,reference_channels,1,bias=False)
        self.query=nn.Parameter(torch.randn(reference_channels)/math.sqrt(reference_channels))
        self.capture_diagnostics=False;self.diagnostics={}

    def reference(self,features):
        _,_,height,width=features.shape
        yy,xx=torch.meshgrid(torch.arange(height,device=features.device,dtype=features.dtype),
            torch.arange(width,device=features.device,dtype=features.dtype),indexing='ij')
        coords=torch.stack(((xx-width//2)/max(width//2,1),(yy-height//2)/max(height//2,1)),0)[None]
        h=self.project(features)
        score=(torch.tanh(self.key(h)+self.position(coords))*self.query[None,:,None,None]).sum(1,keepdim=True)
        alpha=score.flatten(2).softmax(-1).reshape_as(score)
        q=(h*alpha).sum((-2,-1),keepdim=True)
        return q,alpha

    def forward(self,features):
        if features.ndim!=4 or features.shape[1]!=self.channels:raise ValueError('Unexpected SAM feature shape')
        q,alpha=self.reference(features)
        out=self.net(torch.cat((features,q.expand(-1,-1,*features.shape[-2:])),1))
        vector=F.normalize(out[:,:self.channels],dim=1,eps=1e-6)
        logits=(features*vector).sum(1)/.1+out[:,self.channels]
        if self.capture_diagnostics:
            a=alpha.detach().flatten(1)
            self.diagnostics=dict(attention_entropy_mean=float((-(a*a.clamp_min(1e-12).log()).sum(1)).mean()),
                attention_max_mean=float(a.max(1).values.mean()),reference_norm_mean=float(q.detach().flatten(1).norm(dim=1).mean()),
                attention_sum_max_error=float((a.sum(1)-1).abs().max()))
        return logits
