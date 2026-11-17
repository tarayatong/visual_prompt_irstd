"""Differentiable local SAM2 features and exact shallow-prefix early exits."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class LocalMaskEncoder(nn.Module):
    def __init__(self,model,stages=3,clone=False,stage3_blocks=None):
        super().__init__();self.stages=stages;self.prefix=stages<3 or stage3_blocks is not None
        if not self.prefix:
            self.encoder=copy.deepcopy(model.image_encoder) if clone else model.image_encoder
            self.no_mem_embed=nn.Parameter(model.no_mem_embed.detach().clone()) if clone else model.no_mem_embed
        else:
            trunk=model.image_encoder.trunk
            self.patch_embed=copy.deepcopy(trunk.patch_embed) if clone else trunk.patch_embed
            self.pos_embed=nn.Parameter(trunk.pos_embed.detach().clone()) if clone else trunk.pos_embed
            self.pos_embed_window=nn.Parameter(trunk.pos_embed_window.detach().clone()) if clone else trunk.pos_embed_window
            self.stage_ends=trunk.stage_ends[:stages]
            if stage3_blocks is not None:
                assert stages==3 and 1<=stage3_blocks<=trunk.stage_ends[2]-trunk.stage_ends[1]
                self.stage_ends=list(trunk.stage_ends[:2])+[trunk.stage_ends[1]+stage3_blocks]
                self.no_mem_embed=nn.Parameter(model.no_mem_embed.detach().clone()) if clone else model.no_mem_embed
            blocks=list(trunk.blocks)[:self.stage_ends[-1]+1]
            self.blocks=nn.ModuleList(copy.deepcopy(blocks) if clone else blocks)
            neck=model.image_encoder.neck
            assert all(i not in neck.fpn_top_down_levels for i in range(min(stages,2))), 'Shallow FPN must be lateral-only'
            lateral=[neck.convs[-1-i] for i in range(stages)]
            self.lateral=nn.ModuleList(copy.deepcopy(lateral) if clone else lateral)
        self.conv0=copy.deepcopy(model.sam_mask_decoder.conv_s0) if clone else model.sam_mask_decoder.conv_s0
        if stages>=2:self.conv1=copy.deepcopy(model.sam_mask_decoder.conv_s1) if clone else model.sam_mask_decoder.conv_s1

    def forward(self,x):
        if not self.prefix:
            features=self.encoder(x)['backbone_fpn']
            return [self.conv0(features[0]),self.conv1(features[1]),features[2]+self.no_mem_embed.reshape(1,-1,1,1)]
        x=self.patch_embed(x)
        pos=F.interpolate(self.pos_embed,size=x.shape[1:3],mode='bicubic')
        pos=pos+self.pos_embed_window.tile([a//b for a,b in zip(pos.shape,self.pos_embed_window.shape)])
        x=x+pos.permute(0,2,3,1);outputs=[]
        for i,block in enumerate(self.blocks):
            x=block(x)
            if i in self.stage_ends:
                j=len(outputs);value=self.lateral[j](x.permute(0,3,1,2))
                if j==0:value=self.conv0(value)
                elif j==1:value=self.conv1(value)
                else:value=value+self.no_mem_embed.reshape(1,-1,1,1)
                outputs.append(value)
        return outputs
