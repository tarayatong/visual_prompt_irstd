"""Frozen local-SAM utilities shared by retained radial experiments.

This module contains no coordinate correction or experiment training entry.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.run_joint_routing_study import context
from scripts.run_radial_autonomous_queue import ROOT
from scripts.run_objectness_positive_jitter import predictor_for
from visual_prompt.core.candidate_proposals import connected_instances
from visual_prompt.models.radial_profile import DirectionalRadialOperator, RadialProfileConfig

class DifferentiableLocalSAM(nn.Module):
    def __init__(self,model):
        super().__init__();self.sam=model
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        a=torch.arange(-32,32,dtype=torch.float32)
        yy,xx=torch.meshgrid(a,a,indexing='ij')
        self.register_buffer('offsets',torch.stack((xx,yy),-1))

    def forward(self,image,centers):
        pixels=centers[:,None,None,:]+self.offsets[None]
        grid=(pixels+.5)/512*2-1
        crop=F.grid_sample(image.expand(len(centers),-1,-1,-1),grid,align_corners=False,padding_mode='zeros')
        # Original torchvision predictor resizes and normalizes on CPU. Tensor
        # transfers retain autograd and avoid a new CPU/GPU preprocessing confound.
        resized=F.interpolate(crop.cpu(),size=(self.sam.image_size,self.sam.image_size),mode='bilinear',align_corners=False,antialias=True)
        normalized=((resized-self.mean.cpu())/self.std.cpu()).to(image.device)
        backbone=self.sam.forward_image(normalized)
        _,features,_,sizes=self.sam._prepare_backbone_features(backbone)
        if self.sam.directly_add_no_mem_embed:features[-1]=features[-1]+self.sam.no_mem_embed
        features=[f.permute(1,2,0).reshape(len(centers),-1,*size) for f,size in zip(features,sizes)]
        local=F.normalize(torch.cat([F.interpolate(f,size=(64,64),mode='bilinear',align_corners=False) for f in features],1),dim=1)
        return local,crop


class FrozenLocalRuntime:
    def __init__(self):
            self.ids,self.b,self.e0,self.selected,self.records,self.bank,self.f4=context()
            self.predictor=predictor_for(self.b,self.e0.device)
            self.encoder=DifferentiableLocalSAM(self.predictor.model).to(self.e0.device)
            self.radial=DirectionalRadialOperator(1,RadialProfileConfig(fusion='concat',components=('contrast',),
                radii=(1.,2.,3.,4.,6.,8.,12.,16.),opposite_pairs=False)).to(self.e0.device)
            self.images={}

    def image(self,sid):
            if sid not in self.images:
                # Bank access is limited to training IDs until the explicit test phase.
                image,gt,_=self.bank.image(sid)
                rgb=(torch.from_numpy(image.copy()).permute(2,0,1)[None].float()/255).to(self.e0.device)
                gray=torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(np.float32)/255).to(self.e0.device)[None,None]
                path=ROOT/'phase_c_objectness/artifacts/candidate_features'/f'{sid}_score.npy'
                if not path.exists():path=ROOT/'phase_a_proposals/artifacts/scores'/f'{sid}.npy'
                score=torch.from_numpy(np.load(path).astype(np.float32)).to(self.e0.device)[None,None]
                self.images[sid]=dict(rgb=rgb,gray=gray,score=score,gt=gt,instances=connected_instances(gt))
            return self.images[sid]
