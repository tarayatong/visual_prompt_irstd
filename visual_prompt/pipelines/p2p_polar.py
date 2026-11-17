"""Inference for the matched P2P memory/polar-gradient experiment."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from visual_prompt.models.p2p_polar_mask import P2PPolarMask
from visual_prompt.pipelines.p2p_framework import P2PModel,load,local_sam,normalized_feature,mask_descriptor


class P2PPolarModel(P2PModel):
    def __init__(self,dataset,checkpoint,device='cuda:0'):
        ck=load(checkpoint);super().__init__(dataset,ck['base_variant'],device)
        protocol=Path(checkpoint).parent.parent/'protocol.json'
        if protocol.exists():
            expected=json.loads(protocol.read_text())['baseline_sha256']
            assert hashlib.sha256((self.folder/'model.pt').read_bytes()).hexdigest()==expected,'Retained upstream model changed'
        stats=ck['gradient_statistics']
        self.head=P2PPolarMask(self.e0.cpu(),ck['mode'],stats['mean'],stats['std']).to(self.device).eval()
        self.head.load_state_dict(ck['model'])

    @torch.no_grad()
    def local(self,image,y,x):
        sam=local_sam(self.pred(),image,y,x)[0].cpu().half().numpy();ray,valid=self.radial.sample(image,y,x)
        data=dict(sam=sam,ray=ray.numpy(),ray_valid=valid.numpy())
        base=normalized_feature(data,self.ck['radial_mean'].numpy(),self.ck['radial_std'].numpy())
        cpu=torch.from_numpy(base.astype(np.float32))[None]
        descriptor=mask_descriptor(cpu,cpu[:,:352],self.e0.cpu()).to(self.device)
        z=self.head(cpu.to(self.device),descriptor,self.e0,ray[None].to(self.device,dtype=torch.float32),valid[None].to(self.device))
        return (z[0]>0).cpu().numpy().astype(np.uint8)
