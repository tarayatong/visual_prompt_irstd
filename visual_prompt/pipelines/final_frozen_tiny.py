"""Inference for the selected warmup-only Tiny result, without study artifacts."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from visual_prompt.backbones.sam2_adapter import build_sam2_bplus,Sam2SimilarityPredictor
from visual_prompt.core.candidate_proposals import greedy_score_peaks
from visual_prompt.models.full_image_objectness import describe_full_image_points,DESCRIPTOR_NAMES
from visual_prompt.models.lightweight_heads import ObjectnessMLP
from visual_prompt.models.backbone_radial_mask import BackboneRadialMask
from visual_prompt.pipelines.p2p_framework import RadialSampler,paste_local
from visual_prompt.evaluation import eval_sliding_window_semantic_mask as ev
from visual_prompt.evaluation.spatial_alignment import restore_to_original


class FinalFrozenTiny:
    def __init__(self,bundle,dataset,device='cuda:0'):
        self.root=Path(bundle);self.folder=self.root/dataset;self.device=torch.device(device)
        self.config=json.loads((self.root/'config.json').read_text(encoding='utf-8'))
        self.manifest=json.loads((self.folder/'manifest.json').read_text(encoding='utf-8'))
        load=lambda path:torch.load(path,map_location='cpu',weights_only=False)
        model=build_sam2_bplus(str(self.root/self.config['backbone']),self.config['sam_config'],device,512)
        self.predictor=Sam2SimilarityPredictor(model);self.radial=RadialSampler(self.device)
        coarse=load(self.folder/'coarse.pt');self.keys=coarse['keys'].to(self.device)
        self.coarse=ObjectnessMLP(352+len(self.keys)).to(self.device).eval();self.coarse.load_state_dict(coarse['model'])
        self.coarse_mean=coarse['mean'].to(self.device);self.coarse_std=coarse['std'].to(self.device)
        obj=load(self.folder/'obj.pt');self.obj=ObjectnessMLP(1440).to(self.device).eval();self.obj.load_state_dict(obj['model']);self.stats=obj['statistics']
        saved=load(self.folder/'mask.pt');state=saved[self.config['mask_state_keys'][dataset]]
        self.mask=BackboneRadialMask(352,state['radial_mean'],state['radial_std']).to(self.device).eval();self.mask.load_state_dict(state)
        for module in [self.coarse,self.obj,self.mask]:module.requires_grad_(False)

    @torch.no_grad()
    def predict_working(self,image):
        image=np.asarray(Image.fromarray(np.asarray(image,dtype=np.uint8)).convert('RGB').resize((512,512),Image.Resampling.BILINEAR)).copy()
        p=self.predictor;p.set_image(image)
        axis=torch.arange(512,device=self.device,dtype=torch.float32)*2/511-1
        yy,xx=torch.meshgrid(axis,axis,indexing='ij');grid=torch.stack((xx,yy),-1)[None]
        native=p.get_named_sim_features(False)
        layers=[F.grid_sample(f.float(),grid,align_corners=True) for _,f in native]
        raw=F.normalize(torch.cat(layers,1),dim=1)[0].flatten(1).T.contiguous();del layers
        scores=[]
        for v in raw.split(16384):
            features=torch.cat((v,v@self.keys.T),1)
            scores.append(self.coarse((features-self.coarse_mean)/self.coarse_std).sigmoid())
        heat=torch.cat(scores).reshape(512,512).cpu().numpy();del raw,scores
        points=[(y,x) for y,x,_ in greedy_score_peaks(heat,5,7)]
        desc=describe_full_image_points(p,image,heat,points,64,self.config['local_contrast_sigma'],32,encode_image=False)
        raw=np.concatenate([desc[k] for k in DESCRIPTOR_NAMES],1)
        centers=F.normalize(torch.from_numpy(raw[:,6:358]),dim=1).numpy()
        means=F.normalize(torch.from_numpy(raw[:,358:710]),dim=1).numpy()
        t,b=self.stats['prototype_target'],self.stats['prototype_background']
        tc,tm,bc,bm=centers@t,means@t,centers@b,means@b
        full=np.concatenate((raw,np.stack((tc,tm,bc,bm,tc-bc,tm-bm),1)),1)
        x=((full-self.stats['mean'])/self.stats['std']).astype(np.float32)
        prob=self.obj(torch.from_numpy(x).to(self.device)).sigmoid().cpu().numpy();keep=prob>=.1
        canvas=np.zeros((512,512),np.uint8);blank=np.zeros_like(canvas)
        for (y,x),ok in zip(points,keep):
            if not ok:continue
            crop,_=ev.crop_with_padding(image,blank,y-32,x-32,64);p.set_image(crop)
            maps=[F.interpolate(f.half().float(),size=(64,64),mode='bilinear',align_corners=False) for _,f in p.get_named_sim_features(False)]
            sam=F.normalize(torch.cat(maps,1),dim=1).half().float();ray,valid=self.radial.sample(image,y,x)
            mask=(self.mask(sam,ray[None].to(self.device).float(),valid[None].to(self.device))>=0)[0].cpu().numpy().astype(np.uint8)
            paste_local(canvas,mask,y,x)
        return canvas,dict(points=[list(pt) for pt in points],probabilities=prob.tolist(),keep=keep.tolist())

    @torch.no_grad()
    def predict(self,image):
        shape=np.asarray(image).shape[:2];canvas,info=self.predict_working(image)
        return restore_to_original(canvas,shape,discrete=True),info
