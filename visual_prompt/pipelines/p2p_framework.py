"""Retained support-memory retrieval and independently supervised Obj/Mask pipeline.

No imports from exploratory scripts or experiment-cache paths.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from visual_prompt.backbones.sam2_adapter import Sam2SimilarityPredictor,build_sam2_bplus
from visual_prompt.core.candidate_proposals import greedy_score_peaks
from visual_prompt.evaluation import eval_sliding_window_semantic_mask as ev
from visual_prompt.evaluation.spatial_alignment import restore_to_original
from visual_prompt.models.full_image_objectness import describe_full_image_points,DESCRIPTOR_NAMES
from visual_prompt.models.lightweight_heads import ObjectnessMLP
from visual_prompt.models.radial_augmented_heads import RadialMaskVectorMLP
from visual_prompt.models.radial_profile import DirectionalRadialOperator,RadialProfileConfig
from visual_prompt.models.semantic_memory import RecallMemoryReader
from visual_prompt.pipelines.semantic_memory_retrieval import stitched_memory_maps
from visual_prompt.pipelines.semantic_prompt_pipeline import get_named_sim_features,register_sim_feature_hooks

ROOT=Path(__file__).resolve().parents[2]
RETAINED=ROOT/'outputs/retained/p2p'


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def arguments():
    return argparse.Namespace(**json.loads((RETAINED/'sam_args.json').read_text(encoding='utf-8')))


def predictor_for(args):
    model=build_sam2_bplus(args.sam2_ckpt,args.sam2_config,args.device,args.encoder_input_size).eval()
    for p in model.parameters():p.requires_grad_(False)
    predictor=Sam2SimilarityPredictor(model,feature_source=args.feature_source)
    register_sim_feature_hooks(predictor,args.sim_feature_layers);return predictor


def crop_at(image,value,y,x,size=64):return ev.crop_with_padding(image,value,y-size//2,x-size//2,size)


@torch.no_grad()
def local_sam(predictor,image,y,x):
    crop,_=crop_at(image,np.zeros(image.shape[:2],np.uint8),y,x)
    predictor.set_image(crop)
    parts=[F.interpolate(f.float(),size=(64,64),mode='bilinear',align_corners=False) for _,f in get_named_sim_features(predictor,align=False)]
    return F.normalize(torch.cat(parts,1),dim=1)


class RadialSampler:
    def __init__(self,device):
        self.device=device;self.op=DirectionalRadialOperator(1,RadialProfileConfig(fusion='concat',radii=(1.,2.,3.,4.,6.,8.,12.,16.))).to(device).eval()
        yy,xx=torch.meshgrid(torch.arange(-32,32,device=device,dtype=torch.float32),torch.arange(-32,32,device=device,dtype=torch.float32),indexing='ij')
        self.offsets=torch.stack((xx,yy),-1).reshape(1,-1,2)

    @torch.no_grad()
    def sample(self,image,y,x):
        gray=torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(np.float32)/255).to(self.device)[None,None]
        p=self.op.sample_profiles(gray,self.offsets+torch.tensor([[[x,y]]],device=self.device))
        return p.values.flatten(2).transpose(1,2).reshape(256,64,64)[64:128].cpu().half(),p.valid.flatten(2).transpose(1,2).reshape(256,64,64)[64:128].cpu()


def normalized_feature(data,mean,std):
    ray=(data['ray'].astype(np.float32)-np.asarray(mean)[:,None,None])/np.asarray(std)[:,None,None]
    ray[~data['ray_valid']]=0;ray=ray/np.linalg.norm(ray,axis=0,keepdims=True).clip(1e-6)*.25
    return np.concatenate((data['sam'],ray.astype(np.float16)),0).astype(np.float16)


def normalize_f4(raw,stats):
    center=raw[:,6:358];center=center/np.maximum(np.linalg.norm(center,axis=1,keepdims=True),1e-8)
    mean=raw[:,358:710];mean=mean/np.maximum(np.linalg.norm(mean,axis=1,keepdims=True),1e-8)
    t,b=stats['prototype_target'],stats['prototype_background'];tc,tm,bc,bm=center@t,mean@t,center@b,mean@b
    full=np.concatenate((raw,np.stack((tc,tm,bc,bm,tc-bc,tm-bm),1)),1)
    return ((full-stats['mean'])/stats['std']).astype(np.float32)


def mask_descriptor(features,sam,e0):
    h,w=features.shape[-2:];sim=torch.einsum('bchw,c->bhw',sam,e0)
    stats=torch.stack((sim[:,h//2,w//2],sim.amax((-2,-1)),sim.mean((-2,-1)),sim.std((-2,-1))),1)
    return torch.cat((features[:,:,h//2,w//2],features.mean((-2,-1)),stats),1)


def mask_forward(model,features,descriptor,e0):
    output=model.net(descriptor);channels=features.shape[1]
    vector=F.normalize(F.pad(e0,(0,channels-len(e0)))[None]+.25*output[:,:channels],dim=1)
    return torch.einsum('bchw,bc->bhw',features,vector)/.1+output[:,-1,None,None]


def paste_local(canvas,mask,y,x):
    size=mask.shape[0];y0,x0=y-size//2,x-size//2;sy,sx=max(0,-y0),max(0,-x0)
    dy,dx=max(0,y0),max(0,x0);h,w=min(size-sy,canvas.shape[0]-dy),min(size-sx,canvas.shape[1]-dx)
    if h>0 and w>0:canvas[dy:dy+h,dx:dx+w]|=mask[sy:sy+h,sx:sx+w]


class P2PModel:
    def __init__(self,dataset,variant='memory',device='cuda:0',artifact_dir=None,load_mask=True):
        self.folder=Path(artifact_dir) if artifact_dir else RETAINED/dataset/variant
        self.data_folder=Path(artifact_dir) if artifact_dir else RETAINED/dataset
        self.legacy=variant=='legacy_memory_aligned' and artifact_dir is None
        if self.legacy:
            old=load(self.data_folder/'legacy_memory_aligned.pt')
            self.bundle=dict(heads=dict(obj=old['Obj'],mask=old['Mask'],statistics=old['original_statistics'],radial_mean=old['radial_mean'],radial_std=old['radial_std']),reader=dict(channels=old['reader_channels'],model=old['reader']),inference_K=5,peak_radius=7,obj_threshold=.1)
        else:self.bundle=load(self.folder/'model.pt')
        self.ck=self.bundle['heads'];self.args=arguments();self.args.device=device
        self.device=torch.device(device);self.predictor=None;self.radial=RadialSampler(self.device)
        self.e0=load(self.data_folder/'mask_e0.pt').float().flatten().to(self.device)
        self.memory=load(self.data_folder/'memory.pt');self.reader=None
        if self.legacy:
            self.e0=F.normalize(old['e0'].float().flatten(),dim=0).to(self.device);self.memory=old['memory']
        if self.bundle['reader']:
            reader=self.bundle['reader'];self.reader=RecallMemoryReader(reader['channels']).to(self.device).eval();self.reader.load_state_dict(reader['model'])
        self.obj=ObjectnessMLP(1440).to(self.device).eval();self.obj.load_state_dict(self.ck['obj'])
        self.mask=None
        if load_mask:
            self.mask=RadialMaskVectorMLP(352,64,self.e0).to(self.device).eval();self.mask.load_state_dict(self.ck['mask'])

    def pred(self):
        if self.predictor is None:self.predictor=predictor_for(self.args)
        return self.predictor

    def restore(self,mask,shape):
        if self.legacy:return cv2.resize(mask,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
        return restore_to_original(mask,shape,discrete=True)

    @torch.no_grad()
    def scores(self,image):
        if self.legacy:
            maps=stitched_memory_maps(self.pred(),image,self.memory['keys'].to(self.device),self.args)
            return self.reader.score_map(torch.from_numpy(maps).to(self.device)).cpu().numpy(),maps[0]
        canonical,_=ev.compute_stitched_similarity(self.pred(),image,self.e0,self.args)
        if self.reader is None:return canonical,canonical
        maps=stitched_memory_maps(self.pred(),image,self.memory['keys'].to(self.device),self.args)
        maps[0]=canonical
        return self.reader.score_map(torch.from_numpy(maps).to(self.device)).cpu().numpy(),canonical

    @torch.no_grad()
    def local(self,image,y,x):
        sam=local_sam(self.pred(),image,y,x);ray,valid=self.radial.sample(image,y,x)
        if self.legacy:
            r=((ray.to(self.device,dtype=torch.float32)[None]-self.ck['radial_mean'].to(self.device)[None,:,None,None])/self.ck['radial_std'].to(self.device)[None,:,None,None]).masked_fill(~valid.to(self.device)[None],0)
            r=r/r.norm(dim=1,keepdim=True).clamp_min(1e-6)*.25
            return (self.mask(sam,r,self.e0)[0]>0).cpu().numpy().astype(np.uint8)
        data=dict(sam=sam[0].cpu().half().numpy(),ray=ray.numpy(),ray_valid=valid.numpy())
        feature=normalized_feature(data,self.ck['radial_mean'].numpy(),self.ck['radial_std'].numpy())
        f=torch.from_numpy(feature).to(self.device).float()[None]
        return (mask_forward(self.mask,f,mask_descriptor(f,f[:,:352],self.e0),self.e0)[0]>0).cpu().numpy().astype(np.uint8)

    @torch.no_grad()
    def predict(self,image,k=None):
        image=np.array(Image.fromarray(np.asarray(image,dtype=np.uint8)).convert('RGB').resize((512,512),Image.Resampling.BILINEAR),copy=True)
        k=int(k or self.bundle['inference_K']);score,canonical=self.scores(image)
        points=[(y,x) for y,x,_ in greedy_score_peaks(score,k,self.bundle['peak_radius'])]
        parts=[describe_full_image_points(self.pred(),image,canonical,points[i:i+64],64,self.args.local_contrast_sigma,32) for i in range(0,len(points),64)]
        raw=np.concatenate([np.concatenate([d[name] for name in DESCRIPTOR_NAMES],1) for d in parts],0)
        probabilities=self.obj(torch.from_numpy(normalize_f4(raw,self.ck['statistics'])).to(self.device)).sigmoid().cpu().numpy()
        canvas=np.zeros((512,512),np.uint8)
        for (y,x),p in zip(points,probabilities):
            if p>=self.bundle['obj_threshold']:paste_local(canvas,self.local(image,y,x),y,x)
        return canvas,dict(points=points,probabilities=probabilities.tolist())
