"""Port the existing instance/negative memory and Reader into Tiny decoding."""
import copy
from dataclasses import asdict
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from visual_prompt.backbones.local_mask_encoder import LocalMaskEncoder
from visual_prompt.core.semantic_direction import (
    extract_support_semantic_tokens,build_score_token,mean_token,
    mask_to_feature_bool,pool_feature_token,
)
from visual_prompt.models.semantic_memory import RecallMemoryReader


class FrozenTinyMemory:
    """Separate fixed Tiny feature coordinates; never mix historical B+ keys.

    Full-image feature pooling replaces the old128-pixel sliding/crop geometry.
    Key construction, negative correction and Reader retain their original
    implementation. Only the same fixed support5 supply memory labels.
    """
    def __init__(self,native_model,manifest,semantic_config):
        self.manifest=manifest;self.config=semantic_config
        self.encoder=LocalMaskEncoder(native_model.sam,clone=True).cuda().eval()
        self.encoder.requires_grad_(False)
        self.parameters_frozen=sum(p.numel() for p in self.encoder.parameters())
        self.support_ids=list(manifest['support_ids'])
        assert set(self.support_ids)<=set(manifest['train_ids'])
        self.mean=torch.tensor([.485,.456,.406],device='cuda')[:,None,None]
        self.std=torch.tensor([.229,.224,.225],device='cuda')[:,None,None]
        self.preprocess=native_model._preprocess_visual_features
        positives=[];by_image=[];audit=[];negative={'high_freq':[],'bright_flat':[]}
        with torch.no_grad():
            for sid in self.support_ids:
                feature,image=self.features(sid)
                record=manifest['records'][sid]
                original_size=Image.open(record['image']).size
                mask=Image.open(record['mask']).convert('L').resize(original_size,Image.Resampling.NEAREST).resize((640,640),Image.Resampling.NEAREST)
                gt=np.asarray(mask)>0
                labels_count,labels=cv2.connectedComponents(gt.astype(np.uint8),connectivity=8)
                local=[];fhwc=feature[0].permute(1,2,0)
                for iid in range(1,labels_count):
                    imask=torch.from_numpy((labels==iid).astype(np.float32)).cuda()[None,None]
                    selected=mask_to_feature_bool(imask,feature.shape[-2:],threshold=0.,
                        rf_radius_scale=semantic_config.target_rf_radius_scale,
                        rf_radius_min=semantic_config.target_rf_radius_min,
                        rf_radius_max=semantic_config.target_rf_radius_max)
                    token=pool_feature_token(fhwc,selected,semantic_config.target_pool,semantic_config.unit_features)
                    if token is None:raise ValueError(('Empty support instance',sid,iid))
                    positives.append(token);local.append(token)
                    audit.append(dict(sample_id=sid,instance=iid,pixels=int((labels==iid).sum())))
                assert local
                by_image.append(mean_token(local))
                mask_tensor=torch.from_numpy(gt.astype(np.float32)).cuda()[None,None]
                _,negs=extract_support_semantic_tokens(image,gt.astype(np.uint8),mask_tensor,feature,semantic_config)
                for name,value in negs.items():
                    if value is not None:negative[name].append(value)
        original=mean_token(by_image)
        corrected,info=build_score_token(by_image,negative,semantic_config)
        corrected_instances=[build_score_token([t],negative,semantic_config)[0] for t in positives]
        negatives=[t for items in negative.values() for t in items]
        self.keys=F.normalize(torch.cat([original,corrected,*corrected_instances,*negatives],dim=0),dim=1).detach()
        counts={s:sum(r['sample_id']==s for r in audit) for s in self.support_ids}
        self.record=dict(positive_count=len(positives),negative_count=len(negatives),support_ids=self.support_ids,
            positive_weights=[1/(len(counts)*counts[r['sample_id']]) for r in audit],audit=audit,
            corrected_info=info,semantic_config=asdict(semantic_config),key_count=len(self.keys),key_channels=self.keys.shape[1],
            geometry='full-image640 input, native1024 encoding, aligned256 feature grid; no old128-window retrieval',
            key_origin='new frozen pretrained Tiny reference encoder; no B+ keys or Reader task weights reused',
            frozen_reference_parameters=self.parameters_frozen)

    @torch.no_grad()
    def features(self,sid):
        image=np.array(Image.open(self.manifest['records'][sid]['image']).convert('RGB').resize((640,640),Image.Resampling.BILINEAR),dtype=np.float32)
        x=(torch.from_numpy(image.transpose(2,0,1)/255).cuda()-self.mean)/self.std
        samples,_,_,_=self.preprocess(x[None,None],1024)
        layers=self.encoder(samples)
        assert [f.shape[1] for f in layers]==[32,64,256]
        feature=F.normalize(torch.cat([F.interpolate(f.float(),size=(256,256),mode='bilinear',align_corners=False) for f in layers],dim=1),dim=1)
        return feature,image.astype(np.uint8)

    @torch.no_grad()
    def maps(self,sid):
        features,_=self.features(sid)
        return torch.einsum('kc,bchw->bkhw',self.keys,features).cpu()


class RadialAndMemoryFusion(nn.Module):
    """Keep the radial treatment and add the user's differentiable Reader.

    Frozen similarity maps are supplied per query. The zero output projection
    starts at exact radial-only behavior; the Reader is trained through the
    decoder loss. Reader.score_map is deliberately not used: it is no_grad.
    """
    def __init__(self,radial,key_count,sam_channels=32):
        super().__init__();self.radial=radial
        self.reader=RecallMemoryReader(key_count)
        self.memory_projection=nn.Conv2d(1,sam_channels,1,bias=False)
        nn.init.zeros_(self.memory_projection.weight)
        self.maps=None

    def forward(self,features,raw,valid):
        fused=self.radial(features,raw,valid)
        if self.maps is None:raise RuntimeError('Prepare external memory maps for the current query')
        maps=self.maps
        if maps.shape[0]!=features.shape[0]:raise ValueError('Memory-map batch mismatch')
        b,c,h,w=maps.shape
        # Preserve gradients through the existing pointwise Reader.
        score=self.reader(maps.permute(0,2,3,1).reshape(-1,c)).reshape(b,1,h,w)
        if score.shape[-2:]!=features.shape[-2:]:score=F.interpolate(score,size=features.shape[-2:],mode='bilinear',align_corners=False)
        return fused+self.memory_projection(score)
