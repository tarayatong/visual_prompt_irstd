"""Shared frozen feature caches and evaluation for retained objectness heads."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from scripts.run_objectness_positive_jitter import ROOT, load, predictor_for
from scripts.run_phase_b_mask_head import gaussian_map
from scripts.run_global_ablation_201 import write_json, write_csv, encode_local, infer_mask, paste_local, add_metrics
from scripts.run_centered_mask_ablation import summary as mask_summary
from scripts.recompute_objectness_deduplicated import postprocess_metrics
from visual_prompt.data.irstd import raw_irstd_paths
from visual_prompt.models.lightweight_heads import MaskVectorMLP
from visual_prompt.pipelines.semantic_prompt_pipeline import load_rgb_and_mask, get_named_sim_features


def status(out,stage,**fields):
    value=dict(stage=stage,**fields)
    write_json(out/'status.json',value)
    with (out/'events.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)


class FeatureBank:
    def __init__(self,out,baseline,device):
        self.out,self.b,self.device=out,baseline,device
        self.cache=out/'cache/full_image';self.cache.mkdir(parents=True,exist_ok=True)
        self.items={};self.predictor=None

    def get_predictor(self):
        if self.predictor is None:self.predictor=predictor_for(self.b,self.device)
        return self.predictor

    @torch.no_grad()
    def get(self,sid):
        if sid not in self.items:
            path=self.cache/f'{sid}.pt'
            if path.exists():
                item=load(path)
            else:
                image,_,_=load_rgb_and_mask(*raw_irstd_paths(self.b.data_root,sid),(self.b.image_size,self.b.image_size))
                predictor=self.get_predictor();predictor.set_image(image)
                features=get_named_sim_features(predictor,align=False)
                assert [int(f.shape[1]) for _,f in features]==[32,64,256]
                item=dict(layers=[f[0].detach().cpu().half() for _,f in features],
                          gray=torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).copy()),
                          source_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                          feature_protocol='frozen_global_SAM_native_layers_fp16_cache_v1')
                torch.save(item,path)
            self.items[sid]=dict(layers=[v.to(self.device,dtype=torch.float32)[None] for v in item['layers']],
                                 gray=F.pad(item['gray'].to(self.device,dtype=torch.float32)[None,None]/255.,(32,32,32,32)))
        return self.items[sid]

    @torch.no_grad()
    def batch(self,rows,mode):
        n=len(rows);raw=torch.empty((n,1,64,64),device=self.device)
        layers=[torch.empty((n,c,32,32),device=self.device) for c in (32,64,256)] if mode!='raw' else []
        groups=defaultdict(list)
        for i,r in enumerate(rows):groups[r['sample_id']].append(i)
        offsets=torch.linspace(-32,31,32,device=self.device)
        for sid,indices in groups.items():
            item=self.get(sid);grids=[]
            for i in indices:
                y,x=int(rows[i]['y']),int(rows[i]['x'])
                raw[i]=item['gray'][0,:,y:y+64,x:x+64]
                if mode!='raw':
                    ys=2*(y+offsets)/(self.b.image_size-1)-1
                    xs=2*(x+offsets)/(self.b.image_size-1)-1
                    gy,gx=torch.meshgrid(ys,xs,indexing='ij');grids.append(torch.stack((gx,gy),dim=-1))
            if mode!='raw':
                grid=torch.stack(grids)
                for dest,feature in zip(layers,item['layers']):
                    dest[indices]=F.grid_sample(feature.expand(len(indices),-1,-1,-1),grid,align_corners=True,padding_mode='zeros')
        return layers,raw


def select_oof(rows,radius):
    scores=np.asarray([r['score'] for r in rows]);best=None
    for threshold in np.unique(np.r_[0.,scores,1.])[::-1]:
        metric,_=postprocess_metrics(rows,scores,float(threshold),radius)
        key=(metric['f1'],metric['recall'],-metric['fp_per_image'])
        if best is None or key>best[0]:best=(key,float(threshold),metric)
    return best[1],best[2]


def load_mask(bank):
    e0=load(ROOT/'final_heads_50shot/artifacts/edited_target_embedding.pt').float().to(bank.device)
    ck=load(ROOT/'final_heads_50shot/artifacts/mask_final_a2.pt')
    model=MaskVectorMLP(ck['channels'],ck['hidden'],e0,ck.get('variant','m2_residual')).to(bank.device)
    model.load_state_dict(ck['model']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,ck,e0,torch.from_numpy(gaussian_map(64,4.)).float().to(bank.device)


def evaluate(out,mode,ck,bank,test_rows,radius,mask_parts,existing_scores=None,extra_mask_caches=()):
    if existing_scores is None:
        raise ValueError('Retained classifiers must supply their candidate scores.')
    scores=existing_scores
    point_metrics=[];selected={};detail=[]
    thresholds=[('fixed09',.9)]+([('oof',ck['oof_threshold'])] if ck is not None else [])
    for name,threshold in thresholds:
        for label,r in [('raw',0),('point_nms',radius)]:
            metric,decisions=postprocess_metrics(test_rows,scores,threshold,r)
            point_metrics.append(dict(mode=mode,threshold_name=name,threshold=threshold,postprocess=label,**metric))
            if label=='point_nms':
                keep=defaultdict(list)
                for d in decisions:
                    detail.append(dict(mode=mode,threshold_name=name,**d))
                    if d['kept']:keep[d['sample_id']].append((d['y'],d['x']))
                selected[name]=keep
    write_csv(out/'metrics'/f'{mode}_candidate_scores.csv',[dict(**r,score=s) for r,s in zip(test_rows,scores)])
    write_csv(out/'metrics'/f'{mode}_objectness.csv',point_metrics)
    write_csv(out/'metrics'/f'{mode}_nms_decisions.csv',detail)
    by_sid=defaultdict(list)
    for row in test_rows:by_sid[row['sample_id']].append(row)
    model,mc,e0,gaussian=mask_parts;image_rows=[]
    local_cache=out/'cache/local_masks';local_cache.mkdir(parents=True,exist_ok=True)
    old_cache=ROOT/'objectness_positive_jitter_20261027/cache/local_masks'
    for number,(sid,rows) in enumerate(by_sid.items()):
        image,gt,_=load_rgb_and_mask(*raw_irstd_paths(bank.b.data_root,sid),(bank.b.image_size,bank.b.image_size))
        row=dict(sample_id=sid,target_count=int(rows[0]['target_count']),image_pixels=int(gt.size),gt_pixels=int((gt>0).sum()))
        points=sorted({p for keep in selected.values() for p in keep[sid]});masks={}
        for y,x in points:
            filename=f'{sid}_{y}_{x}.npy';path=local_cache/filename;old=old_cache/filename
            if path.exists():local=np.load(path)
            elif old.exists():local=np.load(old)
            elif any((folder/filename).exists() for folder in extra_mask_caches):
                local=np.load(next(folder/filename for folder in extra_mask_caches if (folder/filename).exists()))
            else:
                with torch.no_grad():
                    crop,_,feature=encode_local(bank.get_predictor(),image,y,x,64,64)
                    local=infer_mask(model,feature,crop,e0,gaussian,bank.b,mc['residual_scale'],mc['temperature'])
                np.save(path,local)
            masks[y,x]=local
        for name,keep in selected.items():
            pred=np.zeros_like(gt,dtype=np.uint8)
            for y,x in keep[sid]:paste_local(pred,masks[y,x],y,x)
            add_metrics(row,name,pred,gt)
        image_rows.append(row)
        if (number+1)%25==0:status(out,'evaluating_joint',mode=mode,images_done=number+1,images_total=201)
    summaries=[dict(mode=mode,**mask_summary(image_rows,name)) for name in selected]
    write_csv(out/'metrics'/f'{mode}_image_metrics.csv',image_rows)
    write_csv(out/'metrics'/f'{mode}_joint_summary.csv',summaries)
    return point_metrics,summaries
