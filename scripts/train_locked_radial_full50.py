"""Train only the prelocked independent winners on all 50 training images."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
import yaml

import scripts.run_radial_independent_search as search
from scripts.run_radial_autonomous_queue import HOME, AMPLITUDE, ROOT, read
from scripts.run_objectness_positive_jitter import load
from scripts.run_f4_spatial_concat import F4Bank, JITTER, OLD
from scripts.run_phase_b_mask_head import load_baseline
from scripts.run_global_ablation_201 import write_json
from scripts.run_radial_augmented_heads import MASK_SOURCE, digest
from visual_prompt.evaluation.radial_diagnostics import masked_channel_statistics
from visual_prompt.models.lightweight_heads import ObjectnessMLP, MaskVectorMLP, soft_iou_loss
from visual_prompt.models.radial_augmented_heads import F4RadialObjectness, RadialMaskVectorMLP
from visual_prompt.models.radial_adaptive_heads import AdaptiveRadialObjectness, AdaptiveRadialMask
from visual_prompt.pipelines.semantic_prompt_pipeline import seed_everything

FINAL = HOME/'final_independent'


def make_model(head, selected, e0, device):
    variant = selected['candidate']['variant']; spec = selected['architecture']; n = selected['raw_channels']
    if head == 'obj':
        model = AdaptiveRadialObjectness(n, spec) if spec is not None else ObjectnessMLP(1440) if variant=='A0' else F4RadialObjectness(radial_channels=n)
    else:
        model = AdaptiveRadialMask(n, e0, spec) if spec is not None else MaskVectorMLP(352,128,e0,'m2_residual') if variant=='A0' else RadialMaskVectorMLP(352,n,e0)
    return model.to(device)


def configure(head, selected):
    geometry = selected['geometry']
    search.GEOMETRY = dict(radii=geometry.get('radii'), opposite_pairs=geometry.get('opposite_pairs',False), smooth_sigma=geometry.get('smooth_sigma',0.))
    search.SEED=111; search.ARCHITECTURE=selected['architecture']; search.COMPONENT_OVERRIDE=selected['components']
    search.AMPLITUDE_BASE[head]=selected['radial_amplitude']; search.OUT=FINAL/head
    cache_geometry = selected['candidate']['geometry']
    if cache_geometry:
        signature=hashlib.sha256(json.dumps(cache_geometry,sort_keys=True).encode()).hexdigest()[:12]
        search.CACHE_ROOT=HOME/'cache'/signature
    else:
        search.CACHE_ROOT=AMPLITUDE


def train_obj(selected, ids, baseline, e0, records, lock_sha):
    path=FINAL/'obj/artifacts/selected_full50.pt'
    if path.exists():
        assert load(path)['selection_lock_sha256']==lock_sha
        return
    original=load(JITTER/'artifacts/objectness_target_any_full50.pt')
    assert set(original['sample_ids'])==set(ids) and not original['validation_ids']
    bank=search.TrainOnlyRadialBank(baseline,e0.device,ids); f4bank=F4Bank(records,ids)
    schedule_path=OLD/'metadata/full50_schedule.json'; epochs=read(schedule_path); flat=[r for rows in epochs for r in rows]
    assert {r['sample_id'] for r in flat}<=set(ids) and len(epochs)==100
    raw, valid=bank.point_data(flat); columns=selected['components']; raw,valid=raw[:,columns],valid[:,columns]
    mu,std=masked_channel_statistics(raw,valid)
    f4=torch.from_numpy(f4bank.normalize(f4bank.raw(flat,training=True),original)).to(e0.device)
    rms=float(f4.square().sum(1).mean().sqrt())
    ray=search.transform_radial(torch.from_numpy((raw-mu)/std).to(e0.device),torch.from_numpy(valid).to(e0.device),selected['radial_amplitude'],rms)
    seed_everything(111); model=make_model('obj',selected,e0,e0.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    labels=torch.tensor([r['label'] for r in flat],device=e0.device,dtype=torch.float32)
    history=[]; offset=0; variant=selected['candidate']['variant']
    for epoch,rows in enumerate(epochs):
        losses=[]; model.train()
        for start in range(0,len(rows),32):
            sl=slice(offset+start,offset+min(start+32,len(rows)))
            z=model(f4[sl]) if variant=='A0' else model(f4[sl],ray[sl])
            loss=F.binary_cross_entropy_with_logits(z,labels[sl])
            if hasattr(model,'auxiliary_loss'):loss=loss+model.auxiliary_loss(labels[sl])
            if not torch.isfinite(loss):raise ValueError('Nonfinite full50 Obj loss')
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        offset+=len(rows);history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
    ck=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},radial_mean=mu,radial_std=std,original_rms=rms,
            original_statistics={k:original[k] for k in ('mean','std','prototype_target','prototype_background')},selected=selected,
            selection_lock_sha256=lock_sha,sample_ids=ids,seed=111,epochs=100,history=history,parameters=sum(p.numel() for p in model.parameters()),
            sampling_sha256=digest(schedule_path),official_test_accessed=False)
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(ck,path)
    del model,optimizer,f4,ray;torch.cuda.empty_cache()


def train_mask(selected,ids,baseline,e0,records,lock_sha):
    path=FINAL/'mask/artifacts/selected_full50.pt'
    if path.exists():
        assert load(path)['selection_lock_sha256']==lock_sha
        return
    bank=search.TrainOnlyRadialBank(baseline,e0.device,ids)
    items,sums,squares,counts=search.prepare_masks(bank,ids)
    variant=selected['candidate']['variant'];indices=list(range(len(items)))
    features,descriptors,mu,std,norms=search.mask_features(items,indices,variant,sums,squares,counts,e0)
    pos=[i for i in indices if items[i]['positive']];neg=[i for i in indices if not items[i]['positive']]
    seed_everything(111);model=make_model('mask',selected,e0,e0.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    targets=torch.stack([r['target'] for r in items]).to(e0.device,dtype=torch.float32)
    p_rng,n_rng=random.Random(111),random.Random(112);history=[];schedule=[]
    for epoch in range(120):
        p,n=p_rng.choices(pos,k=128),n_rng.choices(neg,k=128);schedule.append(p+n);losses=[]
        for start in range(0,128,8):
            batch=p[start:start+8]+n[start:start+8]
            logits=search.cached_mask_forward(model,features[batch],descriptors[batch],e0)
            loss=soft_iou_loss(logits[:8],targets[batch[:8]])+F.softplus(logits[8:]).mean()
            if hasattr(model,'auxiliary_loss'):loss=loss+model.auxiliary_loss(logits,targets[batch],8)
            if not torch.isfinite(loss):raise ValueError('Nonfinite full50 Mask loss')
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
    ck=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},radial_mean=mu,radial_std=std,original_rms=1.,selected=selected,
            selection_lock_sha256=lock_sha,sample_ids=ids,seed=111,epochs=120,history=history,parameters=sum(p.numel() for p in model.parameters()),
            sampling_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),official_test_accessed=False)
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(ck,path)
    del model,optimizer,features,descriptors,items;torch.cuda.empty_cache()


def main():
    lock_path=HOME/'selection_lock.json';lock=read(lock_path);lock_sha=digest(lock_path)
    assert not lock['official_test_accessed'] and not lock['joint_inference']
    assert digest(HOME/'selection_review.json')==lock['decision_review_sha256']
    for path,sha in lock['frozen_source_sha256'].items():assert digest(Path(path))==sha
    if (FINAL/'test_started.json').exists():
        raise RuntimeError('Official test has already started; do not refit after test access')
    torch.set_num_threads(2);device=torch.device('cuda:0')
    cfg=yaml.safe_load(Path('experiments/configs/objectness_mask/train_final_heads.yaml').read_text())['args']
    ids=cfg['sample_ids'].split(',');_,baseline=load_baseline(Path(cfg['baseline_config']))
    records=read(ROOT/'phase_c_objectness/metadata/feature_manifest.json')['records'];records=[r for r in records if r['sample_id'] in ids]
    e0=load(ROOT/'final_heads_50shot/artifacts/edited_target_embedding.pt').float().to(device)
    for head,trainer in [('obj',train_obj),('mask',train_mask)]:
        configure(head,lock['heads'][head]);trainer(lock['heads'][head],ids,baseline,e0,records,lock_sha)
        print(json.dumps(dict(head=head,full50_training='complete',official_test_accessed=False)),flush=True)
    write_json(FINAL/'training_audit.json',dict(selection_lock_sha256=lock_sha,train_images=50,seed=111,
                                               original_mask_sha256=digest(MASK_SOURCE/'artifacts/mask_with_rejection.pt'),
                                               official_test_accessed=False,independent_heads=True))


if __name__=='__main__':main()
