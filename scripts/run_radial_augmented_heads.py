"""Independently train Obj and Mask with lossless radial inputs; frozen SAM/e0."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import traceback

import cv2
import numpy as np
import torch
from torch.nn import functional as F
import yaml

from scripts.objectness_features import status
from scripts.run_f4_spatial_concat import F4Bank, JITTER, OLD
from scripts.run_residual_weak_augmentation import CachedFeatureBank
from scripts.run_objectness_positive_jitter import load, read_csv, predictor_for
from scripts.run_phase_b_mask_head import load_baseline
from scripts.run_global_ablation_201 import encode_local, crop_at, paste_local, add_metrics, write_csv, write_json
from scripts.run_centered_mask_ablation import summary
from scripts.run_mask_without_objectness import center_target, forward as old_mask_forward
from scripts.recompute_objectness_deduplicated import postprocess_metrics
from visual_prompt.core.candidate_proposals import connected_instances, snapped_instance_centers
from visual_prompt.data.irstd import raw_irstd_paths
from visual_prompt.models.lightweight_heads import MaskVectorMLP, soft_iou_loss
from visual_prompt.models.radial_augmented_heads import RadialMaskVectorMLP, RadialResidualObjectness
from visual_prompt.models.radial_profile import DirectionalRadialOperator, RadialProfileConfig
from visual_prompt.pipelines.semantic_prompt_pipeline import load_rgb_and_mask, seed_everything

ROOT=Path('outputs/studies/objectness_mask')
MASK_SOURCE=ROOT/'mask_without_objectness_20261027'
OBJ_SOURCE=ROOT/'residual_weak_augmentation_20261027'
OUT=ROOT/'radial_concat_heads_20261028'
DOC=Path('docs/objectness_mask_refinement_experiment_plan.md')
RADIAL_CFG=RadialProfileConfig(fusion='concat')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RadialBank:
    """Gray-only, real-image context; no GT or learned readout in the descriptor."""
    def __init__(self, baseline, device):
        self.b,self.device=baseline,device
        self.op=DirectionalRadialOperator(1,RADIAL_CFG).to(device).eval()
        self.images={};self.point_features={}
        offsets=torch.arange(-32,32,device=device,dtype=torch.float32)
        yy,xx=torch.meshgrid(offsets,offsets,indexing='ij')
        self.offsets=torch.stack((xx,yy),-1).reshape(1,-1,2)

    def image(self,sid):
        if sid not in self.images:
            image,gt,_=load_rgb_and_mask(*raw_irstd_paths(self.b.data_root,sid),(self.b.image_size,self.b.image_size))
            gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
            self.images[sid]=(image,gt,torch.from_numpy(gray).to(self.device)[None,None])
        return self.images[sid]

    @torch.no_grad()
    def points(self,rows):
        grouped=defaultdict(set)
        for r in rows:
            key=(int(r['y']),int(r['x']))
            if (r['sample_id'],*key) not in self.point_features:
                grouped[r['sample_id']].add(key)
        for sid,locations in grouped.items():
            ordered=sorted(locations)
            points=torch.tensor([[(x,y) for y,x in ordered]],dtype=torch.float32,device=self.device)
            values=self.op(self.image(sid)[2],points)[0].cpu().numpy()
            for point,value in zip(ordered,values):
                self.point_features[(sid,*point)]=value.copy()
        return np.stack([self.point_features[(r['sample_id'],int(r['y']),int(r['x']))] for r in rows])

    @torch.no_grad()
    def map(self,sid,y,x):
        points=self.offsets+torch.tensor([[[x,y]]],device=self.device)
        return self.op(self.image(sid)[2],points).transpose(1,2).reshape(1,256,64,64)


def fit_point_statistics(matrix):
    return matrix.mean(0).astype(np.float32),np.maximum(matrix.std(0),.01).astype(np.float32)


def train_mask(radial,e0,ids):
    path=OUT/'artifacts/mask_radial.pt'
    if path.exists():return load(path)
    manifest=read_csv(MASK_SOURCE/'metadata/train_crops.csv')
    items=[];total=torch.zeros(256,dtype=torch.float64);square=total.clone();pixels=0
    for number,r in enumerate(manifest):
        sid,y,x=r['sample_id'],int(r['y']),int(r['x'])
        with np.load(MASK_SOURCE/'cache/train_local'/f'{sid}_{y}_{x}.npz') as data:
            sam=torch.from_numpy(data['feature'].copy())
            mask=torch.from_numpy(data['mask'].astype(np.float32))
        target,iid=center_target(connected_instances(radial.image(sid)[1]).labels,y,x)
        assert np.array_equal(mask.numpy(),target) and bool(iid)==bool(int(r['positive']))
        feature=radial.map(sid,y,x)[0].cpu().half()
        values=feature.double();total+=values.sum((-2,-1));square+=values.square().sum((-2,-1));pixels+=64*64
        items.append(dict(sample_id=sid,feature=sam,radial=feature,mask=mask,positive=int(r['positive'])))
        if (number+1)%40==0:status(OUT,'prepare_mask',crops=number+1,total_crops=len(manifest))
    assert len(items)==320 and sum(r['positive'] for r in items)==172
    assert {r['sample_id'] for r in items}==set(ids)
    mean=(total/pixels).float();std=(square/pixels-(total/pixels).square()).clamp_min(0).sqrt().clamp_min(.01).float()
    pos=[r for r in items if r['positive']];neg=[r for r in items if not r['positive']]
    seed_everything(111);model=RadialMaskVectorMLP(len(e0),256,e0).to(e0.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    p_rng,n_rng=random.Random(111),random.Random(112)
    mu,scale=mean.to(e0.device)[None,:,None,None],std.to(e0.device)[None,:,None,None]
    history=[]
    for epoch in range(120):
        p,n=p_rng.choices(pos,k=128),n_rng.choices(neg,k=128);losses=[]
        for start in range(0,128,8):
            batch=p[start:start+8]+n[start:start+8]
            sam=torch.stack([r['feature'] for r in batch]).to(e0.device,dtype=torch.float32)
            ray=(torch.stack([r['radial'] for r in batch]).to(e0.device,dtype=torch.float32)-mu)/scale
            target=torch.stack([r['mask'] for r in batch]).to(e0.device)
            logits=model(sam,ray,e0)
            a=soft_iou_loss(logits[:8],target[:8]);b=F.softplus(logits[8:]).mean();loss=a+b
            if not torch.isfinite(loss):raise ValueError('Nonfinite Mask loss')
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append((float(a.detach()),float(b.detach())))
        history.append(dict(epoch=epoch+1,positive_iou_loss=float(np.mean([v[0] for v in losses])),negative_BCE=float(np.mean([v[1] for v in losses]))))
        if (epoch+1)%20==0:
            write_csv(OUT/'metrics/mask_history.csv',history);status(OUT,'training_mask',**history[-1])
    checkpoint=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},sam_channels=len(e0),radial_channels=256,
                    radial_mean=mean,radial_std=std,sample_ids=ids,epochs=120,seed=111,history=history,
                    parameters=sum(p.numel() for p in model.parameters()),initialization='fresh_matched_original_SAM_input_columns_new_columns_zero',
                    input='unmodified_local_SAM_352_concat_standardized_radial_256',augmentation=False)
    torch.save(checkpoint,path)
    write_json(OUT/'metadata/mask_training.json',dict(crops=320,positive=172,negative=148,negative_with_other_target=23,
                                                   epochs=120,positive_per_epoch=128,negative_per_epoch=128,updates_per_epoch=16,
                                                   parameters=checkpoint['parameters'],loss='positive_IoU_plus_negative_pixel_BCE'))
    status(OUT,'mask_training_complete',parameters=checkpoint['parameters'])
    return checkpoint


@torch.no_grad()
def predict_obj(model,rows,bank,f4bank,radial,ck):
    model.eval();result=[]
    for start in range(0,len(rows),32):
        batch=rows[start:start+32];layers,raw=bank.batch(batch,'fusion')
        f4=torch.from_numpy(f4bank.normalize(f4bank.raw(batch),ck)).to(bank.device)
        ray=torch.from_numpy((radial.points(batch)-ck['radial_mean'])/ck['radial_std']).to(bank.device)
        result.extend(torch.sigmoid(model(layers,raw,f4,ray)).cpu().tolist())
    return result


def train_obj(fold,records,bank,f4bank,radial):
    path=OUT/'artifacts'/f'obj_radial_{fold}.pt'
    if path.exists():return load(path)
    original=load(JITTER/'artifacts'/f'objectness_target_any_{fold}.pt')
    schedule_path=OLD/'metadata'/f'{fold}_schedule.json'
    epochs=json.loads(schedule_path.read_text());flat=[r for epoch in epochs for r in epoch]
    assert len(epochs)==100 and {r['sample_id'] for r in flat}<=set(original['sample_ids'])
    assert not set(original['sample_ids']) & set(original['validation_ids'])
    rays=radial.points(flat);mean,std=fit_point_statistics(rays)
    ray_tensor=torch.from_numpy((rays-mean)/std).to(bank.device)
    f4=torch.from_numpy(f4bank.normalize(f4bank.raw(flat,training=True),original)).to(bank.device)
    seed_everything(111);model=RadialResidualObjectness(original['model']).to(bank.device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
    offset=0;history=[]
    for number,rows in enumerate(epochs):
        model.train();losses=[]
        for start in range(0,len(rows),32):
            batch=rows[start:start+32];layers,raw=bank.batch(batch,'fusion')
            labels=torch.tensor([r['label'] for r in batch],dtype=torch.float32,device=bank.device)
            logits=model(layers,raw,f4[offset+start:offset+start+len(batch)],ray_tensor[offset+start:offset+start+len(batch)])
            loss=F.binary_cross_entropy_with_logits(logits,labels)
            if not torch.isfinite(loss):raise ValueError('Nonfinite Obj loss')
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
        offset+=len(rows);history.append(dict(epoch=number+1,BCE=float(np.mean(losses))))
        if (number+1)%20==0:
            write_csv(OUT/'metrics'/f'obj_{fold}_history.csv',history)
            status(OUT,'training_obj',fold=fold,**history[-1])
    assert all(torch.equal(v.cpu(),model.base.state_dict()[k].cpu()) for k,v in original['model'].items())
    ck=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},mean=original['mean'],std=original['std'],
            prototype_target=original['prototype_target'],prototype_background=original['prototype_background'],
            radial_mean=mean,radial_std=std,sample_ids=original['sample_ids'],validation_ids=original['validation_ids'],
            epochs=100,seed=111,history=history,parameter_counts=model.parameter_counts(),
            schedule_sha256=digest(schedule_path),base_F4_frozen=True,augmentation=False)
    val=[r for r in records if r['source']=='eval' and r['sample_id'] in original['validation_ids']]
    if val:
        scores=predict_obj(model,val,bank,f4bank,radial,ck)
        ck['predictions']=[dict(**r,score=s) for r,s in zip(val,scores)]
    else:ck['predictions']=[]
    torch.save(ck,path);status(OUT,'obj_fold_complete',fold=fold,parameters=ck['parameter_counts'])
    return ck


def lock_threshold(oof):
    rows=[dict(sample_id=r['sample_id'],y=int(r['y']),x=int(r['x']),assignment=int(r['assignment']),target_count=int(r['target_count'])) for r in oof]
    scores=[r['score'] for r in oof]
    assert len(rows)==225 and len({r['sample_id'] for r in rows})==45
    sweep=[]
    for threshold in np.unique(np.r_[0.,.9,np.asarray(scores)[np.asarray(scores)<=.9]])[::-1]:
        metric,_=postprocess_metrics(rows,scores,float(threshold),16)
        sweep.append(dict(threshold=float(threshold),**metric))
    feasible=[r for r in sweep if r['recall']>=.95]
    maximum=max(r['recall'] for r in sweep)
    chosen=max(feasible if feasible else [r for r in sweep if r['recall']==maximum],key=lambda r:r['threshold'])
    selection=dict(rule='highest_threshold_le_0.9_reaching_95_percent_OOF_recall_else_highest_at_max_recall',
                   threshold=chosen['threshold'],OOF=chosen,maximum_OOF_recall=maximum,test_used=False)
    write_json(OUT/'metadata/new_obj_threshold.json',selection)
    write_csv(OUT/'metrics/new_obj_OOF_sweep.csv',sweep);write_csv(OUT/'metrics/new_obj_OOF_predictions.csv',oof)
    status(OUT,'threshold_locked',**selection)
    return selection


@torch.no_grad()
def evaluate(mask_ck,obj_ck,selection,bank,f4bank,radial,e0,ids):
    # Test data first enter the selection/evaluation stage after OOF lock is saved.
    source=read_csv(OBJ_SOURCE/'metrics/residual_candidate_scores.csv')
    rows=[dict(sample_id=r['sample_id'],y=int(r['y']),x=int(r['x']),assignment=int(r['assignment']),target_count=int(r['target_count'])) for r in source]
    old_scores=[float(r['score']) for r in source]
    assert len(rows)==1005 and len({r['sample_id'] for r in rows})==201 and not set(ids)&{r['sample_id'] for r in rows}
    base=load(JITTER/'artifacts/objectness_target_any_full50.pt')
    obj=RadialResidualObjectness(base['model']).to(e0.device);obj.load_state_dict(obj_ck['model']);obj.eval()
    new_scores=predict_obj(obj,rows,bank,f4bank,radial,obj_ck)
    old_threshold=json.loads((ROOT/'loose_objectness_oof_20261027/metadata/threshold_selection.json').read_text())['threshold']
    decisions={};point_results=[];nms_details=[]
    for label,scores,threshold in [('old_obj_loose',old_scores,old_threshold),('new_obj_loose',new_scores,selection['threshold']),
                                    ('old_obj09',old_scores,.9),('new_obj09',new_scores,.9)]:
        keep=defaultdict(set)
        for post,radius in [('raw',0),('point_nms',16)]:
            metric,detail=postprocess_metrics(rows,scores,threshold,radius)
            point_results.append(dict(system=label,postprocess=post,threshold=threshold,**metric))
            if post=='point_nms':
                for r in detail:
                    nms_details.append(dict(system=label,**r))
                    if r['kept']:keep[r['sample_id']].add((r['y'],r['x']))
        decisions[label]=keep
    write_csv(OUT/'metrics/test_objectness.csv',point_results)
    write_csv(OUT/'metrics/test_nms_decisions.csv',nms_details)
    write_csv(OUT/'metrics/test_candidate_scores.csv',[dict(**r,old_score=a,new_score=b) for r,a,b in zip(rows,old_scores,new_scores)])
    old_ck=load(MASK_SOURCE/'artifacts/mask_with_rejection.pt')
    old_mask=MaskVectorMLP(len(e0),128,e0,'m2_residual').to(e0.device);old_mask.load_state_dict(old_ck['model']);old_mask.eval()
    new_mask=RadialMaskVectorMLP(len(e0),256,e0).to(e0.device);new_mask.load_state_dict(mask_ck['model']);new_mask.eval()
    mu=mask_ck['radial_mean'].to(e0.device)[None,:,None,None];scale=mask_ck['radial_std'].to(e0.device)[None,:,None,None]
    predictor=predictor_for(radial.b,e0.device)
    grouped=defaultdict(list)
    for r in rows:grouped[r['sample_id']].append(r)
    all_images=[];all_rois=[]
    cache=OUT/'cache/evaluation';cache.mkdir(parents=True,exist_ok=True)
    for number,(sid,candidates) in enumerate(grouped.items()):
        path=cache/f'{sid}.json'
        if path.exists():
            data=json.loads(path.read_text(encoding='utf-8'));all_images.append(data['image']);all_rois.extend(data['rois']);continue
        image,gt,_=radial.image(sid);instances=connected_instances(gt)
        centers={(y,x) for _,y,x in snapped_instance_centers(instances)}
        proposals={(r['y'],r['x']) for r in candidates}
        routes={label:keep[sid] for label,keep in decisions.items()}
        routes.update(gt_center=centers,no_obj_top5=proposals)
        unions={kind+'_'+route:np.zeros_like(gt,dtype=np.uint8) for kind in ('old_mask','new_mask') for route in routes}
        roi_rows=[]
        for y,x in sorted(proposals|centers):
            _,_,sam=encode_local(predictor,image,y,x,64,64)
            ray=(radial.map(sid,y,x).half().float()-mu)/scale
            pred_old=(old_mask_forward(old_mask,sam,e0)[0]>0).cpu().numpy().astype(np.uint8)
            pred_new=(new_mask(sam,ray,e0)[0]>0).cpu().numpy().astype(np.uint8)
            target,iid=center_target(instances.labels,y,x);_,local_gt=crop_at(image,gt,y,x,64)
            for kind,pred in [('old_mask',pred_old),('new_mask',pred_new)]:
                roi_rows.append(dict(sample_id=sid,kind=kind,y=y,x=x,candidate=int((y,x) in proposals),positive=int(iid>0),
                                     other_target_pixels=int(((local_gt>0)&(target==0)).sum()),pred_pixels=int(pred.sum()),
                                     target_pixels=int(target.sum()),intersection=int(((pred>0)&(target>0)).sum()),
                                     union=int(((pred>0)|(target>0)).sum())))
                for route,locations in routes.items():
                    if (y,x) in locations:paste_local(unions[kind+'_'+route],pred,y,x)
        row=dict(sample_id=sid,target_count=instances.count,image_pixels=gt.size)
        for system,pred in unions.items():
            add_metrics(row,system,pred,gt)
            assert row[system+'_false_pixels']==row[system+'_pred_pixels']-row[system+'_inter']
        write_json(path,dict(image=row,rois=roi_rows));all_images.append(row);all_rois.extend(roi_rows)
        if (number+1)%20==0:status(OUT,'evaluating',images_done=number+1,images_total=201)
    systems=[kind+'_'+route for kind in ('old_mask','new_mask') for route in routes]
    results=[summary(all_images,system) for system in systems]
    write_csv(OUT/'metrics/test_image_metrics.csv',all_images);write_csv(OUT/'metrics/test_summary.csv',results)
    write_csv(OUT/'metrics/test_roi_metrics.csv',all_rois)
    rejection=[]
    for kind in ('old_mask','new_mask'):
        candidates=[r for r in all_rois if r['kind']==kind and r['candidate']]
        pos=[r for r in candidates if r['positive']];neg=[r for r in candidates if not r['positive']]
        near=[r for r in neg if r['other_target_pixels']]
        rejection.append(dict(kind=kind,positive=len(pos),negative=len(neg),
                              empty_rejection_percent=100*np.mean([r['pred_pixels']==0 for r in neg]),
                              positive_empty_percent=100*np.mean([r['pred_pixels']==0 for r in pos]),
                              other_target_empty_percent=100*np.mean([r['pred_pixels']==0 for r in near]),
                              positive_ROI_IoU=100*sum(r['intersection'] for r in pos)/max(1,sum(r['union'] for r in pos))))
    write_csv(OUT/'metrics/test_rejection.csv',rejection)
    # Reproduce the retained deployed-working-point reference, with all FP pixels.
    reference=next(r for r in read_csv(ROOT/'loose_objectness_oof_20261027/metrics/test_summary.csv') if r['system']=='OOF_loose')
    baseline=next(r for r in results if r['system']=='old_mask_old_obj_loose')
    for key in ('miou_percent','niou_percent','pd_percent','fa_per_million','f1_percent'):
        assert baseline[key]==float(reference[key]),(key,baseline[key],reference[key])
    assert len(all_images)==201 and sum(r['target_count'] for r in all_images)==297
    write_json(OUT/'validation.json',dict(test_images=201,test_targets=297,candidate_points=1005,
                                         independent_training=True,baseline_exactly_reproduced=True,
                                         full_image_FP_verified=True,OOF_threshold_locked_before_test=True,
                                         no_augmentation=True,frozen_SAM_and_e0=True,operator_trainable_parameters=0))
    write_json(OUT/'conclusions.json',dict(language='zh',mask='仅增加径向通道，保持SAM局部编码及正IoU/负BCE监督；完整数字见test_summary/test_rejection。',
                                          obj='保持冻结F4及原残差分支，附加256维候选点剖面；五折只用于阈值选择。',
                                          parameters=dict(mask=mask_ck['parameters'],obj=obj_ck['parameter_counts']),
                                          new_threshold=selection['threshold'],segmentation=results,objectness=point_results,rejection=rejection))
    status(OUT,'complete',threshold=selection['threshold'],parameters=dict(mask=mask_ck['parameters'],obj=obj_ck['parameter_counts']),
           segmentation=results,objectness=[r for r in point_results if r['postprocess']=='point_nms'],rejection=rejection)


def main():
    for folder in ('artifacts','metadata','metrics'):(OUT/folder).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    cfg=yaml.safe_load(Path('experiments/configs/objectness_mask/train_final_heads.yaml').read_text())['args']
    ids=cfg['sample_ids'].split(',');_,b=load_baseline(Path(cfg['baseline_config']));device=torch.device('cuda:0')
    old_threshold=json.loads((ROOT/'loose_objectness_oof_20261027/metadata/threshold_selection.json').read_text())['threshold']
    protocol=dict(sample_ids=ids,train_images=50,test_images=201,seed=111,radial=asdict(RADIAL_CFG),
                  radial_scaling='training_only_channel_mean_std_floor0.01;mask_dense_FP16;Obj_point_FP32',
                  Obj='frozen_F4_plus_original_trainable_CNN_residual_MLP_appended256_radial_point_features;bound2',
                  Mask='SAM352_plus_radial256_vector_MLP;e0_zero_padded;sim_only_SAM',
                  mask_epochs=120,obj_epochs=100,augmentation=False,independent_training=True,
                  old_threshold=old_threshold,new_threshold_rule='five_fold_OOF_recall95_highest_threshold_le0.9',
                  reference_mask_sha256=digest(MASK_SOURCE/'artifacts/mask_with_rejection.pt'),
                  reference_obj_sha256=digest(OBJ_SOURCE/'artifacts/residual_full50.pt'))
    path=OUT/'metadata/protocol.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==protocol
    write_json(path,protocol)
    e0=load(ROOT/'final_heads_50shot/artifacts/edited_target_embedding.pt').float().to(device)
    radial=RadialBank(b,device)
    mask_ck=train_mask(radial,e0,ids)
    records=json.loads((ROOT/'phase_c_objectness/metadata/feature_manifest.json').read_text())['records']
    bank=CachedFeatureBank(OUT,b,device,OLD/'cache/full_image');f4bank=F4Bank(records,ids)
    oof=[]
    for fold in range(5):
        oof.extend(train_obj(f'fold{fold}',records,bank,f4bank,radial)['predictions'])
    selection=lock_threshold(oof)
    obj_ck=train_obj('full50',records,bank,f4bank,radial)
    evaluate(mask_ck,obj_ck,selection,bank,f4bank,radial,e0,ids)
    assert protocol['reference_mask_sha256']==digest(MASK_SOURCE/'artifacts/mask_with_rejection.pt')
    assert protocol['reference_obj_sha256']==digest(OBJ_SOURCE/'artifacts/residual_full50.pt')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output_dir',default=str(OUT))
    OUT=Path(parser.parse_args().output_dir)
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);status(OUT,'failed',error=traceback.format_exc());raise
