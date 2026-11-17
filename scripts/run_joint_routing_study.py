"""OOF routing selection and paired differentiable Obj/Mask training."""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
import yaml

import scripts.run_radial_independent_search as search
from scripts.run_radial_autonomous_queue import HOME, ROOT, read
from scripts.train_locked_radial_full50 import make_model, configure
from scripts.run_objectness_positive_jitter import load, read_csv, predictor_for
from scripts.run_f4_spatial_concat import F4Bank, JITTER
from scripts.run_phase_b_mask_head import load_baseline
from scripts.run_global_ablation_201 import paste_local, add_metrics, encode_local, write_json, write_csv
from scripts.run_centered_mask_ablation import summary
from scripts.run_radial_augmented_heads import digest
from visual_prompt.core.candidate_proposals import scored_point_nms, connected_instances
from visual_prompt.models.lightweight_heads import ObjectnessMLP, soft_iou_loss
from visual_prompt.pipelines.semantic_prompt_pipeline import seed_everything

OUT=ROOT/'joint_routing_20261029'


def routes():
    result=[dict(id='bypass',kind='bypass',threshold=0.,nms=0,weight=0.)]
    for threshold in (.1,.3,.47042956948280334,.7,.9):
        for nms in (0,16):
            result.append(dict(id=f'hard_{threshold:.4f}_nms{nms}',kind='hard',threshold=threshold,nms=nms,weight=0.))
    for weight in (.25,.5):
        result.append(dict(id=f'additive_{weight}',kind='soft',threshold=0.,nms=0,weight=weight))
    return result


def evaluate(rows,logits,objects,truth,route,system):
    grouped=defaultdict(list)
    for i,r in enumerate(rows):grouped[r['sample_id']].append(i)
    images=[];count=0
    for sid,indices in grouped.items():
        gt=truth[sid];pred=np.zeros_like(gt,dtype=np.uint8)
        eligible=[i for i in indices if route['kind']!='hard' or float(torch.sigmoid(torch.tensor(objects[i])) )>=route['threshold']]
        if route['nms']:
            keep=scored_point_nms([(rows[i]['y'],rows[i]['x'],float(objects[i])) for i in eligible],route['nms'])
            eligible=[eligible[k] for k in keep]
        count+=len(eligible)
        for i in eligible:
            z=logits[i]+route['weight']*objects[i]
            paste_local(pred,(z>0).astype(np.uint8),rows[i]['y'],rows[i]['x'])
        row=dict(sample_id=sid,target_count=connected_instances(gt).count,image_pixels=int(gt.size))
        add_metrics(row,'prediction',pred,gt);images.append(row)
    result=summary(images,'prediction');result['system']=system
    return dict(model=system,route=route['id'],ROIs_sent=count,**result),images


def train_pair(name,weight,fold,items,sums,squares,counts,selected,e0,f4bank,ids):
    path=OUT/'artifacts'/f'{name}_{fold}.pt'
    if path.exists():return load(path)
    original=load(JITTER/'artifacts'/f'objectness_target_any_{fold}.pt')
    training=set(ids) if fold=='full50' else set(original['sample_ids'])
    validation=set() if fold=='full50' else set(original['validation_ids'])
    assert not training&validation
    indices=[i for i,r in enumerate(items) if r['sample_id'] in training]
    features,desc,mu,std,_=search.mask_features(items,indices,'C1',sums,squares,counts,e0)
    points=[dict(sample_id=r['sample_id'],y=r['y'],x=r['x'],label=r['positive']) for r in items]
    f4=torch.from_numpy(f4bank.normalize(f4bank.raw(points,training=True),original)).to(e0.device)
    targets=torch.stack([r['target'] for r in items]).to(e0.device,dtype=torch.float32)
    labels=torch.tensor([r['positive'] for r in items],device=e0.device,dtype=torch.float32)
    seed_everything(111);mask=make_model('mask',selected,e0,e0.device)
    seed_everything(111);obj=ObjectnessMLP(1440,128).to(e0.device)
    optimizer=torch.optim.AdamW(list(mask.parameters())+list(obj.parameters()),lr=.001,weight_decay=.0001)
    pos=[i for i in indices if items[i]['positive']];neg=[i for i in indices if not items[i]['positive']]
    pr,nr=random.Random(111),random.Random(112);history=[];schedule=[]
    for epoch in range(120):
        p,n=pr.choices(pos,k=128),nr.choices(neg,k=128);schedule.append(p+n);values=[]
        for start in range(0,128,8):
            batch=p[start:start+8]+n[start:start+8]
            z=search.cached_mask_forward(mask,features[batch],desc[batch],e0)
            o=obj(f4[batch]);fused=z+weight*o[:,None,None]
            segmentation=soft_iou_loss(fused[:8],targets[batch[:8]])+F.softplus(fused[8:]).mean()
            loss=segmentation+F.binary_cross_entropy_with_logits(o,labels[batch])
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();optimizer.step();values.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(values))))
    mask.eval();obj.eval()
    with torch.no_grad():
        val=[i for i,r in enumerate(items) if r['sample_id'] in validation and r['source']!='gt_center']
        # Some GT centers coincide with Top5: select membership explicitly below, not source.
        ck=dict(mask={k:v.detach().cpu() for k,v in mask.state_dict().items()},obj={k:v.detach().cpu() for k,v in obj.state_dict().items()},
                radial_mean=mu,radial_std=std,original_statistics={k:original[k] for k in ('mean','std','prototype_target','prototype_background')},
                train_ids=sorted(training),validation_ids=sorted(validation),weight=weight,selected=selected,epochs=120,seed=111,
                schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),history=history)
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(ck,path)
    del features,desc,f4,targets,mask,obj,optimizer;torch.cuda.empty_cache()
    return ck


@torch.no_grad()
def inference_items(items,indices,mask,obj,ck,e0,f4bank):
    points=[dict(sample_id=items[i]['sample_id'],y=items[i]['y'],x=items[i]['x']) for i in indices]
    f4=torch.from_numpy(f4bank.normalize(f4bank.raw(points),ck['original_statistics'])).to(e0.device)
    o=obj(f4).cpu().numpy();zs=[];sel=ck['selected'];columns=sel['components']
    mu=ck['radial_mean'].to(e0.device)[None,:,None,None];std=ck['radial_std'].to(e0.device)[None,:,None,None]
    for i in indices:
        item=items[i];sam=item['sam'].to(e0.device,dtype=torch.float32)[None]
        raw=item['radial'][columns].to(e0.device,dtype=torch.float32)[None];valid=item['valid'][columns].to(e0.device)[None]
        ray=search.transform_radial((raw-mu)/std,valid,'A2',1.)
        f=torch.cat((sam,ray),1);d=search.mask_descriptor(f,sam,e0)
        zs.append(search.cached_mask_forward(mask,f,d,e0)[0].cpu().numpy())
    return np.stack(zs),o


def context():
    torch.set_num_threads(2);device=torch.device('cuda:0')
    cfg=yaml.safe_load(Path('experiments/configs/objectness_mask/train_final_heads.yaml').read_text())['args']
    ids=cfg['sample_ids'].split(',');_,baseline=load_baseline(Path(cfg['baseline_config']))
    e0=load(ROOT/'final_heads_50shot/artifacts/edited_target_embedding.pt').float().to(device)
    selected=read(HOME/'selection_lock.json')['heads']['mask'];configure('mask',selected);search.OUT=OUT
    records=read(ROOT/'phase_c_objectness/metadata/feature_manifest.json')['records'];records=[r for r in records if r['sample_id'] in ids]
    bank=search.TrainOnlyRadialBank(baseline,device,ids);f4=F4Bank(records,ids)
    return ids,baseline,e0,selected,records,bank,f4


def study():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'selection_lock.json').exists(),'Already locked; do not run new OOF search after test'
    write_json(OUT/'protocol.json',dict(routes=routes(),models=['independent','paired_control','joint_additive'],
               joint_weights=[0.,.25],selection='OOF global IoU then nIoU then Pd then lower Fa',
               original50_only=True,freeze=['SAM','e0'],augmentation=False,joint_epochs=120,
               joint_sampling='same_as_Mask_128positive_128negative',joint_loss='Obj_BCE_plus_fused_positive_IoU_plus_fused_negative_BCE',
               image_level_end_to_end=False,official201_only_after_lock=True))
    ids,baseline,e0,selected,records,bank,f4=context()
    items,sums,squares,counts=search.prepare_masks(bank,ids)
    candidates={(r['sample_id'],int(r['y']),int(r['x'])) for r in records if r['source']=='eval'}
    collected={name:dict(rows=[],mask=[],obj=[]) for name in ('independent','paired_control','joint_additive')};truth={}
    parent=selected['candidate'];folder=Path(parent['output'])/'mask'/parent['variant']
    for fold in range(5):
        old_mask=load(folder/'artifacts'/f'fold{fold}.pt');old_obj=load(JITTER/'artifacts'/f'objectness_target_any_fold{fold}.pt')
        indices=[i for i,r in enumerate(items) if r['sample_id'] in old_obj['validation_ids'] and (r['sample_id'],r['y'],r['x']) in candidates]
        assert len(indices)==5*len(old_obj['validation_ids'])
        for name,weight in [('independent',None),('paired_control',0.),('joint_additive',.25)]:
            if weight is None:
                ck=dict(mask=old_mask['model'],obj=old_obj['model'],radial_mean=old_mask['radial_mean'],radial_std=old_mask['radial_std'],selected=selected,
                        original_statistics={k:old_obj[k] for k in ('mean','std','prototype_target','prototype_background')})
            else:ck=train_pair(name,weight,f'fold{fold}',items,sums,squares,counts,selected,e0,f4,ids)
            mask=make_model('mask',selected,e0,e0.device);mask.load_state_dict(ck['mask']);mask.eval()
            obj=ObjectnessMLP(1440).to(e0.device);obj.load_state_dict(ck['obj']);obj.eval()
            zs,os=inference_items(items,indices,mask,obj,ck,e0,f4)
            collected[name]['rows'] += [dict(sample_id=items[i]['sample_id'],y=items[i]['y'],x=items[i]['x']) for i in indices]
            collected[name]['mask'].extend(zs);collected[name]['obj'].extend(os)
            del mask,obj;torch.cuda.empty_cache()
        for sid in old_obj['validation_ids']:truth[sid]=bank.image(sid)[1]
        print(json.dumps(dict(stage='OOF',fold=fold,models_complete=3)),flush=True)
    results=[];image_rows=[]
    for name,data in collected.items():
        assert len(data['rows'])==225 and len({r['sample_id'] for r in data['rows']})==45
        data['mask']=np.stack(data['mask']);data['obj']=np.asarray(data['obj'])
        for route in routes():
            result,images=evaluate(data['rows'],data['mask'],data['obj'],truth,route,name);results.append(result)
            image_rows.extend([dict(model=name,route=route['id'],**r) for r in images])
    residual=read_csv(ROOT/'residual_weak_augmentation_20261027/metrics/residual_oof_predictions.csv')
    lookup={(r['sample_id'],int(r['y']),int(r['x'])):float(r['score']) for r in residual}
    base=collected['independent'];prob=np.asarray([lookup[(r['sample_id'],r['y'],r['x'])] for r in base['rows']])
    assert len(lookup)==225
    collected['legacy_residual']=dict(rows=base['rows'],mask=base['mask'],obj=np.log(np.clip(prob,1e-7,1-1e-7)/(1-np.clip(prob,1e-7,1-1e-7))))
    legacy_routes=[r for r in routes() if r['kind']!='soft']+[dict(id='legacy_OOF_loose',kind='hard',threshold=.3602827191352844,nms=16,weight=0.)]
    data=collected['legacy_residual']
    for route in legacy_routes:
        result,images=evaluate(data['rows'],data['mask'],data['obj'],truth,route,'legacy_residual');results.append(result)
        image_rows.extend([dict(model='legacy_residual',route=route['id'],**r) for r in images])
    torch.save(dict(predictions=collected,truth=truth),OUT/'OOF_cache.pt')
    control=next(r for r in results if r['model']=='independent' and r['route']=='bypass')
    original=read(folder/'summary.json')['task_metrics']['image_summary'][0]
    assert abs(control['miou_percent']-original['miou_percent'])<1e-8
    winner=max(results,key=lambda r:(r['miou_percent'],r['niou_percent'],r['pd_percent'],-r['fa_per_million']))
    route=next(r for r in routes()+legacy_routes if r['id']==winner['route'])
    write_csv(OUT/'metrics/OOF_comparison.csv',results);write_csv(OUT/'metrics/OOF_images.csv',image_rows)
    write_json(OUT/'selection_lock.json',dict(model=winner['model'],route=route,OOF=winner,seed=111,selected_Mask=selected,
                                             protocol_sha256=digest(OUT/'protocol.json'),test_accessed=False))
    write_json(OUT/'status.json',dict(stage='OOF_locked',winner=winner,official_test_accessed=False))
    print(json.dumps(winner),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['study'],default='study');parser.parse_args();study()
