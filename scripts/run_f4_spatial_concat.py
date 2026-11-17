"""Keep all original F4 inputs and append trainable fused spatial CNN features."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from scripts.run_objectness_positive_jitter import ROOT, load, read_csv
from scripts.objectness_features import FeatureBank, status, select_oof, load_mask, evaluate
from scripts.run_global_ablation_201 import write_csv, write_json
from scripts.run_phase_b_mask_head import load_baseline
from scripts.recompute_objectness_deduplicated import postprocess_metrics
from visual_prompt.models.lightweight_heads import ObjectnessMLP
from visual_prompt.models.spatial_objectness import SpatialObjectness
from visual_prompt.pipelines.semantic_prompt_pipeline import seed_everything

OLD=ROOT/'spatial_obj_cnn_20261027'
JITTER=ROOT/'objectness_positive_jitter_20261027'


class F4Bank:
    """Use the exact original descriptor caches; only prototypes vary by fold."""
    def __init__(self,records,ids):
        self.original={};self.positive={};self.test={}
        for sid in ids:
            with np.load(ROOT/'phase_c_objectness/artifacts/candidate_features'/f'{sid}_desc.npz') as data:
                matrix=np.concatenate([data[k] for k in ('sim','center','mean','max','diff','hand')],axis=1)
            for r in records:
                if r['sample_id']==sid:self.original[sid,int(r['y']),int(r['x'])]=matrix[r['point_index']]
            with np.load(JITTER/'cache/target_pixels'/f'{sid}.npz') as data:
                matrix=np.concatenate([data[k] for k in ('sim','center','mean','max','diff','hand')],axis=1)
                for p,v in zip(data['points'],matrix):self.positive[sid,int(p[0]),int(p[1])]=v

    def raw(self,rows,training=False):
        result=[]
        for row in rows:
            sid=row['sample_id'];key=sid,int(row['y']),int(row['x'])
            if training and int(row['label'])==1:
                result.append(self.positive[key])
            elif key in self.original:
                result.append(self.original[key])
            else:
                if sid not in self.test:
                    with np.load(ROOT/'center_sampled_obj_diagnosis_20261026/cache'/f'{sid}.npz') as data:
                        self.test[sid]={tuple(map(int,p)):v[:1434].copy() for p,v in zip(data['points'],data['matrix'])}
                result.append(self.test[sid][key[1:]])
        return np.stack(result).astype(np.float32)

    @staticmethod
    def normalize(base,checkpoint):
        center=base[:,6:358];center=center/np.maximum(np.linalg.norm(center,axis=1,keepdims=True),1e-8)
        mean=base[:,358:710];mean=mean/np.maximum(np.linalg.norm(mean,axis=1,keepdims=True),1e-8)
        t,b=checkpoint['prototype_target'],checkpoint['prototype_background']
        tc,tm,bc,bm=center@t,mean@t,center@b,mean@b
        full=np.concatenate((base,np.stack((tc,tm,bc,bm,tc-bc,tm-bm),axis=1)),axis=1)
        return ((full-checkpoint['mean'])/checkpoint['std']).astype(np.float32)


@torch.no_grad()
def predict(model,bank,f4bank,checkpoint,rows):
    model.eval();result=[]
    for start in range(0,len(rows),32):
        batch=rows[start:start+32];layers,raw=bank.batch(batch,'fusion')
        f4=torch.from_numpy(f4bank.normalize(f4bank.raw(batch),checkpoint)).to(bank.device)
        result.extend(torch.sigmoid(model(layers,raw,f4)).cpu().tolist())
    return result


def train(out,fold,bank,f4bank,records):
    path=out/'artifacts'/f'f4_fusion_{fold}.pt'
    if path.exists():return load(path)
    original=load(JITTER/'artifacts'/f'objectness_target_any_{fold}.pt')
    schedule_path=OLD/'metadata'/f'{fold}_schedule.json'
    epochs=json.loads(schedule_path.read_text())
    # Precompute the F4 vectors using the original fold's normalization and prototypes.
    flat=[r for epoch in epochs for r in epoch]
    matrix=f4bank.normalize(f4bank.raw(flat,training=True),original)
    matrix=torch.from_numpy(matrix).to(bank.device)
    seed_everything(111);model=SpatialObjectness('fusion',1440).to(bank.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    history=[];offset=0
    for number,rows in enumerate(epochs):
        model.train();losses=[]
        for start in range(0,len(rows),32):
            batch=rows[start:start+32];layers,raw=bank.batch(batch,'fusion')
            f4=matrix[offset+start:offset+start+len(batch)]
            labels=torch.tensor([r['label'] for r in batch],device=bank.device,dtype=torch.float32)
            loss=F.binary_cross_entropy_with_logits(model(layers,raw,f4),labels)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(loss.item())
        offset+=len(rows);history.append(dict(epoch=number+1,loss=float(np.mean(losses))))
        if (number+1)%10==0:
            status(out,'training',fold=fold,epoch=number+1,epochs=100,loss=history[-1]['loss'])
            write_csv(out/'metrics'/f'{fold}_history.csv',history)
    assert offset==len(flat)
    val_ids=set(original['validation_ids'])
    val_rows=[r for r in records if r['source']=='eval' and r['sample_id'] in val_ids]
    scores=predict(model,bank,f4bank,original,val_rows)
    ck=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},mode='fusion',f4_dim=1440,
            mlp_input_dim=1536,parameter_counts=model.parameter_counts(),seed=111,epochs=100,history=history,
            sample_ids=original['sample_ids'],validation_ids=original['validation_ids'],
            mean=original['mean'],std=original['std'],prototype_target=original['prototype_target'],prototype_background=original['prototype_background'],
            schedule_sha256=hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
            input_protocol='original_complete_F4_1440_concat_trainable_fusion_CNN_96',initialization='from_scratch',
            predictions=[dict(**r,score=s) for r,s in zip(val_rows,scores)])
    torch.save(ck,path);return ck




def main(out):
    for folder in ('artifacts','metadata','metrics'):(out/folder).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    device=torch.device('cuda:0');(out/'pid.txt').write_text(str(os.getpid()))
    counts=SpatialObjectness('fusion',1440).parameter_counts();write_json(out/'metadata/parameter_counts.json',counts)
    status(out,'preparing',parameters=counts)
    cfg=yaml.safe_load(Path('experiments/configs/objectness_mask/train_final_heads.yaml').read_text())['args']
    ids=cfg['sample_ids'].split(',');_,b=load_baseline(Path(cfg['baseline_config']))
    records=json.loads((ROOT/'phase_c_objectness/metadata/feature_manifest.json').read_text())['records']
    bank=FeatureBank(out,b,device);bank.cache=OLD/'cache/full_image'
    f4bank=F4Bank(records,ids)
    write_json(out/'metadata/protocol.json',dict(input='standardized_F4_1440_concat_CNN_96',epochs=100,seed=111,
                                               sampling_schedules=str(OLD/'metadata'),fold_statistics=str(JITTER/'artifacts'),
                                               initialization='CNN_and_MLP_from_scratch',threshold=.9,NMS_radius=16,
                                               feature_cache=str(bank.cache),parameter_counts=counts))
    oof=[]
    for fold in ['fold0','fold1','fold2','fold3','fold4']:
        ck=train(out,fold,bank,f4bank,records);oof.extend(ck['predictions'])
    assert len(oof)==225 and len({r['sample_id'] for r in oof})==45
    threshold,_=select_oof(oof,16);oof_rows=[]
    for name,t in [('fixed09',.9),('oof',threshold)]:
        metric,_=postprocess_metrics(oof,[r['score'] for r in oof],t,16)
        oof_rows.append(dict(threshold_name=name,threshold=t,**metric))
    write_csv(out/'metrics/oof_summary.csv',oof_rows);write_csv(out/'metrics/oof_predictions.csv',oof)
    ck=train(out,'full50',bank,f4bank,records);ck['oof_threshold']=threshold
    torch.save(ck,out/'artifacts/f4_fusion_full50.pt')
    test_rows=read_csv(OLD/'metrics/baseline_target_any_candidate_scores.csv')
    test_rows=[dict(sample_id=r['sample_id'],y=int(r['y']),x=int(r['x']),assignment=int(r['assignment']),target_count=int(r['target_count'])) for r in test_rows]
    assert len({r['sample_id'] for r in test_rows})==201 and not set(ids)&{r['sample_id'] for r in test_rows}
    model=SpatialObjectness('fusion',1440).to(device);model.load_state_dict(ck['model'])
    scores=predict(model,bank,f4bank,ck,test_rows)
    # Reproduce the original F4 model before comparing the new head.
    baseline=load(JITTER/'artifacts/objectness_target_any_full50.pt')
    ref=ObjectnessMLP(1440,128).to(device);ref.load_state_dict(baseline['model']);ref.eval()
    with torch.no_grad():
        reproduced=torch.sigmoid(ref(torch.from_numpy(f4bank.normalize(f4bank.raw(test_rows),baseline)).to(device))).cpu().numpy()
    stored=read_csv(OLD/'metrics/baseline_target_any_candidate_scores.csv')
    error=float(np.max(np.abs(reproduced-np.asarray([float(r['score']) for r in stored]))));assert error<1e-4,error
    parts=load_mask(bank)
    points,masks=evaluate(out,'f4_fusion',ck,bank,test_rows,16,parts,existing_scores=scores,extra_mask_caches=[OLD/'cache/local_masks'])
    write_csv(out/'metrics/all_objectness.csv',points);write_csv(out/'metrics/all_joint_summary.csv',masks)
    for fold in ['fold0','fold1','fold2','fold3','fold4','full50']:
        new=load(out/'artifacts'/f'f4_fusion_{fold}.pt');old=load(OLD/'artifacts'/f'fusion_{fold}.pt')
        assert new['schedule_sha256']==old['schedule_sha256']
        assert not set(new['sample_ids'])&set(new['validation_ids'])
    write_json(out/'validation.json',dict(paired_schedules=True,baseline_probability_max_error=error,models=6,test_images=201,targets=297))
    status(out,'complete',models=6,test_images=201,parameters=counts)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output_dir',default=str(ROOT/'f4_spatial_concat_20261027'))
    out=Path(parser.parse_args().output_dir)
    try:main(out)
    except Exception:status(out,'failed',error=traceback.format_exc());raise
