"""Frozen F4 classifier + bounded CNN residual; image augmentation retired."""
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

from scripts.run_f4_spatial_concat import ROOT,OLD,JITTER,F4Bank,load,read_csv
from scripts.objectness_features import FeatureBank,status,select_oof,load_mask,evaluate
from scripts.run_phase_b_mask_head import load_baseline
from scripts.run_global_ablation_201 import write_csv,write_json
from scripts.recompute_objectness_deduplicated import postprocess_metrics
from visual_prompt.models.residual_objectness import ResidualObjectness
from visual_prompt.pipelines.semantic_prompt_pipeline import seed_everything

TRIALS=('residual',)


class CachedFeatureBank(FeatureBank):
    """Reuse original frozen features; retired synthetic images are not accepted."""
    def __init__(self,out,baseline,device,original_cache):
        super().__init__(out,baseline,device)
        self.original_cache=original_cache

    @torch.no_grad()
    def get(self,sid):
        if '__' in sid:
            raise ValueError('Only original images are supported.')
        if sid not in self.items:
            path=self.cache/f'{sid}.pt'
            if not path.exists():path=self.original_cache/f'{sid}.pt'
            if not path.exists():raise FileNotFoundError(path)
            item=load(path)
            self.items[sid]=dict(layers=[v.to(self.device,dtype=torch.float32)[None] for v in item['layers']],
                                 gray=F.pad(item['gray'].to(self.device,dtype=torch.float32)[None,None]/255.,(32,32,32,32)))
        return self.items[sid]


def schedule(out,trial,fold,metadata):
    if trial not in TRIALS:
        raise ValueError('Image augmentation trials have been retired; use residual only.')
    path=out/'metadata'/f'{trial}_{fold}_schedule.json'
    epochs=json.loads((OLD/'metadata'/f'{fold}_schedule.json').read_text())
    result=json.loads(path.read_text()) if path.exists() else epochs
    if result != epochs:
        raise ValueError('No-augmentation schedule must match the original sampling schedule.')
    assert all('__' not in r['sample_id'] for epoch in result for r in epoch)
    counts={'original':sum(len(epoch) for epoch in result),'low':0,'small':0,'both':0,'negative_fallback':0}
    write_json(path,result);write_json(out/'metadata'/f'{trial}_{fold}_sampling_counts.json',counts)
    return result


@torch.no_grad()
def predict(model,bank,f4bank,ck,rows):
    model.eval();scores=[]
    for start in range(0,len(rows),32):
        batch=rows[start:start+32];layers,raw=bank.batch(batch,'fusion')
        f4=torch.from_numpy(f4bank.normalize(f4bank.raw(batch),ck)).to(bank.device)
        scores.extend(torch.sigmoid(model(layers,raw,f4)).cpu().tolist())
    return scores


def train(out,trial,fold,metadata,bank,f4bank,records):
    if trial not in TRIALS:
        raise ValueError('Image augmentation trials have been retired; use residual only.')
    path=out/'artifacts'/f'{trial}_{fold}.pt'
    if path.exists():return load(path)
    original=load(JITTER/'artifacts'/f'objectness_target_any_{fold}.pt')
    epochs=schedule(out,trial,fold,metadata)
    allowed=set(original['sample_ids'])
    assert all(r['sample_id'].split('__')[0] in allowed for e in epochs for r in e)
    flat=[r for epoch in epochs for r in epoch]
    f4=torch.from_numpy(f4bank.normalize(f4bank.raw(flat,training=True),original)).to(bank.device)
    seed_everything(111);model=ResidualObjectness(original['model']).to(bank.device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.0001)
    history=[];offset=0
    for number,rows in enumerate(epochs):
        model.train();losses=[];deltas=[]
        for start in range(0,len(rows),32):
            batch=rows[start:start+32];layers,raw=bank.batch(batch,'fusion')
            labels=torch.tensor([r['label'] for r in batch],dtype=torch.float32,device=bank.device)
            logits,delta=model(layers,raw,f4[offset+start:offset+start+len(batch)],True)
            loss=F.binary_cross_entropy_with_logits(logits,labels)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(loss.item());deltas.append(float(delta.abs().mean().detach().cpu()))
        offset+=len(rows);history.append(dict(epoch=number+1,loss=float(np.mean(losses)),mean_abs_residual=float(np.mean(deltas))))
        if (number+1)%10==0:
            status(out,'training',trial=trial,fold=fold,epoch=number+1,epochs=100,loss=history[-1]['loss'])
            write_csv(out/'metrics'/f'{trial}_{fold}_history.csv',history)
    assert offset==len(flat)
    assert all(torch.equal(model.base.state_dict()[k].cpu(),v.cpu()) for k,v in original['model'].items())
    val=[r for r in records if r['source']=='eval' and r['sample_id'] in original['validation_ids']]
    scores=predict(model,bank,f4bank,original,val)
    ck=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},trial=trial,epochs=100,seed=111,
            bound=2.,parameter_counts=model.parameter_counts(),history=history,
            mean=original['mean'],std=original['std'],prototype_target=original['prototype_target'],prototype_background=original['prototype_background'],
            sample_ids=original['sample_ids'],validation_ids=original['validation_ids'],
            predictions=[dict(**r,score=s) for r,s in zip(val,scores)],
            original_schedule_sha256=hashlib.sha256((OLD/'metadata'/f'{fold}_schedule.json').read_bytes()).hexdigest(),
            actual_schedule_sha256=hashlib.sha256((out/'metadata'/f'{trial}_{fold}_schedule.json').read_bytes()).hexdigest(),
            frozen_base_unchanged=True)
    torch.save(ck,path);return ck




def main(out):
    for folder in ('artifacts','metadata','metrics','cache/full_image'):(out/folder).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    device=torch.device('cuda:0');(out/'pid.txt').write_text(str(os.getpid()))
    cfg=yaml.safe_load(Path('experiments/configs/objectness_mask/train_final_heads.yaml').read_text())['args']
    ids=cfg['sample_ids'].split(',');_,b=load_baseline(Path(cfg['baseline_config']))
    records=json.loads((ROOT/'phase_c_objectness/metadata/feature_manifest.json').read_text())['records']
    bank=CachedFeatureBank(out,b,device,OLD/'cache/full_image');f4bank=F4Bank(records,ids)
    base=load(JITTER/'artifacts/objectness_target_any_full50.pt')
    counts=ResidualObjectness(base['model']).parameter_counts()
    write_json(out/'metadata/parameter_counts.json',counts)
    write_json(out/'metadata/protocol.json',dict(trials=list(TRIALS),sample_ids=ids,epochs=100,seed=111,
                                               frozen_F4=str(JITTER/'artifacts'),residual_bound=2,initial_residual=0,
                                               augmentation_enabled=False,augmented_probability=0.,
                                               extra_shape_transforms=False,
                                               probability_threshold=.9,NMS_radius=16,parameters=counts))
    source=read_csv(OLD/'metrics/baseline_target_any_candidate_scores.csv')
    test=[dict(sample_id=r['sample_id'],y=int(r['y']),x=int(r['x']),assignment=int(r['assignment']),target_count=int(r['target_count'])) for r in source]
    assert len({r['sample_id'] for r in test})==201 and not set(ids)&{r['sample_id'] for r in test}
    mask_parts=load_mask(bank);metadata={};all_points=[];all_masks=[];all_oof=[]
    for trial in TRIALS:
        oof=[]
        for fold in ['fold0','fold1','fold2','fold3','fold4']:
            ck=train(out,trial,fold,metadata,bank,f4bank,records);oof.extend(ck['predictions'])
        assert len(oof)==225
        threshold,_=select_oof(oof,16)
        for name,t in [('fixed09',.9),('oof',threshold)]:
            metric,_=postprocess_metrics(oof,[r['score'] for r in oof],t,16)
            all_oof.append(dict(trial=trial,threshold_name=name,threshold=t,**metric))
        write_csv(out/'metrics/oof_summary.csv',all_oof);write_csv(out/'metrics'/f'{trial}_oof_predictions.csv',oof)
        ck=train(out,trial,'full50',metadata,bank,f4bank,records);ck['oof_threshold']=threshold
        torch.save(ck,out/'artifacts'/f'{trial}_full50.pt')
        model=ResidualObjectness(base['model']).to(device);model.load_state_dict(ck['model'])
        scores=predict(model,bank,f4bank,ck,test)
        points,masks=evaluate(out,trial,ck,bank,test,16,mask_parts,existing_scores=scores,
                              extra_mask_caches=[OLD/'cache/local_masks',ROOT/'f4_spatial_concat_20261027/cache/local_masks'])
        all_points.extend(points);all_masks.extend(masks)
        write_csv(out/'metrics/all_objectness.csv',all_points);write_csv(out/'metrics/all_joint_summary.csv',all_masks)
        status(out,'trial_complete',trial=trial,parameters=counts)
    write_json(out/'validation.json',dict(models=len(TRIALS)*6,epochs=100,train_images=50,test_images=201,GT_instances=297,
                                        original_F4_frozen=True,augmented_validation_images_used_for_training=False))
    status(out,'complete',models=len(TRIALS)*6,test_images=201,parameters=counts)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output_dir',default=str(ROOT/'residual_no_augmentation_20261027'))
    out=Path(parser.parse_args().output_dir)
    try:main(out)
    except Exception:status(out,'failed',error=traceback.format_exc());raise
