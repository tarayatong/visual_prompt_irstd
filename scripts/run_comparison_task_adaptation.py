"""Bounded task adaptation selected on immutable train40/val10; full50 refit once.

All variants are explicitly adapted baselines, never replacements for native records.
Each method runs in a separate process because upstream module names collide.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import random
import gc
import traceback
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from scripts.prepare_comparison import DEFAULT_OUTPUT, write_json
from scripts.run_comparison_recovery_exploration import build_ir, event, save_csv
from scripts.run_comparison_hdnet import ManifestDataset, RPCAManifestDataset
from visual_prompt.evaluation.comparison import image_metrics, summarize

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RUN=ROOT/'outputs/comparison_task_adaptation_20261104'
SPLITS=ROOT/'outputs/comparison_recovery_20261102/splits'


def seed_all(seed=111):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed);torch.set_num_threads(2)


def load_manifest(dataset):
    return json.loads((DEFAULT_OUTPUT/dataset/'protocol/manifest.json').read_text(encoding='utf-8'))


def load_split(dataset):
    return json.loads((SPLITS/dataset/'split.json').read_text(encoding='utf-8'))


def balanced_iou(logits,target,hard_background=False):
    """Foreground/background equal-weight BCE plus per-image soft IoU."""
    axes=tuple(range(1,target.ndim));pos=target.sum(axes);neg=(1-target).sum(axes)
    loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    fg=(loss*target).sum(axes)/pos.clamp_min(1)
    if hard_background:
        flat=(loss*(1-target)).flatten(1)
        bg=flat.topk(min(512,flat.shape[1]),dim=1,sorted=False).values.mean(1)
    else:bg=(loss*(1-target)).sum(axes)/neg.clamp_min(1)
    bce=torch.where(pos>0,.5*(fg+bg),bg)
    prob=logits.sigmoid();inter=(prob*target).sum(axes)
    iou=(inter+1)/(prob.sum(axes)+pos-inter+1)
    return (bce+1-iou).mean()


def ensure_finite_gradients(model):
    parameters=[p for p in model.parameters() if p.grad is not None]
    if not parameters:raise ValueError('no parameter gradients')
    # Infinite clipping bound leaves gradients unchanged; aggregate the finite check
    # to avoid synchronizing CUDA separately for every upstream parameter tensor.
    torch.nn.utils.clip_grad_norm_(parameters,float('inf'),error_if_nonfinite=True)


def native_samples(method,m,ids,data_class):
    view=copy.deepcopy(m);view['test_ids']=list(ids)
    cls=RPCAManifestDataset if method=='RPCANet' else ManifestDataset
    data=cls(view,False,data_class)
    return [data[i] for i in range(len(data))]


@torch.no_grad()
def evaluate_ir(model,method,m,ids,data_class,out,label):
    model.eval();folder=out/label;folder.mkdir(parents=True,exist_ok=True);rows=[]
    samples=native_samples(method,m,ids,data_class)
    for x,_,sid in samples:
        dest=folder/f'{sid}.png'
        if not dest.exists():
            _,logits=model(x[None].cuda(),True)
            prob=logits.sigmoid()[0,0].cpu().numpy()
            h,w=m['records'][sid]['size_hw']
            prob=cv2.resize(prob,(w,h),interpolation=cv2.INTER_LINEAR)
            Image.fromarray((prob>=.5).astype(np.uint8)*255).save(dest)
        rec=m['records'][sid];h,w=rec['size_hw']
        gt=np.asarray(Image.open(rec['mask']).convert('L').resize((w,h),Image.Resampling.NEAREST))>0
        pred=np.asarray(Image.open(dest).convert('L'))>0
        rows.append(image_metrics(pred,gt,sid))
    score=summarize(rows);save_csv(out/'metrics'/f'{label}_per_image.csv',rows)
    return score


def train_ir(args,kind,ids,split,out,max_updates,validate=True):
    seed_all();m=load_manifest(args.dataset)
    # Training construction sees allocated training records only.
    m['records']={sid:m['records'][sid] for sid in split['allocated_ids']}
    model,data_class,native_loss,source=build_ir(args.method)
    samples=native_samples(args.method,m,ids,data_class)
    label_audit=[dict(sample_id=sid,working_pixels=int(y.sum()),original_pixels=int((np.asarray(Image.open(m['records'][sid]['mask']).convert('L'))>0).sum())) for _,y,sid in samples]
    write_json(out/'label_audit.json',label_audit)
    x=torch.stack([r[0] for r in samples]).cuda();y=torch.stack([r[1] for r in samples]).cuda()
    model=model.cuda();lr={'RPCANet':.0005,'HDNet':.005,'MSHNet+NS-FPN':.005,'ISNet':.05}[args.method]
    lr=getattr(args,'lr_override',lr)
    optimizer=(torch.optim.Adam(model.parameters(),lr=lr) if args.method=='RPCANet' else
               torch.optim.Adagrad(model.parameters(),lr=lr,weight_decay=1e-4 if args.method=='ISNet' else 0))
    if getattr(args,'optimizer_override',None)=='Adam':optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    edge_weight=getattr(args,'edge_weight',10.)
    identity='T optimizer schedule adaptation' if kind=='native' else 'A foreground-balanced BCE plus SoftIoU'
    if kind=='hard_background':identity+='; top512 hard-background pixel BCE'
    if args.method=='ISNet' and edge_weight!=10:identity+=f'; A edge BCE weight {edge_weight}'
    cfg=dict(method=args.method,dataset=args.dataset,variant=kind,identity=identity,
             annotation_budget=50,training_ids=ids,validation_ids=split['validation_ids'] if validate else [],
             seed=111,lr=lr,optimizer=type(optimizer).__name__,schedule='constant; isolate loss factor',
             augmentation=False,batch_size=4,input_size=256,max_updates=max_updates,
             initialization='native random constructor; no historical target-trained weights',
             split_sha256=split['sha256'],source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
             source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             selection='val10 IoU, lower Fa, fewer updates; full50 refit uses selected update count',
             official_test_labels_read=False,threshold=.5)
    cfg['edge_BCE_weight']=edge_weight if args.method=='ISNet' else None
    write_json(out/'config.json',cfg)
    updates=0;epoch=0;history=[];validations=[];best_key=None;start=time.monotonic()
    checkpoints={min(max_updates,k*math.ceil(len(ids)/4)) for k in (25,50,100)}|{max_updates}
    while updates<max_updates:
        order=list(range(len(samples)));random.Random(111+epoch).shuffle(order);losses=[]
        model.train()
        for start_index in range(0,len(order),4):
            batch=order[start_index:start_index+4];optimizer.zero_grad(set_to_none=True)
            masks,pred=model(x[batch],epoch>5);target=y[batch]
            loss_fn=(lambda p,t:native_loss(p,t,epoch)) if kind=='native' else (lambda p,t:balanced_iou(p,t,kind=='hard_background'))
            loss=loss_fn(pred,target)
            for mask in masks:
                small=F.adaptive_max_pool2d(target,mask.shape[-2:])
                loss=loss+loss_fn(mask,small)
            loss=loss/(len(masks)+1)
            if args.method=='RPCANet':loss=loss+.01*model.reconstruction_loss
            if args.method=='ISNet':
                from scripts.comparison_isnet_adapter import gradient_magnitude
                edge=model.edge_prediction;edge_target=gradient_magnitude(target).clamp(max=1)
                # Preserve the author's probability-valued edge path exactly in both arms.
                loss=loss+edge_weight*F.binary_cross_entropy(edge,edge_target)+native_loss(edge,edge_target,epoch)
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();ensure_finite_gradients(model)
            optimizer.step();updates+=1;losses.append(float(loss.detach()))
            if updates in checkpoints and validate:
                score=evaluate_ir(model,args.method,m,split['validation_ids'],data_class,out,f'val_step{updates:05d}')
                value=dict(updates=updates,epoch=epoch+1,**score);validations.append(value)
                key=(score['IoU_percent'],-score['Fa_per_million'],-updates)
                if best_key is None or key>best_key:
                    best_key=key;torch.save(dict(model=model.state_dict(),config=cfg,updates=updates),out/'best.pt')
                write_json(out/'validation_history.json',validations)
                event(out,'validation',variant=kind,**value);model.train()
            if updates>=max_updates:break
        epoch+=1;history.append(dict(epoch=epoch,updates=updates,loss=float(np.mean(losses))))
        if epoch%10==0:event(out,'training',variant=kind,**history[-1])
        save_csv(out/'training_curve.csv',history)
    if validate:
        selected=max(validations,key=lambda r:(r['IoU_percent'],-r['Fa_per_million'],-r['updates']))
        report=dict(status='complete',variant=kind,selected_validation=selected,elapsed_seconds=time.monotonic()-start,
                    official_test_labels_read=False)
    else:
        torch.save(dict(model=model.state_dict(),config=cfg,updates=updates),out/'model.pt')
        # Official test labels become accessible only after the validation lock and refit.
        full=load_manifest(args.dataset)
        score=evaluate_ir(model,args.method,full,full['test_ids'],data_class,out,'predictions')
        score.update(method=args.method,dataset=args.dataset,variant=kind,complete_test=True,training_updates=updates,
                     annotation_budget=50,version_identity=cfg['identity'],selection_locked=True)
        write_json(out/'metrics/summary.json',score)
        report=dict(status='complete',test_metrics=score,elapsed_seconds=time.monotonic()-start)
    write_json(out/'report.json',report)
    del model,x,y;torch.cuda.empty_cache()
    return report


def ir_worker(args):
    split=load_split(args.dataset);out=args.output/args.method/args.dataset;out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():return
    reports=[]
    for kind in ('native','balanced'):
        folder=out/'validation'/kind;folder.mkdir(parents=True,exist_ok=True)
        p=folder/'report.json'
        try:
            result=json.loads(p.read_text(encoding='utf-8')) if p.exists() else train_ir(args,kind,split['fit_ids'],split,folder,args.epochs*10)
        except (RuntimeError,ValueError) as exc:
            failure=dict(error=str(exc),traceback=traceback.format_exc(),official_test_labels_read=False)
            write_json(folder/'candidate_failure.json',failure)
            history=folder/'validation_history.json'
            if not history.exists():
                gc.collect();torch.cuda.empty_cache();continue
            valid=json.loads(history.read_text(encoding='utf-8'))
            result=dict(status='validation_checkpoint_retained_after_later_failure',variant=kind,
                        selected_validation=max(valid,key=lambda r:(r['IoU_percent'],-r['Fa_per_million'],-r['updates'])),
                        later_failure=failure,official_test_labels_read=False)
        reports.append(result)
        gc.collect();torch.cuda.empty_cache()
    if not reports:raise RuntimeError('Both validation candidates failed before a scored checkpoint; inspect candidate_failure.json')
    chosen=max(reports,key=lambda r:(r['selected_validation']['IoU_percent'],-r['selected_validation']['Fa_per_million'],-r['selected_validation']['updates']))
    lock=dict(variant=chosen['variant'],updates=chosen['selected_validation']['updates'],validation=chosen['selected_validation'],
              official_test_labels_read=False,rule='IoU, lower Fa, fewer updates',split_sha256=split['sha256'])
    lock_path=out/'selection.json'
    if lock_path.exists():assert json.loads(lock_path.read_text(encoding='utf-8'))==lock
    else:write_json(lock_path,lock)
    folder=out/'full50';folder.mkdir(parents=True,exist_ok=True)
    final=json.loads((folder/'report.json').read_text(encoding='utf-8')) if (folder/'report.json').exists() else train_ir(args,lock['variant'],split['allocated_ids'],split,folder,lock['updates'],False)
    write_json(out/'report.json',dict(status='complete',method=args.method,dataset=args.dataset,selection=lock,final=final))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',required=True)
    parser.add_argument('--dataset',required=True);parser.add_argument('--output',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--epochs',type=int,default=100);args=parser.parse_args()
    assert 1<=args.epochs<=500
    ir_worker(args)


if __name__=='__main__':main()
