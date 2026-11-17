"""Four public architectures x three fixed train50 datasets, seed111, epoch120 only.

Training never reads test masks, selects an epoch, or loads target-task weights.
Run isolated processes because upstream code uses generic package names.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
from typing import Optional, Tuple

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.profile_public_irstd_models import build, EXP, read, save, sha, manifest
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from visual_prompt.evaluation.comparison import image_metrics,summarize

RUN=ROOT/'outputs/comparison_sam_train50_20261109'
METHODS=('IRPruneDet','IR-SAM2','IRSAM','SAMamba')
DATASETS=('SIRST','IRSTD-1k','NUDT-SIRST')


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def seed():
    random.seed(111);np.random.seed(111);torch.manual_seed(111)
    torch.cuda.manual_seed_all(111);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False


def pair(m,sid,size):
    rec=m['records'][sid]
    with Image.open(rec['image']) as im:
        x=np.array(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)
    with Image.open(rec['mask']) as im:
        # Canonicalize 0/1 and 0/255 labels before any resizing.
        y=np.asarray(im.convert('L'))>0
        y=np.array(Image.fromarray(y.astype(np.uint8)*255).resize((size,size),Image.Resampling.NEAREST))>0
    return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(y.astype(np.float32))[None]


def image_only(m,sid,size,method,stats):
    with Image.open(m['records'][sid]['image']) as im:
        x=np.array(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)
    return normalize(torch.from_numpy(x.transpose(2,0,1).copy()),method,stats)


def normalize(x,method,stats):
    if method=='IRSAM':return x
    if method=='IR-SAM2':return (x-stats['mean'])/stats['std']
    return (x/255.-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]


class CachedTrain(Dataset):
    def __init__(self,m,size,method,augment=True):
        self.ids=list(m['train_ids']);self.items=[pair(m,sid,size) for sid in self.ids]
        self.augment=augment
        total=square=n=0.
        for sid in self.ids:
            with Image.open(m['records'][sid]['image']) as im:a=np.asarray(im.convert('L'),dtype=np.float64)
            total+=a.sum();square+=np.square(a).sum();n+=a.size
        self.stats=dict(mean=total/n,std=float(max(square/n-(total/n)**2,1e-12)**.5))
        self.items=[(normalize(x,method,self.stats),y) for x,y in self.items]
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        x,y=self.items[i];x=x.clone();y=y.clone()
        if self.augment:
            if random.random()<.5:x,y=x.flip(-1),y.flip(-1)
            if random.random()<.5:x,y=x.flip(-2),y.flip(-2)
            if random.random()<.5:x,y=x.transpose(-1,-2),y.transpose(-1,-2)
        return x.contiguous(),y.contiguous(),self.ids[i]


def native_csa(source,train):
    path=source/'training/trainer.py';tree=ast.parse(path.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ('CSALoss','LLoss')]
    trainer=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Trainer')
    stats=next(n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name=='_calculate_dataset_statistics')
    env=dict(torch=torch,nn=nn,F=F,cv2=cv2,np=np,random=random,Optional=Optional,Tuple=Tuple)
    exec(compile(ast.Module(body=nodes+[stats],type_ignores=[]),str(path),'exec'),env)
    s,c=env['_calculate_dataset_statistics'](SimpleNamespace(rank=1),train.items)
    epoch=[0]
    criterion=env['CSALoss'](warm_epoch=1,get_epoch_fn=lambda:epoch[0],s_mean=s,c_mean=c,tda_weight=.2)
    return criterion,epoch,dict(s_mean=s,c_mean=c)


def setup(method,m,irsam_edge_weight=10.):
    source=EXP/'sources'/method
    compact_scan = method == 'SAMamba' and m.get('working_size', 0) >= 640
    model,_,identity=build(method,task_weights=False,compact_scan=compact_scan)
    size=m.get('working_size', {'IRPruneDet':512,'IR-SAM2':256,'IRSAM':512,'SAMamba':256}[method])
    batch={'IRPruneDet':4,'IR-SAM2':2,'IRSAM':1,'SAMamba':2}[method]
    identity.update(input_size=size,batch_size=batch,actual_annotation_budget=50,seed=111,
        dataset=m['dataset'],train_annotation_ids=m['train_ids'],manifest_sha256=m['manifest_sha256'],
        training_horizon=120,retained_epochs=[120],threshold=.5,
        test_selection='fixed epoch120 only; no test evaluation or model selection during training',
        augmentation='resize; independent H/V flip and transpose; one pass over all50 each epoch',
        native_input_deviation='SAMamba train/test256 instead of native1024 for 10GB GPU with differentiable reference scan' if method=='SAMamba' else 'native profile input size retained',
        target_task_pretraining_used=False,runner_sha256=sha(Path(__file__)))
    if 'working_size' in m:
        identity['native_input_deviation'] = f'common external training/test resize {size}'
    # No target-trained values survive initialization of the public pruned topology.
    if method=='IRPruneDet':
        for layer in model.modules():
            if isinstance(layer,(nn.Conv2d,nn.ConvTranspose2d,nn.Linear)):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.modules.batchnorm._BatchNorm):layer.reset_parameters();layer.reset_running_stats()
        for p in model.parameters():p.requires_grad_(True)
        identity.update(weights_status='all compact-topology parameter values and BN stats reset; no author task weight values retained',
            initialization='seed111 Kaiming; published hard-pruned channel topology retained; not a reproduction of pruning-stage training')
    elif method=='SAMamba':
        identity['weights_status']='SAM2.1 Hiera-S pretrained trunk only; public target-task weights deliberately not loaded for train50'
    train=CachedTrain(m,size,method)
    loader=DataLoader(train,batch_size=batch,shuffle=True,num_workers=0,drop_last=False)
    identity.update(train_only_statistics=train.stats)
    model.cuda()
    params=[p for p in model.parameters() if p.requires_grad]
    epoch_ref=[0]
    if method=='IRPruneDet':
        lossfn=module(source/'model/loss.py','native_prune_loss').SoftLoULoss()
        optimizer=torch.optim.Adagrad(params,lr=.01,weight_decay=1e-4)
        def objective(x,y):return lossfn(model(x),y)
        def predict(x):return model(x).sigmoid()
        identity.update(loss='author SoftLoULoss; smooth1',optimizer='Adagrad',lr=.01)
    elif method=='IR-SAM2':
        lossfn,epoch_ref,stats=native_csa(source,train)
        Adan=module(source/'training/Adan.py','native_ir_sam2_adan').Adan
        optimizer=Adan(params,lr=.01,weight_decay=1e-4)
        def objective(x,y):return sum(lossfn(z.sigmoid(),y,input_images=x) for z in model(x,True))
        def predict(x):return model(x,True)[0].sigmoid()
        identity.update(loss='author sum CSALoss all heads; warm1; SLS+LLoss+0.2*TDA; statistics from train50 only',
                        optimizer='author Adan',lr=.01,csa_train_statistics=stats)
    elif method=='IRSAM':
        dice=module(source/'utils/loss_mask.py','native_irsam_loss').DICE_loss
        optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        def forward(x):
            masks,edges=model([dict(image=im,original_size=(size,size)) for im in x])
            # The public decoder always returns 512-square masks and edges.
            if masks.shape[-2:] != x.shape[-2:]:
                masks,edges=(F.interpolate(value,size=x.shape[-2:],mode='bilinear',align_corners=False)
                             for value in (masks,edges))
            return masks,edges
        def objective(x,y):
            masks,edges=forward(x)
            gt=np.stack([cv2.Canny((a[0].detach().cpu().numpy()*255).astype(np.uint8),100,200) for a in y])
            edgegt=torch.from_numpy(gt.copy()).to(x.device,dtype=torch.float32)[:,None]/255.
            ld,li=dice(masks,y)
            return ld+li+float(irsam_edge_weight)*F.binary_cross_entropy(edges,edgegt)
        def predict(x):return forward(x)[0].sigmoid()
        identity.update(loss=f'author Dice+SoftIoU+{irsam_edge_weight:g}*edgeBCE; Canny100/200 on training GT only',optimizer='AdamW',lr=1e-4,
                        edge_loss_weight=float(irsam_edge_weight))
    else:
        losses=module(source/'basicseg/loss/basic_loss.py','native_samamba_losses')
        losslist=[losses.SoftIoULoss(),losses.WeightedDiceLoss(),losses.FocalLoss()]
        optimizer=torch.optim.AdamW(params,lr=5e-4,weight_decay=1e-4,betas=(.9,.999))
        def objective(x,y):
            logits=model(x)
            return sum(loss(logits,y) for loss in losslist)
        def predict(x):return model(x).sigmoid()
        identity.update(loss='author SoftIoU+WeightedDice+Focal equal weights',optimizer='AdamW',lr=5e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=120,eta_min=1e-6 if method!='SAMamba' else 1e-5)
    identity.update(schedule='cosine end120; scheduler once each epoch',parameters_trainable=sum(p.numel() for p in params),
                    parameters_total=sum(p.numel() for p in model.parameters()))
    return model,optimizer,scheduler,loader,objective,predict,identity,epoch_ref,train


def evaluate(model,predict,m,folder,config,cp):
    model.eval();rows=[];dest=folder/'epoch_0120';preds=dest/'predictions';preds.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for i,sid in enumerate(m['test_ids']):
            x=image_only(m,sid,config['input_size'],config['method'],config['train_only_statistics'])
            prob=predict(x[None].cuda())[0,0].float().cpu().numpy()
            h,w=m['records'][sid]['size_hw'];prob=cv2.resize(prob,(w,h),interpolation=cv2.INTER_LINEAR)
            assert np.isfinite(prob).all(),sid
            binary=prob>=.5
            with Image.open(m['records'][sid]['mask']) as im:gt=np.array(im.convert('L').resize((w,h),Image.Resampling.NEAREST))>0
            rows.append(image_metrics(binary,gt,sid));Image.fromarray(binary.astype(np.uint8)*255).save(preds/f'{sid}.png')
            if (i+1)%100==0:print(f'eval {config["method"]} {m["dataset"]} {i+1}/{len(m["test_ids"])}',flush=True)
    assert len(rows)==len({r['sample_id'] for r in rows})==len(m['test_ids'])
    assert {p.stem for p in preds.glob('*.png')}==set(m['test_ids'])
    metrics=dest/'metrics';metrics.mkdir(exist_ok=True)
    with (metrics/'per_image.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=summarize(rows);result.update(config)
    result.update(epochs=120,complete_test=True,checkpoint_sha256=sha(cp),prediction_coverage_verified=True,
                  checkpoint=cp.relative_to(ROOT).as_posix(),
                  initialization_checkpoint=config.get('checkpoint'),initialization_checkpoint_sha256=config.get('checkpoint_sha256'),
                  initialization_weights_status=config['weights_status'],weights_status='project train50 seed111 epoch120 task weights',
                  train50_replication_complete=True,evaluation_runner_sha256=sha(Path(__file__)))
    save(metrics/'summary.json',result)
    # Independent recount from compact saved binary masks is done by the queue audit.
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--dataset',choices=DATASETS,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args();seed()
    m=manifest(a.dataset)
    assert len(set(m['train_ids']))==50 and not set(m['train_ids'])&set(m['test_ids'])
    model,opt,sched,loader,objective,predict,config,eref,train=setup(a.method,m)
    config['method']=a.method
    model.cuda()
    folder=RUN/a.dataset/a.method
    if a.smoke:
        model.train();x,y,_=next(iter(loader));torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        opt.zero_grad(set_to_none=True);value=objective(x.cuda(),y.cuda());assert torch.isfinite(value)
        value.backward();params=[p for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert params and torch.isfinite(torch.stack(torch._foreach_norm([p.grad for p in params]))).all()
        opt.step();torch.cuda.synchronize()
        record=dict(method=a.method,dataset=a.dataset,loss=float(value.detach()),seconds=time.perf_counter()-t,
                    peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20,gradient_tensors=len(params),config=config)
        save(RUN/'preflight'/f'{a.method}_{a.dataset}.json',record);print(json.dumps(record),flush=True);return
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'config.json').exists():
        previous=read(folder/'config.json')
        assert {k:v for k,v in previous.items() if k!='runner_sha256'}=={k:v for k,v in config.items() if k!='runner_sha256'},'Run identity changed'
        # Training source hash stays attached to existing checkpoint; evaluator
        # has its own hash, allowing reporting-only repairs without retraining.
        config=previous
    else:save(folder/'config.json',config)
    cp=folder/'checkpoints/latest.pt';cp.parent.mkdir(exist_ok=True);final=cp.parent/'epoch_0120.pt'
    start=0;history=[]
    if cp.exists():
        state=torch.load(cp,map_location='cpu',weights_only=False);assert state['config']==config
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        start=state['epoch'];history=state['history'];random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(start,120):
        model.train();eref[0]=epoch+1;t=time.perf_counter();values=[];seen=[]
        for x,y,ids in loader:
            opt.zero_grad(set_to_none=True);value=objective(x.cuda(),y.cuda())
            if not torch.isfinite(value):raise ValueError(f'Nonfinite loss epoch{epoch+1}')
            value.backward();params=[p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not params or not torch.isfinite(torch.stack(torch._foreach_norm([p.grad for p in params]))).all():raise ValueError(f'Invalid gradients epoch{epoch+1}')
            opt.step();values.append(float(value.detach()));seen.extend(ids)
        assert len(seen)==50 and set(seen)==set(m['train_ids'])
        sched.step();torch.cuda.synchronize()
        history.append(dict(epoch=epoch+1,loss=float(np.mean(values)),seconds=time.perf_counter()-t,unique_supervised_images=50))
        save(folder/'training_history.json',history);save(folder/'status.json',dict(status='training',epoch=epoch+1,maximum_epochs=120))
        if (epoch+1)%5==0 or epoch+1==120:
            state=dict(config=config,epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),history=history,
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            temporary=cp.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(cp)
        print(f'{a.method} {a.dataset} epoch{epoch+1}/120 loss{history[-1]["loss"]:.6f} seconds{history[-1]["seconds"]:.2f}',flush=True)
    torch.save(dict(config=config,epoch=120,model=model.state_dict()),final)
    # Prove evaluation loads the exact retained epoch, not an in-memory variant.
    model.load_state_dict(torch.load(final,map_location='cpu',weights_only=False)['model'],strict=True)
    result=evaluate(model,predict,m,folder,config,final)
    save(folder/'status.json',dict(status='complete',epoch=120,metrics=result))
    print(json.dumps({k:result[k] for k in ['method','dataset','IoU_percent','nIoU_percent','F1_percent','Pd_percent','Fa_per_million']}),flush=True)

if __name__=='__main__':main()
