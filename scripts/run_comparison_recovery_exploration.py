"""Initial recovery exploration: CPU mask audit, immutable splits, serial GPU probes.

No official test labels are read. Existing experiments and checkpoints are read-only.
This is an execution entry point for the first diagnostic batch, not the 44-node plan.
"""
from __future__ import annotations
import argparse
import copy
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.prepare_comparison import DEFAULT_OUTPUT, write_json
from scripts.run_comparison_ifa import build_native
from scripts.run_comparison_hdnet import ManifestDataset, RPCAManifestDataset, RPCANativeOutputs
from visual_prompt.evaluation.comparison import image_metrics, summarize

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_RUN=ROOT/'outputs/comparison_recovery_20261102'
JOBS=[('RPCANet','NUDT-SIRST','RPC-0'),('HDNet','NUDT-SIRST','HD-0'),
      ('MSHNet+NS-FPN','NUDT-SIRST','NS-0'),('ISNet','IRSTD-1k','IS-0')]


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def event(root,stage,**fields):
    value=dict(time_utc=utc_now(),pid=os.getpid(),stage=stage,**fields)
    write_json(root/'status.json',value)
    with (root/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False)+'\n')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def save_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def create_splits(root):
    splits={}
    for dataset in ('IRSTD-1k','NUDT-SIRST','SIRST'):
        manifest=json.loads((DEFAULT_OUTPUT/dataset/'protocol/manifest.json').read_text(encoding='utf-8'))
        ids=manifest['train_ids'];support=set(manifest['support_ids'])
        assert len(ids)==50 and len(support)==5 and support<=set(ids)
        # Stratify validation selection by target pixel area using only allocated labels.
        # Round-robin samples from five size strata; deterministic and without replacement.
        ranked=[]
        for sid in ids:
            rec=manifest['records'][sid]
            mask=np.asarray(Image.open(rec['mask']).convert('L'))>0
            ranked.append((int(mask.sum()),sid))
        eligible=sorted((area,sid) for area,sid in ranked if sid not in support)
        buckets=[list(a) for a in np.array_split(np.asarray(eligible,dtype=object),5)]
        rng=random.Random(20261101)
        val=[]
        for bucket in buckets:
            names=[str(a[1]) for a in bucket];rng.shuffle(names);val.extend(names[:2])
        fit=[sid for sid in ids if sid not in set(val)]
        positive=[sid for _,sid in sorted(ranked) if sid in fit and next(a for a,s in ranked if s==sid)>0]
        assert len(positive)>=8
        # Four smaller and four larger positive examples, each fixed before training.
        small=positive[:len(positive)//2];large=positive[len(positive)//2:]
        probe=random.Random(111).sample(small,4)+random.Random(112).sample(large,4)
        value=dict(dataset=dataset,source_manifest_sha256=manifest['manifest_sha256'],
            allocated_ids=ids,support_ids=manifest['support_ids'],fit_ids=fit,validation_ids=sorted(val),
            diagnostic_ids=probe,split_seed=20261101,stratification='five target-pixel-area strata; two validation images per stratum',
            annotation_budget=50,official_test_labels_read=False)
        value['sha256']=signature(value)
        path=root/'splits'/dataset/'split.json'
        if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==value
        else:write_json(path,value)
        assert len(fit)==40 and len(set(val))==10 and not set(fit)&set(val)
        assert support<=set(fit) and not set(ids)&set(manifest['test_ids'])
        splits[dataset]=value
    return splits


@torch.no_grad()
def ifa_pool_audit(root):
    destination=root/'IFA/IFA-0_1_CPU'
    path=destination/'report.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    model,_=build_native(DEFAULT_OUTPUT)
    feature=model.layer3(model.layer2(model.layer1(model.layer0(torch.zeros(1,3,400,400)))))
    shape=tuple(feature.shape[-2:]);del model,feature
    rows=[]
    for dataset in ('IRSTD-1k','NUDT-SIRST','SIRST'):
        manifest=json.loads((DEFAULT_OUTPUT/dataset/'protocol/manifest.json').read_text(encoding='utf-8'))
        for sid in manifest['train_ids']:
            rec=manifest['records'][sid];size=tuple(reversed(rec['size_hw']))
            raw=np.asarray(Image.open(rec['mask']).convert('L').resize(size,Image.Resampling.NEAREST))>0
            x=torch.from_numpy(raw.astype(np.float32))[None,None]
            train=F.interpolate(x,(400,400),mode='nearest')
            infer=torch.from_numpy((np.asarray(Image.fromarray(raw.astype(np.uint8)*255).resize(
                (400,400),Image.Resampling.NEAREST))>0).astype(np.float32))[None,None]
            for route,mask in [('train',train),('inference',infer)]:
                original=F.interpolate(mask,shape,mode='bilinear',align_corners=True)
                area=F.interpolate(mask,shape,mode='area')
                if mask.sum()>0:assert area.sum()>0
                rows.append(dict(dataset=dataset,sample_id=sid,support5=sid in manifest['support_ids'],route=route,
                    input_pixels=int(mask.sum()),original_pool_mass=float(original.sum()),area_pool_mass=float(area.sum())))
    summary=[]
    for dataset in ('IRSTD-1k','NUDT-SIRST','SIRST'):
        for route in ('train','inference'):
            selected=[r for r in rows if r['dataset']==dataset and r['route']==route]
            summary.append(dict(dataset=dataset,route=route,images=len(selected),
                zero_original=sum(r['original_pool_mass']==0 for r in selected),
                zero_area=sum(r['area_pool_mass']==0 for r in selected),
                support5_zero_original=sum(r['original_pool_mass']==0 for r in selected if r['support5']),
                support5_zero_area=sum(r['area_pool_mass']==0 for r in selected if r['support5'])))
    save_csv(destination/'mask_mass.csv',rows)
    result=dict(experiment_ids=['IFA-0','IFA-1'],scope='CPU mask-path diagnostic; not an accuracy result',
                input_size=[400,400],actual_feature_size=list(shape),summary=summary,
                official_test_labels_read=False,status='complete',
                conclusion='Area pooling preserves nonempty foreground mass; feature discrimination and accuracy still require validation.')
    write_json(path,result)
    return result


def process_alive(pid):
    if os.name!='nt':
        return Path(f'/proc/{pid}').exists()
    # Query only. Never use os.kill(pid,0) on Windows.
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        error=ctypes.get_last_error()
        if error in (87,1168):return False
        if error==5:return True
        return False
    code=ctypes.c_uint32()
    try:
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):return True
        return code.value==259
    finally:kernel.CloseHandle(handle)


def build_ir(method,ns_control=False):
    source=DEFAULT_OUTPUT/'sources'/('NS-FPN' if method=='MSHNet+NS-FPN' else method)
    sys.path.insert(0,str(source.resolve()))
    sys.path.append(str((DEFAULT_OUTPUT/'runtime_packages').resolve()))
    if method=='RPCANet':
        from models import get_model
        from utils.data import augmentation
        from utils.loss import SoftLoULoss
        model=RPCANativeOutputs(get_model('rpcanet'));data_class=augmentation
        criterion=lambda p,y,e:SoftLoULoss()(p,y)
    elif method=='ISNet':
        from scripts.comparison_isnet_adapter import enable_torchvision_dcn,NativeISNetOutputs
        enable_torchvision_dcn(DEFAULT_OUTPUT)
        from ISNet import ISNet
        from loss import SoftLoULoss1
        from utils1.dataset_IRSTD1K import SirstDataset
        model=ISNet([4,4,4],[8,16,32,64])
        def initialize(module):
            if isinstance(module,torch.nn.Conv2d):torch.nn.init.normal_(module.weight,0,.02)
            elif isinstance(module,torch.nn.BatchNorm2d):
                torch.nn.init.normal_(module.weight,1,.02);torch.nn.init.normal_(module.bias,0)
        model.apply(initialize);model=NativeISNetOutputs(model);data_class=SirstDataset
        criterion=lambda p,y,e:SoftLoULoss1()(p,y)
    else:
        if method=='MSHNet+NS-FPN':
            from scripts.comparison_nsfpn_adapter import enable_author_reference_backend
            enable_author_reference_backend()
            from model.MSHNet_NSFPN import MSHNet_NSFPN
            model=MSHNet_NSFPN(3)
        else:model=getattr(importlib.import_module(f'model.{method}'),method)(3)
        from model.loss import SLSIoULoss
        from utils.data import IRSTD_Dataset
        if ns_control:
            # Architecture control: preserve the exact NS-FPN criterion for MSHNet.
            import importlib.util as import_util
            spec=import_util.spec_from_file_location('recovery_ns_loss',DEFAULT_OUTPUT/'sources/NS-FPN/model/loss.py')
            module=import_util.module_from_spec(spec);spec.loader.exec_module(module)
            SLSIoULoss=module.SLSIoULoss
        loss=SLSIoULoss();data_class=IRSTD_Dataset
        criterion=lambda p,y,e:loss(p,y,5,e)
    return model,data_class,criterion,source


def train_probe(args):
    root=args.output_root
    out=root/args.method/args.dataset/args.experiment/'111'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():return
    torch.set_num_threads(2)
    random.seed(111);np.random.seed(111);torch.manual_seed(111);torch.cuda.manual_seed_all(111)
    assert torch.cuda.is_available()
    split=json.loads((root/'splits'/args.dataset/'split.json').read_text(encoding='utf-8'))
    manifest=json.loads((DEFAULT_OUTPUT/args.dataset/'protocol/manifest.json').read_text(encoding='utf-8'))
    ids=split['diagnostic_ids'];view=copy.deepcopy(manifest)
    view['train_ids']=ids;view['test_ids']=ids
    model,data_class,criterion,source=build_ir(args.method,args.ns_control)
    dataset_class=RPCAManifestDataset if args.method=='RPCANet' else ManifestDataset
    # Native validation transform, applied only to the eight training images: no augmentation.
    data=dataset_class(view,False,data_class)
    samples=[data[i] for i in range(len(data))]
    assert len(samples)==8
    label_audit=[dict(sample_id=sid,pixels=int(y.sum()),shape=list(y.shape)) for _,y,sid in samples]
    write_json(out/'label_audit.json',label_audit)
    if any(r['pixels']==0 for r in label_audit):
        write_json(out/'report.json',dict(status='needs_input_repair',reason='positive training image lost all foreground after native resize',
                   images=label_audit,official_test_labels_read=False));return
    x=torch.stack([a[0] for a in samples]).cuda();y=torch.stack([a[1] for a in samples]).cuda()
    model=model.cuda()
    optimizer=(torch.optim.Adam(model.parameters(),lr=5e-4*args.lr_scale) if args.method=='RPCANet'
               else torch.optim.Adagrad(model.parameters(),lr=.05*args.lr_scale,weight_decay=1e-4 if args.method=='ISNet' else 0))
    config=dict(method=args.method,dataset=args.dataset,experiment=args.experiment,
        diagnostic_train_ids=ids,annotation_budget=50,training_images=8,validation_labels_read=False,
        official_test_labels_read=False,seed=111,augmentation=False,maximum_updates=1000,maximum_epochs=500,
        selection='diagnostic only; no formal model selection or test evaluation',
        fit_target_IoU=80,input_size=256,batch_size=4,
        schedule='native base learning rate held constant for isolated fit test; not a full-run schedule comparison',
        initialization='random native constructor; no historical train50 checkpoint',split_sha256=split['sha256'],
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        lr_scale=args.lr_scale,initial_lr=optimizer.param_groups[0]['lr'],edge_BCE_weight=args.edge_weight,
        NS_loss_for_MSHNet_control=args.ns_control)
    write_json(out/'config.json',config)
    curve=[];updates=0;best=0.;start_time=time.monotonic()
    for epoch in range(500):
        for offset in (0,4):
            model.train();optimizer.zero_grad(set_to_none=True)
            masks,pred=model(x[offset:offset+4],epoch>5)
            target=y[offset:offset+4];main=criterion(pred,target,epoch);loss=main
            edge_loss=pred.sum()*0;reconstruction=pred.sum()*0
            if args.method=='ISNet':
                from scripts.comparison_isnet_adapter import gradient_magnitude
                edge=model.edge_prediction;edge_target=gradient_magnitude(target).clamp(max=1)
                edge_loss=args.edge_weight*F.binary_cross_entropy(edge,edge_target)+criterion(edge,edge_target,epoch)
                loss=loss+edge_loss
            if args.method=='RPCANet':
                reconstruction=.01*model.reconstruction_loss;loss=loss+reconstruction
            current=target
            for j,mask in enumerate(masks):
                if j:current=F.max_pool2d(current,2,2)
                loss=loss+criterion(mask,current,epoch)
            loss=loss/(len(masks)+1)
            if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss at update {updates}')
            loss.backward()
            gradients=[p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            grad_norm=float(torch.sqrt(sum(g.detach().square().sum() for g in gradients)))
            optimizer.step();updates+=1
            if updates==1 or updates%50==0:
                model.eval();rows=[]
                with torch.no_grad():
                    for i,(_,gt,sid) in enumerate(samples):
                        _,z=model(x[i:i+1],True)
                        rows.append(image_metrics(torch.sigmoid(z)[0,0].cpu().numpy(),gt[0].numpy()>0,sid))
                metric=summarize(rows);best=max(best,metric['IoU_percent'])
                row=dict(update=updates,epoch=epoch+1,loss=float(loss.detach()),main_loss=float(main.detach()),
                         edge_loss=float(edge_loss.detach()),reconstruction_loss=float(reconstruction.detach()),
                         gradient_norm=grad_norm,logit_min=float(pred.detach().min()),logit_max=float(pred.detach().max()),
                         foreground_fraction=(metric['tp']+metric['fp'])/metric['pixels'],
                         train_IoU=metric['IoU_percent'],train_Pd=metric['Pd_percent'],train_Fa=metric['Fa_per_million'])
                curve.append(row);save_csv(out/'training_curve.csv',curve)
                event(out,'fitting',**row)
                if updates%100==0:
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates=updates,config=config),out/'latest.pt')
                if metric['IoU_percent']>=80:
                    write_json(out/'report.json',dict(status='fit_target_reached',best_train_IoU=best,updates=updates,
                        elapsed_seconds=time.monotonic()-start_time,official_test_labels_read=False));return
    write_json(out/'report.json',dict(status='diagnostic_budget_exhausted',best_train_IoU=best,
        final_train_IoU=curve[-1]['train_IoU'],updates=updates,elapsed_seconds=time.monotonic()-start_time,
        official_test_labels_read=False,next_action='Inspect recorded labels, gradients, logits and loss components before full training.'))


def run_queue(args):
    root=args.output_root;root.mkdir(parents=True,exist_ok=True)
    pidfile=root/'queue_pid.txt'
    if pidfile.exists():
        old=int(pidfile.read_text())
        if old!=os.getpid() and process_alive(old):raise RuntimeError(f'Queue already active: {old}')
    pidfile.write_text(str(os.getpid()))
    write_json(root/'authorization.json',dict(date='2026-11-02',request='启动探索',
        scope='initial CPU audit and four serial GPU fit diagnostics',MPA_resume=False))
    control=root/'dispatch_control.json'
    if not control.exists():write_json(control,dict(launch_enabled=True))
    torch.set_num_threads(2)
    event(root,'preparing_splits');create_splits(root)
    event(root,'IFA_CPU_mask_audit');audit=ifa_pool_audit(root)
    event(root,'IFA_CPU_complete',summary=audit['summary'])
    if args.wait_pid:
        while process_alive(args.wait_pid):
            event(root,'waiting_for_existing_GPU_process',wait_pid=args.wait_pid)
            time.sleep(30)
    results=[]
    for method,dataset,experiment in JOBS:
        while not json.loads(control.read_text(encoding='utf-8')).get('launch_enabled',False):
            event(root,'dispatch_disabled');time.sleep(30)
        job=root/method/dataset/experiment/'111';job.mkdir(parents=True,exist_ok=True)
        if (job/'report.json').exists():
            results.append(dict(method=method,experiment=experiment,report=str(job/'report.json')));continue
        event(root,'running_GPU_diagnostic',method=method,dataset=dataset,experiment=experiment)
        command=[sys.executable,'-u','-m','scripts.run_comparison_recovery_exploration','--worker',
                 '--method',method,'--dataset',dataset,'--experiment',experiment,'--output-root',str(root)]
        with (job/'worker.log').open('a',encoding='utf-8') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            write_json(root/'active_job.json',dict(pid=process.pid,method=method,experiment=experiment,command=command))
            code=process.wait()
        if code:
            write_json(job/'report.json',dict(status='failed',exit_code=code,log=str(job/'worker.log'),
                       official_test_labels_read=False))
        results.append(dict(method=method,experiment=experiment,report=str(job/'report.json'),exit_code=code))
        write_json(root/'batch_results.json',results)
    event(root,'initial_batch_complete',results=results,next='Review diagnostic reports and choose the next single-factor branch under the authorized plan; no official test evaluation.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--method');parser.add_argument('--dataset');parser.add_argument('--experiment')
    parser.add_argument('--lr-scale',type=float,default=1.)
    parser.add_argument('--edge-weight',type=float,default=10.)
    parser.add_argument('--ns-control',action='store_true')
    args=parser.parse_args()
    try:
        if args.worker:train_probe(args)
        else:run_queue(args)
    except Exception:
        print(traceback.format_exc(),flush=True)
        if not args.worker:event(args.output_root,'failed',error=traceback.format_exc())
        raise


if __name__=='__main__':main()
