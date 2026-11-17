"""Single-seed three-domain P2P memory + axial polar residual comparison."""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import yaml

from visual_prompt.core.candidate_proposals import connected_instances,greedy_score_peaks
from visual_prompt.evaluation.comparison import image_metrics,summarize
from visual_prompt.models.center_instance_loss import center_instance_loss
from visual_prompt.models.p2p_polar_mask import P2PPolarMask
from visual_prompt.models.polar_gradient_mask import polar_derivatives
from visual_prompt.pipelines.p2p_framework import (
    P2PModel,ROOT,RETAINED,load,crop_at,local_sam,normalized_feature,
    mask_descriptor,mask_forward,normalize_f4,paste_local,describe_full_image_points,DESCRIPTOR_NAMES,
)
from visual_prompt.pipelines.semantic_prompt_pipeline import get_sim_features,seed_everything

CONFIG=ROOT/'experiments/configs/objectness_mask/p2p_memory_polar_seed111.yaml'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def status(out,stage,**kw):
    value=dict(stage=stage,time=time.time(),**kw);write(out/'status.json',value);print(json.dumps(dict(dataset=out.name,**value)),flush=True)
def read_counts(path):
    from scripts.run_p2p_framework import read_counts as read
    return read(path)

def working(record):
    image=np.array(Image.open(record['image']).convert('RGB').resize((512,512),Image.Resampling.BILINEAR),copy=True)
    gt=(np.asarray(Image.open(record['mask']).convert('L').resize((512,512),Image.Resampling.NEAREST))>=127).astype(np.uint8)
    return image,gt

@torch.no_grad()
def memory_scores_once(model,image):
    """Same scalar and multi-key reductions, reusing each frozen window encoding."""
    from visual_prompt.evaluation import eval_sliding_window_semantic_mask as ev
    args=model.args;h,w=image.shape[:2];step=round(args.window_size*(1-args.overlap))
    total=np.zeros((len(model.memory['keys']),h,w),np.float32);canonical=np.zeros((h,w),np.float32);weight=np.zeros((h,w),np.float32)
    keys=model.memory['keys'].to(model.device);dummy=np.zeros((h,w),np.uint8);p=model.pred()
    for y in ev.sliding_starts(h,args.window_size,step):
        for x in ev.sliding_starts(w,args.window_size,step):
            ev.encode_crop_batch(p,[ev.crop_with_padding(image,dummy,y,x,args.window_size)[0]])
            feature=get_sim_features(p).float()
            single=ev.compute_sim_batch_from_features(p,model.e0,feature,args.window_sim_size)[0].cpu().numpy().astype(np.float32)
            raw=torch.einsum('kc,bchw->bkhw',keys,F.normalize(feature,dim=1))
            if raw.shape[-2:]!=(args.window_sim_size,args.window_sim_size):raw=F.interpolate(raw,size=(args.window_sim_size,args.window_sim_size),mode='bilinear',align_corners=False)
            patches=p.postprocess_masks(raw,p.input_size,p.original_size)[0].cpu().numpy()
            if single.shape!=(args.window_size,args.window_size):single=cv2.resize(single,(args.window_size,args.window_size),interpolation=cv2.INTER_LINEAR)
            canonical[y:y+args.window_size,x:x+args.window_size]+=single
            for k,patch in enumerate(patches):
                if patch.shape!=(args.window_size,args.window_size):patch=cv2.resize(patch,(args.window_size,args.window_size),interpolation=cv2.INTER_LINEAR)
                total[k,y:y+args.window_size,x:x+args.window_size]+=patch
            weight[y:y+args.window_size,x:x+args.window_size]+=1
    canonical/=weight.clip(1e-6);total/=weight.clip(1e-6)[None];total[0]=canonical
    return model.reader.score_map(torch.from_numpy(total).to(model.device)).cpu().numpy(),canonical

class Study:
    def __init__(self,domain,variant,cfg):
        self.cfg=cfg;self.domain=domain;self.variant=variant;self.out=ROOT/cfg['output']/domain;self.out.mkdir(parents=True,exist_ok=True)
        self.m=json.loads((RETAINED/domain/'manifest.json').read_text());self.model=P2PModel(domain,variant);self.device=self.model.device
        self.coord=json.loads((RETAINED/domain/variant/'training_coordinates.json').read_text())
        assert set(r['sample_id'] for r in self.coord)<=set(self.m['train_ids'])
        assert not set(self.m['train_ids'])&set(self.m['test_ids'])
        self.e0=self.model.e0
        protocol=dict(dataset=domain,base_variant=variant,config=cfg,baseline_sha256=sha(RETAINED/domain/variant/'model.pt'),manifest_sha256=sha(RETAINED/domain/'manifest.json'),pool_sha256=sha(RETAINED/domain/variant/'training_coordinates.json'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__).resolve(),ROOT/'visual_prompt/models/p2p_polar_mask.py',ROOT/'visual_prompt/models/polar_gradient_mask.py',ROOT/'visual_prompt/models/center_instance_loss.py')},frozen=['SAM','memory','reader','Obj','e0'],training='fresh Mask; same original fixed120 AdamW/loss/sample schedule; gain and polar additions only',selection='fixed epoch120; no test epoch/config selection',IRSTD='center-biased Mask pool; retained internally-offset Obj stays frozen')
        path=self.out/'protocol.json'
        if path.exists():assert json.loads(path.read_text())==protocol,'Protocol changed; use a new output directory'
        else:write(path,protocol)

    def open_bank(self):
        cache=self.out/'cache';self.base=np.load(cache/'base.npy',mmap_mode='r');self.raw=np.load(cache/'raw.npy',mmap_mode='r');self.rv=np.load(cache/'radial_valid.npy',mmap_mode='r')
        with np.load(cache/'labels.npz') as d:self.desc=d['descriptor'];self.target=d['target'];self.valid=d['valid'];self.positive=d['positive']
        self.stats=load(self.out/'gradient_statistics.pt')

    @torch.no_grad()
    def prepare(self):
        marker=self.out/'cache/complete.json'
        if marker.exists():self.open_bank();return
        cache=marker.parent;cache.mkdir(parents=True,exist_ok=True);n=len(self.coord)
        base=np.lib.format.open_memmap(cache/'base.npy',mode='w+',dtype=np.float16,shape=(n,416,64,64))
        raw=np.lib.format.open_memmap(cache/'raw.npy',mode='w+',dtype=np.float16,shape=(n,64,64,64));rv=np.lib.format.open_memmap(cache/'radial_valid.npy',mode='w+',dtype=np.bool_,shape=(n,64,64,64))
        descriptor=np.empty((n,836),np.float32);target=np.empty((n,64,64),np.uint8);valid=np.empty_like(target,dtype=np.bool_);positive=np.asarray([bool(r['positive']) for r in self.coord])
        sums=torch.zeros(5,8,8,device=self.device,dtype=torch.float64);squares=sums.clone();counts=sums.clone();last=None
        for i,r in enumerate(self.coord):
            sid,y,x=r['sample_id'],r['y'],r['x']
            if sid!=last:image,gt=working(self.m['records'][sid]);inst=connected_instances(gt);last=sid
            sam=local_sam(self.model.pred(),image,y,x)[0].cpu().half().numpy();ray,ray_valid=self.model.radial.sample(image,y,x)
            d=dict(sam=sam,ray=ray.numpy(),ray_valid=ray_valid.numpy());base[i]=normalized_feature(d,self.model.ck['radial_mean'].numpy(),self.model.ck['radial_std'].numpy());raw[i]=d['ray'];rv[i]=d['ray_valid']
            f=torch.from_numpy(np.array(base[i],dtype=np.float32,copy=True))[None];descriptor[i]=mask_descriptor(f,f[:,:352],self.e0.cpu())[0].numpy()
            iid=int(inst.labels[y,x]);assert bool(iid)==bool(r['positive'])
            _,target[i]=crop_at(image,(inst.labels==iid).astype(np.uint8) if iid else np.zeros_like(gt),y,x)
            _,all_gt=crop_at(image,gt,y,x);valid[i]=(all_gt==0)|(target[i]>0)
            fields,masks=polar_derivatives(ray[None].to(self.device,dtype=torch.float32),ray_valid[None].to(self.device));v=fields.double();mask=masks.double()
            sums+=(v*mask).sum((0,4,5));squares+=(v.square()*mask).sum((0,4,5));counts+=mask.sum((0,4,5))
            if (i+1)%250==0:status(self.out,'training_features',crops=i+1,total=n)
        for a in (base,raw,rv):a.flush()
        mean=sums/counts.clamp_min(1);std=(squares/counts.clamp_min(1)-mean.square()).clamp_min(0).sqrt();std=torch.maximum(std,std.new_tensor([.01,.001,.001,.001,.001])[:,None,None]);mean=mean.float().reshape(5,64);std=std.float().reshape(5,64)
        mean[0]=self.model.ck['radial_mean'].to(self.device);std[0]=self.model.ck['radial_std'].to(self.device)
        torch.save(dict(mean=mean.cpu(),std=std.cpu(),count=counts.cpu(),train_ids=self.m['train_ids']),self.out/'gradient_statistics.pt')
        np.savez(cache/'labels.npz',descriptor=descriptor,target=target,valid=valid,positive=positive)
        write(marker,dict(crops=n,positive=int(positive.sum()),negative=int((~positive).sum()),only_train50=True,ignore_other_instances=True,stats_source='matched train ROI pool only'))
        for a in (base,raw,rv):a._mmap.close()
        self.open_bank();status(self.out,'training_features_complete',crops=n)

    def factory(self,mode):
        seed_everything(self.cfg['seed'])
        return P2PPolarMask(self.e0.cpu(),mode,self.stats['mean'],self.stats['std']).to(self.device)

    def batch(self,indices):
        return (torch.from_numpy(np.array(self.base[indices],copy=True)).to(self.device,dtype=torch.float32),torch.from_numpy(self.desc[indices]).to(self.device),torch.from_numpy(np.array(self.raw[indices],copy=True)).to(self.device,dtype=torch.float32),torch.from_numpy(np.array(self.rv[indices],copy=True)).to(self.device))

    def preflight(self):
        fixed,gain,polar=[self.factory(mode) for mode in ('fixed','gain','polar')]
        ix=np.flatnonzero(self.positive)[:2].tolist()+np.flatnonzero(~self.positive)[:2].tolist();args=self.batch(ix)
        with torch.no_grad():
            reference=fixed(args[0],args[1],self.e0,args[2],args[3]);errors={m:float((head(args[0],args[1],self.e0,args[2],args[3])-reference).abs().max()) for m,head in [('gain',gain),('polar',polar)]}
        assert all(v==0 for v in errors.values()),errors
        opt=torch.optim.AdamW(polar.parameters(),lr=.001,weight_decay=.0001)
        t=torch.from_numpy(self.target[ix]).to(self.device).float();v=torch.from_numpy(self.valid[ix]).to(self.device);p=torch.from_numpy(self.positive[ix]).to(self.device)
        for _ in range(6):opt.zero_grad();loss=center_instance_loss(polar(args[0],args[1],self.e0,args[2],args[3]),t,v,p);loss.backward();opt.step()
        gradients={key:sum(float(x.grad.abs().sum()) for x in module.parameters() if x.grad is not None) for key,module in [('radial',polar.encoder.radial),('angular',polar.encoder.angular),('fusion',polar.encoder.fuse),('Mask',polar.net)]};gradients['gain']=float(polar.log_gain.grad.abs().sum());assert all(np.isfinite(x) and x>0 for x in gradients.values()),gradients
        write(self.out/'preflight.json',dict(passed=True,initial_logit_errors=errors,gradient_l1=gradients,residual_parameters=sum(x.numel() for x in polar.encoder.parameters()),discarded_preflight_model=True))
        del fixed,gain,polar,opt,args;gc.collect();torch.cuda.empty_cache()

    def train(self,mode):
        out=self.out/mode;out.mkdir(exist_ok=True);final=out/'model.pt'
        if final.exists():return
        head=self.factory(mode).train();optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
        pos=np.flatnonzero(self.positive).tolist();neg=np.flatnonzero(~self.positive).tolist();pr,nr=random.Random(111),random.Random(112);schedule=[];history=[];start=0
        resume=out/'resume.pt'
        if resume.exists():
            ck=load(resume);head.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);pr.setstate(ck['pr']);nr.setstate(ck['nr']);schedule=ck['schedule'];history=ck['history'];start=ck['epoch']
        for epoch in range(start,120):
            p,n=pr.choices(pos,k=128),nr.choices(neg,k=128);schedule.append(p+n);losses=[]
            for j in range(0,128,8):
                ix=p[j:j+8]+n[j:j+8];f,d,r,rv=self.batch(ix)
                z=head(f,d,self.e0,r,rv);t=torch.from_numpy(self.target[ix]).to(self.device).float();v=torch.from_numpy(self.valid[ix]).to(self.device);positive=torch.from_numpy(self.positive[ix]).to(self.device)
                loss=center_instance_loss(z,t,v,positive);assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
            if (epoch+1)%10==0:
                status(self.out,'training',mode=mode,**history[-1]);torch.save(dict(model=head.state_dict(),optimizer=optimizer.state_dict(),pr=pr.getstate(),nr=nr.getstate(),schedule=schedule,history=history,epoch=epoch+1),resume)
        state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};torch.save(dict(mode=mode,model=state,epoch=120,seed=111,base_variant=self.variant,gradient_statistics=self.stats,sampling_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),baseline_mask_max_weight_difference=max(float((state[k]-v).abs().max()) for k,v in self.model.ck['mask'].items()),protocol_sha256=sha(self.out/'protocol.json')),final)
        write(out/'history.json',history);resume.unlink(missing_ok=True)
        del head,optimizer,f,d,r,rv;gc.collect();torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(self):
        arms={mode:self.factory(mode).eval() for mode in self.cfg['arms']}
        for mode,head in arms.items():head.load_state_dict(load(self.out/mode/'model.pt')['model'])
        rows={mode:[] for mode in ['retained']+list(arms)};expected={r['sample_id']:r for r in read_counts(RETAINED/self.domain/self.variant/'per_image.csv')};mismatch=[]
        for number,sid in enumerate(self.m['test_ids']):
            path=self.out/'evaluation_counts'/f'{sid}.json'
            if path.exists():
                value=json.loads(path.read_text());[rows[k].append(value[k]) for k in rows];continue
            record=self.m['records'][sid];image,_=working(record);score,canonical=memory_scores_once(self.model,image)
            if number==0:
                reference,c=self.model.scores(image);assert np.array_equal(reference,score) and np.array_equal(c,canonical),'one-pass scores differ'
                write(self.out/'score_equivalence.json',dict(passed=True,canonical_exact=True,memory_exact=True))
            points=[(y,x) for y,x,_ in greedy_score_peaks(score,5,7)];desc=describe_full_image_points(self.model.pred(),image,canonical,points,64,self.model.args.local_contrast_sigma,32)
            raw=np.concatenate([desc[k] for k in DESCRIPTOR_NAMES],1);probs=self.model.obj(torch.from_numpy(normalize_f4(raw,self.model.ck['statistics'])).to(self.device)).sigmoid().cpu().tolist()
            canvases={mode:np.zeros((512,512),np.uint8) for mode in rows}
            for (y,x),p in zip(points,probs):
                if p<.1:continue
                sam=local_sam(self.model.pred(),image,y,x)[0].cpu().half().numpy();ray,valid=self.model.radial.sample(image,y,x)
                d=dict(sam=sam,ray=ray.numpy(),ray_valid=valid.numpy());f=normalized_feature(d,self.model.ck['radial_mean'].numpy(),self.model.ck['radial_std'].numpy())
                # Baseline caches computed descriptors on CPU, including fixed statistics.
                fc=torch.from_numpy(f.astype(np.float32))[None];descriptor=mask_descriptor(fc,fc[:,:352],self.e0.cpu()).to(self.device);f=fc.to(self.device);r=ray[None].to(self.device,dtype=torch.float32);rv=valid[None].to(self.device)
                z=mask_forward(self.model.mask,f,mask_descriptor(f,f[:,:352],self.e0),self.e0);paste_local(canvases['retained'],(z[0]>0).cpu().numpy().astype(np.uint8),y,x)
                for mode,head in arms.items():paste_local(canvases[mode],(head(f,descriptor,self.e0,r,rv)[0]>0).cpu().numpy().astype(np.uint8),y,x)
            gt=np.asarray(Image.open(record['mask']).convert('L'))>0;value={}
            for mode,canvas in canvases.items():value[mode]=image_metrics(self.model.restore(canvas,gt.shape),gt,sid);rows[mode].append(value[mode])
            keys=('tp','fp','fn','pixels','targets','predictions','detected','unmatched_pixels');diff={k:[expected[sid][k],value['retained'][k]] for k in keys if expected[sid][k]!=value['retained'][k]}
            if diff:mismatch.append(dict(sample_id=sid,differences=diff))
            write(path,value)
            if (number+1)%20==0:status(self.out,'evaluation',images=number+1,total=len(self.m['test_ids']))
        results={}
        for mode,values in rows.items():
            results[mode]=summarize(values);csv_write(self.out/mode/'per_image.csv',values);write(self.out/mode/'metrics.json',results[mode])
        original=json.loads((RETAINED/self.domain/self.variant/'metrics.json').read_text())
        exact=all(abs(original[k]-results['retained'][k])<1e-9 for k in ('IoU_percent','nIoU_percent','F1_percent','Pd_percent','Fa_per_million'))
        schedules=[load(self.out/mode/'model.pt')['sampling_sha256'] for mode in arms];assert len(set(schedules))==1
        write(self.out/'audit.json',dict(passed=exact,baseline_replay_exact=exact,mismatches=mismatch,paired_sampling=True,sampling_sha256=schedules[0],test_images=len(self.m['test_ids']),seed=111,epochs=120,stats_train_only=True))
        assert exact,('retained baseline replay mismatch',self.domain,mismatch)
        write(self.out/'results.json',results);status(self.out,'complete',results=results)

    def run(self):
        self.prepare();self.preflight()
        for mode in self.cfg['arms']:self.train(mode)
        self.evaluate()

def report(cfg):
    out=ROOT/cfg['output'];records=[];labels={'retained':'原保留记忆模型','fixed':'同协议重训fixed64','gain':'gain64','polar':'gain64＋双轴梯度/对向残差'}
    lines=['# 三域记忆链路＋双轴梯度/对向残差','', '每域仅seed111，冻结记忆/检索/Obj，保持各域原Mask采样、损失、120轮固定AdamW及Top5。仅替换Mask径向模块；本轮各组均从头训练，固定比较120轮。','', '| 数据集 | 设置 | mIoU% | nIoU% | F1% | Pd% | FA/百万↓ |','|---|---|---:|---:|---:|---:|---:|']
    for domain,variant in cfg['domains'].items():
        p=out/domain/'results.json'
        if not p.exists():continue
        values=json.loads(p.read_text());records.append(dict(dataset=domain,variant=variant,results=values,audit=json.loads((out/domain/'audit.json').read_text())))
        for mode in ('retained','fixed','gain','polar'):
            m=values[mode];lines.append('| '+domain+' | '+labels[mode]+' | '+' | '.join(f'{m[k]:.2f}' for k in ('IoU_percent','nIoU_percent','F1_percent','Pd_percent','Fa_per_million'))+' |')
    lines+=['','gain相对fixed检验可学习增益，polar相对gain检验新增双轴/对向残差；polar相对原保留模型衡量接入现有完整链路的总变化。所有结果均为单种子；不据测试结果挑轮次或覆盖原最佳权重。']
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');write(out/'results.json',records)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['SIRST','NUDT-SIRST','IRSTD-1k']);p.add_argument('--report',action='store_true');a=p.parse_args();torch.set_num_threads(2)
    cfg=yaml.safe_load(CONFIG.read_text(encoding='utf-8'));out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    if a.report:report(cfg);return
    for domain,variant in cfg['domains'].items():
        if a.dataset and a.dataset!=domain:continue
        if (out/domain/'results.json').exists():continue
        study=Study(domain,variant,cfg)
        try:study.run()
        except BaseException:status(study.out,'failed',traceback=traceback.format_exc());raise
        del study;gc.collect();torch.cuda.empty_cache();report(cfg)
    report(cfg)

if __name__=='__main__':main()
