"""NUDT→IRSTD label-budget pilot. Each shot starts from the same source checkpoint."""
import argparse
import gc
import json
import random
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from scripts import run_motivation_current_ours as own
from scripts import run_background_replacement_baselines as ir
from visual_prompt.evaluation.comparison import image_metrics, summarize
from visual_prompt.models.tiny_shared_memory import (
    descriptor, pool_feature_token, mean_token, build_score_token, SemanticDirectionConfig,
)

bg = own.bg
OUT = bg.ROOT/'outputs/nudt_irstd_shot_curve_20261112'
SOURCE, TARGET = 'NUDT-SIRST', 'IRSTD-1k'
METHODS = [*bg.METHODS, 'Ours']
SHOTS = (0,1,5,10,50)
STEPS = 100
SEED = 111


def setup():
    m = bg.manifest(TARGET)
    order = sorted(m['train_ids']); random.Random(SEED).shuffle(order)
    assert len(order)==len(set(order))==50 and not set(order)&set(m['test_ids'])
    checkpoints = {}
    for method in METHODS:
        if method=='Ours':
            r=own.selected()[SOURCE]; p=bg.ROOT/r['checkpoint']; sha=r['sha256']; cfg=None
        else:
            p,cfg,meta=bg.checkpoint(method,SOURCE)
            locked=own.read_json(bg.OUT/'protocol.json')['checkpoints']
            r=next(r for r in locked if r['method']==method and r['domain']==SOURCE); sha=r['sha256']
            assert all(r.get(k)==v for k,v in meta.items() if k in ('epoch','updates'))
        assert bg.digest(p)==sha
        checkpoints[method]=dict(path=str(p),sha256=sha,config=str(cfg) if cfg else None)
    protocol=dict(source=SOURCE,target=TARGET,seed=SEED,steps=STEPS,batch=4,lr=1e-4,shot_ids={str(k):order[:k] for k in SHOTS},
        test_ids=m['test_ids'],checkpoints=checkpoints,threshold=.5,shots='nested prefixes of a fixed shuffle of all 50 target training IDs',
        ours_training='one labeled target query per episode; distinct target support drawn across all K at K>=2; at K=1 use a source training support and the single target query',
        ours_evaluation='0: frozen source five supports; K>0: all K target supports contribute to memory (fixed 19 slots). Official target test images are queries only.',
        baseline_training='full model, frozen BatchNorm statistics, AdamW with shared foreground-balanced BCE plus SoftIoU; effective batch 4',
        adaptation='all positive shots independently reset to their method’s source checkpoint; fixed number of updates, no target test selection',
        baseline_versions='NUDT source DNANet 500, ILNet and SCTransNet 125, RPCANet main-table adapted 1000 updates. All 0-shot values recomputed from those exact same weights.',
        runner_sha256=bg.digest(__file__))
    path=OUT/'protocol.json'
    if path.exists(): assert own.read_json(path)==protocol, 'Existing run has a different locked protocol'
    else:bg.dump(path,protocol)
    return protocol


def sample_stream(ids):
    rng=random.Random(SEED)
    while True:
        a=list(ids);rng.shuffle(a)
        yield from a


@torch.no_grad()
def target_memory(e, pairs, ids):
    # Preserve the original key builder's pooling and slot selection while
    # encoding in small chunks; all K images enter the global keys.
    positives, image_tokens = [], []
    negatives = {'high_freq':[], 'bright_flat':[]}
    for offset in range(0,len(ids),4):
        chunk=ids[offset:offset+4]
        bk=e.backbone(torch.stack([pairs[i][0] for i in chunk])[None])
        for i,sid in enumerate(chunk):
            feature=descriptor(e.model,bk,i)[0].permute(1,2,0)
            region=pairs[sid][2]
            local=[pool_feature_token(feature,x.to(feature.device),'mean_max',True) for x in region['instances']]
            positives.extend(local);image_tokens.append(mean_token(local))
            for name,x in zip(negatives,region['negatives']):
                token=pool_feature_token(feature,x.to(feature.device),'mean',True)
                if token is not None:negatives[name].append(token)
        del bk
    cfg=SemanticDirectionConfig(); corrected,_=build_score_token(image_tokens,negatives,cfg)
    keys=corrected.new_zeros(19,352);keys[0]=mean_token(image_tokens)[0];keys[1]=corrected[0]
    for i,p in enumerate(positives[:8],2):keys[i]=build_score_token([p],negatives,cfg)[0][0]
    for i,p in enumerate([p for v in negatives.values() for p in v][:9],10):keys[i]=p[0]
    return e.model.prototype_memory.prepare(keys.detach())


def train_ours(e, ids, steps):
    from util.promptable_utils import build_prompt_dict
    from util.losses import loss_masks
    pairs=e.pairs(TARGET,ids)
    source=e.pairs(SOURCE,bg.manifest(SOURCE)['train_ids']) if len(ids)==1 else None
    # Cyclic pairing allows every target image to participate on both sides,
    # without ever presenting its own label as its support mask.
    queries=sample_stream(ids); supports=sample_stream(source) if source else None
    support_map={q:ids[(i+1)%len(ids)] for i,q in enumerate(ids)}
    params=[p for p in e.model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.)
    e.model.train();e.cache=None; seen_q=set(); seen_s=set()
    for step in range(steps):
        batch=[]
        for _ in range(4):
            q=next(queries);s=next(supports) if source else support_map[q]
            assert source is not None or s!=q
            batch.append(((source or pairs)[s],pairs[q]));seen_q.add(q);seen_s.add(s)
        opt.zero_grad(set_to_none=True)
        for i in (0,2):
            part=batch[i:i+2]
            x=torch.stack([torch.stack((s[0],q[0])) for s,q in part]).cuda()
            y=torch.stack([torch.stack((s[1],q[1])) for s,q in part]).cuda()
            e.model.support_geometry=[[s[2]] for s,_ in part]
            branch=e.model.radial_branch;branch.begin([(None,None)]*2)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prompt=build_prompt_dict(y,'mask',1,True,torch.device('cuda'))
                pred=e.model(x,prompt)['pred_masks'];branch.finish()
                loss=sum(loss_masks(pred,y[:,-1:],num_frames=1).values())*.5
            assert torch.isfinite(loss)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params,.1,error_if_nonfinite=True);opt.step()
        if (step+1)%20==0: own.log(event='update',method='Ours',shot=len(ids),step=step+1)
    if steps==STEPS:
        assert seen_q==set(ids)
        assert seen_s==set(source if source else ids)
    e.model.eval();e.cache=target_memory(e,pairs,ids)
    return seen_q,seen_s


def train_ir(model, method, cfg, ids, steps):
    m=bg.manifest(TARGET); pairs={}
    for i in ids:
        im,gt,_=bg.read(TARGET,i); x=ir.transform(im,method,cfg,m['records'][i]['image'])
        h,w=x.shape[-2:]
        y=cv2.resize(gt,(w,h),interpolation=cv2.INTER_NEAREST_EXACT)
        pairs[i]=(x,torch.from_numpy(y[None]).float())
    model.train()
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.)
    stream=sample_stream(ids);seen=set()
    for step in range(steps):
        batch=[next(stream) for _ in range(4)];seen.update(batch)
        opt.zero_grad(set_to_none=True)
        micro=1 if method=='ILNet' else 2
        for i in range(0,4,micro):
            part=batch[i:i+micro]
            x=torch.stack([pairs[s][0] for s in part]).cuda()
            y=torch.stack([pairs[s][1] for s in part]).cuda()
            p=model(x).float();assert p.shape==y.shape
            # Balanced loss prevents a tiny target from being overwhelmed by
            # the background during these short, unaugmented runs.
            q=p.clamp(1e-6,1-1e-6)
            dims=(1,2,3);fg=y.sum(dims);n=(1-y).sum(dims)
            pos=-(y*q.log()).sum(dims)/fg.clamp_min(1)
            neg=-((1-y)*torch.log1p(-q)).sum(dims)/n.clamp_min(1)
            overlap=(p*y).sum(dims); union=(p+y-p*y).sum(dims)
            loss=(torch.where(fg>0,.5*(pos+neg),neg)+1-(overlap+1)/(union+1)).mean()*(micro/4)
            assert torch.isfinite(loss)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if (step+1)%20==0:own.log(event='update',method=method,shot=len(ids),step=step+1)
    if steps==STEPS:assert seen==set(ids)
    model.eval();return seen,set()


def wait_gpu():
    while True:
        line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]
        used,util=map(int,line.split(','))
        if used<1100 and util<15:return
        own.log(event='waiting_for_gpu',used_mib=used,utilization=util)
        time.sleep(30)


def worker(method, protocol, smoke=False):
    m=bg.manifest(TARGET); info=protocol['checkpoints'][method]
    k_values=(1,50) if smoke else SHOTS
    for k in k_values:
        dest=OUT/('smoke' if smoke else 'results')/method/f'shot{k:02d}'
        if (dest/'summary.json').exists():continue
        own.base.seed_all()
        ids=protocol['shot_ids'][str(k)];steps=(1 if smoke else STEPS) if k else 0
        if method=='Ours':
            e=own.Engine(SOURCE,own.selected()[SOURCE]); model=e.model
            predict=lambda im,sid:e.predict(im)
        else:
            cfg=json.loads(open(info['config'],encoding='utf-8').read());model=ir.build(method).cuda()
            state=torch.load(info['path'],map_location='cpu',weights_only=False)
            model.load_state_dict(state['model'],strict=True);del state;model.eval()
            def predict(im,sid):
                x=ir.transform(im,method,cfg,m['records'][sid]['image'])
                p=model(x[None].cuda())[0,0].cpu().numpy()
                return cv2.resize(p,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_LINEAR)
        own.log(event='begin',method=method,shot=k,steps=steps,smoke=smoke)
        start=time.perf_counter();q,s=set(),set()
        if k:
            q,s=train_ours(e,ids,steps) if method=='Ours' else train_ir(model,method,cfg,ids,steps)
        torch.cuda.synchronize(); adaptation_seconds=time.perf_counter()-start if k else 0.
        rows=[];test_ids=protocol['test_ids'][:3] if smoke else protocol['test_ids']
        with torch.no_grad():
            for sid in test_ids:
                im,gt,_=bg.read(TARGET,sid)
                rows.append(image_metrics(predict(im,sid),gt,sid))
        assert [r['sample_id'] for r in rows]==test_ids
        own.save_csv(dest/'per_image.csv',rows)
        value=dict(method=method,shot=k,steps=steps,source_sha256=info['sha256'],train_ids=ids,queries_seen=sorted(q),support_ids_seen=sorted(s),
                   adaptation_seconds=adaptation_seconds,**summarize(rows))
        bg.dump(dest/'summary.json',value);own.log(event='done',method=method,shot=k,iou=value['IoU_percent'],smoke=smoke)
        del predict,model
        if method=='Ours':del e
        gc.collect();torch.cuda.empty_cache()


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=own.read_json(OUT/'protocol.json');rows=[]
    fig,ax=plt.subplots(figsize=(7.4,4.6));plt.rcParams['pdf.fonttype']=42
    for method in METHODS:
        points=[]
        for k in SHOTS:
            path=OUT/'results'/method/f'shot{k:02d}'/'summary.json'
            if not path.exists():continue
            r=own.read_json(path)
            assert r['source_sha256']==p['checkpoints'][method]['sha256'] and r['train_ids']==p['shot_ids'][str(k)]
            assert r['images']==len(p['test_ids']) and r['steps']==(STEPS if k else 0)
            if k:assert set(r['queries_seen'])==set(r['train_ids'])
            points.append(r);rows.append({a:r[a] for a in ('method','shot','IoU_percent','nIoU_percent','Pd_percent','Fa_per_million','adaptation_seconds')})
        if points:
            ax.plot([SHOTS.index(r['shot']) for r in points],[r['IoU_percent'] for r in points],marker='o',label=method,lw=2.5 if method=='Ours' else 1.6)
    if not rows:return
    ax.set(xticks=range(len(SHOTS)),xticklabels=SHOTS,xlabel='Labeled IRSTD training images (shots)',ylabel='IRSTD test IoU (%)',title='NUDT → IRSTD few-shot adaptation')
    ax.grid(alpha=.2);ax.spines[['right','top']].set_visible(False);ax.legend(ncol=2,frameon=False)
    fig.text(.14,.012,'Nested target splits (seed 111); 100 updates per nonzero budget.',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1))
    for ext in ('png','pdf'):fig.savefig(OUT/f'shot_curve.{ext}',dpi=200)
    plt.close(fig);own.save_csv(OUT/'shot_curve.csv',rows)
    if len(rows)==len(METHODS)*len(SHOTS):bg.dump(OUT/'complete.json',dict(status='complete',cells=len(rows)))


def main():
    a=argparse.ArgumentParser();a.add_argument('--method',choices=METHODS);a.add_argument('--smoke',action='store_true');a.add_argument('--preflight',action='store_true');a.add_argument('--render',action='store_true');args=a.parse_args()
    if args.render:render();return
    p=setup()
    if args.preflight:print('Verified five source checkpoints and five nested target splits');return
    if args.method:worker(args.method,p,args.smoke);return
    for method in METHODS:
        wait_gpu()
        for smoke in (True,False):
            if args.smoke and not smoke:break
            subprocess.run([sys.executable,'-m','scripts.run_nudt_irstd_shot_curve','--method',method,*(['--smoke'] if smoke else [])],cwd=bg.ROOT,check=True)
        if not args.smoke:render()


if __name__=='__main__':main()
