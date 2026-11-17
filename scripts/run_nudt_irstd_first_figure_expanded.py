"""Add five recent IRSTD controls to the paired NUDT→IRSTD shot curves."""
import argparse
import gc
import subprocess
import sys

import cv2
import torch

from scripts import run_nudt_irstd_multimethod_retention as prev

bg,own,curve=prev.bg,prev.own,prev.curve
ROOT=bg.ROOT
OUT=curve.OUT/'first_figure_ten_methods'
NEW=('HDNet','SAM-SPL','SAMamba','MSHNet+NS-FPN','IR-SAM2')
YEARS={'HDNet':2025,'SAM-SPL':2025,'SAMamba':2025,'MSHNet+NS-FPN':2026,'IR-SAM2':2026}
TASK=ROOT/'outputs/comparison_task_adaptation_20261104'
RECENT=ROOT/'outputs/comparison_recent_a_20261104'
SAM=ROOT/'outputs/comparison_sam_train50_20261109'


def files(method):
    if method in ('HDNet','MSHNet+NS-FPN'):
        d=TASK/method/curve.SOURCE/'full50'
        return d/'model.pt',d/'config.json',d/'metrics/summary.json',1000
    d=(RECENT if method=='SAM-SPL' else SAM)/curve.SOURCE/method
    return d/'checkpoints/epoch_0120.pt',d/'config.json',d/'epoch_0120/metrics/summary.json',120


def setup():
    p,source,target=prev.setup()
    audit=own.read_json(prev.OUT/'audit.json');assert audit['status']=='passed' and audit['protocol_sha256']==bg.digest(curve.OUT/'protocol.json')
    candidates={}
    for method in NEW:
        path,cfg,summary,horizon=files(method);s=own.read_json(summary);c=own.read_json(cfg)
        assert s['images']==664 and s['tp']>0 and s['pixels']-s['tp']-s['fp']-s['fn']>0
        assert s['IoU_percent']>=40 and s['Fa_per_million']<500, (method,'unexpected source-health record')
        assert c.get('train_annotation_ids',c.get('training_ids'))==source['train_ids']
        sha=bg.digest(path)
        if s.get('checkpoint_sha256'):assert s['checkpoint_sha256']==sha
        candidates[method]=dict(checkpoint=str(path),checkpoint_sha256=sha,config=str(cfg),source_summary=str(summary),
                                year=YEARS[method],archived_source_iou=s['IoU_percent'],source_training_budget=horizon,
                                source_budget_unit='updates' if horizon==1000 else 'epochs')
    protocol=dict(source=curve.SOURCE,target=curve.TARGET,shot_ids=p['shot_ids'],test_ids=p['test_ids'],source_test_ids=source['test_ids'],
                  methods=[*bg.METHODS,*NEW,'Ours'],new_methods=candidates,adapter_only_in_figure=False,
                  selection='Recent IRSTD methods in the comparison table with healthy retained NUDT source checkpoints. Selection fixed before target adaptation; individual low shot scores are not grounds for exclusion.',
                  optimizer='AdamW lr 1e-4, effective batch 4, gradient clip 1; frozen BN running statistics, 100 updates; foreground-balanced BCE + SoftIoU on final mask',
                  freeze='Preserve author-frozen backbones on SAM models; train remaining native parameters. Existing baselines are reused unchanged.',
                  original_protocol_sha256=bg.digest(curve.OUT/'protocol.json'))
    dest=OUT/'protocol.json'
    if dest.exists():assert own.read_json(dest)==protocol
    else:bg.dump(dest,protocol)
    return protocol,source,target


class ProbabilityModel(torch.nn.Module):
    def __init__(self, model, forward):
        super().__init__();self.model=model;self.predict=forward
    def forward(self,x):return self.predict(x)


def build(method,source,device):
    if method in ('HDNet','MSHNet+NS-FPN'):
        from scripts.run_comparison_recovery_exploration import build_ir
        m,*_=build_ir(method)
        return m,lambda x:m(x,True)[1].sigmoid()
    if method=='SAM-SPL':
        from scripts.comparison_added_irstd_adapter import setup as setup_spl
        m,*_=setup_spl(method,source,device)
        return m,lambda x:m(x)[0].sigmoid()
    from scripts.profile_public_irstd_models import build as build_public
    m,f,_=build_public(method,task_weights=False)
    return m,f


def make_transform(method,cfg):
    if method in ('HDNet','MSHNet+NS-FPN'):
        return lambda im:curve.ir.transform(im,method,cfg)
    if method in ('IR-SAM2','SAMamba'):
        from scripts.train_sam_extension_train50 import normalize
        from PIL import Image
        import numpy as np
        def transform(im):
            x=np.array(Image.fromarray(im).resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)
            return normalize(torch.from_numpy(x.transpose(2,0,1).copy()),method,cfg['train_only_statistics'])
        return transform
    from PIL import Image
    import numpy as np
    stats=cfg['train_only_statistics']
    def transform(im):
        x=np.array(Image.fromarray(im).resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(((x-stats['mean'])/stats['std']).transpose(2,0,1)))
    return transform


def train(model,method,target,ids,steps,transform):
    pairs={}
    for sid in ids:
        im,gt,_=bg.read(curve.TARGET,sid);x=transform(im)
        h,w=x.shape[-2:]; y=cv2.resize(gt,(w,h),interpolation=cv2.INTER_NEAREST_EXACT)
        pairs[sid]=(x,torch.from_numpy(y[None]).float())
    model.train()
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    params=[p for p in model.parameters() if p.requires_grad];assert params
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.)
    stream=curve.sample_stream(ids);seen=set();micro=1 if method in ('IR-SAM2','SAMamba','SAM-SPL') else 2
    for i in range(steps):
        batch=[next(stream) for _ in range(4)];seen.update(batch);opt.zero_grad(set_to_none=True)
        for j in range(0,4,micro):
            ids_j=batch[j:j+micro]
            x=torch.stack([pairs[s][0] for s in ids_j]).cuda();y=torch.stack([pairs[s][1] for s in ids_j]).cuda()
            p=model(x).float();assert p.shape==y.shape
            q=p.clamp(1e-6,1-1e-6);dims=(1,2,3);fg=y.sum(dims);n=(1-y).sum(dims)
            pos=-(y*q.log()).sum(dims)/fg.clamp_min(1);neg=-((1-y)*torch.log1p(-q)).sum(dims)/n.clamp_min(1)
            overlap=(p*y).sum(dims);union=(p+y-p*y).sum(dims)
            loss=(torch.where(fg>0,.5*(pos+neg),neg)+1-(overlap+1)/(union+1)).mean()*(micro/4)
            assert torch.isfinite(loss);loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
        if (i+1)%20==0:own.log(event='update',method=method,shot=len(ids),step=i+1)
    if steps==curve.STEPS:assert seen==set(ids)
    return sorted(seen)


@torch.no_grad()
def evaluate(model,m,method,k,transform,ids=None):
    domain=m['dataset'];rows=[];model.eval()
    ids=m['test_ids'] if ids is None else ids
    for i,sid in enumerate(ids,1):
        im,gt,_=bg.read(domain,sid)
        prob=model(transform(im)[None].cuda())[0,0].float().cpu().numpy()
        prob=cv2.resize(prob,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_LINEAR)
        rows.append(curve.image_metrics(prob,gt,sid))
        if i%200==0:own.log(event='test',method=method,shot=k,domain=domain,images=i,total=len(ids))
    return rows


def worker(p,source,target,method,smoke):
    rec=p['new_methods'][method];cfg=own.read_json(ROOT/rec['config'])
    saved=torch.load(rec['checkpoint'],map_location='cpu',weights_only=False)
    key='updates' if rec['source_budget_unit']=='updates' else 'epoch'
    assert saved[key]==rec['source_training_budget']
    own.base.seed_all()
    m,forward=build(method,source,torch.device('cuda'));m.cuda()
    model=ProbabilityModel(m,forward).cuda();transform=make_transform(method,cfg)
    for k in ((1,) if smoke else curve.SHOTS):
        dest=OUT/('smoke' if smoke else 'results')/method/f'shot{k:02d}'
        if (dest/'summary.json').exists():continue
        own.base.seed_all();m.load_state_dict(saved['model'],strict=True);model.eval()
        ids=p['shot_ids'][str(k)];steps=(1 if smoke else curve.STEPS) if k else 0
        own.log(event='begin',method=method,shot=k,steps=steps,smoke=smoke)
        seen=train(model,method,target,ids,steps,transform) if k else []
        # Preflight reads only allocated training images, not test labels.
        t=evaluate(model,target,method,k,transform,ids=ids if smoke else None)
        s=evaluate(model,source,method,k,transform,ids=source['train_ids'][:1] if smoke else None)
        value=dict(method=method,shot=k,steps=steps,train_ids=ids,seen_train_ids=seen,
                   source_sha256=rec['checkpoint_sha256'],source=curve.summarize(s),target=curve.summarize(t))
        if not k:
            assert abs(value['source']['IoU_percent']-rec['archived_source_iou'])<.25, (method,'source replay differs',value['source']['IoU_percent'])
        own.save_csv(dest/'source_per_image.csv',s);own.save_csv(dest/'target_per_image.csv',t);bg.dump(dest/'summary.json',value)
        own.log(event='done',method=method,shot=k,source_iou=value['source']['IoU_percent'],target_iou=value['target']['IoU_percent'],smoke=smoke)
        gc.collect();torch.cuda.empty_cache()


def label_curve_ends(ax, endpoints):
    ordered = sorted(endpoints, key=lambda endpoint: endpoint[1])
    gap = 4.0
    positions = [1.8]
    for _, value, _ in ordered:
        positions.append(max(value, positions[-1] + gap))
    positions = positions[1:]
    positions[-1] = min(positions[-1], 82.0)
    for i in range(len(positions) - 2, -1, -1):
        positions[i] = min(positions[i], positions[i + 1] - gap)

    for (method, value, color), label_y in zip(ordered, positions):
        ours = method == 'Ours'
        ax.plot([4.06, 4.25], [value, label_y], color=color, lw=0.9, alpha=.8, clip_on=False)
        ax.text(4.30, label_y, method, color=color, va='center', fontsize=8,
                fontweight='bold' if ours else 'normal')


def render(p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows=[];counts={'source':0,'target':0}
    old=own.csv_rows(prev.OUT/'source_retention_all_methods.csv')
    for method in p['methods']:
        initial=None
        for k in curve.SHOTS:
            if method in NEW:
                d=OUT/'results'/method/f'shot{k:02d}';r=own.read_json(d/'summary.json')
                assert r['method']==method and r['shot']==k and r['steps']==(curve.STEPS if k else 0)
                assert r['train_ids']==p['shot_ids'][str(k)] and set(r['seen_train_ids'])==set(r['train_ids'])
                assert r['source_sha256']==p['new_methods'][method]['checkpoint_sha256']
                counts['source']+=prev.recount(d/'source_per_image.csv',p['source_test_ids'],r['source'])
                counts['target']+=prev.recount(d/'target_per_image.csv',p['test_ids'],r['target'])
                s,t=r['source']['IoU_percent'],r['target']['IoU_percent']
            else:
                r=next(r for r in old if r['method']==method and int(r['shot'])==k)
                s,t=float(r['source_iou']),float(r['target_iou'])
            if k==0:initial=s
            rows.append(dict(method=method,shot=k,target_iou=t,source_iou=s,source_change_pp=s-initial))
    assert counts=={'source':16600,'target':5025} and len(rows)==50
    # A method that never makes useful target predictions does not get a paper line.
    failed=[m for m in p['methods'] if max(r['target_iou'] for r in rows if r['method']==m)<1]
    assert not failed, ('Fully failed methods require replacement',failed)
    own.save_csv(OUT/'first_figure_ten_methods.csv',rows)
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(13.2,4.8),sharey=True)
    styles=[('#4E6D8C','s'),('#A46169','D'),('#717983','v'),('#A76B39','P'),('#80729F','^'),
            ('#8F705E','X'),('#437C8C','p'),('#956083','h'),('#737E4E','<'),('#08724E','o')]
    stages = range(len(curve.SHOTS))
    for ax,key,title in zip(axes,('target_iou','source_iou'),('Target: IRSTD test','Source: NUDT test')):
        endpoints = []
        for (color,marker),method in zip(styles,p['methods']):
            y=[r[key] for r in rows if r['method']==method];assert len(y)==len(stages)
            ours = method == 'Ours'
            ax.plot(stages, y, color=color, marker=marker, lw=2.8 if ours else 1.35,
                    ms=6 if ours else 4.5, zorder=5 if ours else 2)
            endpoints.append((method, y[-1], color))
            if ours:
                for x, value in zip(stages, y):
                    ax.annotate(f'{value:.2f}', (x, value), xytext=(0, 7), textcoords='offset points',
                                ha='center', va='bottom', color=color, fontsize=8.5, fontweight='bold',
                                bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=1.1), zorder=6)
        label_curve_ends(ax, endpoints)
        ax.set(xticks=stages,xticklabels=curve.SHOTS,title=title,xlabel='Labeled IRSTD training images (shots)',
               xlim=(-.23,5.55),ylim=(0,85))
        ax.grid(axis='y',alpha=.20);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Test IoU (%)')
    fig.suptitle('NUDT → IRSTD: few-shot adaptation and source retention',fontweight='bold',y=.965)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUT/f'first_figure_ten_methods.{ext}',dpi=210)
    plt.close(fig)
    bg.dump(OUT/'audit.json',dict(status='passed',methods=len(p['methods']),new_predictions_recounted=counts,adapter_only_excluded=True,
        failed_methods_in_figure=failed,existing_results_sha256=bg.digest(prev.OUT/'source_retention_all_methods.csv')))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--method',choices=NEW);a.add_argument('--smoke',action='store_true');a.add_argument('--render',action='store_true');args=a.parse_args()
    p,source,target=setup()
    if args.render:render(p)
    elif args.method:worker(p,source,target,args.method,args.smoke)
    else:
        curve.wait_gpu()
        for method in NEW:
            for smoke in (True,False):
                subprocess.run([sys.executable,'-u','-m','scripts.run_nudt_irstd_first_figure_expanded','--method',method,*(['--smoke'] if smoke else [])],cwd=ROOT,check=True)
        render(p)
