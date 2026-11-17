"""Extend NUDT source retention to the four infrared methods in the shot study."""
import argparse
import gc
import subprocess
import sys

import cv2
import torch

from scripts import run_nudt_irstd_four_way_retention as paired

curve,own,bg=paired.curve,paired.own,paired.bg
ir=curve.ir
OUT=paired.OUT/'all_methods'
LABELS={**{m:m for m in bg.METHODS},'adapter':'Adapter-only','ours':'Ours'}


def setup():
    p,_,_=paired.four.fixed.comparison.load_reference()
    audit=own.read_json(paired.OUT/'audit.json')
    assert audit['status']=='passed' and audit['previous_protocol_sha256']==bg.digest(curve.OUT/'protocol.json')
    source=bg.manifest(curve.SOURCE);target=bg.manifest(curve.TARGET)
    assert len(source['test_ids'])==len(set(source['test_ids']))==664
    assert target['test_ids']==p['test_ids'] and len(p['test_ids'])==201
    for method in bg.METHODS:
        cp=p['checkpoints'][method];assert bg.digest(cp['path'])==cp['sha256']
        for k in curve.SHOTS:
            d=curve.OUT/'results'/method/f'shot{k:02d}';s=own.read_json(d/'summary.json')
            assert s['source_sha256']==cp['sha256'] and s['train_ids']==p['shot_ids'][str(k)]
            assert s['steps']==(curve.STEPS if k else 0) and s['images']==201
    return p,source,target


@torch.no_grad()
def evaluate(model,method,cfg,manifest,k):
    domain=manifest['dataset'];rows=[]
    for i,sid in enumerate(manifest['test_ids'],1):
        im,gt,_=bg.read(domain,sid)
        x=ir.transform(im,method,cfg,manifest['records'][sid]['image'])
        prob=model(x[None].cuda())[0,0].cpu().numpy()
        prob=cv2.resize(prob,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_LINEAR)
        rows.append(curve.image_metrics(prob,gt,sid))
        if i%200==0:own.log(event='test',method=method,shot=k,domain=domain,images=i,total=len(manifest['test_ids']))
    return rows


def worker(p,source,target,method):
    cp=p['checkpoints'][method];cfg=own.read_json(bg.ROOT/cp['config'])
    state=torch.load(cp['path'],map_location='cpu',weights_only=False)['model']
    for k in curve.SHOTS:
        dest=OUT/'results'/method/f'shot{k:02d}'
        if (dest/'summary.json').exists():continue
        own.base.seed_all();model=ir.build(method).cuda()
        model.load_state_dict(state,strict=True);model.eval()
        ids=p['shot_ids'][str(k)];steps=curve.STEPS if k else 0
        own.log(event='begin',method=method,shot=k,steps=steps)
        if k:
            seen,_=curve.train_ir(model,method,cfg,ids,steps)
            assert seen==set(ids)
        t=evaluate(model,method,cfg,target,k)
        s=evaluate(model,method,cfg,source,k)
        prior=curve.OUT/'results'/method/f'shot{k:02d}'
        old=own.read_json(prior/'summary.json');old_rows=own.csv_rows(prior/'per_image.csv')
        assert [r['sample_id'] for r in old_rows]==p['test_ids']
        summary=dict(method=method,shot=k,updates=steps,train_ids=ids,source_sha256=cp['sha256'],
            source=curve.summarize(s),target=curve.summarize(t),archived_target_iou=old['IoU_percent'],
            target_replay_identical_images=paired.matching_counts(t,old_rows))
        own.save_csv(dest/'source_per_image.csv',s);own.save_csv(dest/'target_replay_per_image.csv',t)
        bg.dump(dest/'summary.json',summary)
        own.log(event='done',method=method,shot=k,source_iou=summary['source']['IoU_percent'],target_iou=summary['target']['IoU_percent'])
        del model;gc.collect();torch.cuda.empty_cache()


def recount(path,ids,expected):
    rows=own.csv_rows(path);assert [r['sample_id'] for r in rows]==ids
    for r in rows:
        for a in ('tp','fp','fn','pixels','gt_pixels','targets','predictions','detected','unmatched_pixels'):r[a]=int(r[a])
        r['image_iou']=float(r['image_iou'])
    assert all(expected[a]==v for a,v in curve.summarize(rows).items())
    return len(rows)


def render(p,source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows=[]; counts={'source':0,'target':0};exact=0
    for method in bg.METHODS:
        initial=None
        for k in curve.SHOTS:
            d=OUT/'results'/method/f'shot{k:02d}';r=own.read_json(d/'summary.json')
            assert r['method']==method and r['shot']==k and r['train_ids']==p['shot_ids'][str(k)]
            assert r['source_sha256']==p['checkpoints'][method]['sha256'] and r['updates']==(curve.STEPS if k else 0)
            counts['source']+=recount(d/'source_per_image.csv',source['test_ids'],r['source'])
            counts['target']+=recount(d/'target_replay_per_image.csv',p['test_ids'],r['target'])
            exact+=r['target_replay_identical_images']
            score=r['source']['IoU_percent']
            if not k:initial=score
            rows.append(dict(method=LABELS[method],shot=k,source_iou=score,source_change_pp=score-initial,target_iou=r['target']['IoU_percent']))
    existing=own.csv_rows(paired.OUT/'source_retention.csv')
    assert len(existing)==10
    for r in existing:
        rows.append(dict(method=LABELS[r['method']],shot=int(r['shot']),**{k:float(r[k]) for k in ('source_iou','source_change_pp','target_iou')}))
    assert counts=={'source':13280,'target':4020} and len(rows)==30
    own.save_csv(OUT/'source_retention_all_methods.csv',rows)
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(8.2,4.8))
    styles=[('#467eb0','s'),('#cb5c66','D'),('#6e8189','v'),('#e68a35','P'),('#786bb1','^'),('#218054','o')]
    for name,(color,marker) in zip(LABELS.values(),styles):
        values=[r['source_iou'] for r in rows if r['method']==name]
        assert len(values)==5
        ax.plot(range(5),values,marker=marker,color=color,label=name,lw=2.7 if name=='Ours' else 1.9)
    values=[r['source_iou'] for r in rows]
    ax.set(xticks=range(5),xticklabels=curve.SHOTS,xlabel='Labeled IRSTD training images (shots)',ylabel='NUDT test IoU (%)',
           title='NUDT performance after IRSTD fine-tuning',ylim=(max(0,5*(int(min(values)//5)-1)),min(100,5*(int(max(values)//5)+2))))
    ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,-.32),fontsize=9)
    fig.text(.5,.043,'Same IRSTD shot splits and NUDT test set. Each nonzero budget uses 100 updates.',ha='center',fontsize=8)
    fig.text(.5,.018,'Ours and adapter-only use the original five NUDT support images.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.068,1,1))
    for ext in ('png','pdf'):fig.savefig(OUT/f'source_retention_all_methods.{ext}',dpi=210)
    plt.close(fig)
    bg.dump(OUT/'audit.json',dict(status='passed',new_predictions_recounted=counts,target_replay_identical_images=exact,
        reused_verified_ours_and_adapter_sha256=bg.digest(paired.OUT/'source_retention.csv'),protocol_sha256=bg.digest(curve.OUT/'protocol.json')))
    for row in rows:own.log(**row)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--method',choices=bg.METHODS);a.add_argument('--preflight',action='store_true');a.add_argument('--render',action='store_true');args=a.parse_args()
    p,source,target=setup()
    if args.preflight:print('Verified four source weights, locked shot IDs, and the existing retention audit.')
    elif args.render:render(p,source)
    elif args.method:worker(p,source,target,args.method)
    else:
        curve.wait_gpu()
        for method in bg.METHODS:
            subprocess.run([sys.executable,'-u','-m','scripts.run_nudt_irstd_multimethod_retention','--method',method],cwd=bg.ROOT,check=True)
        render(p,source)
