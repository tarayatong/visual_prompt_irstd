"""Matched pretrained and trained SAM2 Tiny curves on the locked IRSTD support splits."""
import argparse
import gc
import math
import time

import torch

from scripts import run_nudt_irstd_fixed_support_recipe as fixed

bg,own,curve,legacy = fixed.bg,fixed.own,fixed.curve,fixed.legacy
OUT=fixed.comparison.OUT/'four_way'
ORIGINAL= fixed.OLD.parents[2]/'memory_native_radial_off/epoch_0000/metrics'


def checkpoint(p):
    rows=[r for r in own.csv_rows(own.INDEX) if (r['dataset'],r['seed'],r['variant'])==(curve.SOURCE,'111','native')]
    assert len(rows)==1
    row=rows[0];path=bg.ROOT/row['checkpoint'];sha=bg.digest(path)
    assert sha==row['sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    cfg=state['config'];m=bg.manifest(curve.SOURCE)
    assert state['epoch']==120 and cfg['variant']=='native' and cfg['training_horizon']==120
    assert cfg['manifest_sha256']==m['manifest_sha256'] and cfg.get('training_random_seed',cfg.get('seed'))==111
    assert cfg['train_annotation_ids']==m['train_ids']
    combined=torch.load(p['checkpoints']['Ours']['path'],map_location='cpu',weights_only=False)['config']
    for key in ('manifest_sha256','initialization_sha256','train_annotation_ids','training_horizon','batch_size'):
        assert combined[key]==cfg[key]
    return state['model'],sha,cfg['initialization_sha256']


@torch.no_grad()
def base_run(p,m,initial_sha):
    legacy.base.seed_all()
    model=fixed.build_without_adapter()
    sha=fixed.comparison.state_hash(model)
    archived=own.read_json(ORIGINAL/'summary.json')
    assert archived['initialization_sha256']==initial_sha and archived['adapter_source']=='identity initialization'
    assert archived['memory']=='native' and archived['radial']=='off'
    pred=legacy.make_predictor(model,dict(m,support_ids=archived['support_ids']),'native',False,support_batch_size=5)
    for r in own.csv_rows(ORIGINAL/'per_image.csv')[:4]:
        got=fixed.load_test(r['sample_id'],pred)
        assert all(got[key]==int(r[key]) for key in ('tp','fp','fn','detected')), (r['sample_id'],got)
    del pred
    own.log(event='base_historical_replay_passed',images=4)
    for k in curve.SHOTS[1:]:
        ids=p['shot_ids'][str(k)];dest=OUT/'base'/f'shot{k:02d}'
        if (dest/'summary.json').exists():
            assert own.read_json(dest/'summary.json')['state_sha256']==sha
            continue
        own.log(event='begin',variant='base',shot=k,updates=0)
        start=time.perf_counter()
        pred=legacy.make_predictor(model,dict(m,support_ids=ids),'native',False,support_batch_size=5)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        rows=evaluate(p,pred,'base',k)
        assert fixed.comparison.state_hash(model)==sha
        result=dict(shot=k,updates=0,support_ids=ids,state_sha256=sha,adapter_parameters=0,
                    initialization_sha256=initial_sha,adaptation_seconds=seconds,**curve.summarize(rows))
        save(dest,rows,result,'base');del pred


def pairs(domain,ids):
    m=bg.manifest(domain)
    assert set(ids)<=set(m['train_ids']) and not set(ids)&set(m['test_ids'])
    out={}
    for sid in ids:
        x,y=legacy.base.raw_pair(m,sid,640,True)
        out[sid]=(own.tensor(x),torch.from_numpy(y))
    return out


def train_adapter(model,ids):
    from util.promptable_utils import build_prompt_dict
    from util.losses import loss_masks

    target=pairs(curve.TARGET,ids)
    source=pairs(curve.SOURCE,bg.manifest(curve.SOURCE)['train_ids']) if len(ids)==1 else None
    stream=curve.sample_stream(ids);source_stream=curve.sample_stream(source) if source else None
    support_map={q:ids[(i+1)%len(ids)] for i,q in enumerate(ids)}
    params=[q for name,q in model.named_parameters() if q.requires_grad and 'adapter' in name]
    assert params and len(params)==sum(q.requires_grad for q in model.parameters())
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.)
    seen_q=set();seen_s=set();model.train()
    for step in range(curve.STEPS):
        episode=[]
        for _ in range(4):
            q=next(stream);s=next(source_stream) if source else support_map[q]
            assert source is not None or s!=q
            episode.append(((source or target)[s],target[q]));seen_q.add(q);seen_s.add(s)
        opt.zero_grad(set_to_none=True)
        for i in (0,2):
            part=episode[i:i+2]
            x=torch.stack([torch.stack((s[0],q[0])) for s,q in part]).cuda()
            y=torch.stack([torch.stack((s[1],q[1])) for s,q in part]).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(x,build_prompt_dict(y,'mask',1,True,torch.device('cuda')))['pred_masks']
                # Native forward includes both frames; score only the last query.
                assert pred.shape[0]==4
                loss=sum(loss_masks(pred,y,num_frames=1).values())*.5
            assert torch.isfinite(loss);loss.backward()
        torch.nn.utils.clip_grad_norm_(params,.1,error_if_nonfinite=True);opt.step()
        model.memory_bank={}
        if (step+1)%20==0:own.log(event='update',variant='adapter',shot=len(ids),step=step+1)
    assert seen_q==set(ids) and seen_s==set(source or ids)
    model.eval();return sorted(seen_q),sorted(seen_s)


@torch.no_grad()
def evaluate(p,pred,variant,shot):
    rows=[]
    for i,sid in enumerate(p['test_ids'],1):
        rows.append(fixed.load_test(sid,pred))
        if i%50==0:own.log(event='test',variant=variant,shot=shot,images=i,total=len(p['test_ids']))
    return rows


def save(dest,rows,result,variant):
    own.save_csv(dest/'per_image.csv',rows);bg.dump(dest/'summary.json',result)
    own.log(event='done',variant=variant,shot=result['shot'],iou=result['IoU_percent'])


def adapter_run(p,m,weights,sha):
    for k in curve.SHOTS:
        dest=OUT/'adapter'/f'shot{k:02d}'
        if (dest/'summary.json').exists():
            assert own.read_json(dest/'summary.json')['source_sha256']==sha
            continue
        legacy.base.seed_all();model,*rest=legacy.base.setup_sansa(m,torch.device('cuda'));del rest
        names={n for n,q in model.named_parameters() if q.requires_grad}
        assert set(weights)==names and all('adapter' in n for n in names)
        model.load_state_dict(weights,strict=False);model.eval()
        ids=p['shot_ids'][str(k)];q=[];s=[]
        own.log(event='begin',variant='adapter',shot=k,updates=curve.STEPS if k else 0)
        start=time.perf_counter()
        if k:q,s=train_adapter(model,ids)
        else:
            src=bg.manifest(curve.SOURCE);assert not set(src['support_ids'])&set(m['records'])
        with torch.no_grad():
            view=dict(m,support_ids=ids) if k else dict(m,records={**src['records'],**m['records']},support_ids=src['support_ids'])
            pred=legacy.make_predictor(model,view,'native',False,support_batch_size=5)
            torch.cuda.synchronize();seconds=time.perf_counter()-start if k else 0.
            rows=evaluate(p,pred,'adapter',k)
        result=dict(shot=k,updates=curve.STEPS if k else 0,source_sha256=sha,train_ids=ids,support_ids=view['support_ids'],
                    queries_seen=q,support_ids_seen=s,adaptation_seconds=seconds,**curve.summarize(rows))
        save(dest,rows,result,'adapter');del pred,model
        gc.collect();torch.cuda.empty_cache()


def read_score(folder,p,reference,k):
    result=own.read_json(folder/'summary.json');test=own.csv_rows(folder/'per_image.csv')
    assert result['shot']==k and [r['sample_id'] for r in test]==p['test_ids']
    for row,old in zip(test,reference):
        assert all(row[a]==old[a] for a in ('pixels','gt_pixels','targets'))
        for a in ('tp','fp','fn','pixels','gt_pixels','targets','predictions','detected','unmatched_pixels'):row[a]=int(row[a])
        row['image_iou']=float(row['image_iou'])
    assert all(math.isclose(result[a],v,rel_tol=1e-12,abs_tol=1e-12) for a,v in curve.summarize(test).items() if isinstance(v,(int,float)))
    return result


def render(p,ref,sha,initial_sha):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows=[];states=set();count=0
    for k in curve.SHOTS:
        name=f'shot{k:02d}';a=read_score(OUT/'adapter'/name,p,ref[k][1],k);count+=201
        assert a['source_sha256']==sha and a['updates']==(curve.STEPS if k else 0)
        ids=p['shot_ids'][str(k)];b=f=None
        if k:
            b=read_score(OUT/'base'/name,p,ref[k][1],k);f=read_score(fixed.OUT/name,p,ref[k][1],k);count+=402
            assert all(z['adapter_parameters']==z['updates']==0 and z['support_ids']==ids and z['initialization_sha256']==initial_sha for z in (b,f))
            states.add(b['state_sha256'])
            assert a['train_ids']==a['support_ids']==ids and set(a['queries_seen'])==set(ids)
        rows.append(dict(shot=k,base_iou=b['IoU_percent'] if b else None,base_fixed_iou=f['IoU_percent'] if f else None,
                         adapter_iou=a['IoU_percent'],ours_iou=ref[k][0]['IoU_percent']))
    assert count==2613 and len(states)==1
    own.save_csv(OUT/'shot_four_way.csv',rows)
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(8.1,4.95))
    lines=[('base_iou','Base: SAM2, no adapter','#737780','--','o'),
           ('base_fixed_iou','Base + fixed recipe, no adapter','#b9652e','--','D'),
           ('adapter_iou','Base + trained adapter','#377caa','-','s'),
           ('ours_iou','Ours: trained adapter + learned settings','#218054','-','o')]
    for key,label,color,ls,marker in lines:
        subset=[(i,r[key]) for i,r in enumerate(rows) if r[key] is not None]
        ax.plot(*zip(*subset),ls=ls,marker=marker,color=color,label=label,lw=2.4)
    ax.set(xticks=range(len(rows)),xticklabels=curve.SHOTS,xlabel='Labeled IRSTD support images (shots)',ylabel='IRSTD test IoU (%)',
           title='NUDT → IRSTD: fixed vs. trained adaptation',ylim=(0,75))
    ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower right',frameon=False,fontsize=9)
    fig.text(.13,.042,'Dashed: pretrained SAM2 with no adapter. Solid: NUDT-trained, then 100 IRSTD updates per nonzero budget.',fontsize=8)
    fig.text(.13,.016,'Same supports and test queries. Native memory uses its original attention window; no 0-shot point for dashed lines.',fontsize=8)
    fig.tight_layout(rect=(0,.075,1,1))
    for ext in ('png','pdf'):fig.savefig(OUT/f'shot_four_way.{ext}',dpi=210)
    plt.close(fig)
    bg.dump(OUT/'audit.json',dict(status='passed',new_predictions_recounted=count,matched_support_and_test_ids=True,
            native_source_sha256=sha,pretrained_sha256=initial_sha,ours_reference_sha256=bg.digest(curve.OUT/'shot_curve.csv')))
    for row in rows:own.log(**row)


def main():
    a=argparse.ArgumentParser();a.add_argument('--preflight',action='store_true');a.add_argument('--render',action='store_true');args=a.parse_args()
    p,ref,previous,old_rows,m=fixed.setup();weights,sha,initial_sha=checkpoint(p)
    if args.preflight:
        print('Verified the matching native and combined NUDT checkpoints, same initialization, source train IDs and horizon.')
        return
    if not args.render:
        fixed.run(p,ref,previous,old_rows,m)
        gc.collect();torch.cuda.empty_cache()
        base_run(p,m,initial_sha)
        gc.collect();torch.cuda.empty_cache()
        adapter_run(p,m,weights,sha)
    render(p,ref,sha,initial_sha)


if __name__=='__main__':main()
