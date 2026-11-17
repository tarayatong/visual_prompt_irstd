"""Freeze the audited Ours checkpoint and swap supports on the original shot splits."""
import argparse
import hashlib
import math
import time

import torch

from scripts import run_nudt_irstd_shot_curve as curve

bg, own = curve.bg, curve.own
OUT = curve.OUT/'support_only'


def load_reference():
    p = own.read_json(curve.OUT/'protocol.json')
    audit = own.read_json(curve.OUT/'audit.json')
    assert audit['status']=='passed' and audit['cells']==25
    for file in ('protocol.json','shot_curve.csv'):
        assert bg.digest(curve.OUT/file)==audit['artifact_sha256'][file]
    assert bg.digest(curve.__file__)==p['runner_sha256']
    assert p['source']==curve.SOURCE and p['target']==curve.TARGET
    m=bg.manifest(curve.TARGET)
    assert p['test_ids']==m['test_ids'] and len(p['test_ids'])==201
    checkpoints=p['checkpoints']['Ours']
    assert bg.digest(checkpoints['path'])==checkpoints['sha256']==own.selected()[curve.SOURCE]['sha256']
    source_row=own.selected()[curve.SOURCE]
    assert bg.digest(bg.ROOT/source_row['checkpoint'])==checkpoints['sha256']
    ref={}; exported={int(r['shot']):float(r['IoU_percent']) for r in own.csv_rows(curve.OUT/'shot_curve.csv') if r['method']=='Ours'}
    for k in curve.SHOTS:
        ids=p['shot_ids'][str(k)]
        assert len(ids)==len(set(ids))==k and set(ids)<=set(m['train_ids'])
        assert not set(ids)&set(p['test_ids'])
        dest=curve.OUT/'results/Ours'/f'shot{k:02d}'
        result=own.read_json(dest/'summary.json')
        assert result['source_sha256']==checkpoints['sha256'] and result['train_ids']==ids
        assert result['steps']==(curve.STEPS if k else 0) and result['images']==201
        assert result['IoU_percent']==exported[k]
        rows=own.csv_rows(dest/'per_image.csv')
        assert [r['sample_id'] for r in rows]==p['test_ids']
        ref[k]=(result,rows)
    return p,source_row,ref


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    return h.hexdigest()


@torch.no_grad()
def run(p, source_row, ref):
    curve.wait_gpu()
    e=own.Engine(curve.SOURCE,source_row)
    e.model.eval().requires_grad_(False)
    start_hash=state_hash(e.model)
    for k in curve.SHOTS[1:]:
        ids=p['shot_ids'][str(k)];dest=OUT/f'shot{k:02d}'
        if (dest/'summary.json').exists():
            s=own.read_json(dest/'summary.json')
            assert s['support_ids']==ids and s['model_state_sha256']==start_hash
            continue
        own.log(event='begin',shot=k,updates=0)
        start=time.perf_counter()
        pairs=e.pairs(curve.TARGET,ids)
        e.cache=curve.target_memory(e,pairs,ids)
        torch.cuda.synchronize()
        memory_seconds=time.perf_counter()-start
        rows=[]
        for sid in p['test_ids']:
            im,gt,_=bg.read(curve.TARGET,sid)
            rows.append(curve.image_metrics(e.predict(im),gt,sid))
        assert state_hash(e.model)==start_hash, 'Frozen weights or buffers changed'
        assert all(param.grad is None for param in e.model.parameters())
        own.save_csv(dest/'per_image.csv',rows)
        result=dict(shot=k,updates=0,source_sha256=e.sha,model_state_sha256=start_hash,
                    support_ids=ids,memory_build_seconds=memory_seconds,**curve.summarize(rows))
        bg.dump(dest/'summary.json',result)
        own.log(event='done',shot=k,iou=result['IoU_percent'],memory_build_seconds=memory_seconds)
        e.cache=None
    return render(p,ref)


def render(p,ref):
    baseline=ref[0][0]['IoU_percent']; rows=[]; verified=0; state_hashes=set()
    for k in curve.SHOTS:
        trained=ref[k][0]['IoU_percent']; seconds=0.
        if k:
            d=OUT/f'shot{k:02d}'; result=own.read_json(d/'summary.json'); test=own.csv_rows(d/'per_image.csv')
            assert result['support_ids']==p['shot_ids'][str(k)] and result['updates']==0 and result['shot']==k
            assert result['source_sha256']==p['checkpoints']['Ours']['sha256']
            assert [r['sample_id'] for r in test]==p['test_ids']
            state_hashes.add(result['model_state_sha256'])
            for row,prior in zip(test,ref[k][1]):
                assert all(row[key]==prior[key] for key in ('sample_id','gt_pixels','pixels','targets'))
                for key in ('tp','fp','fn','pixels','gt_pixels','targets','predictions','detected','unmatched_pixels'):
                    row[key]=int(row[key])
                row['image_iou']=float(row['image_iou'])
            assert all(math.isclose(result[key],v,rel_tol=1e-12,abs_tol=1e-12) for key,v in curve.summarize(test).items() if isinstance(v,(int,float)))
            verified+=len(test)
            frozen=result['IoU_percent'];seconds=result['memory_build_seconds']
        else:frozen=baseline
        rows.append(dict(shot=k,support_only_iou=frozen,fine_tuned_iou=trained,
                         support_only_gain_pp=frozen-baseline,fine_tuning_extra_pp=trained-frozen,
                         memory_build_seconds=seconds))
    assert verified==804 and len(state_hashes)==1
    own.save_csv(OUT/'support_only_vs_fine_tuned.csv',rows)

    plot_comparison(rows,OUT,'support_only_vs_fine_tuned')
    bg.dump(OUT/'audit.json',dict(status='passed',frozen_test_predictions_checked=verified,weight_state_sha256=next(iter(state_hashes)),
        same_checkpoint_support_and_test_ids=True,baseline_reused_from_audited_run=True,
        original_protocol_sha256=bg.digest(curve.OUT/'protocol.json'),original_curve_sha256=bg.digest(curve.OUT/'shot_curve.csv')))
    for r in rows:own.log(**r)


def plot_comparison(rows, out, stem, *, legacy=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.7,4.8))
    x=list(range(len(rows))); colors=('#397cb0','#218054'); baseline=rows[0]['support_only_iou']
    for key,color,style,name in (
        ('support_only_iou',colors[0],'--','Target supports only · 0 updates'),
        ('fine_tuned_iou',colors[1],'-','Target supports + fine-tuning · 100 updates'),
    ):
        ax.plot(x,[r[key] for r in rows],style,marker='o',color=color,label=name,lw=2.4)
    ax.scatter([0],[baseline],color='#454c55',s=36,zorder=5)
    ax.annotate(f'{baseline:.2f}',(0,baseline),xytext=(0,9),textcoords='offset points',ha='center',color='#454c55',fontsize=9)
    for i,r in enumerate(rows[1:],1):
        a,b=r['support_only_iou'],r['fine_tuned_iou']
        for value,color,sign in ((a,colors[0],1 if a>b else -1),(b,colors[1],1 if b>=a else -1)):
            ax.annotate(f'{value:.2f}',(i,value),xytext=(0,9*sign),textcoords='offset points',ha='center',va='bottom' if sign>0 else 'top',color=color,fontsize=9)
    if legacy:
        values=[r['legacy_fixed_iou'] for r in rows[1:]]
        color='#bb6630'
        ax.plot(x[1:],values,':',marker='D',color=color,label='Pretrained Tiny + fixed support recipe · 0 updates',lw=2.5)
        for i,value in enumerate(values,1):
            ax.annotate(f'{value:.2f}',(i,value),xytext=(0,-10),textcoords='offset points',ha='center',va='top',color=color,fontsize=9)
    hi=max(max(r['support_only_iou'],r['fine_tuned_iou'],r.get('legacy_fixed_iou') or 0) for r in rows)
    ax.set(xticks=x,xticklabels=curve.SHOTS,xlabel='Labeled IRSTD training images (shots)',ylabel='IRSTD test IoU (%)',
           title='NUDT → IRSTD: support swap vs. fine-tuning',ylim=(0,min(100,hi+10)))
    ax.grid(axis='y',alpha=.23);ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower right',frameon=False,fontsize=9)
    fig.text(.15,.03 if legacy else .015,'Same nested support IDs and 201 test queries. At 0-shot, Ours uses source supports.',fontsize=8)
    if legacy:
        fig.text(.15,.008,'Fixed recipe: all supports inform correction; dense attention keeps its native window.',fontsize=8)
    fig.tight_layout(rect=(0,.06 if legacy else .04,1,1))
    for ext in ('png','pdf'):fig.savefig(out/f'{stem}.{ext}',dpi=210)
    plt.close(fig)


def main():
    a=argparse.ArgumentParser();a.add_argument('--preflight',action='store_true');args=a.parse_args()
    p,source_row,ref=load_reference()
    if args.preflight:
        print('Verified source checkpoint, four support prefixes, and 201 test queries.')
        return
    run(p,source_row,ref)


if __name__=='__main__':main()
