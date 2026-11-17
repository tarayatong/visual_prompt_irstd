"""Replay the locked target-shot adapters and test their source-domain retention."""
import argparse
import gc
import time

import cv2
import torch

from scripts import run_nudt_irstd_four_way as four

bg,own,curve,legacy=four.bg,four.own,four.curve,four.legacy
OUT=four.OUT/'source_retention'
METHODS=('adapter','ours')


def setup():
    p,target_ref,_,_,target=four.fixed.setup()
    audit=own.read_json(four.OUT/'audit.json');assert audit['status']=='passed'
    weights,sha,initial_sha=four.checkpoint(p);assert sha==audit['native_source_sha256']
    source=bg.manifest(curve.SOURCE);assert len(source['test_ids'])==664
    archive={}
    for method,variant in zip(METHODS,('native','combined')):
        rows=[r for r in own.csv_rows(own.INDEX) if (r['dataset'],r['seed'],r['variant'])==(curve.SOURCE,'111',variant)]
        assert len(rows)==1
        entry=rows[0];s=own.read_json(bg.ROOT/entry['summary'])
        test=own.csv_rows(bg.ROOT/entry['per_image_counts'])
        assert [r['sample_id'] for r in test]==source['test_ids']
        assert s['checkpoint_sha256']==entry['sha256']==(sha if method=='adapter' else p['checkpoints']['Ours']['sha256'])
        assert s['support_annotation_ids']==source['support_ids'] and s['images']==664
        assert float(entry['IoU_percent'])==s['IoU_percent']
        archive[method]=(s,test)
    for k in curve.SHOTS:
        folder=four.OUT/'adapter'/f'shot{k:02d}'
        target_ref_adapter=four.read_score(folder,p,target_ref[k][1],k)
        assert target_ref_adapter['source_sha256']==sha and target_ref_adapter['train_ids']==p['shot_ids'][str(k)]
    return p,target_ref,target,weights,sha,source,archive


@torch.no_grad()
def evaluate(domain,ids,predict,method,k):
    rows=[]
    for i,sid in enumerate(ids,1):
        im,gt,_=bg.read(domain,sid)
        rows.append(curve.image_metrics(predict(sid,im),gt,sid))
        if i%100==0:own.log(event='test',method=method,shot=k,domain=domain,images=i,total=len(ids))
    return rows


def native_predictor(model,m,ids):
    with torch.no_grad():
        pred=legacy.make_predictor(model,dict(m,support_ids=ids),'native',False,support_batch_size=5)
    def predict(sid,im):
        return cv2.resize(pred(sid).cpu().numpy(),(im.shape[1],im.shape[0]),interpolation=cv2.INTER_LINEAR)
    return predict


def matching_counts(rows,prior):
    keys=('sample_id','tp','fp','fn','detected','predictions','unmatched_pixels')
    return sum(all(str(a[k])==str(b[k]) for k in keys) for a,b in zip(rows,prior))


def run(p,target_ref,target,weights,sha,source,archive):
    curve.wait_gpu()
    for method in METHODS:
        for k in curve.SHOTS[1:]:
            dest=OUT/method/f'shot{k:02d}';ids=p['shot_ids'][str(k)]
            if (dest/'summary.json').exists():continue
            legacy.base.seed_all();start=time.perf_counter()
            own.log(event='begin',method=method,shot=k)
            if method=='ours':
                engine=own.Engine(curve.SOURCE,own.selected()[curve.SOURCE]);model=engine.model;src_sha=engine.sha
                q,s=curve.train_ours(engine,ids,curve.STEPS)
                predict=lambda sid,im:engine.predict(im)
                t_before,t_prior=target_ref[k]
            else:
                model,*rest=legacy.base.setup_sansa(target,torch.device('cuda'));del rest
                assert set(weights)=={name for name,v in model.named_parameters() if v.requires_grad}
                model.load_state_dict(weights,strict=False);model.eval();src_sha=sha
                q,s=four.train_adapter(model,ids)
                predict=native_predictor(model,target,ids)
                folder=four.OUT/'adapter'/f'shot{k:02d}'
                t_before=own.read_json(folder/'summary.json');t_prior=own.csv_rows(folder/'per_image.csv')
            train_seconds=time.perf_counter()-start
            assert set(q)==set(ids)
            target_rows=evaluate(curve.TARGET,p['test_ids'],predict,method,k)
            same=matching_counts(target_rows,t_prior)
            t_now=curve.summarize(target_rows)
            own.log(event='target_replay',method=method,shot=k,matching_images=same,old_iou=t_before['IoU_percent'],new_iou=t_now['IoU_percent'])
            del predict
            if method=='ours':
                engine.cache=None
                assert list(engine.support_ids)==source['support_ids']
                predict=lambda sid,im:engine.predict(im)
            else:predict=native_predictor(model,source,source['support_ids'])
            source_rows=evaluate(curve.SOURCE,source['test_ids'],predict,method,k)
            summary=dict(method=method,shot=k,updates=curve.STEPS,source_sha256=src_sha,
                train_ids=ids,source_support_ids=source['support_ids'],train_seconds=train_seconds,
                source=curve.summarize(source_rows),target=t_now,archived_target_iou=t_before['IoU_percent'],
                target_replay_identical_images=same)
            own.save_csv(dest/'source_per_image.csv',source_rows);own.save_csv(dest/'target_replay_per_image.csv',target_rows)
            bg.dump(dest/'summary.json',summary)
            own.log(event='done',method=method,shot=k,source_iou=summary['source']['IoU_percent'],target_iou=t_now['IoU_percent'])
            del predict,model
            if method=='ours':del engine
            gc.collect();torch.cuda.empty_cache()
    render(p,target_ref,source,archive)


def render(p,target_ref,source,archive):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows=[];counts={'source':0,'target':0};identical=0
    def recount(path,ids,expected):
        test=own.csv_rows(path);assert [r['sample_id'] for r in test]==ids
        for r in test:
            for key in ('tp','fp','fn','pixels','gt_pixels','targets','predictions','detected','unmatched_pixels'):r[key]=int(r[key])
            r['image_iou']=float(r['image_iou'])
        assert all(expected[key]==v for key,v in curve.summarize(test).items() if isinstance(v,(int,float)))
        return len(test)
    for method in METHODS:
        initial=archive[method][0]['IoU_percent']
        for k in curve.SHOTS:
            if k:
                d=OUT/method/f'shot{k:02d}';z=own.read_json(d/'summary.json')
                assert z['method']==method and z['shot']==k and z['train_ids']==p['shot_ids'][str(k)]
                assert z['source_support_ids']==source['support_ids'] and z['updates']==curve.STEPS
                assert z['source_sha256']==archive[method][0]['checkpoint_sha256']
                counts['source']+=recount(d/'source_per_image.csv',source['test_ids'],z['source'])
                counts['target']+=recount(d/'target_replay_per_image.csv',p['test_ids'],z['target'])
                identical+=z['target_replay_identical_images']
                a=z['source']['IoU_percent'];b=z['target']['IoU_percent']
            else:
                a=initial
                b=(target_ref[k][0] if method=='ours' else own.read_json(four.OUT/'adapter/shot00/summary.json'))['IoU_percent']
            rows.append(dict(method=method,shot=k,source_iou=a,source_change_pp=a-initial,target_iou=b))
    assert counts=={'source':5312,'target':1608}
    own.save_csv(OUT/'source_retention.csv',rows)
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True)
    for ax,key,title in zip(axes,('target_iou','source_iou'),('Target: IRSTD test','Source: NUDT test')):
        for method,label,color,marker in [('adapter','Adapter-only','#377caa','s'),('ours','Ours','#218054','o')]:
            v=[r[key] for r in rows if r['method']==method]
            ax.plot(range(len(v)),v,marker=marker,color=color,label=label,lw=2.2)
        ax.set(xticks=range(len(curve.SHOTS)),xticklabels=curve.SHOTS,title=title,xlabel='Labeled IRSTD training images (shots)',ylim=(0,85))
        ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Test IoU (%)');axes[0].legend(frameon=False,loc='lower right')
    fig.suptitle('Target adaptation and source retention',fontweight='bold')
    fig.text(.105,.02,'For the source test, both models use their original five NUDT support images. Each nonzero budget uses 100 updates.',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.94))
    for ext in ('png','pdf'):fig.savefig(OUT/f'source_retention.{ext}',dpi=210)
    plt.close(fig)
    bg.dump(OUT/'audit.json',dict(status='passed',new_predictions_recounted=counts,target_replay_identical_images=identical,
        total_target_replay_images=1608,source_zero_shot_baselines='audited original NUDT test results',
        previous_protocol_sha256=bg.digest(curve.OUT/'protocol.json')))
    for row in rows:own.log(**row)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--preflight',action='store_true');args=a.parse_args()
    inputs=setup()
    if args.preflight:print('Verified exact source checkpoints, archived NUDT baselines, and the eight target-adaptation configurations.')
    else:run(*inputs)
