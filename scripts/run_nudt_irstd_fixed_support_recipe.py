"""Replay the pretrained Tiny fixed dense-memory recipe on the locked IRSTD shots."""
import argparse
import math
import sys
import time

import cv2
import torch

from scripts import eval_tiny_fixed_memory_radial as legacy
from scripts import run_nudt_irstd_support_only as comparison
from scripts import run_nudt_irstd_shot_curve as curve

bg, own = curve.bg, curve.own
OUT = comparison.OUT/'fixed_recipe'
STEM = 'support_only_three_way'
OLD = bg.ROOT/'outputs/tiny_dense_target_correction_20261111/IRSTD-1k/memory_native_radial_fixed_correction_0.25_gain_0.25/epoch_0000/metrics'


def setup():
    p,_,ref=comparison.load_reference()
    previous=own.read_json(OLD/'summary.json')
    assert previous['memory']=='native' and previous['radial']=='fixed'
    assert previous['fixed_radial_scale']==.1 and previous['target_correction']==.25 and previous['foreground_gain']==.25
    assert previous['training_horizon']==0 and previous['adapter_source']=='identity initialization'
    m=bg.manifest(curve.TARGET)
    assert previous['manifest_sha256']==m['manifest_sha256']
    assert bg.digest(legacy.base.RUN/'weights/sam2_hiera_tiny.pt')==previous['initialization_sha256']
    old_rows=own.csv_rows(OLD/'per_image.csv')
    assert [r['sample_id'] for r in old_rows]==p['test_ids']
    comparison_audit=own.read_json(comparison.OUT/'audit.json')
    assert comparison_audit['status']=='passed' and comparison_audit['original_protocol_sha256']==bg.digest(curve.OUT/'protocol.json')
    return p,ref,previous,old_rows,m


def load_test(sid,predict):
    im,gt,_=bg.read(curve.TARGET,sid)
    prob=predict(sid).cpu().numpy()
    prob=cv2.resize(prob,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_LINEAR)
    return curve.image_metrics(prob,gt,sid)


def build_predictor(model,m,ids):
    return legacy.make_predictor(model,dict(m,support_ids=ids),'native',True,
                                 target_correction=.25,foreground_gain=.25,support_batch_size=5)


def build_without_adapter():
    run_root=legacy.base.RUN
    sys.path[:0]=[str(run_root/'sources/SANSA'),str(run_root/'runtime_packages')]
    from util import path_utils
    from models.sansa.sansa import build_sansa
    weight=run_root/'weights/sam2_hiera_tiny.pt'
    path_utils.SAM2_PATHS_CONFIG['tiny']=(str(weight),'../sam2/sam2_configs/sam2_hiera_t.yaml')
    model=build_sansa('tiny',[],.3,'cuda').cuda().eval()
    assert not any('adapter' in name for name,_ in model.named_parameters())
    return model.requires_grad_(False)


@torch.inference_mode()
def run(p,ref,previous,old_rows,m):
    curve.wait_gpu()
    legacy.base.seed_all()
    model=build_without_adapter()
    legacy.install_decoder_feature_radial_gated(model,'feature_decoder_gated',torch.zeros(64),torch.ones(64),fixed_scale=.1,active_stages=(0,1))
    assert sum(x.numel() for x in model.radial_branch.parameters())==0
    model.eval().requires_grad_(False)
    sha=comparison.state_hash(model)
    # Original five supports, with the original five-frame encoder batch.
    predict=build_predictor(model,m,previous['support_ids'])
    for expected in old_rows[:4]:
        actual=load_test(expected['sample_id'],predict)
        assert all(actual[k]==int(expected[k]) for k in ('tp','fp','fn','targets','detected')), (expected['sample_id'],actual)
    del predict
    own.log(event='historical_replay_passed',images=4)
    for k in curve.SHOTS[1:]:
        dest=OUT/f'shot{k:02d}'; ids=p['shot_ids'][str(k)]
        if (dest/'summary.json').exists():
            assert own.read_json(dest/'summary.json')['state_sha256']==sha
            continue
        own.log(event='begin',shot=k,updates=0)
        start=time.perf_counter()
        predict=build_predictor(model,m,ids)
        torch.cuda.synchronize()
        build_seconds=time.perf_counter()-start
        rows=[]
        for i,sid in enumerate(p['test_ids'],1):
            rows.append(load_test(sid,predict))
            if i%50==0:own.log(event='test',shot=k,images=i,total=len(p['test_ids']))
        assert comparison.state_hash(model)==sha
        assert all(x.grad is None for x in model.parameters())
        own.save_csv(dest/'per_image.csv',rows)
        # Native SAM2 stores all supports but attends to the first frame plus
        # the latest frames admitted by its spatial/pointer windows.
        spatial=[ids[0],*ids[max(1,k-model.sam.num_maskmem+1):]]
        pointers=[ids[0],*ids[max(1,k-model.sam.max_obj_ptrs_in_encoder+1):]] if model.sam.use_obj_ptrs_in_encoder else []
        result=dict(shot=k,updates=0,foreground_multiplier=1.25,target_correction=.25,fixed_radial_scale=.1,
                    support_ids=ids,correction_support_count=k,spatial_attention_support_ids=spatial,
                    pointer_attention_support_ids=pointers,initialization_sha256=previous['initialization_sha256'],adapter_parameters=0,
                    state_sha256=sha,memory_build_seconds=build_seconds,**curve.summarize(rows))
        bg.dump(dest/'summary.json',result)
        own.log(event='done',shot=k,iou=result['IoU_percent'],dense_frames=len(spatial),pointer_frames=len(pointers))
        del predict
    render(p,ref,previous)


def render(p,ref,previous):
    source=own.csv_rows(comparison.OUT/'support_only_vs_fine_tuned.csv')
    assert tuple(int(r['shot']) for r in source)==curve.SHOTS
    states=set();count=0;rows=[]
    for k,r in zip(curve.SHOTS,source):
        row={name:float(v) for name,v in r.items() if name!='shot'};row['shot']=k
        assert row['fine_tuned_iou']==ref[k][0]['IoU_percent']
        score=None;dense_frames=None
        if k:
            d=OUT/f'shot{k:02d}';s=own.read_json(d/'summary.json');test=own.csv_rows(d/'per_image.csv')
            assert s['support_ids']==p['shot_ids'][str(k)] and s['correction_support_count']==k and s['updates']==0
            assert s['initialization_sha256']==previous['initialization_sha256'] and s['adapter_parameters']==0
            assert (s['foreground_multiplier'],s['target_correction'],s['fixed_radial_scale'])==(1.25,.25,.1)
            assert [x['sample_id'] for x in test]==p['test_ids']
            for x,y in zip(test,ref[k][1]):
                assert all(x[a]==y[a] for a in ('pixels','gt_pixels','targets'))
                for a in ('tp','fp','fn','pixels','gt_pixels','targets','predictions','detected','unmatched_pixels'):x[a]=int(x[a])
                x['image_iou']=float(x['image_iou'])
            assert all(math.isclose(s[a],v,rel_tol=1e-12,abs_tol=1e-12) for a,v in curve.summarize(test).items() if isinstance(v,(int,float)))
            states.add(s['state_sha256']);count+=len(test);score=s['IoU_percent'];dense_frames=len(s['spatial_attention_support_ids'])
        row.update(legacy_fixed_iou=score,legacy_dense_attention_frames=dense_frames);rows.append(row)
    assert count==804 and len(states)==1
    own.save_csv(comparison.OUT/f'{STEM}.csv',rows)
    comparison.plot_comparison(rows,comparison.OUT,STEM,legacy=True)
    bg.dump(OUT/'audit.json',dict(status='passed',predictions_checked=count,weights_frozen=True,historical_replay_images=4,
        state_sha256=next(iter(states)),initialization_sha256=previous['initialization_sha256'],original_protocol_sha256=bg.digest(curve.OUT/'protocol.json')))
    for row in rows:own.log(**row)


def main():
    a=argparse.ArgumentParser();a.add_argument('--preflight',action='store_true');args=a.parse_args()
    inputs=setup()
    if args.preflight:
        print('Verified old recipe and pretrained backbone hash, shared support sets, and all test IDs.')
        return
    run(*inputs)


if __name__=='__main__':main()
