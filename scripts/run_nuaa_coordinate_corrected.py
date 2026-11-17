"""Rerun every NUAA table entry under one pixel-center restoration protocol.

Frozen historical checkpoints/episode settings, no hyperparameter search.
Each model is isolated in its own process to preserve native import namespaces.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from scripts.prepare_comparison import ROOT, DEFAULT_OUTPUT, write_json
from scripts.run_global_ablation_201 import write_csv, paste_local
from visual_prompt.evaluation.comparison import image_metrics, summarize
from visual_prompt.evaluation.spatial_alignment import restore_to_original, SPATIAL_PROTOCOL

OUT=ROOT/'outputs/nuaa_coordinate_corrected_20261101'
METHODS=('Ours','HDNet','MSHNet','DNANet','SCTransNet','RPCANet','ISNet','ILNet','MSHNet+NS-FPN',
         'IFA_50label','MPA','ABCDFSS','TVGTANet')
INFRARED=set(METHODS[1:9])


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def manifest():return read(DEFAULT_OUTPUT/'SIRST/protocol/manifest.json')


def old_folder(method):
    if method=='Ours':return ROOT/'outputs/final6108_target_domains_20261101/SIRST'
    rows=list(csv.DictReader((ROOT/'experiments/evidence/comparison_single_dataset.csv').open(encoding='utf-8-sig')))
    r=next(r for r in rows if r['dataset']=='SIRST' and r['method']==method)
    return (ROOT/r['source']).parent.parent


def ground_truth(m,sid):
    r=m['records'][sid];h,w=r['size_hw']
    return np.array(Image.open(r['mask']).convert('L').resize((w,h),Image.Resampling.NEAREST))>0


def finish(method, *, checkpoint=None, checkpoint_hash=None):
    m=manifest();out=OUT/'SIRST'/method;old=old_folder(method)
    discrete=method not in INFRARED
    rows=[];legacy=[];historical=[];matches=0
    for sid in m['test_ids']:
        native=np.load(out/'working_predictions'/f'{sid}.npy')
        rec=m['records'][sid];h,w=rec['size_hw'];gt=ground_truth(m,sid)
        pred=restore_to_original(native,(h,w),discrete=discrete)
        stored=np.load(out/'predictions'/f'{sid}.npy')
        assert stored.shape==gt.shape and np.array_equal(pred,stored),sid
        rows.append(image_metrics(stored,gt,sid))
        previous=cv2.resize(native,(w,h),interpolation=cv2.INTER_NEAREST if discrete else cv2.INTER_LINEAR)
        legacy.append(image_metrics(previous,gt,sid))
        old_path=old/'predictions'/f'{sid}_K5.npy' if method=='Ours' else old/'predictions'/f'{sid}.npy'
        old_pred=np.load(old_path)
        historical.append(image_metrics(old_pred,gt,sid))
        matches+=int(np.array_equal(old_pred>=.5,previous>=.5))
    assert len(rows)==86 and {r['sample_id'] for r in rows}==set(m['test_ids'])
    assert {p.stem for p in (out/'predictions').glob('*.npy')}==set(m['test_ids'])
    if checkpoint is not None:assert sha(checkpoint)==checkpoint_hash
    s=summarize(rows);s.update(method=method,dataset='SIRST',complete_test=True,spatial_protocol=SPATIAL_PROTOCOL,
        correction_only=True,offline_retraining=False,gradient_training=method in ('ABCDFSS','TVGTANet'),episode_adaptation=method in ('ABCDFSS','TVGTANet'),
        actual_annotation_budget=5 if method in ('ABCDFSS','TVGTANet','MPA') else 50,
        manifest_sha256=m['manifest_sha256'],source_run=str(old.relative_to(ROOT)),
        original_checkpoint=str(checkpoint.relative_to(ROOT)) if checkpoint else None,
        checkpoint_sha256=checkpoint_hash,legacy_replay=summarize(legacy),historical=summarize(historical),
        legacy_binary_predictions_equal_images=matches,legacy_binary_predictions_total=86,
        native_resolution=np.load(out/'working_predictions'/f'{m["test_ids"][0]}.npy').shape)
    write_csv(out/'metrics/per_image.csv',rows)
    write_csv(out/'metrics/legacy_replay_per_image.csv',legacy)
    write_json(out/'metrics/summary.json',s)
    print(json.dumps(dict(method=method,stage='complete',IoU=s['IoU_percent'],legacy_binary_equal=matches),ensure_ascii=False),flush=True)


def build_infrared(method):
    # Execute only the existing native-constructor section, before checkpoint,
    # optimizer, training, and evaluation code. This keeps all adapters identical.
    import scripts.run_comparison_hdnet as original
    path=Path(original.__file__);tree=ast.parse(path.read_text(encoding='utf-8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    def assignment(node,name):
        return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
    start=next(i for i,n in enumerate(main.body) if assignment(n,'source'))
    end=next(i for i,n in enumerate(main.body) if assignment(n,'commit'))
    assert start<end
    ns=dict(vars(original));ns['args']=SimpleNamespace(root=DEFAULT_OUTPUT,method=method)
    exec(compile(ast.Module(body=main.body[start:end],type_ignores=[]),str(path),'exec'),ns)
    return ns


def infrared(method):
    m=manifest();old=old_folder(method);out=OUT/'SIRST'/method
    config=read(old/'config.json');cp=old/'checkpoints/latest.pt';before=sha(cp)
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    assert ck['manifest_sha256']==m['manifest_sha256']
    ns=build_infrared(method);model=ns['HDNet'](3).cuda().eval();model.load_state_dict(ck['model'],strict=True)
    if method=='ILNet':m['train50_rgb_stats']=config['normalization']
    if method=='SCTransNet':m['train50_gray_stats']=config['normalization']
    dataset=ns['dataset_class'](m,False,ns['IRSTD_Dataset'])
    write_json(out/'config.json',dict(original_config=config,spatial_protocol=SPATIAL_PROTOCOL,
        checkpoint=str(cp),checkpoint_sha256=before,epoch=ck['epoch'],training=False,
        input_geometry='unchanged native full-image resize; pixel-center image interpolation',
        continuous_restore='bilinear then original threshold .5'))
    with torch.no_grad():
        for i in range(len(dataset)):
            x,_,sid=dataset[i];dest=out/'predictions'/f'{sid}.npy'
            if dest.exists():continue
            _,logits=model(x[None].cuda(),True)
            probability=(logits if ns['prediction_is_probability'] else torch.sigmoid(logits))[0,0].cpu().numpy()
            assert np.isfinite(probability).all()
            np.save(out/'working_predictions'/f'{sid}.npy',probability)
            np.save(dest,restore_to_original(probability,m['records'][sid]['size_hw'],discrete=False))
            if (i+1)%10==0:print(json.dumps(dict(method=method,images=i+1,total=86)),flush=True)
    finish(method,checkpoint=cp,checkpoint_hash=before)


def fss(method):
    from scripts.run_fss50 import native_model
    m=manifest();out=OUT/'SIRST'/method;old=old_folder(method)
    cp=old/'checkpoints/latest.pt';before=sha(cp);ck=torch.load(cp,map_location='cpu',weights_only=False)
    native_name='IFA' if method=='IFA_50label' else 'MPA'
    model,_,_=native_model(DEFAULT_OUTPUT,native_name);model.cuda().eval();model.load_state_dict(ck['model'],strict=True)
    config=read(old/'config.json')
    assert config['manifest_sha256']==m['manifest_sha256']
    ids=m['train_ids'] if method=='IFA_50label' else m['support_ids'];model.shot=len(ids)
    transform=transforms.Compose([transforms.Resize((400,400)),transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    sx=[];sy=[]
    for sid in ids:
        rec=m['records'][sid];sx.append(transform(Image.open(rec['image']).convert('RGB'))[None].cuda())
        mask=np.array(Image.open(rec['mask']).convert('L').resize((400,400),Image.Resampling.NEAREST))>0
        sy.append(torch.from_numpy(mask.astype(np.float32))[None].cuda())
    write_json(out/'config.json',dict(original_config=config,spatial_protocol=SPATIAL_PROTOCOL,
        checkpoint=str(cp),checkpoint_sha256=before,epoch=ck['epoch'],training=False,inference_support_ids=ids,
        discrete_restore='nearest_exact; native argmax unchanged'))
    with torch.no_grad():
        for i,sid in enumerate(m['test_ids']):
            dest=out/'predictions'/f'{sid}.npy'
            if dest.exists():continue
            rec=m['records'][sid];query=transform(Image.open(rec['image']).convert('RGB'))[None].cuda()
            prediction=model(sx,sy,[query],[None])[0] if method=='MPA' else model(sx,sy,query,None)[0]
            native=prediction.argmax(1)[0].float().cpu().numpy()
            np.save(out/'working_predictions'/f'{sid}.npy',native)
            np.save(dest,restore_to_original(native,rec['size_hw'],discrete=True))
            del query,prediction
            if (i+1)%10==0:print(json.dumps(dict(method=method,images=i+1,total=86)),flush=True)
    finish(method,checkpoint=cp,checkpoint_hash=before)


def ours():
    m=manifest();old=old_folder('Ours');out=OUT/'SIRST/Ours';cp=old/'artifacts/heads.pt';before=sha(cp)
    for sid in m['test_ids']:
        case=read(old/'cache/cases'/f'{sid}.json');working=np.zeros((512,512),np.uint8)
        for p in case['candidates']:
            if p['kept']:paste_local(working,np.load(old/'cache/local_masks'/f'{sid}_{p["y"]}_{p["x"]}.npy'),p['y'],p['x'])
        np.save(out/'working_predictions'/f'{sid}.npy',working)
        np.save(out/'predictions'/f'{sid}.npy',restore_to_original(working,m['records'][sid]['size_hw'],discrete=True))
    write_json(out/'config.json',dict(spatial_protocol=SPATIAL_PROTOCOL,reference='final_6108',
        checkpoint=str(cp),checkpoint_sha256=before,TopK=5,Obj_threshold=.1,NMS=0,Mask_threshold=0,
        frozen_ROI_masks_reused=True,training=False))
    finish('Ours',checkpoint=cp,checkpoint_hash=before)


def worker(method):
    torch.set_num_threads(2);random.seed(111);np.random.seed(111);torch.manual_seed(111);torch.cuda.manual_seed_all(111)
    out=OUT/'SIRST'/method
    for sub in ('predictions','working_predictions','metrics'):(out/sub).mkdir(parents=True,exist_ok=True)
    if (out/'metrics/summary.json').exists() and 'legacy_replay' in read(out/'metrics/summary.json'):return
    if method in INFRARED:infrared(method)
    elif method=='Ours':ours()
    elif method in ('MPA','IFA_50label'):fss(method)
    else:
        args=[sys.executable,'-u','-m','scripts.run_comparison_abcdfss','--method',method,'--datasets','SIRST','--coordinate-output',str(OUT)]
        subprocess.run(args,check=True,cwd=ROOT)
        finish(method)


def queue():
    OUT.mkdir(exist_ok=True)
    snapshot=OUT/'previous_table';snapshot.mkdir(exist_ok=True)
    for p in (ROOT/'docs/comparison_three_datasets.md',ROOT/'experiments/tables/comparison_three_datasets.tex',ROOT/'experiments/evidence/comparison_three_datasets.json'):
        if not (snapshot/p.name).exists():shutil.copyfile(p,snapshot/p.name)
    protocol=dict(spatial_protocol=SPATIAL_PROTOCOL,dataset='SIRST',methods=list(METHODS),test_images=86,
        manifest_sha256=manifest()['manifest_sha256'],test_selection=False,parameter_search=False,
        original_training_checkpoints_frozen=True,train50_support5_manifests_unchanged=True,
        continuous_predictions='bilinear pixel-center restore then original .5 threshold',
        discrete_predictions='nearest_exact pixel-center restore, native argmax/threshold unchanged',
        input_resolution_and_normalization='retain each recorded native setting',
        ABCDFSS_TVGTANet='repeat original deterministic per-query adaptation; reuse identical cached CAMs')
    write_json(OUT/'protocol.json',protocol)
    failures=[]
    for method in METHODS:
        write_json(OUT/'status.json',dict(stage='running',current_method=method,failures=failures))
        result=subprocess.run([sys.executable,'-u','-m','scripts.run_nuaa_coordinate_corrected','--method',method],cwd=ROOT)
        if result.returncode:failures.append(method)
    write_json(OUT/'status.json',dict(stage='failed' if failures else 'inference_complete',failures=failures))
    if failures:raise RuntimeError(f'Coordinate reruns need repair: {failures}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS);args=p.parse_args()
    if args.method:worker(args.method)
    else:queue()


if __name__=='__main__':main()
