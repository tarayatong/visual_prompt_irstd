"""Retain the exact user-selected comparison; all deletes are journaled and scoped."""
from __future__ import annotations
import argparse
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/retention_comparison_20261105'
BASE=ROOT/'outputs/comparison_50shot_5support'
RECENT=ROOT/'outputs/comparison_recent_a_20261104'
ADAPT=ROOT/'outputs/comparison_task_adaptation_20261104'
RECOVERY=ROOT/'outputs/comparison_recovery_20261102'
DATASETS=('SIRST','IRSTD-1k','NUDT-SIRST')
METHODS=('ABCDFSS','IFA','TVGTANet','SANSA','PAHNet','MPA','HERA','ISNet','DNANet','MSHNet','SCTransNet','RPCANet','ILNet','HDNet','SAM-SPL','MLP-Net','MSHNet+NS-FPN')
REMOVED=('Matcher','PI-CLIP','DSV-LFS','PConv+SD','SFDTNet','IOVarNet')
SOURCE_REMOVE=('Matcher','PI-CLIP','DSV-LFS','PConv','SFDTNet','IOVarNet','DGNet','ISA','CLIP')
USER_IOU={
 'ABCDFSS':(14.65,7.07,9.36),'IFA':(37.68,19.55,29.20),'TVGTANet':(21.87,11.62,20.73),
 'SANSA':(62.08,62.48,65.96),'PAHNet':(10.31,6.24,17.72),'MPA':(26.67,1.52,24.79),'HERA':(3.64,None,1.66),
 'ISNet':(56.66,5.06,36.52),'DNANet':(57.35,58.87,62.20),'MSHNet':(60.08,55.52,57.61),
 'SCTransNet':(59.03,56.35,63.72),'RPCANet':(46.31,43.41,17.37),'ILNet':(63.21,61.78,56.57),
 'HDNet':(62.28,59.29,61.61),'SAM-SPL':(54.25,30.82,46.80),'MLP-Net':(21.98,6.83,16.86),'MSHNet+NS-FPN':(58.35,53.17,53.00)}
ADAPTED={('ABCDFSS','SIRST'),('ABCDFSS','NUDT-SIRST'),('IFA','IRSTD-1k'),('TVGTANet','NUDT-SIRST'),
         ('ISNet','IRSTD-1k'),('RPCANet','NUDT-SIRST'),('HDNet','NUDT-SIRST'),('MSHNet+NS-FPN','NUDT-SIRST')}
FIELDS=('IoU','nIoU','F1','Pd','Fa')


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def relative(p):return p.resolve().relative_to(ROOT).as_posix()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def inside(p,base):return p.resolve().is_relative_to(base.resolve())
def checkpoint(folder,epoch):
 candidates=[folder/'checkpoints'/f'epoch_{epoch:04d}.pt',folder/'checkpoints/latest.pt']
 return next(p for p in candidates if p.exists())


def make_plan():
 OUT.mkdir(exist_ok=True)
 for p in (ROOT/'docs/comparison_three_datasets.md',ROOT/'experiments/tables/comparison_three_datasets.tex',ROOT/'experiments/tables/comparison_three_datasets.csv',ROOT/'experiments/evidence/comparison_three_datasets.json',ROOT/'docs/comparison_task_adaptation_results.md',ROOT/'experiments/evidence/comparison_task_adaptation_20261104.json'):
  dest=OUT/'before'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():shutil.copy2(p,dest)
 evidence=read(ROOT/'experiments/evidence/comparison_three_datasets.json')
 lookup={(r['method'],r['dataset']):r for r in evidence['final_table_results']}
 rows=[];kept_weights=set();kept_runs=set();warnings=[]
 for method in METHODS:
  for i,dataset in enumerate(DATASETS):
   old=lookup[method,dataset];value=USER_IOU[method][i]
   if value is None:
    rows.append(dict(method=method,dataset=dataset,values=None,status='not_reported_by_user',checkpoint=None));continue
   assert f"{old['values']['IoU']:.2f}"==f'{value:.2f}',(method,dataset,old['values'])
   row=dict(old);row['table_adaptation_star']=(method,dataset) in ADAPTED
   if row['table_adaptation_star']:
    folder=ADAPT/method/dataset
    if method=='ISNet':folder=ADAPT/'followups'/method/dataset
    weight=folder/'full50/model.pt' if method not in ('ABCDFSS','TVGTANet') else None
    epoch=None
   elif method in ('SANSA','PAHNet','SAM-SPL','MLP-Net'):
    folder=RECENT/dataset/method;weight=checkpoint(folder,120);epoch=120
   elif method=='HERA':
    folder=BASE/dataset/method;weight=folder/'checkpoints/support_adaptation.pt';epoch=None
   else:
    folder=BASE/('fss50_v2' if method=='IFA' else 'full16')/dataset/method
    reported=str(old.get('epochs',''))
    epoch=125 if dataset=='NUDT-SIRST' and method in ('SCTransNet','ILNet') else int(reported) if reported.isdigit() else 500
    weight=checkpoint(folder,epoch) if method not in ('ABCDFSS','TVGTANet') else None
   assert folder.exists(),folder
   kept_runs.add(folder.resolve())
   config=folder/'full50/config.json' if row['table_adaptation_star'] and weight else folder/'config.json'
   if row['table_adaptation_star'] and method=='IFA':config=folder/'full50/training_config.json'
   if not config.exists():config=folder/'selection.json'
   row.update(run_directory=relative(folder),checkpoint=relative(weight) if weight else None,
              expected_epoch=epoch,config=relative(config) if config.exists() else None,
              metric_source_exists=(ROOT/row['source']).exists(),
              weight_kind='support_adaptation_plus_frozen_backbone' if method=='HERA' else 'per_query_adaptation_no_static_task_checkpoint' if weight is None else 'one_retained_task_checkpoint')
   if weight:
    assert weight.exists(),weight;kept_weights.add(weight.resolve())
   if not row['metric_source_exists']:
    warnings.append(dict(method=method,dataset=dataset,missing_source=row['source'],note='Historical serialized metrics retained; raw metric directory was already absent before this cleanup. No new inference claimed.'))
   dest=OUT/'selected'/dataset/method
   write(dest/'record.json',row)
   if config.exists():shutil.copy2(config,dest/config.name)
   for name in ('selection.json','reuse.json','protocol.json'):
    if (folder/name).exists() and (folder/name)!=config:shutil.copy2(folder/name,dest/name)
   rows.append(row)
 # Our retained ablations/Top-k and all MPCNet assets are explicitly outside deletion scope.
 ours=[]
 for ds,cp,summary,variant in (
  ('SIRST','outputs/final_memory_radial_ablation_20261104/SIRST/M2/learned/seed111/model.pt','outputs/final_memory_radial_ablation_20261104/SIRST/M2/evaluation_counts','learned'),
  ('IRSTD-1k','outputs/final_memory_radial_ablation_20261104/IRSTD-1k/M2/learned/seed111/model.pt','outputs/final_memory_radial_ablation_20261104/IRSTD-1k/M2/evaluation_counts','learned'),
  ('NUDT-SIRST','outputs/no_center_memory_radial_ablation_20261105/NUDT-SIRST/learned/seed111/model.pt','outputs/no_center_memory_radial_ablation_20261105/NUDT-SIRST/evaluation_counts','learned')):
  p=ROOT/cp;assert p.exists();ours.append(dict(method='Ours',dataset=ds,checkpoint=cp,evaluation_directory=summary,variant=variant))
 write(OUT/'ours_protected.json',ours)
 # Collect only method-owned folders at known comparison locations.
 delete_dirs=set()
 for root in (BASE,RECENT,ADAPT):
  for name in SOURCE_REMOVE:
   p=root/'sources'/name
   if p.exists():delete_dirs.add(p.resolve())
  for ds in DATASETS:
   for base in (root/ds,root/'full16'/ds,root/'fss50_v2'/ds):
    for name in (*METHODS,*REMOVED):
     p=base/name
     if p.exists() and p.resolve() not in kept_runs:delete_dirs.add(p.resolve())
  if root==ADAPT:
   for name in METHODS:
    for ds in DATASETS:
     p=root/name/ds
     if p.exists() and p.resolve() not in kept_runs:delete_dirs.add(p.resolve())
   for name in METHODS:
    for ds in DATASETS:
     p=root/'followups'/name/ds
     if p.exists() and p.resolve() not in kept_runs:delete_dirs.add(p.resolve())
 for name in ('IFA','HDNet','RPCANet','MSHNet+NS-FPN','ISNet'):
  if (RECOVERY/name).exists():delete_dirs.add((RECOVERY/name).resolve())
 for folder in kept_runs:
  for p in folder.iterdir():
   if p.is_dir() and (p.name=='validation' or p.name.startswith('full50_')):delete_dirs.add(p.resolve())
 # Only weights within known external-method folders are candidates for pruning.
 delete_files=set()
 inventory=read(OUT/'initial_weight_inventory.json')
 for record in inventory:
  p=(ROOT/record['path']).resolve()
  if p in kept_weights:continue
  if any(inside(p,r) for r in (BASE,RECENT,ADAPT,RECOVERY)):
   parts=set(p.parts)
   if parts.intersection(set(METHODS)|set(REMOVED)) or inside(p,RECOVERY):delete_files.add(p)
 for name in ('dinov2_vitl14_pretrain.pth','sam_vit_h_4b8939.pth','hdmnet_coco_split0_resnet50_5shot.pth','pspnet_coco_split0_resnet50.pth','resnet50_v2_hdmnet.pth'):
  p=BASE/'weights'/name
  if p.exists():delete_files.add(p.resolve())
 for name in ('dinov2_vitl14_pretrain.pth','dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'):
  p=BASE/'runtime_cache/torch_hub/checkpoints'/name
  if p.exists():delete_files.add(p.resolve())
 if (RECENT/'weights/ViT-B-32.pt').exists():delete_files.add((RECENT/'weights/ViT-B-32.pt').resolve())
 for name in ('run_comparison_matcher.py','comparison_piclip_adapter.py','download_piclip_dependencies.py'):
  p=ROOT/'scripts'/name
  if p.exists():delete_files.add(p.resolve())
 # Drop nested entries; avoid counting or deleting a file twice.
 delete_dirs={p for p in delete_dirs if not any(p!=q and inside(p,q) for q in delete_dirs)}
 delete_files={p for p in delete_files if not any(inside(p,q) for q in delete_dirs)}
 assert not any(inside(w,d) or w==d for w in kept_weights for d in delete_dirs|delete_files)
 candidates={p for d in delete_dirs for p in d.rglob('*') if p.is_file()}|delete_files
 bytes_deleted=sum(p.stat().st_size for p in candidates)
 plan=dict(created=datetime.now().isoformat(timespec='seconds'),scope='user supplied17 external methods plus Ours; MPCNet separately protected',
  selected_methods=list(METHODS),supplemental_protected_method='MPCNet',removed_methods=list(REMOVED),
  selected_rows=rows,selected_checkpoints=sorted(relative(p) for p in kept_weights),
  retained_run_directories=sorted(relative(p) for p in kept_runs),
  delete_directories=sorted(relative(p) for p in delete_dirs),delete_files=sorted(relative(p) for p in delete_files),
  delete_file_count=len(candidates),delete_bytes=bytes_deleted,legacy_missing_metric_sources=warnings,
  protected_prefixes=['outputs/comparison_mpcnet_20261105','outputs/final_memory_radial_ablation_20261104','outputs/no_center_memory_radial_ablation_20261105','outputs/retained','outputs/topk_best_three_domains_20261105'],
  shared_dependency_weights=[relative(p) for p in (BASE/'weights').glob('*') if p.suffix in ('.pt','.pth') and p.resolve() not in delete_files])
 write(OUT/'retention_plan.json',plan)
 print(json.dumps(dict(selected_cells=sum(r['values'] is not None for r in rows),unique_external_checkpoints=len(kept_weights),missing_legacy_sources=len(warnings),delete_files=len(candidates),delete_GiB=round(bytes_deleted/2**30,3)),ensure_ascii=False))


def verify():
 import torch
 torch.set_num_threads(2)
 plan=read(OUT/'retention_plan.json');audit=[]
 for rel in plan['selected_checkpoints']:
  p=ROOT/rel;s=torch.load(p,map_location='cpu',weights_only=False)
  rows=[r for r in plan['selected_rows'] if r.get('checkpoint')==rel]
  for row in rows:
   if row['expected_epoch'] is not None:assert s['epoch']==row['expected_epoch'],(rel,s.get('epoch'))
  weights=s.get('model') or s.get('adapted_parameters')
  assert isinstance(weights,dict) and weights,rel
  tensors=[v for v in weights.values() if torch.is_tensor(v)]
  assert tensors and all(torch.isfinite(t).all() for t in tensors),rel
  audit.append(dict(path=rel,sha256=sha(p),epoch=s.get('epoch'),tensors=len(tensors),passed=True))
  del s,weights,tensors;gc.collect()
 write(OUT/'selected_checkpoint_audit.json',audit)
 print(json.dumps(dict(checkpoints=len(audit),passed=True)))


def apply():
 plan=read(OUT/'retention_plan.json');audit=read(OUT/'selected_checkpoint_audit.json')
 assert len(audit)==len(plan['selected_checkpoints']) and all(a['passed'] for a in audit)
 for a in audit:assert sha(ROOT/a['path'])==a['sha256']
 allowed=(BASE,RECENT,ADAPT,RECOVERY,ROOT/'scripts')
 targets=[ROOT/p for p in plan['delete_directories']+plan['delete_files']]
 for p in targets:
  assert p.resolve()!=ROOT and any(inside(p,b) and p.resolve()!=b.resolve() for b in allowed),p
  assert not any(inside(p,ROOT/x) or inside(ROOT/x,p) for x in plan['protected_prefixes']),p
  assert not p.is_symlink() and not getattr(p,'is_junction',lambda:False)(),p
  assert not any(inside(ROOT/k,p) or (ROOT/k).resolve()==p.resolve() for k in plan['selected_checkpoints']),p
 files={p for rel in plan['delete_directories'] for p in (ROOT/rel).rglob('*') if p.is_file()}|{ROOT/p for p in plan['delete_files']}
 for p in files:
  assert inside(p,ROOT) and not any(inside(p,ROOT/x) for x in plan['protected_prefixes']),p
 journal=OUT/'deletion_journal.jsonl';count=total=0
 with journal.open('a',encoding='utf-8') as f:
  for p in sorted(files):
   if not p.exists():continue
   size=p.stat().st_size
   try:p.unlink()
   except PermissionError:
    # Git object files can be read-only on Windows; target scope is checked above.
    p.chmod(stat.S_IREAD|stat.S_IWRITE);p.unlink()
   count+=1;total+=size
   f.write(json.dumps(dict(path=relative(p),bytes=size),ensure_ascii=False)+'\n')
 for rel in plan['delete_directories']:
  p=ROOT/rel
  if p.exists():shutil.rmtree(p)
 for a in audit:assert (ROOT/a['path']).exists() and sha(ROOT/a['path'])==a['sha256']
 result=dict(completed=True,deleted_files=count,deleted_bytes=total,selected_checkpoints_verified=len(audit),MPCNet_protected=True,plan_sha256=sha(OUT/'retention_plan.json'))
 write(OUT/'cleanup_completed.json',result);print(json.dumps(result))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--apply',action='store_true');a=p.parse_args()
 if a.verify:verify()
 elif a.apply:apply()
 else:make_plan()
