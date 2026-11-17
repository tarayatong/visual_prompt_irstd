"""Single retained P2P inference and evidence-verification entry."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from visual_prompt.pipelines.p2p_framework import P2PModel,RETAINED
from visual_prompt.evaluation.comparison import image_metrics,summarize
from visual_prompt.evaluation.spatial_alignment import restore_to_original


def read_counts(path):
    rows=[]
    with Path(path).open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            for key,value in list(r.items()):
                if key=='sample_id':continue
                try:r[key]=float(value) if any(c in value for c in '.eE') else int(value)
                except (ValueError,TypeError):pass
            rows.append(r)
    return rows


def verify():
    catalog=json.loads((RETAINED/'catalog.json').read_text());checked=0
    for r in catalog:
        p=RETAINED/r['dataset']/r['variant'];assert hashlib.sha256((p/'model.pt').read_bytes()).hexdigest()==r['sha256']
        rows=read_counts(p/'per_image.csv');m=summarize(rows)
        for key in ('IoU_percent','nIoU_percent','Pd_percent','Fa_per_million'):assert abs(m[key]-r['metrics'][key])<1e-8
        manifest=json.loads((RETAINED/r['dataset']/'manifest.json').read_text())
        assert {x['sample_id'] for x in rows}==set(manifest['test_ids'])
        assert not set(manifest['train_ids'])&set(manifest['test_ids']);checked+=len(rows)
    cleanup=RETAINED/'cleanup_manifest.json'
    if cleanup.exists():
        audit=json.loads(cleanup.read_text(encoding='utf-8'))
        for asset in audit['retained_assets']:
            path=RETAINED.parents[2]/asset['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==asset['sha256'],str(path)
        if audit['status']=='complete':assert all(not Path(t['absolute_path']).exists() for t in audit['targets'])
    print(json.dumps(dict(passed=True,models=len(catalog),images_recounted=checked,retained_assets_checked=bool(cleanup.exists()))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['IRSTD-1k','SIRST','NUDT-SIRST'],default='IRSTD-1k')
    p.add_argument('--variant',default='frozen');p.add_argument('--model-dir',type=Path);p.add_argument('--predict');p.add_argument('--output',type=Path);p.add_argument('--evaluate',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--verify',action='store_true');p.add_argument('--k',type=int)
    a=p.parse_args();torch.set_num_threads(2)
    if a.verify:verify();return
    model=P2PModel(a.dataset,a.variant,artifact_dir=a.model_dir)
    if a.predict:
        if a.output is None:p.error('--predict requires --output')
        image=np.asarray(Image.open(a.predict).convert('RGB'));mask,info=model.predict(image,a.k)
        result=model.restore(mask,image.shape[:2]);a.output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(result.astype(np.uint8)*255).save(a.output);print(json.dumps(info));return
    if a.evaluate:
        m=json.loads((model.data_folder/'manifest.json').read_text());ids=m['test_ids'][:a.limit] if a.limit else m['test_ids'];rows=[]
        for sid in ids:
            image=np.asarray(Image.open(m['records'][sid]['image']).convert('RGB'));gt=np.asarray(Image.open(m['records'][sid]['mask']).convert('L'))>0
            mask,_=model.predict(image,a.k);rows.append(image_metrics(model.restore(mask,gt.shape),gt,sid))
        result=summarize(rows)
        if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(metrics=result,per_image=rows),indent=2)+'\n')
        print(json.dumps(result,indent=2));return
    p.error('Choose --verify, --predict, or --evaluate')


if __name__=='__main__':main()
