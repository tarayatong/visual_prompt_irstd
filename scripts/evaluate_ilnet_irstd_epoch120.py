"""Evaluate the preserved seed111 train50 rerun snapshot at epoch120."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/'outputs/comparison_50shot_5support'
sys.path[:0]=[str(BASE/'sources/ILNet'),str(BASE/'runtime_packages')]
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from model.ilnet import ILNet_L
from utils.data import SirstDataset
from scripts.comparison_ilnet_adapter import ILNetManifestDataset,ILNetNativeOutputs
from visual_prompt.evaluation.comparison import image_metrics,summarize


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    torch.set_num_threads(2)
    source=ROOT/'outputs/motivation_three_sources_20261108/source_preparation/ILNet_IRSTD-1k125'
    output=ROOT/'outputs/comparison_ilnet_epoch120_20261109/IRSTD-1k/ILNet'
    m=read(BASE/'IRSTD-1k/protocol/manifest.json');protocol=read(source/'protocol.json')
    cp=source/'latest.pt';state=torch.load(cp,map_location='cpu',weights_only=False)
    assert state['epoch']==120 and len(state['history'])==120
    assert [r['epoch'] for r in state['history']]==list(range(1,121))
    assert state['protocol_sha256']==sha(source/'protocol.json')
    assert state['manifest_sha256']==m['manifest_sha256']==protocol['manifest_sha256']
    assert protocol['source_annotation_ids']==m['train_ids'] and protocol['seed']==111
    assert len(m['train_ids'])==50 and len(m['test_ids'])==201
    assert not set(m['train_ids']) & set(m['test_ids'])
    m['train50_rgb_stats']=protocol['normalization']
    provenance=dict(method='ILNet',dataset='IRSTD-1k',epoch=120,seed=111,train_images=50,
        source_checkpoint=cp.relative_to(ROOT).as_posix(),source_checkpoint_sha256=sha(cp),
        source_protocol=protocol,manifest_sha256=m['manifest_sha256'],runner_sha256=sha(Path(__file__)),
        note='Available seed111 rerun snapshot, not the deleted intermediate checkpoint of the historical500-epoch run; original500-horizon cosine retained, evaluated at120. No retraining or test selection.',
        input_size=512,threshold=.5)
    save(output/'config.json',provenance)
    save(output/'training_history.json',state['history'])
    retained=output/'checkpoints/epoch_0120.pt';retained.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(epoch=120,model=state['model'],provenance=provenance),retained)
    model=ILNetNativeOutputs(ILNet_L());model.load_state_dict(state['model'],strict=True)
    model.cuda().eval();predictions=output/'epoch_0120/predictions';predictions.mkdir(parents=True,exist_ok=True)
    rows=[]
    with torch.inference_mode():
        for x,_,ids in DataLoader(ILNetManifestDataset(m,False,SirstDataset),batch_size=1,num_workers=0):
            sid=ids[0];_,z=model(x.cuda(),True);assert torch.isfinite(z).all()
            h,w=m['records'][sid]['size_hw']
            prob=cv2.resize(z[0,0].sigmoid().cpu().numpy(),(w,h),interpolation=cv2.INTER_LINEAR)
            pred=prob>=.5
            gt=np.array(Image.open(m['records'][sid]['mask']).convert('L').resize((w,h),Image.Resampling.NEAREST))>0
            rows.append(image_metrics(pred,gt,sid))
            Image.fromarray(pred.astype(np.uint8)*255).save(predictions/f'{sid}.png')
    recount=[]
    assert {p.stem for p in predictions.glob('*.png')}==set(m['test_ids'])
    for sid in m['test_ids']:
        pred=np.array(Image.open(predictions/f'{sid}.png'))>0
        h,w=m['records'][sid]['size_hw'];assert pred.shape==(h,w)
        gt=np.array(Image.open(m['records'][sid]['mask']).convert('L').resize((w,h),Image.Resampling.NEAREST))>0
        recount.append(image_metrics(pred,gt,sid))
    assert summarize(rows)==summarize(recount)
    result=dict(summarize(recount),method='ILNet',dataset='IRSTD-1k',epochs=120,seed=111,
        actual_annotation_budget=50,complete_test=True,checkpoint=retained.relative_to(ROOT).as_posix(),
        checkpoint_sha256=sha(retained),source_checkpoint_sha256=provenance['source_checkpoint_sha256'],
        predictions_recount_verified=True,note=provenance['note'])
    metrics=output/'epoch_0120/metrics';save(metrics/'summary.json',result)
    with (metrics/'per_image.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(recount[0]));writer.writeheader();writer.writerows(recount)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
