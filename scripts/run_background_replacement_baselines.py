"""Frozen background robustness evaluation; one upstream method per process."""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from scripts.background_replacement_data import (BASE,OUT,DOMAINS,METHODS,ROOT,manifest,read,compose,checkpoint,digest,dump)
from visual_prompt.evaluation.comparison import image_metrics,summarize

class Wrapper(torch.nn.Module):
    def __init__(self,native,method):
        super().__init__();self.native=native;self.method=method
    def forward(self,x):
        y=self.native(x)
        if self.method=='RPCANet': y=y[1]
        elif isinstance(y,(tuple,list)): y=y[0] if self.method=='ILNet' else y[-1]
        return y if self.method=='SCTransNet' else y.sigmoid()

def build(method):
    sys.path.insert(0,str(BASE/'sources'/method));sys.path.append(str(BASE/'runtime_packages'))
    if method=='DNANet':
        from model.model_DNANet import DNANet,Res_CBAM_block
        native=DNANet(1,3,Res_CBAM_block,[2,2,2,2],[16,32,64,128,256],deep_supervision=True)
    elif method=='SCTransNet':
        from model.SCTransNet import SCTransNet,get_CTranS_config
        native=SCTransNet(get_CTranS_config(),mode='train',deepsuper=True)
    elif method=='RPCANet':
        from models import get_model
        native=get_model('rpcanet')
    else:
        from model.ilnet import ILNet_L
        native=ILNet_L()
    return Wrapper(native,method)

def transform(image,method,cfg,original_path=None):
    if method=='RPCANet':
        if original_path:
            gray=cv2.imread(str(original_path),cv2.IMREAD_GRAYSCALE)
        else:
            # Match the author's IMREAD_GRAYSCALE decoding for constructed PNG inputs.
            ok,encoded=cv2.imencode('.png',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            assert ok
            gray=cv2.imdecode(encoded,cv2.IMREAD_GRAYSCALE)
        x=cv2.resize(gray,(256,256),interpolation=cv2.INTER_LINEAR).astype('float32')[None]/255
    elif method=='SCTransNet':
        gray=np.asarray(Image.fromarray(image).convert('L'),dtype='float32')
        stats=cfg['normalization'];gray=(gray-stats['mean'])/stats['std']
        x=cv2.resize(gray,(256,256),interpolation=cv2.INTER_LINEAR)[None]
    else:
        size=512 if method=='ILNet' else 256
        x=np.asarray(Image.fromarray(image).resize((size,size),Image.Resampling.BILINEAR),dtype='float32').transpose(2,0,1)/255
        stats=cfg['normalization'] if method=='ILNet' else dict(mean=[.485,.456,.406],std=[.229,.224,.225])
        x=(x-np.array(stats['mean'],dtype='float32')[:,None,None])/np.array(stats['std'],dtype='float32')[:,None,None]
    return torch.from_numpy(np.ascontiguousarray(x))

def save_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

@torch.inference_mode()
def worker(method,limit):
    torch.set_num_threads(2);torch.manual_seed(111);np.random.seed(111)
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    model=build(method).cuda().eval()
    for domain in DOMAINS:
        p,c,_=checkpoint(method,domain)
        locked=next(r for r in protocol['checkpoints'] if r['method']==method and r['domain']==domain)
        assert digest(p)==locked['sha256']
        assert digest(BASE/domain/'protocol/manifest.json')==protocol['manifests'][domain]
        cfg=json.loads(c.read_text(encoding='utf-8'))
        state=torch.load(p,map_location='cpu',weights_only=False)
        for key in ('epoch','updates'):
            if key in locked: assert state[key]==locked[key]
        model.load_state_dict(state['model'],strict=True);del state
        folder=OUT/('smoke' if limit else 'results')/method/domain;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'summary.json').exists():
            prior=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            assert prior['checkpoint_sha256']==locked['sha256']
            print(f'SKIP completed {method} {domain}',flush=True);continue
        ids=manifest(domain)['test_ids'];ids=ids[:limit] if limit else ids
        rows=[];start=time.monotonic()
        for index,sid in enumerate(ids):
            im,gt,_=read(domain,sid)
            conditions=[('original',-1,im,dict(donor_domain='',donor_id='',replicate=-1,input_sha256=__import__('hashlib').sha256(im.tobytes()).hexdigest(),gain=[],offset=[],changed_background_fraction=0.))]
            for dd in DOMAINS:
                for rep in range(2):
                    synthetic,meta=compose(domain,sid,dd,rep)
                    conditions.append((dd,rep,synthetic,meta))
            # Native sizes equal within a method, allowing a small inference batch.
            for offset in range(0,len(conditions),2):
                part=conditions[offset:offset+2]
                x=torch.stack([transform(r[2],method,cfg,manifest(domain)['records'][sid]['image'] if r[0]=='original' else None) for r in part]).cuda()
                probabilities=model(x)[:,0].cpu().numpy()
                for (condition,rep,_,meta),prob in zip(part,probabilities):
                    prob=cv2.resize(prob,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_LINEAR)
                    row=dict(method=method,dataset=domain,condition=condition,**meta,**image_metrics(prob,gt,sid))
                    row['gain']=json.dumps(row['gain']);row['offset']=json.dumps(row['offset']);rows.append(row)
            if (index+1)%25==0 or index+1==len(ids):
                event=dict(method=method,dataset=domain,images=index+1,total=len(ids),elapsed_seconds=round(time.monotonic()-start,1))
                dump(OUT/'status.json',event);print(json.dumps(event),flush=True)
        save_csv(folder/'per_image.csv',rows)
        scores={}
        for condition in ['original']+DOMAINS:
            reps=[-1] if condition=='original' else [0,1]
            per=[summarize([r for r in rows if r['condition']==condition and r['replicate']==rep]) for rep in reps]
            # Average independent replicate-level dataset metrics, avoiding pooling random repetitions.
            scores[condition]=dict(replicates=per,**{k:float(np.mean([r[k] for r in per])) for k in ('IoU_percent','nIoU_percent','Pd_percent','Fa_per_million')})
        cross=float(np.mean([scores[d]['IoU_percent'] for d in DOMAINS if d!=domain]))
        result=dict(method=method,dataset=domain,images=len(ids),checkpoint_sha256=locked['sha256'],checkpoint=str(p),training={k:locked[k] for k in ('epoch','updates','adaptation') if k in locked},scores=scores,cross_background_mean_iou=cross,drop_from_original_pp=scores['original']['IoU_percent']-cross,drop_from_same_domain_control_pp=scores[domain]['IoU_percent']-cross,elapsed_seconds=time.monotonic()-start)
        dump(folder/'summary.json',result)
        print(json.dumps(dict(method=method,dataset=domain,status='complete',original_iou=scores['original']['IoU_percent'],cross_background_mean_iou=cross,drop_pp=result['drop_from_original_pp'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if a.method: worker(a.method,a.limit)
    else:
        for method in METHODS:
            subprocess.run([sys.executable,'-m','scripts.run_background_replacement_baselines','--method',method,'--limit',str(a.limit)],cwd=ROOT,check=True)
        dump(OUT/('smoke_complete.json' if a.limit else 'complete.json'),dict(status='complete',methods=METHODS,domains=DOMAINS))

if __name__=='__main__':main()
