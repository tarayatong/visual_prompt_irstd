"""Evaluate the retained combined Tiny checkpoints under the three first-figure protocols."""
import argparse
import csv
import hashlib
import json
import random
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from scripts import background_replacement_data as bg
from scripts import run_recent_a_comparison as base
from scripts.run_tiny_ablation_combined import restore, state_for
from visual_prompt.evaluation.comparison import image_metrics, summarize
from visual_prompt.models.tiny_prototype_feature_radial import install_prototype_feature_radial
from visual_prompt.models.tiny_prototype_memory_replacement import install_prototype_memory
from visual_prompt.models.tiny_shared_memory import build_episode_keys, support_regions

ROOT = bg.ROOT
OUT = ROOT/'outputs/motivation_current_ours_20261111'
DOMAINS = bg.DOMAINS
INDEX = ROOT/'experiments/evidence/tiny_memory_radial_retained.csv'
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temp.replace(path)


def selected():
    rows = {r['dataset']: r for r in csv_rows(INDEX) if r['seed']=='111' and r['variant']=='combined'}
    assert set(rows)==set(DOMAINS)
    return rows


def tensor(im):
    return (torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2,0,1)/255)-MEAN)/STD


class Engine:
    def __init__(self, domain, row):
        base.seed_all()
        self.domain = domain
        m = bg.manifest(domain)
        self.path = ROOT/row['checkpoint']; self.sha = bg.digest(self.path)
        assert self.sha == row['sha256']
        state = torch.load(self.path, map_location='cpu', weights_only=False)
        config = state['config']
        assert state['epoch']==120 and config['variant']=='combined'
        assert config['manifest_sha256']==m['manifest_sha256'] and config.get('training_random_seed',config.get('seed'))==111
        assert config['initialization_sha256']==bg.digest(base.RUN/'weights/sam2_hiera_tiny.pt')
        self.model, *_ = base.setup_sansa(m, torch.device('cuda'))
        install_prototype_memory(self.model); install_prototype_feature_radial(self.model)
        restore(self.model, state['model']); self.model.eval()
        assert sum(p.numel() for p in self.model.parameters() if p.requires_grad)==1160372
        assert sum(p.numel() for p in self.model.parameters())==38589397
        self.support = self.pairs(domain, m['support_ids'])
        self.support_ids = tuple(m['support_ids'])
        self.cache = None

    @staticmethod
    def pairs(domain, ids):
        m = bg.manifest(domain)
        assert set(ids)<=set(m['train_ids']) and not set(ids)&set(m['test_ids'])
        out = {}
        for sid in ids:
            x,y = base.raw_pair(m,sid,640,True)
            out[sid] = (tensor(x), torch.from_numpy(y), support_regions(x.astype(np.uint8), y>0))
        return out

    def backbone(self, batch):
        x,_,_,sizes = self.model._preprocess_visual_features(batch.cuda(),1024)
        return self.model._forward_backbone(x,sizes)

    @torch.no_grad()
    def predict(self, im):
        if self.cache is None:
            xs,_,regions = zip(*(self.support[i] for i in self.support_ids))
            bk = self.backbone(torch.stack(xs)[None])
            keys = build_episode_keys(self.model,bk,range(5),list(regions))
            self.cache = self.model.prototype_memory.prepare(keys)
        x = tensor(Image.fromarray(im).resize((640,640), Image.Resampling.BILINEAR))
        bk = self.backbone(x[None,None])
        branch = self.model.radial_branch
        branch.begin([(None,None)])
        decoded = self.model.prototype_memory.decode(self.model,bk,0,self.cache)
        branch.finish()
        prob = F.interpolate(decoded.masks.float(), (640,640),mode='bilinear',align_corners=False)[0,0].sigmoid().cpu().numpy()
        return cv2.resize(prob,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_LINEAR)

    def adapt(self, domain, steps, out):
        from util.promptable_utils import build_prompt_dict
        from util.losses import loss_masks
        base.seed_all()
        ids = bg.manifest(domain)['support_ids']
        pairs = self.pairs(domain, ids)
        rng = random.Random(111)
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params,lr=1e-4,weight_decay=0.)
        self.model.train(); self.cache=None
        losses=[]
        for step in range(steps):
            queries=rng.sample(ids,4)
            supports=[rng.choice([s for s in ids if s!=q]) for q in queries]
            episodes=[(pairs[s],pairs[q]) for s,q in zip(supports,queries)]
            x=torch.stack([torch.stack([s[0],q[0]]) for s,q in episodes]).cuda()
            y=torch.stack([torch.stack([s[1],q[1]]) for s,q in episodes]).cuda()
            self.model.support_geometry=[[s[2]] for s,_ in episodes]
            self.model.radial_branch.begin([(None,None)]*len(queries))
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prompt=build_prompt_dict(y,'mask',1,True,torch.device('cuda'))
                pred=self.model(x,prompt)['pred_masks']
                self.model.radial_branch.finish()
                loss=sum(loss_masks(pred,y[:,-1:],num_frames=1).values())
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(params,.1,error_if_nonfinite=True); opt.step()
            losses.append(float(loss.detach()))
            if (step+1)%20==0: log(event='adapt',source=self.domain,domain=domain,step=step+1,total=steps)
        self.model.eval(); self.cache=None
        if steps==100:
            out.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(model=state_for(self.model), source_sha256=self.sha, domain=domain, ids=ids, steps=steps, losses=losses),out)


def log(**fields):
    print(json.dumps(fields),flush=True)


def test(e, target, out, limit=0):
    if (out/'summary.json').exists(): return read_json(out/'summary.json')
    ids = bg.manifest(target)['test_ids']; ids=ids[:limit] if limit else ids
    rows=[]; start=time.monotonic()
    for i,sid in enumerate(ids):
        im,gt,_=bg.read(target,sid)
        rows.append(image_metrics(e.predict(im),gt,sid))
        if (i+1)%50==0 or i+1==len(ids):
            log(event='test',path=str(out.relative_to(OUT)),images=i+1,total=len(ids),seconds=round(time.monotonic()-start,1))
    save_csv(out/'per_image.csv',rows)
    result=dict(source=e.domain,target=target,source_sha256=e.sha,**summarize(rows))
    bg.dump(out/'summary.json',result); log(event='test_done',path=str(out.relative_to(OUT)),iou=result['IoU_percent'])
    return result


def background(e, out, limit):
    if (out/'summary.json').exists(): return
    d=e.domain; ids=bg.manifest(d)['test_ids']; ids=ids[:limit] if limit else ids
    baseline=csv_rows(bg.OUT/'results/DNANet'/d/'per_image.csv')
    expected={(r['sample_id'],r['condition'],int(r['replicate'])):r['input_sha256'] for r in baseline}
    partial=out/'per_image.csv'
    done=csv_rows(partial) if partial.exists() else []
    if done:
        assert len(done)%7==0
        assert [r['sample_id'] for r in done[::7]]==ids[:len(done)//7]
    start=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    with partial.open('a',encoding='utf-8',newline='') as stream:
        writer=None
        for i,sid in enumerate(ids[len(done)//7:],start=len(done)//7):
            im,gt,_=bg.read(d,sid)
            variants=[('original',im,dict(donor_domain='',donor_id='',replicate=-1,gain=[],offset=[],input_sha256=hashlib.sha256(im.tobytes()).hexdigest(),changed_background_fraction=0.))]
            for donor in DOMAINS:
                for rep in range(2):
                    x,meta=bg.compose(d,sid,donor,rep); variants.append((donor,x,meta))
            rows=[]
            for condition,x,meta in variants:
                assert meta['input_sha256']==expected[sid,condition,meta['replicate']]
                rows.append(dict(method='Ours',dataset=d,condition=condition,**meta,**image_metrics(e.predict(x),gt,sid)))
            if writer is None:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                if not done: writer.writeheader()
            writer.writerows(rows);stream.flush()
            if (i+1)%25==0 or i+1==len(ids):log(event='background',domain=d,images=i+1,total=len(ids),seconds=round(time.monotonic()-start,1))
    rows=csv_rows(partial)
    assert len(rows)==7*len(ids)
    for r in rows:
        for k in ('replicate','tp','fp','fn','pixels','gt_pixels','targets','predictions','detected','unmatched_pixels'):r[k]=int(r[k])
        r['image_iou']=float(r['image_iou'])
        assert r['input_sha256']==expected[r['sample_id'],r['condition'],r['replicate']]
    scores={}
    for condition in ['original',*DOMAINS]:
        reps=[]
        for rep in ([-1] if condition=='original' else [0,1]):
            rr=[r for r in rows if (r['condition'],r['replicate'])==(condition,rep)]
            assert [r['sample_id'] for r in rr]==ids
            reps.append(summarize(rr))
        scores[condition]=dict(replicates=reps,**{k:float(np.mean([r[k] for r in reps])) for k in ('IoU_percent','nIoU_percent','Pd_percent','Fa_per_million')})
    cross=float(np.mean([scores[q]['IoU_percent'] for q in DOMAINS if q!=d]))
    bg.dump(out/'summary.json',dict(method='Ours',dataset=d,images=len(ids),checkpoint=str(e.path),checkpoint_sha256=e.sha,
        training=dict(epoch=120,seed=111,adaptation='combined 19-token memory and dual-scale radial gate'),scores=scores,cross_background_mean_iou=cross,
        drop_from_original_pp=scores['original']['IoU_percent']-cross,drop_from_same_domain_control_pp=scores[d]['IoU_percent']-cross))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','transfer','retention','background'],required=True);a=p.parse_args()
    rows=selected()
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(checkpoints=rows,threshold=.5,background_recipe=bg.RECIPE,transfer='frozen source weights and source five supports; zero target labels',
        retention='two sequential stages of 100 AdamW updates each, lr 1e-4, batch 4, BF16; one support per episode distinct from query; only five new domain training images; evaluate with original five supports and no source replay',
        domains=DOMAINS,seed=111,epoch=120,runner_sha256=bg.digest(ROOT/'scripts/run_motivation_current_ours.py'))
    bg.dump(OUT/'protocol.json',protocol)
    for i,d in enumerate(DOMAINS):
        e=Engine(d,rows[d]);log(event='loaded',domain=d,phase=a.phase)
        if a.phase=='smoke':
            test(e,d,OUT/'smoke'/d/'original',3)
            background(e,OUT/'smoke'/d/'background',1)
            e.adapt(DOMAINS[(i+1)%3],1,OUT/'unused.pt')
            test(e,d,OUT/'smoke'/d/'after_update',1)
        elif a.phase=='transfer':
            for target in [d,*[q for q in DOMAINS if q!=d]]:
                test(e,target,OUT/'transfer'/d/target)
        elif a.phase=='retention':
            for j,target in enumerate((DOMAINS+DOMAINS)[i+1:i+3],start=2):
                dest=OUT/'retention'/d/f'stage{j}';cp=dest/'model.pt'
                if cp.exists():
                    state=torch.load(cp,map_location='cpu',weights_only=False)
                    assert state['source_sha256']==e.sha and state['domain']==target and state['steps']==100
                    restore(e.model,state['model']);e.cache=None
                else:e.adapt(target,100,cp)
                test(e,d,dest)
        else: background(e,OUT/'background'/d,0)
        del e;torch.cuda.empty_cache()
    bg.dump(OUT/f'{a.phase}_complete.json',dict(status='complete',phase=a.phase))


if __name__=='__main__':main()
