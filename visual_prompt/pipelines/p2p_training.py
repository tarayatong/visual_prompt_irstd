"""Final two-stage training recipe, independent of retired experiment scripts.

Rebuilds disposable frozen-feature caches in a new output directory. Historical
checkpoints remain the metric reference; full retraining of this reorganized
entry is not claimed. Positive samples never depend on retrieval TopK.
"""
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from visual_prompt.pipelines.p2p_framework import (
    P2PModel, RETAINED, load, ev, describe_full_image_points, DESCRIPTOR_NAMES,
    local_sam, crop_at, normalized_feature, normalize_f4, mask_descriptor,
    mask_forward, ObjectnessMLP, RadialMaskVectorMLP, RecallMemoryReader,
    stitched_memory_maps, greedy_score_peaks,
)
from visual_prompt.core.candidate_proposals import connected_instances, snapped_instance_centers, point_instance_id
from visual_prompt.core.semantic_direction import build_score_token, extract_support_semantic_tokens, mean_token
from visual_prompt.pipelines.semantic_prompt_pipeline import get_sim_features, seed_everything
from visual_prompt.models.center_instance_loss import center_instance_loss

READER_EPOCHS=50
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state_dict(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def status(out,stage,**kw):print(json.dumps(dict(stage=stage,**kw)),flush=True)
def multi_score(maps,memory):
    p=maps[2:2+memory['positive_count']];weights=np.asarray(memory['positive_weights'],np.float32)[:,None,None];mx=p.max(0)
    return mx+np.log((np.exp((p-mx)*20)*weights).sum(0).clip(1e-10))/20

class P2PTraining(P2PModel):
    def __init__(self,dataset,variant,output,device='cuda:0'):
        if variant not in ('frozen','memory'):raise ValueError('Final paired training supports frozen/memory')
        self.out=Path(output).resolve()
        if self.out.exists():raise FileExistsError('Choose a new training output directory')
        if self.out.is_relative_to(RETAINED.resolve()):raise ValueError('Retained results cannot be overwritten')
        super().__init__(dataset,variant,device)
        self.out.mkdir(parents=True)
        self.m=json.loads((self.data_folder/'manifest.json').read_text());self.ids=self.m['train_ids'];self.support=self.m['support_ids']
        self.allowed=set(self.ids);assert set(self.support)<=self.allowed
        self.b=self.args;self.images={};self.variant=variant
        write_json(self.out/'manifest.json',self.m)
        write_json(self.out/'protocol.json',dict(recipe='retained GT-fixed paired recipe',seed=111,reader_epochs=50,heads_epochs=120,train_negative_K=60,inference_K=5,positive='GT snapped center plus one deterministic inside point',Mask_e0='original frozen edited support vector',full_retraining_of_reorganized_entry_validated=False))
        torch.save(self.e0.cpu(),self.out/'mask_e0.pt');torch.save(self.memory,self.out/'memory.pt')

    def image(self,sid):
        assert sid in self.allowed,('Unauthorized split access',sid)
        if sid not in self.images:
            r=self.m['records'][sid]
            image=np.array(Image.open(r['image']).convert('RGB').resize((512,512),Image.Resampling.BILINEAR),copy=True)
            gt=(np.asarray(Image.open(r['mask']).convert('L').resize((512,512),Image.Resampling.NEAREST))>=127).astype(np.uint8)
            self.images[sid]=image,gt
        return self.images[sid]

    @torch.no_grad()
    def frozen_score(self,sid):
        path=self.out/'cache/frozen'/f'{sid}.npy'
        if path.exists():return np.load(path)
        score,_=ev.compute_stitched_similarity(self.pred(),self.image(sid)[0],self.e0,self.b)
        path.parent.mkdir(parents=True,exist_ok=True);np.save(path,score);return score

    @torch.no_grad()
    def maps(self,sid):
        path=self.out/'cache/maps'/f'{sid}.npy'
        if path.exists():return np.load(path)
        maps=stitched_memory_maps(self.pred(),self.image(sid)[0],self.memory['keys'].to(self.device),self.b)
        maps[0]=self.frozen_score(sid);path.parent.mkdir(parents=True,exist_ok=True);np.save(path,maps);return maps

    @torch.no_grad()
    def score(self,sid,arm):
        if self.variant=='frozen':return self.frozen_score(sid)
        return self.reader.score_map(torch.from_numpy(self.maps(sid)).to(self.device)).cpu().numpy()

    @torch.no_grad()
    def raw(self,sid,points,key):
        parts=[describe_full_image_points(self.pred(),self.image(sid)[0],self.frozen_score(sid),points[i:i+64],64,self.b.local_contrast_sigma,32) for i in range(0,len(points),64)]
        return np.concatenate([np.concatenate([p[k] for k in DESCRIPTOR_NAMES],1) for p in parts],0)

    @torch.no_grad()
    def local(self,sid,y,x):
        path=self.out/'cache/local'/f'{sid}_{y}_{x}.npz'
        if path.exists():
            with np.load(path) as d:return {k:d[k].copy() for k in d.files}
        sam=local_sam(self.pred(),self.image(sid)[0],y,x);ray,valid=self.radial.sample(self.image(sid)[0],y,x)
        data=dict(sam=sam[0].cpu().half().numpy(),ray=ray.numpy(),ray_valid=valid.numpy())
        path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,**data);return data

    @torch.no_grad()
    def support_memory(self):
        path = self.out/'memory.pt'
        if path.exists():
            return load(path)
        config = ev.build_config(self.b)
        targets, by_image, audit = [], [], []
        negative = {'high_freq': [], 'bright_flat': []}
        predictor = self.pred()
        for sid in self.support:
            image, gt = self.image(sid)
            inst, local = connected_instances(gt), []
            for iid, y, x in snapped_instance_centers(inst):
                crop, mask = ev.crop_with_padding(image, (inst.labels == iid).astype(np.uint8),
                    y-self.b.window_size//2, x-self.b.window_size//2, self.b.window_size)
                assert mask.any()
                pad = predictor.set_image(crop, np.repeat(mask[:, :, None]*255, 3, axis=2).astype(np.uint8))
                token, _ = extract_support_semantic_tokens(crop, mask, pad, get_sim_features(predictor), config)
                assert token is not None
                local.append(token); targets.append(token)
                audit.append(dict(sample_id=sid, instance=iid, full_pixels=int((inst.labels == iid).sum()), retained_pixels=int(mask.sum())))
            assert local
            by_image.append(mean_token(local))
            pad = predictor.set_image(image, np.repeat(gt[:, :, None]*255, 3, axis=2).astype(np.uint8))
            _, negs = extract_support_semantic_tokens(image, gt, pad, get_sim_features(predictor), config)
            for key, value in negs.items():
                if value is not None:
                    negative[key].append(value)
        corrected, info = build_score_token(by_image, negative, config)
        corrected_instances = [build_score_token([t], negative, config)[0] for t in targets]
        neg = [t for ts in negative.values() for t in ts]
        keys = F.normalize(torch.cat([self.e0[None], corrected]+corrected_instances+neg, dim=0), dim=1)
        counts = [sum(a['sample_id'] == sid for a in audit) for sid in self.support]
        weights = [1/(len(counts)*count) for count in counts for _ in range(count)]
        memory = dict(keys=keys.cpu(), positive_count=len(targets), positive_weights=weights,
                      negative_count=len(neg), support_ids=self.support, audit=audit, corrected_info=info)
        torch.save(memory, path)
        return memory

    def train_reader(self):
        self.memory = self.support_memory()
        path = self.out/'reader.pt'
        if path.exists():
            ck = load(path)
        else:
            examples = {}
            for number, sid in enumerate(self.ids):
                maps = self.maps(sid)
                inst = connected_instances(self.image(sid)[1])
                groups = [torch.from_numpy(maps[:, inst.labels == iid].T.copy()).to(self.device)
                          for iid in range(1, inst.count+1)]
                safe = cv2.dilate((inst.labels > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) == 0
                points = set()
                for score in (maps[0], maps[1], multi_score(maps, self.memory)):
                    points.update((y, x) for y, x, _ in greedy_score_peaks(score, 60, 7) if safe[y, x])
                yy, xx = np.where(safe)
                code = int(sid) if sid.isdecimal() else int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(111+code)
                points.update((int(yy[i]), int(xx[i])) for i in rng.choice(len(yy), min(128, len(yy)), replace=False))
                assert points and groups
                yy, xx = np.asarray(sorted(points)).T
                examples[sid] = groups, torch.from_numpy(maps[:, yy, xx].T.copy()).to(self.device)
                status(self.out, 'reader_features', images=number+1, total=50)
            seed_everything(111)
            model = RecallMemoryReader(len(self.memory['keys'])).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            history = []
            for epoch in range(READER_EPOCHS):
                order = list(self.ids); random.Random(111+epoch).shuffle(order); losses = []
                for sid in order:
                    groups, background = examples[sid]
                    neg = model(background).topk(min(32, len(background))).values
                    terms = []
                    for group in groups:
                        score = model(group)
                        pos = .02*(torch.logsumexp(score/.02, 0)-math.log(len(score)))
                        terms.append(F.softplus((neg-pos+.03)/.05).mean())
                    loss = torch.stack(terms).mean(); assert torch.isfinite(loss)
                    optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                    losses.append(float(loss.detach()))
                history.append(dict(epoch=epoch+1, loss=float(np.mean(losses))))
                if (epoch+1) % 10 == 0:
                    status(self.out, 'train_reader', **history[-1], epochs=READER_EPOCHS)
            ck = dict(model=state_dict(model), train_ids=self.ids, epoch=READER_EPOCHS, channels=len(self.memory['keys']), history=history)
            torch.save(ck, path)
            del model, optimizer, examples
        assert set(ck['train_ids']) == set(self.ids) and ck['epoch'] == READER_EPOCHS
        self.reader = RecallMemoryReader(ck['channels']).to(self.device).eval()
        self.reader.load_state_dict(ck['model'])
        for p in self.reader.parameters():
            p.requires_grad_(False)
        gc.collect(); torch.cuda.empty_cache()

    def build_pool(self, arm):
        path = self.out/arm/'pool.pt'
        if path.exists():
            return load(path)
        pool, coverage = [], []
        for number, sid in enumerate(self.ids):
            image, gt = self.image(sid); inst = connected_instances(gt)
            rng = random.Random(111 ^ int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16))
            positive = set()
            for iid, y, x in snapped_instance_centers(inst):
                positive.add((y, x)); coordinates = np.argwhere(inst.labels == iid)
                if len(coordinates) > 1:
                    choices = [tuple(map(int, p)) for p in coordinates if tuple(p) != (y, x)]
                    positive.add(rng.choice(choices))
            peaks = [(y, x) for y, x, _ in greedy_score_peaks(self.score(sid, arm), 60, 7)]
            negative = {(y, x) for y, x in peaks if not gt[y, x]}
            all_points = sorted(positive | set(peaks))
            raw = self.raw(sid, all_points, 'train_'+arm)
            lookup = {tuple(p): i for i, p in enumerate(all_points)}
            for y, x in sorted(positive | negative):
                d = self.local(sid, y, x); iid = int(inst.labels[y, x])
                _, target = crop_at(image, (inst.labels == iid).astype(np.uint8) if iid else np.zeros_like(gt), y, x, 64)
                _, all_gt = crop_at(image, gt, y, x, 64); valid = (all_gt == 0) | (target > 0)
                ray, rv = d['ray'].astype(np.float64), d['ray_valid']
                pool.append(dict(sample_id=sid, y=y, x=x, positive=int(bool(iid)), raw=raw[lookup[y, x]],
                    target=target, valid=valid, ray_sum=(ray*rv).sum((1, 2)),
                    ray_square=(ray*ray*rv).sum((1, 2)), ray_count=rv.sum((1, 2))))
            coverage.append(dict(sample_id=sid, targets=inst.count, GT_positive_instances=inst.count,
                candidate_exact=len({point_instance_id(inst.labels, y, x, 0) for y, x in peaks}-{0})))
            status(self.out, 'GT_fixed_pool', arm=arm, images=number+1, total=50, crops=len(pool))
        path.parent.mkdir(parents=True, exist_ok=True); torch.save(pool, path)
        write_json(path.parent/'pool_counts.json', dict(positive=sum(r['positive'] for r in pool),
            negative=sum(not r['positive'] for r in pool), images=50,
            targets=sum(r['targets'] for r in coverage), positive_GT_coverage=sum(r['GT_positive_instances'] for r in coverage),
            candidate_exact=sum(r['candidate_exact'] for r in coverage)))
        write_csv(path.parent/'train_candidate_coverage.csv', coverage)
        return pool

    def train_heads(self,key="full50",validation_ids=()):
        path=self.out/'heads.pt'
        if path.exists():return load(path)
        train_ids=set(self.ids)-set(validation_ids);indices=[i for i,r in enumerate(self.pool) if r['sample_id'] in train_ids]
        assert set(self.support)<=train_ids
        raw=np.stack([r['raw'] for r in self.pool]);labels=np.asarray([r['positive'] for r in self.pool])
        centers=raw[:,6:358];centers=centers/np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-8)
        pos=[i for i in indices if labels[i]];neg=[i for i in indices if not labels[i]]
        t=centers[pos].mean(0);b=centers[neg].mean(0);t/=max(np.linalg.norm(t),1e-8);b/=max(np.linalg.norm(b),1e-8)
        means=raw[:,358:710];means=means/np.maximum(np.linalg.norm(means,axis=1,keepdims=True),1e-8)
        tc,tm,bc,bm=centers@t,means@t,centers@b,means@b
        full=np.concatenate((raw,np.stack((tc,tm,bc,bm,tc-bc,tm-bm),1)),1)
        stats=dict(mean=full[indices].mean(0,keepdims=True),std=full[indices].std(0,keepdims=True).clip(1e-5),prototype_target=t,prototype_background=b)
        f4=torch.from_numpy(normalize_f4(raw,stats)).to(self.device)
        # Optional independent Obj positive sampler; default paired recipe is unchanged.
        obj_bank = self.prepare_offset_obj(stats) if hasattr(self,'prepare_offset_obj') else None
        sums=np.stack([self.pool[i]['ray_sum'] for i in indices]).sum(0);square=np.stack([self.pool[i]['ray_square'] for i in indices]).sum(0)
        count=np.stack([self.pool[i]['ray_count'] for i in indices]).sum(0).clip(1)
        mu=(sums/count).astype(np.float32);std=np.sqrt(np.maximum(0,square/count-(sums/count)**2)).clip(.01).astype(np.float32)
        # Reuse one bounded CPU/disk feature store, rather than retaining each fold.
        scratch=self.out/'scratch';scratch.mkdir(exist_ok=True)
        features=np.lib.format.open_memmap(scratch/'features.npy',mode='w+',dtype=np.float16,shape=(len(self.pool),416,64,64))
        desc=np.zeros((len(self.pool),836),np.float32)
        for j,i in enumerate(indices):
            r=self.pool[i];d=self.local(r['sample_id'],r['y'],r['x'])
            f=normalized_feature(d,mu,std)
            features[i]=f
            with torch.no_grad():desc[i]=mask_descriptor(torch.from_numpy(f.astype(np.float32))[None],torch.from_numpy(f[:352].astype(np.float32))[None],self.e0.cpu())[0].numpy()
            if (j+1)%500==0:status(self.out,'prepare_fold_features',key=key,crops=j+1,total=len(indices))
        features.flush();desc=torch.from_numpy(desc).to(self.device)
        targets=torch.from_numpy(np.stack([r['target'] for r in self.pool])).to(self.device).float()
        valid=torch.from_numpy(np.stack([r['valid'] for r in self.pool])).to(self.device)
        labels=torch.from_numpy(labels).to(self.device).bool()
        seed_everything(111);mask=RadialMaskVectorMLP(352,64,self.e0).to(self.device)
        seed_everything(111);obj=ObjectnessMLP(1440).to(self.device)
        optimizer=torch.optim.AdamW(list(mask.parameters())+list(obj.parameters()),lr=.001,weight_decay=.0001)
        pr,nr=random.Random(111),random.Random(112);history=[];schedule=[]
        epochs=getattr(self,'head_epochs',120)
        start=0
        resume=self.out/'training_latest.pt'
        if getattr(self,'resumable',False) and resume.exists():
            saved=load(resume);mask.load_state_dict(saved['mask']);obj.load_state_dict(saved['obj'])
            optimizer.load_state_dict(saved['optimizer']);start=saved['epoch'];history=saved['history'];schedule=saved['schedule']
            pr.setstate(saved['positive_rng']);nr.setstate(saved['negative_rng'])
            torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        for epoch in range(start,epochs):
            p,n=pr.choices(pos,k=128),nr.choices(neg,k=128);schedule.append(p+n);values=[]
            for s in range(0,128,8):
                ix=p[s:s+8]+n[s:s+8]
                f=torch.from_numpy(np.array(features[ix],copy=True)).to(self.device,dtype=torch.float32)
                z=mask_forward(mask,f,desc[ix],self.e0)
                obj_x=f4[ix] if obj_bank is None else torch.cat((obj_bank[0][obj_bank[1][epoch,s:s+8]],f4[n[s:s+8]]))
                o=obj(obj_x)
                loss=center_instance_loss(z,targets[ix],valid[ix],labels[ix])+F.binary_cross_entropy_with_logits(o,labels[ix].float())
                assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();optimizer.step();values.append(float(loss.detach()))
            history.append(dict(epoch=epoch+1,loss=float(np.mean(values))))
            if getattr(self,'resumable',False) and ((epoch+1)%25==0 or epoch+1==epochs):
                saved=dict(mask=state_dict(mask),obj=state_dict(obj),optimizer=optimizer.state_dict(),epoch=epoch+1,
                    history=history,schedule=schedule,positive_rng=pr.getstate(),negative_rng=nr.getstate(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                torch.save(saved,resume.with_suffix('.tmp'));resume.with_suffix('.tmp').replace(resume)
                torch.save(dict(mask=saved['mask'],obj=saved['obj'],epoch=epoch+1),self.out/f'epoch_{epoch+1:04d}.pt')
                write_json(self.out/'status.json',dict(stage='training',epoch=epoch+1,total=epochs,loss=history[-1]['loss']))
            if (epoch+1)%30==0:status(self.out,'stage1_training',key=key,**history[-1])
        ck=dict(mask=state_dict(mask),obj=state_dict(obj),radial_mean=torch.from_numpy(mu),radial_std=torch.from_numpy(std),statistics=stats,
                train_ids=sorted(train_ids),validation_ids=sorted(validation_ids),positive_pool=len(pos),negative_pool=len(neg),
                schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),epochs=epochs,seed=111,
                protocol_sha256=digest(self.out/'protocol.json'))
        path.parent.mkdir(exist_ok=True);torch.save(ck,path);write_csv(self.out/'heads'/f'{key}_history.csv',history)
        del mask,obj,optimizer,f4,desc,targets,valid,labels,f;features._mmap.close();del features;gc.collect();torch.cuda.empty_cache()
        return ck
