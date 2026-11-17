"""Matched center-only Mask ablation: MLP, spatial conv, edge1 and edge3."""
from __future__ import annotations
import argparse
import csv
import json
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scripts.run_phase_b_mask_head import load_baseline
from scripts.run_global_ablation_201 import encode_local, paste_local, add_metrics, write_csv, write_json
from visual_prompt.models.region_mask_heads import RegionMaskHead, inner_edge
from visual_prompt.models.lightweight_heads import soft_iou_loss
from visual_prompt.backbones.sam2_adapter import Sam2SimilarityPredictor, build_sam2_bplus
from visual_prompt.core.candidate_proposals import connected_instances, snapped_instance_centers
from visual_prompt.data.irstd import RawIRSTDDataset
from visual_prompt.pipelines.semantic_prompt_pipeline import load_rgb_and_mask, register_sim_feature_hooks, seed_everything

ROOT=Path('outputs/studies/objectness_mask')
OUT=ROOT/'mask_center_ablation_20261026'
VARIANTS=('plain','conv','edge1','edge3')


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def status(stage,**extra):
    row=dict(stage=stage,updated=time.strftime('%Y-%m-%d %H:%M:%S'),**extra)
    write_json(OUT/'status.json',row);print(row,flush=True)




def training_items():
    manifest=json.loads((ROOT/'phase_b_mask_head/metadata/feature_manifest.json').read_text())
    records=manifest['records'];items=[]
    for r in records:
        data=np.load(ROOT/'phase_b_mask_head/artifacts/local_features'/f'{r["key"]}.npz')
        assert r['crop_truncated']==0
        items.append(dict(feature=torch.from_numpy(data['feature'].astype(np.float16)),mask=torch.from_numpy(data['mask'].astype(np.float32)),sample_id=r['sample_id']))
    assert len(items)==78 and len({r['sample_id'] for r in items})==50
    return items


def loss_terms(final,main,edge,target,kind):
    # Identical full-mask terms in ALL variants, including the MLP-only control.
    result=.5*soft_iou_loss(final,target)+.5*soft_iou_loss(main,target)
    if kind.startswith('edge'):
        boundary=inner_edge(target,int(kind[-1]))
        raw=F.binary_cross_entropy_with_logits(edge,boundary,reduction='none')
        positive=boundary>0;negative=~positive;zero=raw.sum()*0
        boundary_loss=(raw[positive].mean() if positive.any() else zero)+(raw[negative].mean() if negative.any() else zero)
        result=result+.3*boundary_loss
    return result


def train_variant(kind,items,e0,device,epochs):
    path=OUT/'artifacts'/f'mask_{kind}.pt'
    if path.exists():return
    seed_everything(111);rng=random.Random(111)
    model=RegionMaskHead(len(e0),e0,kind).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    history=[]
    status('training',variant=kind,epoch=0,epochs=epochs)
    for epoch in range(epochs):
        # Replacement keeps 128 samples / 16 updates for all four 78-crop controls.
        order=rng.choices(range(len(items)),k=128);rng.shuffle(order);losses=[]
        for start in range(0,128,8):
            features=[];masks=[]
            for index in order[start:start+8]:
                item=items[index];rotation=rng.randrange(4);flip=rng.randrange(2)
                feature=torch.rot90(item['feature'],rotation,(-2,-1));target=torch.rot90(item['mask'],rotation,(-2,-1))
                if flip:feature=feature.flip(-1);target=target.flip(-1)
                features.append(feature);masks.append(target)
            features=torch.stack(features).float().to(device);target=torch.stack(masks).float().to(device)
            final,main,edge=model(features,e0)
            loss=loss_terms(final,main,edge,target,kind)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(loss.item())
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses))))
        if (epoch+1)%20==0:
            status('training',variant=kind,epoch=epoch+1,epochs=epochs,loss=history[-1]['loss'])
            write_json(OUT/'metrics'/f'{kind}_history.json',history)
    checkpoint=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},variant=kind,epochs=epochs,seed=111,sample_ids=sorted({r['sample_id'] for r in items}),e0=e0.cpu(),parameters=sum(p.numel() for p in model.parameters()),protocol='GT-center-only local encoding;78 positive crops;no position jitter or empty negative crops;shared full mask loss;no mask NMS',history=history)
    torch.save(checkpoint,path)
    write_json(OUT/'metrics'/f'{kind}_history.json',history)
    status('trained',variant=kind,epochs=epochs)


def summary(rows,system):
    inter=sum(r[system+'_inter'] for r in rows);pred=sum(r[system+'_pred_pixels'] for r in rows);gt=sum(r['gt_pixels'] for r in rows);targets=sum(r['target_count'] for r in rows)
    return dict(system=system,images=len(rows),targets=targets,miou_percent=100*inter/max(1,pred+gt-inter),niou_percent=100*np.mean([r[system+'_iou'] for r in rows]),pd_percent=100*sum(r[system+'_detected'] for r in rows)/targets,fa_per_million=1e6*(pred-inter)/sum(r['image_pixels'] for r in rows),f1_percent=200*inter/max(1,pred+gt))


@torch.no_grad()
def evaluate(e0,device):
    if (OUT/'metrics/summary.csv').exists():return
    status('evaluating',images=201)
    _,b=load_baseline(Path('experiments/configs/sam2/main.yaml'))
    sam=build_sam2_bplus(b.sam2_ckpt,b.sam2_config,str(device),b.encoder_input_size)
    sam.eval()
    for p in sam.parameters():p.requires_grad_(False)
    predictor=Sam2SimilarityPredictor(sam,feature_source=b.feature_source)
    register_sim_feature_hooks(predictor,b.sim_feature_layers)
    models={}
    for kind in VARIANTS:
        m=RegionMaskHead(len(e0),e0,kind).to(device);m.load_state_dict(load(OUT/'artifacts'/f'mask_{kind}.pt')['model']);m.eval();models[kind]=m
    selected=defaultdict(list)
    for r in csv.DictReader((ROOT/'global_ablation_test201/metrics/top5_objectness_scores.csv').open()):
        if int(r['objectness_kept']):selected[r['sample_id']].append((int(r['y']),int(r['x'])))
    dataset=RawIRSTDDataset(b.data_root,b.split,b.image_size);rows=[]
    for index in tqdm(range(len(dataset)),desc='centered-mask matched evaluation'):
        sid=dataset.sample_ids[index];image,gt,_=load_rgb_and_mask(dataset.images[index],dataset.masks[index],(b.image_size,b.image_size))
        instances=connected_instances(gt);centers=[(y,x) for _,y,x in snapped_instance_centers(instances)]
        points=list(dict.fromkeys(centers+selected[sid]));features=[]
        for y,x in points:
            _,_,f=encode_local(predictor,image,y,x,64,64);features.append(f[0])
        features=torch.stack(features);lookup={p:i for i,p in enumerate(points)}
        row=dict(sample_id=sid,target_count=instances.count,image_pixels=gt.size)
        for kind,model in models.items():
            masks=[]
            for start in range(0,len(features),8):masks.extend((model(features[start:start+8],e0)[0]>0).cpu().numpy().astype(np.uint8))
            for mode,locations in [('gt_center',centers),('fixed_old_obj',selected[sid])]:
                pred=np.zeros_like(gt,dtype=np.uint8)
                for y,x in locations:paste_local(pred,masks[lookup[y,x]],y,x)
                add_metrics(row,kind+'_'+mode,pred,gt)
        rows.append(row)
    results=[summary(rows,k+'_'+mode) for k in VARIANTS for mode in ['gt_center','fixed_old_obj']]
    write_csv(OUT/'metrics/image_metrics.csv',rows);write_csv(OUT/'metrics/summary.csv',results)
    lines=['改动：仅GT中心训练，四组同数据、增强、更新预算和完整mask损失；所有参数预先固定，未按test选模型。','', '| 结构 | 中心mIoU | 中心Pd | 中心Fa/M | 中心F1 | 原Obj联合推理mIoU |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for kind in VARIANTS:
        center=next(r for r in results if r['system']==kind+'_gt_center');actual=next(r for r in results if r['system']==kind+'_fixed_old_obj')
        lines.append(f'| {kind} | {center["miou_percent"]:.2f}% | {center["pd_percent"]:.2f}% | {center["fa_per_million"]:.3f} | {center["f1_percent"]:.2f}% | {actual["miou_percent"]:.2f}% |')
    lines+=['','结论：先比较本轮plain与conv，再比较conv与edge1/edge3；历史73.13%使用A2小偏移，是参考而非本轮同数据控制组。单种子111，不宣称稳定增益。']
    status('complete',results=results)


def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--output_dir',default=str(OUT));p.add_argument('--epochs',type=int,default=120);args=p.parse_args();OUT=Path(args.output_dir)
    for folder in ['artifacts','metrics','metadata']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);seed_everything(111)
    write_json(OUT/'metadata/protocol.json',dict(training_images=50,instances=78,sampling='GT centers only',background_crops=0,point_jitter=0,epochs=args.epochs,samples_per_epoch=128,batch_size=8,seed=111,loss='0.5 IoU(final)+0.5 IoU(main);edge-only extra0.3 balanced BCE',variants=list(VARIANTS),edge_thicknesses=[1,3],mask_threshold_logit=0,sam_frozen=True,e0_frozen=True,extra_nms=False,selection='predeclared epochs and losses;no test tuning'))
    e0=load(ROOT/'final_heads_50shot/artifacts/edited_target_embedding.pt').float();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');e0=e0.to(device)
    items=training_items()
    for kind in VARIANTS:train_variant(kind,items,e0,device,args.epochs)
    del items
    evaluate(e0,device)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True);status('failed',error=repr(exc));traceback.print_exc();raise
