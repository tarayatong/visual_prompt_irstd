"""Controlled Obj-only training with GT-center, radius-5 and any-target positives."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from scripts.run_phase_b_mask_head import load_baseline, gaussian_map
from scripts.run_phase_c_objectness import (
    build_matrix, descriptor, detection_metrics, load_arrays, prototypes, select_threshold,
)
from scripts.run_global_ablation_201 import (
    add_metrics, encode_local, infer_mask, paste_local, write_csv, write_json,
)
from scripts.run_centered_mask_ablation import summary as mask_summary
from scripts.recompute_objectness_deduplicated import jitter_reports
from visual_prompt.backbones.sam2_adapter import Sam2SimilarityPredictor, build_sam2_bplus
from visual_prompt.core.candidate_proposals import connected_instances, snapped_instance_centers
from visual_prompt.data.irstd import raw_irstd_paths
from visual_prompt.models.full_image_objectness import describe_full_image_points
from visual_prompt.models.lightweight_heads import MaskVectorMLP, ObjectnessMLP
from visual_prompt.pipelines.semantic_prompt_pipeline import load_rgb_and_mask, register_sim_feature_hooks, seed_everything

ROOT = Path('outputs/studies/objectness_mask')
VARIANTS = ('center', 'radius5', 'target_any')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def status(out, stage, **fields):
    value = dict(stage=stage, **fields)
    write_json(out / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def predictor_for(b, device):
    sam = build_sam2_bplus(b.sam2_ckpt, b.sam2_config, str(device), b.encoder_input_size)
    sam.eval()
    for p in sam.parameters():
        p.requires_grad_(False)
    predictor = Sam2SimilarityPredictor(sam, feature_source=b.feature_source)
    register_sim_feature_hooks(predictor, b.sim_feature_layers)
    return predictor


def prepare_positives(out, b, sample_ids, records, arrays, device):
    """Cache every labeled target pixel; one full-image encoding per training image."""
    cache = out / 'cache' / 'target_pixels'
    cache.mkdir(parents=True, exist_ok=True)
    predictor = None
    positive_rows, groups, audit = [], [], []
    original = {(r['sample_id'], int(r['assignment'])): r for r in records if r['source'] == 'gt'}
    for number, sid in enumerate(sample_ids):
        image, gt, _ = load_rgb_and_mask(*raw_irstd_paths(b.data_root, sid), (b.image_size, b.image_size))
        instances = connected_instances(gt)
        points = np.argwhere(gt > 0).astype(np.int32)
        signature = hashlib.sha256(image.tobytes() + gt.tobytes()).hexdigest()
        path = cache / f'{sid}.npz'
        if path.exists():
            with np.load(path) as data:
                assert str(data['signature']) == signature
                assert np.array_equal(data['points'], points)
                desc = {k: data[k].copy() for k in ('sim', 'center', 'mean', 'max', 'diff', 'hand')}
        else:
            if predictor is None:
                predictor = predictor_for(b, device)
            score = np.load(ROOT / 'phase_c_objectness/artifacts/candidate_features' / f'{sid}_score.npy')
            # At most the annotated pixels of one image are held as ROI features.
            with torch.no_grad():
                desc = describe_full_image_points(predictor, image, score, [tuple(map(int, p)) for p in points], 64, b.local_contrast_sigma, 32)
            np.savez_compressed(path, points=points, signature=signature, **desc)
        key = sid + ':target_pixels'
        arrays[key] = desc
        lookup = {tuple(p): i for i, p in enumerate(points)}
        for iid, cy, cx in snapped_instance_centers(instances):
            old = original[sid, iid]
            assert (cy, cx) == (old['y'], old['x'])
            yy, xx = np.where(instances.labels == iid)
            row_indices = []
            for y, x in zip(yy, xx):
                row_indices.append(len(positive_rows))
                positive_rows.append(dict(sample_id=key, image_id=sid, point_index=lookup[y, x],
                                          y=int(y), x=int(x), assignment=iid, label=1, source='target_pixel'))
            center_index = next(i for i in row_indices if positive_rows[i]['y'] == cy and positive_rows[i]['x'] == cx)
            radius_indices = [i for i in row_indices if (positive_rows[i]['y']-cy)**2 + (positive_rows[i]['x']-cx)**2 <= 25]
            groups.append(dict(sample_id=sid, instance_id=iid, center=[center_index], radius5=radius_indices, target_any=row_indices))
            error = max(float(np.max(np.abs(desc[k][lookup[cy, cx]] - arrays[sid][k][old['point_index']]))) for k in desc)
            if error > 2e-5:
                raise ValueError(f'Original full-image descriptor mismatch: {sid}/{iid}: {error}')
            audit.append(dict(sample_id=sid, instance_id=iid, pixels=len(row_indices), radius5_pixels=len(radius_indices),
                              center_descriptor_max_error=error, same_radius5_and_any=int(len(radius_indices)==len(row_indices))))
        status(out, 'caching_positive_points', images_done=number+1, images_total=len(sample_ids), points=len(positive_rows))
    del predictor
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    assert len(groups) == 78
    write_csv(out / 'metrics/positive_sampling_audit.csv', audit)
    write_json(out / 'metadata/positive_groups.json', groups)
    return positive_rows, groups


def train_one(out, variant, fold, train_ids, val_ids, records, arrays, positive_rows, groups, device):
    path = out / 'artifacts' / f'objectness_{variant}_{fold}.pt'
    if path.exists():
        return load(path)
    train_ids = set(train_ids)
    proto = prototypes(records, arrays, train_ids)
    fit = [r for r in records if r['sample_id'] in train_ids and
           (r['source']=='gt' or (r['source']=='hard' and r['label']==0))]
    matrix = build_matrix(fit, arrays, 'f4', proto)
    mean, std = matrix.mean(0, keepdims=True), np.maximum(matrix.std(0, keepdims=True), 1e-5)
    negatives = [r for r in fit if r['label']==0]
    pos_groups = [g for g in groups if g['sample_id'] in train_ids]
    # Standardization/prototypes come only from the identical center+hard-negative training pool.
    pos_matrix = torch.from_numpy((build_matrix(positive_rows, arrays, 'f4', proto)-mean)/std).to(device)
    neg_matrix = torch.from_numpy((build_matrix(negatives, arrays, 'f4', proto)-mean)/std).to(device)
    seed_everything(111)
    model = ObjectnessMLP(matrix.shape[1], 128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    # Separate RNGs keep negatives and minibatch ordering identical across positive policies.
    pos_rng, neg_rng, order_rng = random.Random(111), random.Random(111), random.Random(111)
    history = []
    for epoch in range(100):
        pos_indices = [pos_rng.choice(g[variant]) for g in pos_groups]
        neg_indices = [neg_rng.randrange(len(negatives)) for _ in pos_groups]
        x = torch.cat((pos_matrix[pos_indices], neg_matrix[neg_indices]))
        y = torch.cat((torch.ones(len(pos_groups), device=device), torch.zeros(len(pos_groups), device=device)))
        order = list(range(len(y))); order_rng.shuffle(order)
        losses = []
        model.train()
        for start in range(0, len(order), 32):
            idx = order[start:start+32]
            loss = F.binary_cross_entropy_with_logits(model(x[idx]), y[idx])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(dict(epoch=epoch+1, loss=float(np.mean(losses))))
        if (epoch+1) % 20 == 0:
            status(out, 'training', variant=variant, fold=fold, epoch=epoch+1, epochs=100, loss=history[-1]['loss'])
    model.eval()
    val_rows = [r for r in records if r['sample_id'] in val_ids and r['source']=='eval']
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy((build_matrix(val_rows, arrays, 'f4', proto)-mean)/std).to(device))).cpu().numpy() if val_rows else []
    predictions = [dict(**r, score=float(s)) for r,s in zip(val_rows,scores)]
    checkpoint = dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()}, input_dim=matrix.shape[1], hidden=128,
                      mean=mean, std=std, prototype_target=proto[0], prototype_background=proto[1], descriptor='f4',
                      feature_protocol='full_image_encode_then_roi_sample_v1', threshold=.9, threshold_source='predeclared_fixed_0.9',
                      positive_policy=variant, sampling='one positive per instance and equal unchanged hard negatives per epoch',
                      sample_ids=sorted(train_ids), validation_ids=sorted(val_ids), seed=111, epochs=100,
                      positive_instances=len(pos_groups), hard_negative_pool=len(negatives), history=history, predictions=predictions)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    write_csv(out / 'metrics' / f'{variant}_{fold}_history.csv', history)
    return checkpoint


def score_matrix(model, checkpoint, base, device):
    """Recompute prototype similarities for this training split; preserve all other F4 inputs."""
    matrix = base.copy()
    center, mean = matrix[:,6:358], matrix[:,358:710]
    center = center / np.maximum(np.linalg.norm(center,axis=1,keepdims=True),1e-8)
    mean = mean / np.maximum(np.linalg.norm(mean,axis=1,keepdims=True),1e-8)
    t,b = checkpoint['prototype_target'], checkpoint['prototype_background']
    tc,tm,bc,bm = center@t,mean@t,center@b,mean@b
    matrix[:,-6:] = np.stack((tc,tm,bc,bm,tc-bc,tm-bm),axis=1)
    with torch.no_grad():
        return model(torch.from_numpy((matrix-checkpoint['mean'])/checkpoint['std']).to(device)).cpu().numpy()


def evaluate(out, checkpoints, b, device):
    source = read_csv(ROOT/'global_ablation_test201/metrics/top5_objectness_scores.csv')
    by_sid = defaultdict(list)
    for r in source:
        by_sid[r['sample_id']].append(r)
    target_counts = {r['sample_id']:int(r['target_count']) for r in read_csv(ROOT/'global_ablation_test201/metrics/image_metrics.csv')}
    models = {}
    for variant, ck in checkpoints.items():
        model = ObjectnessMLP(ck['input_dim'], ck['hidden']).to(device)
        model.load_state_dict(ck['model']); model.eval(); models[variant]=model
    selections, point_rows, metric_rows, image_rows = {}, [], [], []
    for variant,ck in checkpoints.items():
        rows,scores=[],[]
        for sid, candidates in by_sid.items():
            with np.load(ROOT/'center_sampled_obj_diagnosis_20261026/cache'/f'{sid}.npz') as data:
                lookup={tuple(p):i for i,p in enumerate(data['points'])}
                matrix=data['matrix'][[lookup[int(r['y']),int(r['x'])] for r in candidates]]
            logits=score_matrix(models[variant],ck,matrix,device)
            probabilities=torch.sigmoid(torch.from_numpy(logits)).numpy()
            for r,logit,p in zip(candidates,logits,probabilities):
                row=dict(sample_id=sid,y=int(r['y']),x=int(r['x']),assignment=int(r['assignment_dilate3']),target_count=target_counts[sid])
                rows.append(row); scores.append(float(p))
                point_rows.append(dict(variant=variant,**row,logit=float(logit),score=float(p)))
        scores=np.asarray(scores)
        for threshold_name,threshold in [('fixed09',.9),('oof',ck['oof_threshold'])]:
            metric=detection_metrics(rows,scores,threshold)
            keep=[r for r,p in zip(rows,scores) if p>=threshold]
            background=sum(r['assignment']==0 for r in keep)
            metric_rows.append(dict(variant=variant,threshold_name=threshold_name,threshold=threshold,
                                    **metric,background=background,duplicates=len(keep)-metric['tp']-background,selected=len(keep)))
            selected=defaultdict(list)
            for r in keep:
                selected[r['sample_id']].append((r['y'],r['x']))
            selections[f'{variant}_{threshold_name}']=selected
    write_csv(out/'metrics/test_objectness.csv',metric_rows)
    write_csv(out/'metrics/test_candidate_scores.csv',point_rows)
    status(out,'evaluating_joint_masks',images_done=0,images_total=len(by_sid))
    predictor=predictor_for(b,device)
    e0=load(ROOT/'final_heads_50shot/artifacts/edited_target_embedding.pt').float().to(device)
    mc=load(ROOT/'final_heads_50shot/artifacts/mask_final_a2.pt')
    mask=MaskVectorMLP(mc['channels'],mc['hidden'],e0,'m2_residual').to(device)
    mask.load_state_dict(mc['model']); mask.eval()
    gaussian=torch.from_numpy(gaussian_map(64,4.)).float().to(device)
    local_cache=out/'cache/local_masks';local_cache.mkdir(parents=True,exist_ok=True)
    for number,sid in enumerate(by_sid):
        image,gt,_=load_rgb_and_mask(*raw_irstd_paths(b.data_root,sid),(b.image_size,b.image_size))
        points=sorted({p for selected in selections.values() for p in selected[sid]})
        row=dict(sample_id=sid,target_count=target_counts[sid],image_pixels=int(gt.size),gt_pixels=int((gt>0).sum()))
        point_masks={}
        for y,x in points:
            path=local_cache/f'{sid}_{y}_{x}.npy'
            if path.exists():
                local=np.load(path)
            else:
                with torch.no_grad():
                    crop,_,feature=encode_local(predictor,image,y,x,64,64)
                    local=infer_mask(mask,feature,crop,e0,gaussian,b,mc['residual_scale'],mc['temperature'])
                np.save(path,local)
            point_masks[y,x]=local
        for system,selected in selections.items():
            pred=np.zeros_like(gt,dtype=np.uint8)
            for y,x in selected[sid]:paste_local(pred,point_masks[y,x],y,x)
            add_metrics(row,system,pred,gt)
        image_rows.append(row)
        if (number+1)%10==0 or number+1==len(by_sid):
            status(out,'evaluating_joint_masks',images_done=number+1,images_total=len(by_sid))
            write_csv(out/'metrics/test_image_metrics.csv',image_rows)
    summaries=[mask_summary(image_rows,system) for system in selections]
    write_csv(out/'metrics/test_joint_summary.csv',summaries)
    postprocessed=jitter_reports(out)
    return [r for r in postprocessed if r['postprocess']=='point_nms'],summaries




def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output_dir',default=str(ROOT/'objectness_positive_jitter_20261027'))
    args=parser.parse_args();out=Path(args.output_dir)
    for directory in ['artifacts','metadata','metrics']:(out/directory).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this queued experiment')
    device=torch.device('cuda:0')
    status(out,'preparing',variants=list(VARIANTS))
    config=yaml.safe_load(Path('experiments/configs/objectness_mask/train_final_heads.yaml').read_text())['args']
    sample_ids=config['sample_ids'].split(',');support=set(config['support_ids'].split(','))
    _,b=load_baseline(Path(config['baseline_config']))
    records=json.loads((ROOT/'phase_c_objectness/metadata/feature_manifest.json').read_text())['records']
    arrays=load_arrays(ROOT/'phase_c_objectness/artifacts/candidate_features',sample_ids)
    folds=json.loads((ROOT/'phase_c_objectness_refine/metadata/folds.json').read_text())['validation_folds']
    assert len(set(x for f in folds for x in f))==45 and not support.intersection(x for f in folds for x in f)
    test_ids={r['sample_id'] for r in read_csv(ROOT/'global_ablation_test201/metrics/top5_objectness_scores.csv')}
    assert len(test_ids)==201 and not set(sample_ids)&test_ids
    write_json(out/'metadata/protocol.json',dict(variants=list(VARIANTS),sample_ids=sample_ids,support_ids=sorted(support),folds=folds,
                                                seed=111,epochs=100,fixed_threshold=.9,negative_policy='unchanged original high-sim pool',
                                                point_policy='uniform same-instance pixels, radius5 uses squared Euclidean distance <=25',
                                                positive_count_per_instance_per_epoch=1,exact_center_mixture=False))
    positive_rows,groups=prepare_positives(out,b,sample_ids,records,arrays,device)
    checkpoints={};oof_summary=[];oof_predictions=[]
    # Train the two requested policies first, followed by the matched center control.
    for variant in ['radius5','target_any','center']:
        oof=[]
        for index,val_ids in enumerate(folds):
            train_ids=set(sample_ids)-set(val_ids)
            assert support<=train_ids
            ck=train_one(out,variant,f'fold{index}',train_ids,set(val_ids),records,arrays,positive_rows,groups,device)
            oof.extend(ck['predictions'])
        assert len(oof)==225
        scores=np.asarray([r['score'] for r in oof],dtype=np.float32)
        threshold,metric,_=select_threshold(oof,scores)
        for name,t in [('fixed09',.9),('oof',threshold)]:
            oof_summary.append(dict(variant=variant,threshold_name=name,threshold=t,**detection_metrics(oof,scores,t)))
        oof_predictions.extend(dict(variant=variant,**r) for r in oof)
        ck=train_one(out,variant,'full50',sample_ids,set(),records,arrays,positive_rows,groups,device)
        ck['oof_threshold']=float(threshold);ck['oof_metric']=metric
        torch.save(ck,out/'artifacts'/f'objectness_{variant}_full50.pt');checkpoints[variant]=ck
        write_csv(out/'metrics/oof_summary.csv',oof_summary)
        write_csv(out/'metrics/oof_candidate_predictions.csv',oof_predictions)
        status(out,'variant_training_complete',variant=variant,oof_threshold=threshold,oof_f1=metric['f1'])
    metric_rows,summaries=evaluate(out,checkpoints,b,device)
    status(out,'complete',training_models=18,test_images=201,variants=list(VARIANTS))


if __name__=='__main__':
    try:main()
    except Exception:
        # The background process retains the full traceback in its stderr log.
        parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--output_dir',default=str(ROOT/'objectness_positive_jitter_20261027'))
        args,_=parser.parse_known_args();status(Path(args.output_dir),'failed',error=traceback.format_exc())
        raise
