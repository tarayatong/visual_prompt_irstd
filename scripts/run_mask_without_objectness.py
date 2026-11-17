"""Top-5 local SAM masks without Objectness, including explicit empty-mask training."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import traceback

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from scripts.run_phase_b_mask_head import load_baseline
from scripts.run_global_ablation_201 import crop_at, encode_local, paste_local, add_metrics, write_csv, write_json
from scripts.run_centered_mask_ablation import summary
from scripts.run_objectness_positive_jitter import load, read_csv, predictor_for
from scripts.objectness_features import status
from visual_prompt.core.candidate_proposals import connected_instances, snapped_instance_centers, scored_point_nms
from visual_prompt.data.irstd import raw_irstd_paths
from visual_prompt.models.lightweight_heads import MaskVectorMLP, soft_iou_loss
from visual_prompt.pipelines.semantic_prompt_pipeline import load_rgb_and_mask, seed_everything

ROOT = Path('outputs/studies/objectness_mask')
KINDS = ('positive_only', 'with_rejection')


def center_target(labels, y, x):
    """Only the instance containing the prompt pixel is foreground; no snapping."""
    iid = int(labels[y, x])
    target = (labels == iid).astype(np.uint8) if iid else np.zeros(labels.shape, np.uint8)
    _, crop = crop_at(np.zeros((*labels.shape, 3), np.uint8), target, y, x, 64)
    return crop, iid


def forward(model, feature, e0):
    sim = torch.einsum('bchw,c->bhw', feature, e0)
    zeros = torch.zeros_like(sim)
    return model(feature, e0, sim, zeros, zeros, .25, .1)




@torch.no_grad()
def prepare(out, baseline, ids, predictor):
    records = json.loads((ROOT/'phase_c_objectness/metadata/feature_manifest.json').read_text())['records']
    candidates = defaultdict(list)
    for row in records:
        if row['source'] == 'eval':
            candidates[row['sample_id']].append(row)
    cache = out/'cache/train_local'
    cache.mkdir(parents=True, exist_ok=True)
    items, manifest = [], []
    for number, sid in enumerate(ids):
        image, gt, _ = load_rgb_and_mask(*raw_irstd_paths(baseline.data_root, sid), (baseline.image_size, baseline.image_size))
        instances = connected_instances(gt)
        points = {(y, x): 'gt_center' for _, y, x in snapped_instance_centers(instances)}
        for row in candidates[sid]:
            points.setdefault((row['y'], row['x']), 'sim_top5')
        for (y, x), source in sorted(points.items()):
            target, iid = center_target(instances.labels, y, x)
            _, all_gt = crop_at(image, gt, y, x, 64)
            row = dict(sample_id=sid, y=y, x=x, source=source, positive=int(iid > 0),
                       instance_id=iid, target_pixels=int(target.sum()),
                       other_target_pixels=int(((all_gt > 0) & (target == 0)).sum()))
            path = cache/f'{sid}_{y}_{x}.npz'
            if path.exists():
                data = np.load(path)
                assert np.array_equal(data['mask'], target)
                feature = data['feature'].copy()
            else:
                _, _, feature = encode_local(predictor, image, y, x, 64, 64)
                feature = feature[0].cpu().half().numpy()
                np.savez_compressed(path, feature=feature, mask=target)
            assert bool(target.any()) == bool(iid)
            assert not np.any((target > 0) & (all_gt == 0))
            items.append(dict(**row, feature=torch.from_numpy(feature), mask=torch.from_numpy(target.astype(np.float32))))
            manifest.append(row)
        status(out, 'prepare_training', images_done=number+1, images_total=len(ids))
    assert len({r['sample_id'] for r in manifest}) == 50
    write_csv(out/'metadata/train_crops.csv', manifest)
    counts = dict(crops=len(items), positive=sum(r['positive'] for r in manifest),
                  negative=sum(not r['positive'] for r in manifest),
                  negative_with_other_target=sum(not r['positive'] and r['other_target_pixels'] > 0 for r in manifest))
    assert counts['negative_with_other_target'] > 0
    write_json(out/'metadata/training_counts.json', counts)
    return items


def train(out, kind, items, e0, epochs):
    path = out/'artifacts'/f'mask_{kind}.pt'
    if path.exists():
        return load(path)
    positive = [r for r in items if r['positive']]
    negative = [r for r in items if not r['positive']]
    seed_everything(111)
    model = MaskVectorMLP(len(e0), 128, e0, 'm2_residual').to(e0.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    pos_rng, neg_rng = random.Random(111), random.Random(112)
    history = []
    for epoch in range(epochs):
        # Same 128 positive draws and 16 updates in both controls; no augmentation.
        pos = pos_rng.choices(positive, k=128)
        neg = neg_rng.choices(negative, k=128)
        values = []
        for start in range(0, 128, 8):
            batch = pos[start:start+8]
            if kind == 'with_rejection':
                batch = batch + neg[start:start+8]
            features = torch.stack([r['feature'] for r in batch]).to(e0.device, dtype=torch.float32)
            targets = torch.stack([r['mask'] for r in batch]).to(e0.device)
            logits = forward(model, features, e0)
            positive_loss = soft_iou_loss(logits[:8], targets[:8])
            # Empty-mask IoU has near-zero gradients; explicit negative pixel BCE is needed.
            negative_loss = F.softplus(logits[8:]).mean() if kind == 'with_rejection' else logits.sum()*0
            loss = positive_loss + negative_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            values.append((float(positive_loss.detach()), float(negative_loss.detach())))
        history.append(dict(epoch=epoch+1, positive_iou_loss=float(np.mean([v[0] for v in values])),
                            negative_bce=float(np.mean([v[1] for v in values]))))
        if (epoch+1) % 20 == 0:
            write_csv(out/'metrics'/f'{kind}_history.csv', history)
            status(out, 'training', kind=kind, epochs=epochs, **history[-1])
    ck = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
              channels=len(e0), hidden=128, variant='m2_residual', residual_scale=.25, temperature=.1,
              sample_ids=sorted({r['sample_id'] for r in items}), seed=111, epochs=epochs,
              augmentation=False, kind=kind, history=history, parameters=sum(p.numel() for p in model.parameters()))
    torch.save(ck, path)
    return ck


@torch.no_grad()
def evaluate(out, baseline, predictor, e0, checkpoints, train_ids):
    models = {}
    for kind, ck in checkpoints.items():
        model = MaskVectorMLP(len(e0), ck['hidden'], e0, 'm2_residual').to(e0.device)
        model.load_state_dict(ck['model']); model.eval(); models[kind] = model
    source = read_csv(ROOT/'residual_weak_augmentation_20261027/metrics/residual_candidate_scores.csv')
    grouped = defaultdict(list)
    for row in source:
        grouped[row['sample_id']].append(dict(y=int(row['y']), x=int(row['x']), score=float(row['score'])))
    assert len(grouped) == 201 and sum(map(len, grouped.values())) == 1005
    assert not set(grouped) & set(train_ids)
    all_images, all_rois = [], []
    image_cache = out/'cache/evaluation'
    image_cache.mkdir(parents=True, exist_ok=True)
    for number, (sid, points) in enumerate(grouped.items()):
        saved = image_cache/f'{sid}.json'
        if saved.exists():
            cached = json.loads(saved.read_text(encoding='utf-8'))
            all_images.append(cached['image']); all_rois.extend(cached['rois']); continue
        image, gt, _ = load_rgb_and_mask(*raw_irstd_paths(baseline.data_root, sid), (baseline.image_size, baseline.image_size))
        instances = connected_instances(gt)
        centers = [(y, x) for _, y, x in snapped_instance_centers(instances)]
        proposals = [(p['y'], p['x']) for p in points]
        # Auxiliary paired comparison only; the proposed route uses all five points.
        eligible = [p for p in points if p['score'] >= .9]
        keep = scored_point_nms([(p['y'], p['x'], p['score']) for p in eligible], 16)
        routed = [(eligible[i]['y'], eligible[i]['x']) for i in keep]
        locations = list(dict.fromkeys(proposals + centers))
        unions = {kind+'_'+route: np.zeros_like(gt, dtype=np.uint8)
                  for kind in models for route in ('no_obj_top5', 'residual_obj09_nms16', 'gt_center')}
        roi_rows = []
        for y, x in locations:
            _, _, feature = encode_local(predictor, image, y, x, 64, 64)
            target, iid = center_target(instances.labels, y, x)
            _, local_gt = crop_at(image, gt, y, x, 64)
            for kind, model in models.items():
                pred = (forward(model, feature, e0)[0] > 0).cpu().numpy().astype(np.uint8)
                inter = int(((pred > 0) & (target > 0)).sum())
                roi_rows.append(dict(sample_id=sid, kind=kind, y=y, x=x, candidate=int((y, x) in proposals),
                                     positive=int(iid > 0), other_target_pixels=int(((local_gt > 0) & (target == 0)).sum()),
                                     pred_pixels=int(pred.sum()), target_pixels=int(target.sum()), intersection=inter,
                                     union=int(((pred > 0) | (target > 0)).sum()),
                                     other_instance_fp=int(((pred > 0) & (local_gt > 0) & (target == 0)).sum())))
                for route, chosen in [('no_obj_top5', proposals), ('residual_obj09_nms16', routed), ('gt_center', centers)]:
                    if (y, x) in chosen:
                        paste_local(unions[kind+'_'+route], pred, y, x)
        row = dict(sample_id=sid, target_count=instances.count, image_pixels=gt.size)
        for system, pred in unions.items():
            add_metrics(row, system, pred, gt)
            assert row[system+'_false_pixels'] == int(((pred > 0) & (gt == 0)).sum())
        write_json(saved, dict(image=row, rois=roi_rows))
        all_images.append(row); all_rois.extend(roi_rows)
        if (number+1) % 10 == 0:
            status(out, 'evaluating', images_done=number+1, images_total=201)
    systems = [kind+'_'+route for kind in models for route in ('no_obj_top5', 'residual_obj09_nms16', 'gt_center')]
    results = [summary(all_images, system) for system in systems]
    write_csv(out/'metrics/image_metrics.csv', all_images)
    write_csv(out/'metrics/roi_metrics.csv', all_rois)
    write_csv(out/'metrics/summary.csv', results)
    diagnostics = []
    for kind in models:
        rows = [r for r in all_rois if r['kind'] == kind and r['candidate']]
        pos = [r for r in rows if r['positive']]
        neg = [r for r in rows if not r['positive']]
        near = [r for r in neg if r['other_target_pixels']]
        diagnostics.append(dict(kind=kind, positive_rois=len(pos), negative_rois=len(neg),
                                empty_rejection_percent=100*np.mean([r['pred_pixels'] == 0 for r in neg]),
                                positive_empty_percent=100*np.mean([r['pred_pixels'] == 0 for r in pos]),
                                negative_with_other_target=len(near),
                                other_target_empty_rejection_percent=100*np.mean([r['pred_pixels'] == 0 for r in near]),
                                positive_roi_global_iou_percent=100*sum(r['intersection'] for r in pos)/max(1, sum(r['union'] for r in pos))))
    write_csv(out/'metrics/rejection_summary.csv', diagnostics)
    lines = ['Top5直接原图64×64裁剪→冻结SAM2重新编码→Mask→映射回原图并集；主路线不调用Obj、不加后置mask筛选/NMS。新训两组均无数据增强、相同正样本及16次更新/epoch；原A2仅为历史权重参考。201张测试/297实例，阈值logit>0固定；未用test调参。', '',
             '| 设置 | mIoU | nIoU | Pd | 全图Fa/M | 像素F1 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        lines.append(f'| {r["system"]} | {r["miou_percent"]:.2f}% | {r["niou_percent"]:.2f}% | {r["pd_percent"]:.2f}% | {r["fa_per_million"]:.3f} | {r["f1_percent"]:.2f}% |')
    lines += ['', '| Mask | 错误候选全零率 | 正候选误拒率 | 含其他目标负ROI全零率 |', '| --- | ---: | ---: | ---: |']
    for r in diagnostics:
        lines.append(f'| {r["kind"]} | {r["empty_rejection_percent"]:.2f}% | {r["positive_empty_percent"]:.2f}% | {r["other_target_empty_rejection_percent"]:.2f}% |')
    control = next(r for r in results if r['system'] == 'positive_only_no_obj_top5')
    proposed = next(r for r in results if r['system'] == 'with_rejection_no_obj_top5')
    lines += ['', f'简结：加入拒绝监督后，无Obj的mIoU变化{proposed["miou_percent"]-control["miou_percent"]:+.2f}个百分点，Fa变化{proposed["fa_per_million"]-control["fa_per_million"]:+.3f}/M。严格全零率与全图指标同时报告，避免用减小预测面积冒充有效拒绝。单种子、全50图固定epoch，不宣称稳定增益。完整结果：`{out.as_posix()}/metrics/`。']
    write_json(out/'validation.json', dict(test_images=201, GT_instances=297, candidate_rois=1005,
                                           train_test_disjoint=True, full_image_FP_verified=True,
                                           inference_uses_GT=False, diagnostic_labels_only_use_GT=True,
                                           no_augmentation=True, mask_union=True))
    status(out, 'complete', results=results, rejection=diagnostics)


def main(out, epochs):
    for folder in ('artifacts', 'metadata', 'metrics'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    cfg = yaml.safe_load(Path('experiments/configs/objectness_mask/train_final_heads.yaml').read_text())['args']
    ids = cfg['sample_ids'].split(',')
    _, baseline = load_baseline(Path(cfg['baseline_config']))
    protocol = dict(train_ids=ids, train_images=50, test_images=201, epochs=epochs, seed=111,
                    positive_source='GT centers and actual training sim Top5 on GT pixels',
                    negative_source='actual training sim Top5 on background pixels, including ROIs containing other targets',
                    label='only instance at exact center pixel; background center gets all-zero mask',
                    augmentation=False, trainable='MaskVectorMLP only', ROI=64, SAM_input=baseline.encoder_input_size,
                    inference='all existing sim Top5, raw ROI SAM reencoding, mask logit>0, full image union',
                    paired_control='same positive draws and optimizer updates, positive IoU vs positive IoU + negative BCE',
                    auxiliary='same masks with frozen no-augmentation residual Obj0.9/NMS16; GT center diagnostic only',
                    checkpoint_selection='fixed epochs; no test tuning; no validation-selected checkpoint')
    protocol_path = out/'metadata/protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text(encoding='utf-8')) == protocol
    write_json(protocol_path, protocol)
    e0 = load(ROOT/'final_heads_50shot/artifacts/edited_target_embedding.pt').float().to('cuda:0')
    predictor = predictor_for(baseline, e0.device)
    items = prepare(out, baseline, ids, predictor)
    checkpoints = {kind: train(out, kind, items, e0, epochs) for kind in KINDS}
    del items
    checkpoints['original_A2'] = load(ROOT/'final_heads_50shot/artifacts/mask_final_a2.pt')
    evaluate(out, baseline, predictor, e0, checkpoints, ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default=str(ROOT/'mask_without_objectness_20261027'))
    parser.add_argument('--epochs', type=int, default=120)
    args = parser.parse_args()
    out = Path(args.output_dir)
    try:
        main(out, args.epochs)
    except Exception:
        out.mkdir(parents=True, exist_ok=True)
        status(out, 'failed', error=traceback.format_exc())
        raise
