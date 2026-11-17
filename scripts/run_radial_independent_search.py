"""Training-side five-fold radial search. No official-test or joint-inference entry."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from scripts.run_f4_spatial_concat import F4Bank, JITTER, OLD
from scripts.run_radial_augmented_heads import RadialBank, MASK_SOURCE, digest
from scripts.run_objectness_positive_jitter import load, read_csv
from scripts.run_phase_b_mask_head import load_baseline
from scripts.run_global_ablation_201 import crop_at, paste_local, add_metrics, write_csv, write_json
from scripts.run_centered_mask_ablation import summary
from scripts.run_mask_without_objectness import center_target
from scripts.objectness_features import status
from scripts.recompute_objectness_deduplicated import postprocess_metrics
from visual_prompt.core.candidate_proposals import connected_instances, snapped_instance_centers
from visual_prompt.evaluation.radial_diagnostics import RadialDiagnostics, masked_channel_statistics
from visual_prompt.models.lightweight_heads import ObjectnessMLP, MaskVectorMLP, soft_iou_loss
from visual_prompt.models.radial_augmented_heads import F4RadialObjectness, RadialMaskVectorMLP
from visual_prompt.models.radial_profile import DirectionalRadialOperator, RadialProfileConfig
from visual_prompt.models.radial_adaptive_heads import AdaptiveRadialObjectness, AdaptiveRadialMask
from visual_prompt.pipelines.semantic_prompt_pipeline import seed_everything

ROOT = Path('outputs/studies/objectness_mask')
OUT = ROOT/'radial_independent_search_20261029'
CACHE_ROOT = OUT
AMPLITUDE_BASE = {'obj': 'A1', 'mask': 'A1'}
SEED = 111
COMPONENT_OVERRIDE = None
GEOMETRY = {'radii': None, 'opposite_pairs': False, 'smooth_sigma': 0.}
ARCHITECTURE = None


def component_indices(variant):
    if COMPONENT_OVERRIDE is not None and variant != 'A0':
        return COMPONENT_OVERRIDE
    if variant == 'B1':
        return list(range(64, 128))
    if variant == 'B2':
        return list(range(64, 192))
    if variant == 'B3':
        return list(range(64, 256))
    return list(range(256))


def norm_variant(variant, head):
    return AMPLITUDE_BASE[head] if variant.startswith(('B', 'C', 'D', 'E')) else variant


def transform_radial(ray, valid, variant, original_rms):
    ray = ray.masked_fill(~valid, 0)
    if variant in ('A0', 'A4'):
        return torch.zeros_like(ray)
    if variant in ('A2', 'A3'):
        alpha = .25 if variant == 'A2' else 1.
        ray = ray/ray.norm(dim=1, keepdim=True).clamp_min(1e-6)*(alpha*original_rms)
    return ray


def mask_descriptor(features, sam, e0):
    h, w = features.shape[-2:]
    sim = torch.einsum('bchw,c->bhw', sam, e0)
    stats = torch.stack((sim[:, h//2, w//2], sim.amax((-2, -1)), sim.mean((-2, -1)), sim.std((-2, -1))), 1)
    return torch.cat((features[:, :, h//2, w//2], features.mean((-2, -1)), stats), 1)


def cached_mask_forward(model, features, descriptor, e0, zero_radial=False):
    if hasattr(model, 'forward_features'):
        return model.forward_features(features, e0, zero_radial=zero_radial)
    output = model.net(descriptor); channels = features.shape[1]
    initial = F.pad(e0, (0, channels-len(e0)))
    vector = F.normalize(initial[None]+.25*output[:, :channels], dim=1)
    return torch.einsum('bchw,bc->bhw', features, vector)/.1+output[:, -1, None, None]


class TrainOnlyRadialBank(RadialBank):
    def __init__(self, baseline, device, allowed):
        super().__init__(baseline, device)
        self.allowed = set(allowed); self.valid_points = {}; self.accessed = set()
        if GEOMETRY['radii'] is not None:
            self.op = DirectionalRadialOperator(1, RadialProfileConfig(fusion='concat', radii=tuple(GEOMETRY['radii']))).to(device).eval()
        self.smoothed = {}

    def image(self, sid):
        if sid not in self.allowed:
            raise ValueError(f'Official-test/non-training image access forbidden: {sid}')
        self.accessed.add(sid)
        result = super().image(sid)
        if GEOMETRY['smooth_sigma'] > 0:
            if sid not in self.smoothed:
                import cv2
                gray = result[2][0, 0].cpu().numpy()
                blurred = cv2.GaussianBlur(gray, (0, 0), GEOMETRY['smooth_sigma'], borderType=cv2.BORDER_REFLECT_101)
                self.smoothed[sid] = torch.from_numpy(blurred.copy()).to(self.device)[None, None]
            return result[0], result[1], self.smoothed[sid]
        return result

    @staticmethod
    def encode_pairs(values, valid):
        if not GEOMETRY['opposite_pairs']:
            return values, valid
        # Values and validity are N,C or C,H,W; retain four types, eight directions/eight radii.
        if values.ndim == 2:
            v = values.reshape(-1, 4, 8, 8); m = valid.reshape(-1, 4, 8, 8)
            a, b = v[:, :, :4], v[:, :, 4:]
            good = m[:, :, :4] & m[:, :, 4:]
            value = torch.cat(((a+b)/2, (a-b)/2), 2)
            mask = torch.cat((good, good), 2)
        else:
            v = values.reshape(4, 8, 8, *values.shape[-2:]); m = valid.reshape_as(v)
            a, b = v[:, :4], v[:, 4:]
            good = m[:, :4] & m[:, 4:]
            value = torch.cat(((a+b)/2, (a-b)/2), 1)
            mask = torch.cat((good, good), 1)
        return value.masked_fill(~mask, 0).reshape_as(values), mask.reshape_as(valid)

    @torch.no_grad()
    def point_data(self, rows):
        grouped = defaultdict(set)
        for r in rows:
            key = (r['sample_id'], int(r['y']), int(r['x']))
            if key not in self.point_features:
                grouped[key[0]].add(key[1:])
        for sid, locations in grouped.items():
            ordered = sorted(locations)
            points = torch.tensor([[(x, y) for y, x in ordered]], dtype=torch.float32, device=self.device)
            p = self.op.sample_profiles(self.image(sid)[2], points)
            value, mask = self.encode_pairs(p.values.flatten(2)[0], p.valid.flatten(2)[0])
            values = value.cpu().numpy(); valid = mask.cpu().numpy()
            for loc, value, mask in zip(ordered, values, valid):
                self.point_features[(sid, *loc)] = value.copy(); self.valid_points[(sid, *loc)] = mask.copy()
        keys = [(r['sample_id'], int(r['y']), int(r['x'])) for r in rows]
        return np.stack([self.point_features[k] for k in keys]), np.stack([self.valid_points[k] for k in keys])

    @torch.no_grad()
    def map_data(self, sid, y, x):
        points = self.offsets+torch.tensor([[[x, y]]], device=self.device)
        p = self.op.sample_profiles(self.image(sid)[2], points)
        values = p.values.flatten(2).transpose(1, 2).reshape(256, 64, 64)
        valid = p.valid.flatten(2).transpose(1, 2).reshape(256, 64, 64)
        values, valid = self.encode_pairs(values, valid)
        return values.cpu().half(), valid.cpu()


def obj_curves(rows):
    scores = [r['score'] for r in rows]; sweep = []
    for threshold in np.unique(np.r_[0., .9, 1., scores])[::-1]:
        metric, _ = postprocess_metrics(rows, scores, float(threshold), 16)
        sweep.append(dict(threshold=float(threshold), **metric))
    eligible = [r for r in sweep if r['threshold'] <= .9]
    feasible = [r for r in eligible if r['recall'] >= .95]
    maximum = max(r['recall'] for r in eligible)
    chosen = max(feasible or [r for r in eligible if r['recall'] == maximum], key=lambda r: r['threshold'])
    fixed, _ = postprocess_metrics(rows, scores, .9, 16)
    result = dict(fixed09=fixed, loose=chosen, threshold=chosen['threshold'])
    images = len({r['sample_id'] for r in rows})
    for budget in (.25, .5):
        valid = [r for r in sweep if r['background']/images <= budget]
        result[f'recall_at_bg_{budget}'] = max((r['recall'] for r in valid), default=0.)
    result['candidate_pool_recall'] = postprocess_metrics(rows, [1.]*len(rows), 0., 0)[0]['recall']
    return result, sweep


def obj_fold(variant, fold, records, f4bank, radial):
    folder = OUT/'obj'/variant; path = folder/'artifacts'/f'{fold}.pt'
    if path.exists() and (folder/'metrics'/f'{fold}.json').exists():
        return json.loads((folder/'metrics'/f'{fold}.json').read_text(encoding='utf-8'))
    original = load(JITTER/'artifacts'/f'objectness_target_any_{fold}.pt')
    schedule_path = OLD/'metadata'/f'{fold}_schedule.json'; epochs = json.loads(schedule_path.read_text())
    flat = [r for epoch in epochs for r in epoch]
    train_ids, val_ids = set(original['sample_ids']), set(original['validation_ids'])
    assert not train_ids & val_ids and {r['sample_id'] for r in flat} <= train_ids
    columns = component_indices(variant)
    raw, valid = radial.point_data(flat); raw, valid = raw[:, columns], valid[:, columns]
    mu, std = masked_channel_statistics(raw, valid)
    f4 = torch.from_numpy(f4bank.normalize(f4bank.raw(flat, training=True), original)).to(radial.device)
    original_rms = float(f4.square().sum(1).mean().sqrt())
    ray = transform_radial(torch.from_numpy((raw-mu)/std).to(radial.device), torch.from_numpy(valid).to(radial.device), norm_variant(variant, 'obj'), original_rms)
    seed_everything(SEED)
    model = (AdaptiveRadialObjectness(len(columns), ARCHITECTURE) if ARCHITECTURE is not None else
             ObjectnessMLP(1440) if variant == 'A0' else F4RadialObjectness(radial_channels=len(columns))).to(radial.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    labels = torch.tensor([r['label'] for r in flat], dtype=torch.float32, device=radial.device)
    history = []; offset = 0; started = time.perf_counter()
    for epoch, rows in enumerate(epochs):
        model.train(); losses = []
        for start in range(0, len(rows), 32):
            sl = slice(offset+start, offset+min(start+32, len(rows)))
            logits = model(f4[sl]) if variant == 'A0' else model(f4[sl], ray[sl])
            loss = F.binary_cross_entropy_with_logits(logits, labels[sl])
            if hasattr(model, 'auxiliary_loss'):
                loss = loss+model.auxiliary_loss(labels[sl])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite Obj loss')
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        offset += len(rows); history.append(dict(epoch=epoch+1, BCE=float(np.mean(losses))))
        if (epoch+1) % 20 == 0:
            status(OUT, 'training', head='obj', variant=variant, fold=fold, **history[-1])
    val = [r for r in records if r['source'] == 'eval' and r['sample_id'] in val_ids]
    rv, vv = radial.point_data(val); rv, vv = rv[:, columns], vv[:, columns]
    fv = torch.from_numpy(f4bank.normalize(f4bank.raw(val), original)).to(radial.device)
    r = transform_radial(torch.from_numpy((rv-mu)/std).to(radial.device), torch.from_numpy(vv).to(radial.device), norm_variant(variant, 'obj'), original_rms)
    model.eval()
    with torch.no_grad():
        z = model(fv) if variant == 'A0' else model(fv, r)
        z0 = z if variant == 'A0' else model.forward_radial_zero(fv, r) if hasattr(model, 'forward_radial_zero') else model(fv, torch.zeros_like(r))
        scores = z.sigmoid().cpu().tolist()
        representation = model.radial_representation(r) if hasattr(model, 'radial_representation') else r
    diagnostics = RadialDiagnostics()
    for i, row in enumerate(val):
        diagnostics.add(row['sample_id'], 'target' if row['assignment'] else 'background', representation[i:i+1].cpu().numpy(), [float(z[i])], [float(z0[i])])
    predictions = [dict(**r0, score=s, logit=float(z[i]), zero_logit=float(z0[i])) for i, (r0, s) in enumerate(zip(val, scores))]
    point_metrics, sweep = obj_curves(predictions)
    result = dict(head='obj', variant=variant, fold=fold, point_metrics=point_metrics,
                  diagnostics=diagnostics.result(radial_absent=variant == 'A0'), predictions=predictions,
                  train_ids=sorted(train_ids), validation_ids=sorted(val_ids), train_points=len(flat),
                  parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-started)
    ck = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, radial_mean=mu, radial_std=std,
              original_rms=original_rms, variant=variant, component_indices=columns, amplitude=norm_variant(variant, 'obj'), fold=fold, epochs=100, seed=SEED,
              sample_ids=sorted(train_ids), validation_ids=sorted(val_ids), schedule_sha256=digest(schedule_path),
              normalization_train_only=True, original_statistics={k: original[k] for k in ('mean', 'std', 'prototype_target', 'prototype_background')})
    ck['architecture'] = ARCHITECTURE; ck['geometry'] = GEOMETRY
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(ck, path)
    write_csv(folder/'metrics'/f'{fold}_history.csv', history)
    write_csv(folder/'metrics'/f'{fold}_curve.csv', sweep); write_json(folder/'metrics'/f'{fold}.json', result)
    status(OUT, 'fold_complete', head='obj', variant=variant, fold=fold, diagnostics=result['diagnostics'], seconds=result['seconds'])
    return result


def prepare_masks(radial, ids):
    manifest = read_csv(MASK_SOURCE/'metadata/train_crops.csv'); items = []
    cache = CACHE_ROOT/'cache/training_radial.pt'
    signature = digest(MASK_SOURCE/'metadata/train_crops.csv')
    cached = load(cache) if cache.exists() else None
    if cached is not None:
        assert cached['manifest_sha256'] == signature
        assert cached.get('geometry', {'radii': None, 'opposite_pairs': False, 'smooth_sigma': 0.}) == GEOMETRY
    sums = []; squares = []; counts = []; points_error = 0.
    for number, row in enumerate(manifest):
        sid, y, x = row['sample_id'], int(row['y']), int(row['x'])
        assert sid in ids
        with np.load(MASK_SOURCE/'cache/train_local'/f'{sid}_{y}_{x}.npz') as d:
            sam = torch.from_numpy(d['feature'].copy()); target = torch.from_numpy(d['mask'].astype(np.uint8))
        image, gt, _ = radial.image(sid); inst = connected_instances(gt)
        expected, iid = center_target(inst.labels, y, x)
        assert np.array_equal(target.numpy(), expected) and bool(iid) == bool(int(row['positive']))
        _, local_gt = crop_at(image, gt, y, x, 64)
        ray, valid = (cached['radial'][number], cached['valid'][number]) if cached is not None else radial.map_data(sid, y, x)
        point, _ = radial.point_data([dict(sample_id=sid, y=y, x=x)])
        points_error = max(points_error, float(np.max(np.abs(ray[:, 32, 32].float().numpy()-point[0]))))
        value = ray.double(); mask = valid.double()
        sums.append((value*mask).sum((-2, -1))); squares.append((value.square()*mask).sum((-2, -1))); counts.append(mask.sum((-2, -1)))
        items.append(dict(sample_id=sid, y=y, x=x, source=row['source'], positive=int(row['positive']),
                          sam=sam, target=target, other=torch.from_numpy(((local_gt > 0)&(expected == 0)).copy()),
                          radial=ray, valid=valid))
        if (number+1) % 40 == 0:
            status(OUT, 'preparing_training_ROIs', ROIs_done=number+1, ROIs_total=len(manifest))
    assert len(items) == 320 and {r['sample_id'] for r in items} == set(ids)
    if cached is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(manifest_sha256=signature, geometry=GEOMETRY, radial=torch.stack([r['radial'] for r in items]),
                        valid=torch.stack([r['valid'] for r in items])), cache)
    q0 = dict(training_images=50, training_ROIs=len(items), positive_ROIs=sum(r['positive'] for r in items),
              radial_point_map_max_FP16_error=points_error, true_context=True, invalid_excluded_from_statistics=True,
              official_test_read=False, joint_inference=False)
    write_json(OUT/'metadata/preflight.json', q0)
    return items, torch.stack(sums), torch.stack(squares), torch.stack(counts)


@torch.no_grad()
def mask_features(items, train_indices, variant, sums, squares, counts, e0):
    count = counts[train_indices].sum(0).clamp_min(1)
    mean = sums[train_indices].sum(0)/count
    std = (squares[train_indices].sum(0)/count-mean.square()).clamp_min(0).sqrt().clamp_min(.01)
    columns = component_indices(variant)
    mean, std = mean[columns], std[columns]
    mu, scale = mean.float().to(e0.device)[None, :, None, None], std.float().to(e0.device)[None, :, None, None]
    channels = 352 if variant == 'A0' else 352+len(columns)
    features = torch.empty((len(items), channels, 64, 64), device=e0.device)
    descriptors = torch.empty((len(items), 2*channels+4), device=e0.device)
    norm_rows = []
    for start in range(0, len(items), 8):
        batch = items[start:start+8]
        sam = torch.stack([r['sam'] for r in batch]).to(e0.device, dtype=torch.float32)
        raw = torch.stack([r['radial'][columns] for r in batch]).to(e0.device, dtype=torch.float32)
        valid = torch.stack([r['valid'][columns] for r in batch]).to(e0.device)
        ray = transform_radial((raw-mu)/scale, valid, norm_variant(variant, 'mask'), 1.)
        z = sam if variant == 'A0' else torch.cat((sam, ray), 1)
        features[start:start+len(batch)] = z; descriptors[start:start+len(batch)] = mask_descriptor(z, sam, e0)
        for item, sn, rn in zip(batch, sam.norm(dim=1).mean((-2, -1)).tolist(), ray.norm(dim=1).mean((-2, -1)).tolist()):
            norm_rows.append(dict(sample_id=item['sample_id'], positive=item['positive'], SAM_norm=sn, radial_norm=rn))
    return features, descriptors, mean.float(), std.float(), norm_rows


@torch.no_grad()
def evaluate_mask(model, variant, val_indices, items, features, descriptors, e0, radial, records):
    diagnostics = RadialDiagnostics(); roi_rows = []; image_rows = []
    grouped = defaultdict(list)
    candidates = {(r['sample_id'], int(r['y']), int(r['x'])) for r in records if r['source'] == 'eval'}
    for i in val_indices:
        grouped[items[i]['sample_id']].append(i)
    model.eval()
    for sid, indices in grouped.items():
        image, gt, _ = radial.image(sid); inst = connected_instances(gt)
        centers = {(y, x) for _, y, x in snapped_instance_centers(inst)}
        unions = {k: np.zeros_like(gt, dtype=np.uint8) for k in ('top5_no_obj', 'gt_center_diagnostic')}
        for i in indices:
            item = items[i]; f = features[i:i+1]
            logits = cached_mask_forward(model, f, descriptors[i:i+1], e0)[0]
            f0 = f.clone()
            if variant != 'A0':
                f0[:, 352:] = 0
            desc0 = mask_descriptor(f0, f0[:, :352], e0)
            zero_logits = cached_mask_forward(model, f0, desc0, e0, zero_radial=True)[0]
            z = logits.cpu().numpy(); z0 = zero_logits.cpu().numpy(); pred = z > 0
            target = item['target'].numpy() > 0; other = item['other'].numpy()
            background = ~(target | other)
            radial_repr = model.radial_representation(f[:, 352:]) if hasattr(model, 'radial_representation') else f[:, 352:]
            ray = radial_repr[0].permute(1, 2, 0).cpu().numpy() if variant != 'A0' else np.empty((64, 64, 0))
            # One mean/square per ROI then per image; fixed pixel subset, no global pixel pooling.
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f'{sid}:{item["y"]}:{item["x"]}'.encode()).digest()[:4], 'little'))
            for group, mask in [('target', target), ('background', background), ('other_instance', other)]:
                locations = np.argwhere(mask)
                if len(locations) > 64:
                    locations = locations[rng.choice(len(locations), 64, replace=False)]
                if len(locations):
                    yy, xx = locations.T
                    diagnostics.add(sid, group, ray[yy, xx], z[yy, xx], z0[yy, xx])
            inter = int((pred & target).sum()); union = int((pred | target).sum())
            candidate = (sid, item['y'], item['x']) in candidates
            roi_rows.append(dict(sample_id=sid, y=item['y'], x=item['x'], candidate=int(candidate), positive=item['positive'],
                                 other_target_pixels=int(other.sum()), pred_pixels=int(pred.sum()), target_pixels=int(target.sum()),
                                 inter=inter, union=union, FP_pixels=int((pred & ~target).sum()),
                                 iou=inter/max(1, union), max_logit=float(z.max())))
            if candidate:
                paste_local(unions['top5_no_obj'], pred.astype(np.uint8), item['y'], item['x'])
            if (item['y'], item['x']) in centers:
                paste_local(unions['gt_center_diagnostic'], pred.astype(np.uint8), item['y'], item['x'])
        assert sum((sid, items[i]['y'], items[i]['x']) in candidates for i in indices) == 5
        row = dict(sample_id=sid, target_count=inst.count, image_pixels=gt.size)
        for name, pred in unions.items():
            add_metrics(row, name, pred, gt)
            assert row[name+'_false_pixels'] == row[name+'_pred_pixels']-row[name+'_inter']
        image_rows.append(row)
    pos = [r for r in roi_rows if r['positive']]; neg = [r for r in roi_rows if not r['positive']]
    near = [r for r in neg if r['other_target_pixels']]
    metrics = dict(positive_ROI_global_IoU=100*sum(r['inter'] for r in pos)/max(1, sum(r['union'] for r in pos)),
                   positive_ROI_mean_IoU=100*float(np.mean([r['iou'] for r in pos])),
                   negative_empty_percent=100*float(np.mean([not r['pred_pixels'] for r in neg])),
                   negative_FP_mean=float(np.mean([r['FP_pixels'] for r in neg])),
                   other_target_negative_empty_percent=100*float(np.mean([not r['pred_pixels'] for r in near])) if near else None,
                   positive_empty_percent=100*float(np.mean([not r['pred_pixels'] for r in pos])))
    return dict(task_metrics=metrics, diagnostics=diagnostics.result(radial_absent=variant == 'A0'),
                other_instance_diagnostics=diagnostics.result(negative='other_instance', radial_absent=variant == 'A0'),
                roi_rows=roi_rows, image_rows=image_rows,
                image_summary=[summary(image_rows, name) for name in unions])


def mask_fold(variant, fold, items, sums, squares, counts, e0, radial, records):
    folder = OUT/'mask'/variant; path = folder/'artifacts'/f'{fold}.pt'
    if path.exists() and (folder/'metrics'/f'{fold}.json').exists():
        return json.loads((folder/'metrics'/f'{fold}.json').read_text(encoding='utf-8'))
    original = load(JITTER/'artifacts'/f'objectness_target_any_{fold}.pt')
    train_ids, val_ids = set(original['sample_ids']), set(original['validation_ids'])
    assert not train_ids & val_ids
    train_indices = [i for i, r in enumerate(items) if r['sample_id'] in train_ids]
    val_indices = [i for i, r in enumerate(items) if r['sample_id'] in val_ids]
    pos = [i for i in train_indices if items[i]['positive']]; neg = [i for i in train_indices if not items[i]['positive']]
    features, descriptors, mu, std, norms = mask_features(items, train_indices, variant, sums, squares, counts, e0)
    seed_everything(SEED)
    model = (AdaptiveRadialMask(len(component_indices(variant)), e0, ARCHITECTURE) if ARCHITECTURE is not None else
             MaskVectorMLP(352, 128, e0, 'm2_residual') if variant == 'A0' else RadialMaskVectorMLP(352, len(component_indices(variant)), e0)).to(e0.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    targets = torch.stack([r['target'] for r in items]).to(e0.device, dtype=torch.float32)
    p_rng, n_rng = random.Random(111), random.Random(112)
    history = []; started = time.perf_counter(); schedule = []
    for epoch in range(120):
        p, n = p_rng.choices(pos, k=128), n_rng.choices(neg, k=128); losses = []
        schedule.append(p+n)
        for start in range(0, 128, 8):
            indices = p[start:start+8]+n[start:start+8]
            logits = cached_mask_forward(model, features[indices], descriptors[indices], e0)
            a = soft_iou_loss(logits[:8], targets[indices[:8]])
            b = F.softplus(logits[8:]).mean(); loss = a+b
            if hasattr(model, 'auxiliary_loss'):
                loss = loss+model.auxiliary_loss(logits, targets[indices], 8)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite Mask loss')
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append((float(a.detach()), float(b.detach())))
        history.append(dict(epoch=epoch+1, positive_IoU_loss=float(np.mean([r[0] for r in losses])), negative_BCE=float(np.mean([r[1] for r in losses]))))
        if (epoch+1) % 20 == 0:
            write_csv(folder/'metrics'/f'{fold}_history.csv', history)
            status(OUT, 'training', head='mask', variant=variant, fold=fold, **history[-1], seconds=time.perf_counter()-started)
    result = evaluate_mask(model, variant, val_indices, items, features, descriptors, e0, radial, records)
    result.update(head='mask', variant=variant, fold=fold, train_ids=sorted(train_ids), validation_ids=sorted(val_ids),
                  parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-started)
    ck = dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, variant=variant, fold=fold,
              radial_mean=mu, radial_std=std, original_rms=1., component_indices=component_indices(variant),
              amplitude=norm_variant(variant, 'mask'), sample_ids=sorted(train_ids), validation_ids=sorted(val_ids),
              normalization_train_only=True, epochs=120, seed=SEED, train_ROIs=len(train_indices), positive_ROIs=len(pos), negative_ROIs=len(neg),
              sampling_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest())
    ck['architecture'] = ARCHITECTURE; ck['geometry'] = GEOMETRY
    path.parent.mkdir(parents=True, exist_ok=True); torch.save(ck, path)
    write_json(folder/'metrics'/f'{fold}.json', result); write_csv(folder/'metrics'/f'{fold}_norms.csv', norms)
    status(OUT, 'fold_complete', head='mask', variant=variant, fold=fold, metrics=result['task_metrics'], diagnostics=result['diagnostics'], seconds=result['seconds'])
    del model, optimizer, features, descriptors, targets
    torch.cuda.empty_cache()
    return result


def aggregate(head, variant, results):
    assert len(results) == 5
    ids = [sid for r in results for sid in r['validation_ids']]
    assert len(ids) == len(set(ids)) == 45
    folder = OUT/head/variant
    diagnostics = {k: (float(np.mean([r['diagnostics'][k] for r in results])) if all(r['diagnostics'][k] is not None for r in results) else None)
                   for k in ('Fisher_J', 'margin_gain', 'target_logit_change', 'background_logit_change')}
    if head == 'obj':
        predictions = [v for r in results for v in r['predictions']]
        task, sweep = obj_curves(predictions)
        write_csv(folder/'metrics/OOF_predictions.csv', predictions); write_csv(folder/'metrics/OOF_curve.csv', sweep)
        sentences = [f'五折独立Obj：固定0.9 P/R/F1={100*task["fixed09"]["precision"]:.2f}/{100*task["fixed09"]["recall"]:.2f}/{100*task["fixed09"]["f1"]:.2f}%。',
                     f'背景FP/图≤0.25与≤0.5时召回为{100*task["recall_at_bg_0.25"]:.2f}%与{100*task["recall_at_bg_0.5"]:.2f}%，两个中间量按折均值另列。',
                     '本组只做独立验证，保留待同波父对照比较，不读取201测试集或运行联合推理。']
    else:
        rois = [v for r in results for v in r['roi_rows']]; images = [v for r in results for v in r['image_rows']]
        pos = [r for r in rois if r['positive']]; neg = [r for r in rois if not r['positive']]
        task = dict(positive_ROI_global_IoU=100*sum(r['inter'] for r in pos)/max(1, sum(r['union'] for r in pos)),
                    positive_ROI_mean_IoU=100*float(np.mean([r['iou'] for r in pos])),
                    negative_empty_percent=100*float(np.mean([not r['pred_pixels'] for r in neg])),
                    negative_FP_mean=float(np.mean([r['FP_pixels'] for r in neg])),
                    positive_empty_percent=100*float(np.mean([not r['pred_pixels'] for r in pos])),
                    image_summary=[summary(images, name) for name in ('top5_no_obj', 'gt_center_diagnostic')])
        write_csv(folder/'metrics/OOF_ROIs.csv', rois); write_csv(folder/'metrics/OOF_images.csv', images)
        sentences = [f'五折独立Mask：正ROI累计IoU={task["positive_ROI_global_IoU"]:.2f}%，逐ROI均值={task["positive_ROI_mean_IoU"]:.2f}%。',
                     f'负ROI全零率={task["negative_empty_percent"]:.2f}%，平均FP={task["negative_FP_mean"]:.2f}像素，正例误拒={task["positive_empty_percent"]:.2f}%。',
                     '固定Top5全部送入Mask，不经过Obj；独立优势待同波比较，不运行联合推理。']
    value = dict(head=head, variant=variant, OOF_images=45, task_metrics=task, diagnostics_fold_mean=diagnostics,
                 parameters=results[0]['parameters'], folds=[{k: r[k] for k in ('fold', 'diagnostics', 'seconds')} for r in results],
                 Chinese_conclusion=sentences, official_test_accessed=False, joint_inference=False)
    write_json(folder/'summary.json', value); status(OUT, 'configuration_complete', head=head, variant=variant, metrics=task, diagnostics=diagnostics)
    return value


def main(args):
    OUT.mkdir(parents=True, exist_ok=True); (OUT/'pid.txt').write_text(str(os.getpid()))
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    cfg = yaml.safe_load(Path('experiments/configs/objectness_mask/train_final_heads.yaml').read_text())['args']
    ids = cfg['sample_ids'].split(','); _, baseline = load_baseline(Path(cfg['baseline_config']))
    records = json.loads((ROOT/'phase_c_objectness/metadata/feature_manifest.json').read_text())['records']
    records = [r for r in records if r['sample_id'] in ids]
    radial = TrainOnlyRadialBank(baseline, torch.device('cuda:0'), ids)
    e0 = load(ROOT/'final_heads_50shot/artifacts/edited_target_embedding.pt').float().to(radial.device)
    protocol = dict(wave='Q2_content' if args.variants.startswith('B') else 'Q0_Q1_amplitude', variants=args.variants.split(','), heads=args.heads.split(','),
                    train_ids=ids, OOF_images=45, official_test_access=False, joint_inference=False,
                    feature_statistics='corresponding_training_fold_only_invalid_samples_excluded',
                    metric_definitions='J_and_paired_radial_zero_margin_plus_independent_tasks',
                    seed=SEED, immutable_source_mask_sha256=digest(MASK_SOURCE/'artifacts/mask_with_rejection.pt'))
    if any(v.startswith(('B', 'C', 'D', 'E')) for v in protocol['variants']):
        protocol['amplitude_parent'] = AMPLITUDE_BASE
        protocol['parent_selection_sha256'] = digest(Path(args.amplitude_parent)/'metadata/selected_A.json')
    if args.seed != 111 or any(v.startswith(('C', 'D', 'E')) for v in protocol['variants']):
        protocol['geometry'] = GEOMETRY
        protocol['component_override'] = COMPONENT_OVERRIDE
    if ARCHITECTURE is not None:
        protocol['architecture'] = ARCHITECTURE
    path = OUT/'metadata/protocol.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == protocol
    write_json(path, protocol); status(OUT, 'started', **protocol)
    all_results = []; f4bank = F4Bank(records, ids) if 'obj' in protocol['heads'] else None
    for variant in protocol['variants']:
        if 'obj' in protocol['heads']:
            results = [obj_fold(variant, f'fold{fold}', records, f4bank, radial) for fold in range(5)]
            all_results.append(aggregate('obj', variant, results))
    if 'mask' in protocol['heads']:
        items, sums, squares, counts = prepare_masks(radial, ids)
        for variant in protocol['variants']:
            results = [mask_fold(variant, f'fold{fold}', items, sums, squares, counts, e0, radial, records) for fold in range(5)]
            all_results.append(aggregate('mask', variant, results))
    assert radial.accessed <= set(ids)
    assert protocol['immutable_source_mask_sha256'] == digest(MASK_SOURCE/'artifacts/mask_with_rejection.pt')
    next_wave = 'Q3_geometry_selection_from_independent_OOF' if args.variants.startswith('B') else 'Q2_content_selection_from_independent_OOF'
    write_json(OUT/'summary.json', dict(results=all_results, wave_complete=True, next_wave=next_wave))
    write_json(OUT/'validation.json', dict(only_training_images_accessed=True, accessed_image_ids=sorted(radial.accessed),
                                          OOF_images=45, independent_heads=True, official_test_read=False,
                                          joint_inference=False, source_mask_unchanged=True,
                                          models=len(protocol['variants'])*len(protocol['heads'])*5))
    status(OUT, 'wave_complete', configurations=len(all_results), models=len(all_results)*5, next_wave=next_wave)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output_dir', default=str(OUT))
    parser.add_argument('--variants', default='A0,A1,A2,A3,A4'); parser.add_argument('--heads', default='obj,mask')
    parser.add_argument('--amplitude-parent')
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--cache-root')
    parser.add_argument('--radii')
    parser.add_argument('--opposite-pairs', action='store_true')
    parser.add_argument('--smooth-sigma', type=float, default=0.)
    parser.add_argument('--components', choices=['all', 'contrast', 'contrast_first', 'contrast_first_second'])
    parser.add_argument('--architecture-json')
    args = parser.parse_args(); OUT = Path(args.output_dir)
    SEED = args.seed
    ARCHITECTURE = json.loads(Path(args.architecture_json).read_text(encoding='utf-8')) if args.architecture_json else None
    GEOMETRY = dict(radii=[float(v) for v in args.radii.split(',')] if args.radii else None,
                    opposite_pairs=args.opposite_pairs, smooth_sigma=args.smooth_sigma)
    if args.components:
        mapping = {'all': list(range(256)), 'contrast': list(range(64, 128)),
                   'contrast_first': list(range(64, 192)), 'contrast_first_second': list(range(64, 256))}
        COMPONENT_OVERRIDE = mapping[args.components]
    if not set(args.variants.split(',')) <= {'A0', 'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3',
                                           'D1', 'D1_linear', 'D2', 'D2_zero', 'E1', 'E2', 'E3', 'E4_1', 'E4_3'}:
        parser.error('Only implemented amplitude/content configurations are accepted')
    if any(v.startswith(('B', 'C', 'D', 'E')) for v in args.variants.split(',')):
        if not args.amplitude_parent:
            parser.error('Content wave requires an amplitude parent and a separate output directory')
        CACHE_ROOT = Path(args.amplitude_parent)
        if CACHE_ROOT.resolve() == OUT.resolve():
            parser.error('Content outputs must not overwrite amplitude outputs')
        AMPLITUDE_BASE = json.loads((CACHE_ROOT/'metadata/selected_A.json').read_text(encoding='utf-8'))['selected_amplitude']
    else:
        CACHE_ROOT = OUT
    if args.cache_root:
        CACHE_ROOT = Path(args.cache_root)
    if any(GEOMETRY.values()) and CACHE_ROOT.resolve() == Path(args.amplitude_parent or '.').resolve():
        parser.error('Changed geometry requires a distinct cache directory')
    if not set(args.heads.split(',')) <= {'obj', 'mask'}:
        parser.error('Only independent Obj and Mask heads are accepted')
    try:
        main(args)
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True); status(OUT, 'failed', error=traceback.format_exc()); raise
