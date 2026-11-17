"""Inspect completed content ablations using saved training OOF evidence only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_global_ablation_201 import write_csv, write_json
from scripts.run_objectness_positive_jitter import load, read_csv
from scripts.run_radial_independent_search import component_indices

ROOT = Path('outputs/studies/objectness_mask')
PARENT = ROOT/'radial_independent_search_20261029'
OUT = ROOT/'radial_content_search_20261029'


def counter_summary(rows, system):
    inter = sum(int(r[system+'_inter']) for r in rows)
    pred = sum(int(r[system+'_pred_pixels']) for r in rows)
    gt = sum(int(r['gt_pixels']) for r in rows)
    return dict(miou_percent=100*inter/max(1, pred+gt-inter),
                niou_percent=100*np.mean([float(r[system+'_iou']) for r in rows]),
                pd_percent=100*sum(int(r[system+'_detected']) for r in rows)/sum(int(r['target_count']) for r in rows),
                fa_per_million=1e6*sum(int(r[system+'_false_pixels']) for r in rows)/sum(int(r['image_pixels']) for r in rows),
                f1_percent=200*inter/max(1, pred+gt))


def main():
    assert json.loads((OUT/'status.json').read_text())['stage'] == 'wave_complete'
    protocol = json.loads((OUT/'metadata/protocol.json').read_text())
    assert not protocol['official_test_access'] and not protocol['joint_inference']
    comparisons = []; fold_comparisons = []; final_results = []
    for head in ('obj', 'mask'):
        baseline = json.loads((PARENT/head/'A0/summary.json').read_text(encoding='utf-8'))
        parent_name = protocol['amplitude_parent'][head]
        for name in ('B1', 'B2', 'B3'):
            folder = OUT/head/name
            d = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            columns = component_indices(name); ids = []
            for fold in range(5):
                ck = load(folder/'artifacts'/f'fold{fold}.pt')
                parent = load(PARENT/head/parent_name/'artifacts'/f'fold{fold}.pt')
                assert not set(ck['sample_ids']) & set(ck['validation_ids'])
                assert ck['sample_ids'] == parent['sample_ids'] and ck['validation_ids'] == parent['validation_ids']
                ids.extend(ck['validation_ids'])
                assert ck['component_indices'] == columns and ck['amplitude'] == parent_name
                for key in ('radial_mean', 'radial_std'):
                    np.testing.assert_array_equal(np.asarray(ck[key]), np.asarray(parent[key])[columns])
                if head == 'mask':
                    assert ck['sampling_sha256'] == parent['sampling_sha256']
                else:
                    assert ck['schedule_sha256'] == parent['schedule_sha256']
                new = json.loads((folder/'metrics'/f'fold{fold}.json').read_text(encoding='utf-8'))
                old = json.loads((PARENT/head/'A0/metrics'/f'fold{fold}.json').read_text(encoding='utf-8'))
                if head == 'mask':
                    delta = new['task_metrics']['positive_ROI_global_IoU']-old['task_metrics']['positive_ROI_global_IoU']
                    image_delta = new['image_summary'][0]['miou_percent']-old['image_summary'][0]['miou_percent']
                else:
                    delta = 100*(new['point_metrics']['fixed09']['recall']-old['point_metrics']['fixed09']['recall'])
                    image_delta = None
                fold_comparisons.append(dict(head=head, variant=name, fold=fold, independent_primary_delta=delta,
                                             top5_mIoU_delta=image_delta, Fisher_J=new['diagnostics']['Fisher_J'],
                                             margin_gain=new['diagnostics']['margin_gain']))
            assert len(ids) == len(set(ids)) == 45
            t, z, base = d['task_metrics'], d['diagnostics_fold_mean'], baseline['task_metrics']
            if head == 'obj':
                rows = read_csv(folder/'metrics/OOF_predictions.csv')
                assert len(rows) == 225 and len({r['sample_id'] for r in rows}) == 45
                for metric in (t['fixed09'], t['loose']):
                    assert abs(metric['precision']-metric['tp']/max(1, metric['tp']+metric['fp'])) < 1e-12
                    assert abs(metric['recall']-metric['tp']/70) < 1e-12
                # Independently reconstruct the paired, image-balanced margin from saved logits.
                margins = []
                for fold in range(5):
                    val_ids = set(load(folder/'artifacts'/f'fold{fold}.pt')['validation_ids'])
                    entries = [r for r in rows if r['sample_id'] in val_ids]
                    means = {}
                    for label in (0, 1):
                        images = {r['sample_id'] for r in entries if bool(int(r['assignment'])) == bool(label)}
                        means[label] = np.mean([np.mean([float(r['logit'])-float(r['zero_logit']) for r in entries
                                                        if r['sample_id'] == sid and bool(int(r['assignment'])) == bool(label)]) for sid in images])
                    margins.append(means[1]-means[0])
                assert abs(np.mean(margins)-z['margin_gain']) < 1e-9
                d['Chinese_conclusion'] = [f'相对无径向A0，背景≤0.25/图时召回变化{100*(t["recall_at_bg_0.25"]-base["recall_at_bg_0.25"]):+.2f}个百分点；固定0.9 F1变化{100*(t["fixed09"]["f1"]-base["fixed09"]["f1"]):+.2f}个百分点。',
                                           f'J={z["Fisher_J"]:.4f}、间隔增益={z["margin_gain"]:.4f}，但同预算召回未超过完整径向父A1，不能以J提高证明检出更好。',
                                           '本组不替换A1；保留成分负对照，后续优先核查弱折及尺度/对向关系，201图与联合推理未运行。']
                comparisons.append(dict(head=head, variant=name, Fisher_J=z['Fisher_J'], margin_gain=z['margin_gain'],
                                        primary=t['recall_at_bg_0.25']*100, secondary=t['recall_at_bg_0.5']*100,
                                        primary_delta_vs_A0=100*(t['recall_at_bg_0.25']-base['recall_at_bg_0.25'])))
            else:
                rois = read_csv(folder/'metrics/OOF_ROIs.csv'); images = read_csv(folder/'metrics/OOF_images.csv')
                assert len(images) == 45 and sum(int(r['target_count']) for r in images) == 70
                assert sum(int(r['candidate']) for r in rois) == 225
                pos = [r for r in rois if int(r['positive'])]; neg = [r for r in rois if not int(r['positive'])]
                assert abs(100*sum(int(r['inter']) for r in pos)/sum(int(r['union']) for r in pos)-t['positive_ROI_global_IoU']) < 1e-9
                assert abs(100*np.mean([int(r['pred_pixels']) == 0 for r in neg])-t['negative_empty_percent']) < 1e-9
                for kind in t['image_summary']:
                    actual = counter_summary(images, kind['system'])
                    for key, value in actual.items():
                        assert abs(value-kind[key]) < 1e-9
                for label, group in [('pure_background', [r for r in neg if not int(r['other_target_pixels'])]),
                                     ('other_target', [r for r in neg if int(r['other_target_pixels'])])]:
                    t[label+'_negative_empty_percent'] = 100*float(np.mean([int(r['pred_pixels']) == 0 for r in group])) if group else None
                d['Chinese_conclusion'] = [f'相对A0，正ROI累计IoU变化{t["positive_ROI_global_IoU"]-base["positive_ROI_global_IoU"]:+.2f}个百分点，负ROI全零率变化{t["negative_empty_percent"]-base["negative_empty_percent"]:+.2f}个百分点。',
                                           f'固定Top5全图mIoU变化{t["image_summary"][0]["miou_percent"]-base["image_summary"][0]["miou_percent"]:+.2f}个百分点；J={z["Fisher_J"]:.4f}、间隔增益={z["margin_gain"]:.4f}，折间稳定性另列。',
                                           'B1保留为多种子复验候选，B2/B3不升级；尚未进行201张最终测试或联合推理。']
                comparisons.append(dict(head=head, variant=name, Fisher_J=z['Fisher_J'], margin_gain=z['margin_gain'],
                                        primary=t['positive_ROI_global_IoU'], secondary=t['negative_empty_percent'],
                                        primary_delta_vs_A0=t['positive_ROI_global_IoU']-base['positive_ROI_global_IoU']))
            write_json(folder/'summary.json', d); final_results.append(d)
    # Paired bootstrap is conditional on the fitted single-seed fold models, not a seed-retraining CI.
    old = read_csv(PARENT/'mask/A0/metrics/OOF_images.csv'); new = read_csv(OUT/'mask/B1/metrics/OOF_images.csv')
    old_lookup = {r['sample_id']: r for r in old}; new_lookup = {r['sample_id']: r for r in new}
    ids = sorted(old_lookup); assert ids == sorted(new_lookup)
    def arr(lookup):
        return np.array([[float(lookup[s]['top5_no_obj_inter']), float(lookup[s]['top5_no_obj_pred_pixels']),
                          float(lookup[s]['gt_pixels'])] for s in ids])
    a, b = arr(old_lookup), arr(new_lookup)
    def iou(x):
        inter, pred, gt = x.sum(0)
        return 100*inter/max(1, pred+gt-inter)
    rng = np.random.default_rng(20261029); differences = []
    for _ in range(5000):
        indices = rng.integers(0, len(ids), len(ids))
        differences.append(iou(b[indices])-iou(a[indices]))
    fold4_ids = set(load(OUT/'mask/B1/artifacts/fold4.pt')['validation_ids'])
    selected = [i for i, sid in enumerate(ids) if sid not in fold4_ids]
    stability = dict(B1_vs_A0_top5_mIoU_delta=iou(b)-iou(a),
                     paired_image_bootstrap_95percent_interval=np.percentile(differences, [2.5, 97.5]).tolist(),
                     bootstrap_repetitions=5000, includes_retraining_seed_variance=False,
                     excluding_fold4_top5_mIoU_delta=iou(b[selected])-iou(a[selected]),
                     positive_ROI_IoU_improved_folds=sum(r['independent_primary_delta'] > 0 for r in fold_comparisons if r['head']=='mask' and r['variant']=='B1'),
                     top5_mIoU_improved_folds=sum(r['top5_mIoU_delta'] > 0 for r in fold_comparisons if r['head']=='mask' and r['variant']=='B1'))
    write_csv(OUT/'content_comparison.csv', comparisons); write_csv(OUT/'fold_comparison.csv', fold_comparisons)
    write_json(OUT/'stability.json', stability)
    wave = json.loads((OUT/'summary.json').read_text(encoding='utf-8')); wave['results'] = final_results; wave['stability'] = stability
    write_json(OUT/'summary.json', wave)
    validation = json.loads((OUT/'validation.json').read_text())
    validation.update(all_fold_statistics_equal_parent_channel_slices=True, same_sampling_as_parent=True,
                      independent_counter_metrics_verified=True, Obj_margin_independently_recomputed=True,
                      unique_45_OOF_images_225_Top5_candidates_verified=True)
    write_json(OUT/'validation.json', validation)
    print(json.dumps(dict(stability=stability, completed_models=30, next_direction=dict(Obj='keep_A1_geometry_and_weak_fold_analysis',
                                                                                    Mask='B1_paired_multiseed_confirmation_before_more_complexity')),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
