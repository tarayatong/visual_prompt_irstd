"""Report Obj target P/R/F1 after a common point NMS selected on training OOF."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json

import numpy as np

from scripts.run_global_ablation_201 import write_csv, write_json
from scripts.run_phase_c_objectness import detection_metrics
from visual_prompt.core.candidate_proposals import scored_point_nms

ROOT = Path('outputs/studies/objectness_mask')
OUT = ROOT / 'objectness_dedup_metrics_20261027'
PROTOCOL = OUT / 'protocol.json'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def postprocess_metrics(rows, scores, threshold, radius):
    """Evaluate all images/GT after threshold -> prediction-only point NMS."""
    if len(rows) != len(scores):
        raise ValueError('Rows and scores must have identical length')
    groups = defaultdict(list)
    selected = np.zeros(len(rows), dtype=np.float32)
    decisions = []
    for index, (row, score) in enumerate(zip(rows, scores)):
        if float(score) >= threshold:
            groups[row['sample_id']].append(index)
    before = sum(len(v) for v in groups.values())
    for sid, indices in groups.items():
        points = [(int(rows[i]['y']), int(rows[i]['x']), float(scores[i])) for i in indices]
        local_kept = set(scored_point_nms(points, radius))
        for local, index in enumerate(indices):
            row = rows[index]
            selected[index] = float(local in local_kept)
            decisions.append(dict(sample_id=sid, y=int(row['y']), x=int(row['x']), score=float(scores[index]),
                                  kept=int(local in local_kept), assignment=int(row['assignment'])))
    # Retain rows for images with no selected points, so their targets still count.
    metric = detection_metrics(rows, selected, .5)
    background = sum(row['assignment']==0 for row,keep in zip(rows,selected) if keep)
    kept_count = int(selected.sum())
    metric.update(selected_before=before, selected=kept_count, suppressed=before-kept_count,
                  background=int(background), duplicates=kept_count-metric['tp']-background)
    return metric, decisions


def ensure_protocol():
    if PROTOCOL.exists():
        return json.loads(PROTOCOL.read_text())
    manifest = json.loads((ROOT/'phase_c_objectness/metadata/feature_manifest.json').read_text())
    counts = {r['sample_id']:r['target_count'] for r in manifest['records']}
    rows = [dict(sample_id=r['sample_id'], y=int(r['y']), x=int(r['x']), assignment=int(r['assignment']),
                 target_count=counts[r['sample_id']], score=float(r['score']))
            for r in read_csv(ROOT/'phase_c_objectness_refine/metrics/candidate_predictions.csv') if r['experiment']=='e9r_f4']
    assert len(rows)==225 and len({r['sample_id'] for r in rows})==45
    scores = [r['score'] for r in rows]
    sweep = []
    for radius in (0,8,12,16):
        metric,_ = postprocess_metrics(rows,scores,.9,radius)
        sweep.append(dict(radius_pixels=radius,**metric))
    eligible = [r for r in sweep if r['radius_pixels']>0]
    best = max(eligible,key=lambda r:(r['f1'],r['recall'],-r['radius_pixels']))
    protocol = dict(method='score_ordered_point_NMS',radius_pixels=best['radius_pixels'],
                    score='Obj probability',distance='original image Euclidean distance <= radius',
                    selection='original center baseline pooled five-fold OOF at fixed threshold 0.9',
                    OOF_images=45,OOF_targets=70,radius_candidates_pixels=[8,12,16],
                    selection_key='target F1, then recall, then smaller radius',
                    common_across_models=True,uses_test_for_selection=False,uses_GT_to_suppress=False,
                    mask_dependency=False,scope='Obj metric postprocessing only; historical joint masks unchanged')
    write_csv(OUT/'oof_radius_selection.csv',sweep)
    write_json(PROTOCOL,protocol)
    return protocol


def jitter_reports(directory, protocol=None):
    protocol = protocol or ensure_protocol()
    source = read_csv(directory/'metrics/test_candidate_scores.csv')
    raw_summary = read_csv(directory/'metrics/test_objectness.csv')
    summaries, decisions = [], []
    for setting in raw_summary:
        variant = setting['variant']
        subset = [r for r in source if r['variant']==variant]
        rows = [dict(sample_id=r['sample_id'],y=int(r['y']),x=int(r['x']),assignment=int(r['assignment']),
                     target_count=int(r['target_count'])) for r in subset]
        scores = [float(r['score']) for r in subset]
        threshold = float(setting['threshold'])
        for label,radius in [('raw',0),('point_nms',protocol['radius_pixels'])]:
            metric,detail=postprocess_metrics(rows,scores,threshold,radius)
            summaries.append(dict(variant=variant,threshold_name=setting['threshold_name'],threshold=threshold,
                                  postprocess=label,nms_radius_pixels=radius,**metric))
            if label=='point_nms':
                decisions.extend(dict(variant=variant,threshold_name=setting['threshold_name'],**r) for r in detail)
    write_csv(directory/'metrics/test_objectness_deduplicated.csv',summaries)
    write_csv(directory/'metrics/test_objectness_nms_decisions.csv',decisions)
    write_json(directory/'metadata/objectness_metric_postprocess.json',protocol)
    return summaries


def baseline_reports(protocol):
    counts = {r['sample_id']:int(r['target_count']) for r in read_csv(ROOT/'global_ablation_test201/metrics/image_metrics.csv')}
    source = read_csv(ROOT/'global_ablation_test201/metrics/top5_objectness_scores.csv')
    rows = [dict(sample_id=r['sample_id'],y=int(r['y']),x=int(r['x']),assignment=int(r['assignment_dilate3']),
                 target_count=counts[r['sample_id']]) for r in source]
    scores = [float(r['objectness_score']) for r in source]
    summaries=[];decisions=[]
    for name,threshold in [('fixed09',.9),('oof',float(source[0]['objectness_threshold']))]:
        for label,radius in [('raw',0),('point_nms',protocol['radius_pixels'])]:
            metric,detail=postprocess_metrics(rows,scores,threshold,radius)
            summaries.append(dict(variant='original_center',threshold_name=name,threshold=threshold,
                                  postprocess=label,nms_radius_pixels=radius,**metric))
            if label=='point_nms':decisions.extend(dict(threshold_name=name,**r) for r in detail)
    write_csv(OUT/'original_baseline.csv',summaries)
    write_csv(OUT/'original_baseline_nms_decisions.csv',decisions)
    return summaries




def main():
    protocol=ensure_protocol()
    rows=baseline_reports(protocol)
    rows+=jitter_reports(ROOT/'objectness_positive_jitter_20261027',protocol)
    write_csv(OUT/'all_test_objectness.csv',rows)
    print(json.dumps(dict(protocol=protocol,rows=len(rows)),ensure_ascii=False))


if __name__=='__main__':main()
