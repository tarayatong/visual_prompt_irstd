"""Strict original-resolution metrics shared by all comparison methods."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

PROTOCOL_VERSION = "comparison_v2_centroid_lt3_full_image_pixel_fp_fa"


def refresh_full_image_fa(result):
    """Recompute cached results from sufficient statistics before reporting."""
    value = dict(result)
    if 'Fa_per_million' not in value:
        return value
    if 'fp' not in value or 'pixels' not in value or value['pixels'] <= 0:
        raise ValueError('Cannot restate Fa without full-image fp/pixels; re-evaluate saved predictions')
    value['Fa_per_million'] = 1e6*value['fp']/value['pixels']
    value['pixel_FP_per_million'] = value['Fa_per_million']
    value['protocol_version'] = PROTOCOL_VERSION
    return value


def image_metrics(prediction, target, sample_id="", threshold=0.5):
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError(f"Shape mismatch for {sample_id}: {prediction.shape} / {target.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite prediction: {sample_id}")
    pred, gt = prediction >= threshold, target > 0
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    pn, pl, ps, pc = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    gn, gl, gs, gc = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
    npred, ngt = pn - 1, gn - 1
    matched = set()
    if npred and ngt:
        distances = np.linalg.norm(gc[1:, None] - pc[None, 1:], axis=-1)
        valid = distances < 3.0
        # A bonus larger than all possible distance costs prioritizes cardinality.
        bonus = 3.0 * (min(npred, ngt) + 1)
        costs = np.where(valid, distances - bonus, 0.0)
        rows, cols = linear_sum_assignment(costs)
        matched = {int(c) + 1 for r, c in zip(rows, cols) if valid[r, c]}
    unmatched_pixels = sum(int(ps[i, cv2.CC_STAT_AREA]) for i in range(1, pn) if i not in matched)
    union = tp + fp + fn
    return dict(sample_id=sample_id, tp=tp, fp=fp, fn=fn,
                image_iou=tp / union if union else 1.0,
                pixels=int(gt.size), gt_pixels=int(gt.sum()),
                targets=ngt, predictions=npred, detected=len(matched),
                unmatched_pixels=unmatched_pixels)


def summarize(rows):
    if not rows:
        raise ValueError("Cannot evaluate an empty test set")
    totals = {k: sum(r[k] for r in rows) for k in
              ("tp", "fp", "fn", "pixels", "targets", "predictions", "detected", "unmatched_pixels")}
    tp, fp, fn = (totals[k] for k in ("tp", "fp", "fn"))
    return dict(protocol_version=PROTOCOL_VERSION, images=len(rows),
                IoU_percent=100 * tp / (tp + fp + fn) if tp + fp + fn else 100.0,
                nIoU_percent=100 * float(np.mean([r["image_iou"] for r in rows])),
                F1_percent=200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 100.0,
                Pd_percent=100 * totals["detected"] / totals["targets"] if totals["targets"] else None,
                Fa_per_million=1e6 * fp / totals["pixels"],
                pixel_FP_per_million=1e6 * fp / totals["pixels"],
                FP_per_image=(totals["predictions"] - totals["detected"]) / len(rows),
                **totals)
