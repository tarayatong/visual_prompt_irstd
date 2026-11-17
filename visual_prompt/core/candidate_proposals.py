"""Candidate-point and instance metrics shared by the few-shot studies."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class InstanceMap:
    labels: np.ndarray
    count: int
    areas: tuple[int, ...]
    centroids_yx: tuple[tuple[float, float], ...]


def connected_instances(mask: np.ndarray) -> InstanceMap:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    return InstanceMap(
        labels=labels.astype(np.int32),
        count=max(0, count - 1),
        areas=tuple(int(v) for v in stats[1:, cv2.CC_STAT_AREA]),
        centroids_yx=tuple((float(y), float(x)) for x, y in centroids[1:]),
    )


def greedy_score_peaks(score: np.ndarray, k: int, radius: int = 0) -> list[tuple[int, int, float]]:
    """Return deterministic score peaks with circular greedy suppression."""
    if k <= 0:
        return []
    flat = score.reshape(-1)
    if radius <= 0:
        order = np.argsort(-flat, kind="stable")[:k]
        ys, xs = np.unravel_index(order, score.shape)
        return [(int(y), int(x), float(score[y, x])) for y, x in zip(ys, xs)]

    work = score.astype(np.float32, copy=True)
    points = []
    for _ in range(k):
        index = int(np.argmax(work.reshape(-1)))
        value = float(work.reshape(-1)[index])
        if not np.isfinite(value):
            break
        y, x = np.unravel_index(index, work.shape)
        points.append((int(y), int(x), float(score[y, x])))
        cv2.circle(work, (int(x), int(y)), int(radius), float("-inf"), thickness=-1)
    return points


def point_instance_id(labels: np.ndarray, y: int, x: int, tolerance: int = 0) -> int:
    direct = int(labels[y, x])
    if direct > 0 or tolerance <= 0:
        return direct
    height, width = labels.shape
    y0, y1 = max(0, y - tolerance), min(height, y + tolerance + 1)
    x0, x1 = max(0, x - tolerance), min(width, x + tolerance + 1)
    local = labels[y0:y1, x0:x1]
    ys, xs = np.nonzero(local)
    if not len(xs):
        return 0
    distances = (ys + y0 - y) ** 2 + (xs + x0 - x) ** 2
    best = int(np.argmin(distances))
    if float(distances[best]) > float(tolerance * tolerance):
        return 0
    return int(local[ys[best], xs[best]])


def scored_point_nms(
    points: list[tuple[int, int, float]], radius: float,
) -> list[int]:
    """Return input indices kept by score-ordered, circular point NMS, without GT.

    Points contain (y, x, score) in original image pixels. Suppress distances <=
    radius; break score ties by coordinates and then original input order.
    """
    if not np.isfinite(radius) or radius < 0:
        raise ValueError('Point NMS radius must be finite and nonnegative')
    if any(not np.isfinite(point).all() for point in points):
        raise ValueError('Point NMS coordinates and scores must be finite')
    order = sorted(range(len(points)), key=lambda i: (-points[i][2], points[i][0], points[i][1], i))
    if radius == 0:
        return order
    kept = []
    for index in order:
        y, x, _ = points[index]
        if all((y-points[j][0])**2 + (x-points[j][1])**2 > radius**2 for j in kept):
            kept.append(index)
    return kept


def candidate_counts(
    instances: InstanceMap,
    points: list[tuple[int, int, float]],
    tolerance: int = 0,
) -> dict:
    assignments = [point_instance_id(instances.labels, y, x, tolerance) for y, x, _ in points]
    counts = np.bincount(assignments, minlength=instances.count + 1)
    recalled = int(np.count_nonzero(counts[1:]))
    duplicates = int(np.maximum(counts[1:] - 1, 0).sum())
    return {
        "recalled_targets": recalled,
        "target_count": instances.count,
        "false_candidates": int(counts[0]),
        "duplicate_candidates": duplicates,
        "all_targets_recalled": int(instances.count > 0 and recalled == instances.count),
        "assignments": assignments,
    }


def snapped_instance_centers(instances: InstanceMap) -> list[tuple[int, int, int]]:
    result = []
    for component_id, (cy, cx) in enumerate(instances.centroids_yx, start=1):
        y, x = int(round(cy)), int(round(cx))
        if instances.labels[y, x] != component_id:
            ys, xs = np.nonzero(instances.labels == component_id)
            distances = (ys - cy) ** 2 + (xs - cx) ** 2
            best = int(np.argmin(distances))
            y, x = int(ys[best]), int(xs[best])
        result.append((component_id, y, x))
    return result


def greedy_instance_matches(pred: np.ndarray, gt: np.ndarray, iou_threshold: float) -> dict:
    pred_instances = connected_instances(pred)
    gt_instances = connected_instances(gt)
    pairs = []
    for gt_id in range(1, gt_instances.count + 1):
        gt_mask = gt_instances.labels == gt_id
        for pred_id in range(1, pred_instances.count + 1):
            pred_mask = pred_instances.labels == pred_id
            intersection = int(np.logical_and(gt_mask, pred_mask).sum())
            if intersection <= 0:
                continue
            union = int(np.logical_or(gt_mask, pred_mask).sum())
            pairs.append((intersection / union, gt_id, pred_id))
    pairs.sort(reverse=True)
    used_gt, used_pred = set(), set()
    for iou, gt_id, pred_id in pairs:
        if iou < iou_threshold or gt_id in used_gt or pred_id in used_pred:
            continue
        used_gt.add(gt_id)
        used_pred.add(pred_id)
    tp = len(used_gt)
    return {
        "tp": tp,
        "fp": pred_instances.count - tp,
        "fn": gt_instances.count - tp,
        "pred_instances": pred_instances.count,
        "gt_instances": gt_instances.count,
    }


def centroid_detection_counts(pred: np.ndarray, gt: np.ndarray, distance_threshold: float = 3.0) -> dict:
    pred_instances = connected_instances(pred)
    gt_instances = connected_instances(gt)
    pairs = []
    for gt_id, gt_center in enumerate(gt_instances.centroids_yx, start=1):
        for pred_id, pred_center in enumerate(pred_instances.centroids_yx, start=1):
            distance = float(np.linalg.norm(np.asarray(gt_center) - np.asarray(pred_center)))
            if distance < distance_threshold:
                pairs.append((distance, gt_id, pred_id))
    pairs.sort()
    used_gt, used_pred = set(), set()
    for _, gt_id, pred_id in pairs:
        if gt_id in used_gt or pred_id in used_pred:
            continue
        used_gt.add(gt_id)
        used_pred.add(pred_id)
    unmatched_pixels = sum(
        area for pred_id, area in enumerate(pred_instances.areas, start=1) if pred_id not in used_pred
    )
    # Fa is a full-image pixel false-positive rate, independent of instance matching.
    false_pixels = int(np.count_nonzero((pred > 0) & ~(gt > 0)))
    return {
        "detected_targets": len(used_gt),
        "target_count": gt_instances.count,
        "false_alarm_pixels": int(false_pixels),
        "unmatched_prediction_pixels": int(unmatched_pixels),
    }
