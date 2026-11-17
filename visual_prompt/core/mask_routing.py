"""Conservative, GT-free duplicate suppression for masks in shared coordinates."""
from __future__ import annotations

from collections.abc import Sequence
import math
import numpy as np


def instance_mask_nms(
    masks: Sequence[np.ndarray], scores: Sequence[float], *,
    iou_threshold: float = .5, containment_threshold: float = .8,
    centroid_radius_scale: float = 1., centroid_floor: float = 2.,
) -> tuple[list[int], list[dict]]:
    """Suppress only overlapping, centroid-consistent predictions of one object.

    All masks must be in the same full-image coordinate system. ROI-box overlap
    and prompt distance alone never establish duplication. Empty masks are
    omitted. Ranking uses supplied scores with deterministic index tie breaking.
    The returned audit records every suppression; it does not assert that GT
    recall is preserved. Thresholds require training/validation assessment.
    """
    if len(masks) != len(scores):
        raise ValueError('Masks and scores must have equal lengths')
    if not 0 < iou_threshold <= 1 or not 0 < containment_threshold <= 1:
        raise ValueError('Overlap thresholds must lie in (0, 1]')
    if not np.isfinite([centroid_radius_scale, centroid_floor]).all() or min(centroid_radius_scale, centroid_floor) < 0:
        raise ValueError('Centroid gate settings must be finite and nonnegative')
    if not np.isfinite(scores).all():
        raise ValueError('Scores must be finite')
    binary = [np.asarray(m) > 0 for m in masks]
    if any(m.ndim != 2 for m in binary) or any(m.shape != binary[0].shape for m in binary):
        raise ValueError('Masks must share one two-dimensional coordinate system')
    areas = [int(m.sum()) for m in binary]
    centers = [np.argwhere(m).mean(0) if a else None for m,a in zip(binary,areas)]
    order = sorted((i for i,a in enumerate(areas) if a), key=lambda i:(-float(scores[i]),i))
    kept, audit = [], []
    for i in order:
        for j in kept:
            intersection = int(np.count_nonzero(binary[i] & binary[j]))
            if not intersection:
                continue
            iou = intersection/(areas[i]+areas[j]-intersection)
            containment = intersection/min(areas[i],areas[j])
            distance = float(np.linalg.norm(centers[i]-centers[j]))
            gate = max(centroid_floor,centroid_radius_scale*math.sqrt(min(areas[i],areas[j])/math.pi))
            if (iou >= iou_threshold or containment >= containment_threshold) and distance <= gate:
                audit.append(dict(suppressed=i,kept=j,IoU=iou,intersection_over_min_area=containment,
                                  centroid_distance=distance,centroid_gate=gate))
                break
        else:
            kept.append(i)
    return kept,audit
