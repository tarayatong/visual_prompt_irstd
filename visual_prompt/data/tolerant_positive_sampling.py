"""Candidate tolerance without changing segmentation ground truth.

Distances are Euclidean distances to the nearest ORIGINAL foreground pixel in
working-image coordinates. Ties use row-major foreground order, matching
core.candidate_proposals.point_instance_id. A background point in the tolerance
band belongs to its nearest instance; it is never also a negative candidate.
"""
import numpy as np


def positive_regions(labels, radius=3):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not labels.size or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Expected nonempty integer instance-label image')
    if np.any(labels < 0):
        raise ValueError('Instance labels must be nonnegative')
    if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError('Radius must be a nonnegative integer')
    h, w = labels.shape
    owner = labels.astype(np.int32, copy=True)
    best = np.full((h, w), radius*radius+1, np.int32)
    best[labels > 0] = 0
    padded = np.pad(labels, radius)
    # Ascending dy,dx makes equal-distance ties agree with the scalar matcher.
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            distance = dy*dy+dx*dx
            if distance > radius*radius:
                continue
            source = padded[radius+dy:radius+dy+h, radius+dx:radius+dx+w]
            update = (source > 0) & (distance < best)
            owner[update] = source[update]
            best[update] = distance
    best[owner == 0] = -1
    return owner, best


def _crop(array, y, x, size):
    result = np.zeros((size, size), dtype=array.dtype)
    top, left = y-size//2, x-size//2
    y0, x0 = max(0, top), max(0, left)
    y1, x1 = min(array.shape[0], top+size), min(array.shape[1], left+size)
    result[y0-top:y1-top, x0-left:x1-left] = array[y0:y1, x0:x1]
    return result


def mask_supervision(labels, owner, y, x, crop_size=64):
    """Assign an offset candidate to original GT; ignore other GT instances."""
    labels, owner = np.asarray(labels), np.asarray(owner)
    if owner.shape != labels.shape or labels.ndim != 2:
        raise ValueError('Owner and original labels must have the same 2D shape')
    if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
        raise ValueError('Candidate must be inside the working image')
    if crop_size <= 0:
        raise ValueError('Positive crop size required')
    iid = int(owner[y, x])
    original_target = (labels == iid).astype(np.uint8) if iid else np.zeros_like(labels, dtype=np.uint8)
    target = _crop(original_target, y, x, crop_size)
    all_gt = _crop((labels > 0).astype(np.uint8), y, x, crop_size)
    if iid and not target.any():
        raise ValueError('Positive candidate crop contains none of its assigned original target')
    return dict(instance_id=iid, positive=bool(iid), target=target,
                valid=(all_gt == 0) | (target > 0))
