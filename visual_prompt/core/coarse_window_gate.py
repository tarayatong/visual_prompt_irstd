"""Select fixed overlapping fine windows from a coarse full-image score map."""
import math
import numpy as np


WINDOWS = tuple((y, x) for y in range(0, 385, 64) for x in range(0, 385, 64))


def window_maxima(score):
    if score.shape != (512, 512) or not np.isfinite(score).all():
        raise ValueError('Expected a finite 512x512 coarse score')
    return np.asarray([score[y:y+128, x:x+128].max() for y, x in WINDOWS])


def coverage(selected):
    result = np.zeros((512, 512), dtype=bool)
    for index in selected:
        y, x = WINDOWS[int(index)]
        result[y:y+128, x:x+128] = True
    return result


def instance_survival_scores(labels, maxima):
    """Threshold at which each GT instance still intersects a selected window.

    Used ONLY on training labels to calibrate the threshold. At inference the
    gate uses window maxima alone, with no labels and no minimum-window fallback.
    """
    scores = []
    for iid in range(1, int(labels.max())+1):
        touching = [i for i, (y, x) in enumerate(WINDOWS)
                    if np.any(labels[y:y+128, x:x+128] == iid)]
        scores.append(float(np.max(maxima[touching])))
    return scores


def calibrated_threshold(instance_scores, target_fraction):
    if not 0 < target_fraction <= 1 or not len(instance_scores):
        raise ValueError('Need instances and a coverage target in (0,1]')
    required = math.ceil(len(instance_scores) * target_fraction)
    return float(np.sort(instance_scores)[::-1][required-1])
