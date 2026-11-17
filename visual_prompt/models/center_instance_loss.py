"""Center-instance supervision with optional exclusion of other GT instances."""
import torch
from torch.nn import functional as F


def center_instance_loss(logits, target, valid, positive):
    """Positive soft IoU plus negative pixel BCE, normalized over valid pixels.

    `target` contains only the instance under the candidate point. `valid`
    excludes other real instances when ignore supervision is requested.
    """
    valid = valid.to(logits.dtype)
    target = target.to(logits.dtype)
    probability = logits.sigmoid() * valid
    intersection = (probability * target).sum((-2, -1))
    union = probability.sum((-2, -1)) + (target * valid).sum((-2, -1)) - intersection
    per_positive = 1 - (intersection + 1e-6) / (union + 1e-6)
    per_negative = (F.softplus(logits) * valid).sum((-2, -1)) / valid.sum((-2, -1)).clamp_min(1)
    zero = logits.sum() * 0
    return ((per_positive[positive].mean() if positive.any() else zero)
            + (per_negative[~positive].mean() if (~positive).any() else zero))
