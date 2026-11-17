"""Image-balanced representation and logit diagnostics for held-out samples."""
from collections import defaultdict

import numpy as np


class RadialDiagnostics:
    def __init__(self):
        self.groups = defaultdict(lambda: defaultdict(list))

    def add(self, image_id, group, features, logits, zero_logits):
        features = np.asarray(features, dtype=np.float64)
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        zero_logits = np.asarray(zero_logits, dtype=np.float64).reshape(-1)
        if not len(logits):
            return
        if features.ndim != 2 or len(features) != len(logits) or logits.shape != zero_logits.shape:
            raise ValueError('Diagnostics require aligned features and paired logits')
        if not all(np.isfinite(v).all() for v in (features, logits, zero_logits)):
            raise ValueError('Nonfinite diagnostic input')
        self.groups[group][image_id].append(dict(mean=features.mean(0),
                                                square=float(np.square(features).sum(1).mean()),
                                                logit=float(logits.mean()), zero_logit=float(zero_logits.mean())))

    def moments(self, group):
        images = self.groups.get(group, {})
        if not images:
            return None
        per_image = []
        for entries in images.values():
            per_image.append(dict(mean=np.mean([e['mean'] for e in entries], 0),
                                  square=np.mean([e['square'] for e in entries]),
                                  logit=np.mean([e['logit'] for e in entries]),
                                  zero_logit=np.mean([e['zero_logit'] for e in entries])))
        mean = np.mean([e['mean'] for e in per_image], 0)
        return dict(mean=mean, trace=max(0., float(np.mean([e['square'] for e in per_image])-np.square(mean).sum())),
                    logit=float(np.mean([e['logit'] for e in per_image])),
                    zero_logit=float(np.mean([e['zero_logit'] for e in per_image])), images=len(images))

    def result(self, positive='target', negative='background', radial_absent=False):
        a, b = self.moments(positive), self.moments(negative)
        if a is None or b is None:
            return dict(Fisher_J=None, margin_gain=None, target_logit_change=None, background_logit_change=None,
                        positive_images=0 if a is None else a['images'], negative_images=0 if b is None else b['images'])
        return dict(Fisher_J=None if radial_absent else float(np.square(a['mean']-b['mean']).sum()/(a['trace']+b['trace']+1e-6)),
                    margin_gain=float((a['logit']-b['logit'])-(a['zero_logit']-b['zero_logit'])),
                    target_logit_change=float(a['logit']-a['zero_logit']),
                    background_logit_change=float(b['logit']-b['zero_logit']),
                    positive_images=a['images'], negative_images=b['images'])


def masked_channel_statistics(values, valid):
    """Fit channels from training observations only; invalid samples stay excluded."""
    values, valid = np.asarray(values), np.asarray(valid, dtype=bool)
    if values.shape != valid.shape or values.ndim != 2:
        raise ValueError('Expected observations by channels and matching validity')
    count = valid.sum(0)
    total = np.where(valid, values, 0).sum(0, dtype=np.float64)
    square = np.where(valid, np.square(values.astype(np.float64)), 0).sum(0)
    mean = total/np.maximum(count, 1)
    std = np.sqrt(np.maximum(0., square/np.maximum(count, 1)-mean**2))
    return mean.astype(np.float32), np.maximum(std, .01).astype(np.float32)
