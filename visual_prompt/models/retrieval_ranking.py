"""Instance-pooled versus foreground-pixel retrieval ranking objectives."""
import math
import torch
from torch.nn import functional as F


def retrieval_ranking_loss(model, groups, background, mode, foreground=None):
    negative=model(background).topk(min(32,len(background))).values
    if mode=='instance_mean':
        terms=[]
        for group in groups:
            score=model(group)
            positive=.02*(torch.logsumexp(score/.02,0)-math.log(len(score)))
            terms.append(F.softplus((negative-positive+.03)/.05).mean())
        return torch.stack(terms).mean()
    if mode=='pixel_mean':
        if foreground is None:foreground=torch.cat(groups,0)
        positive=model(foreground)
        return F.softplus((negative[None]-positive[:,None]+.03)/.05).mean()
    raise ValueError(mode)
