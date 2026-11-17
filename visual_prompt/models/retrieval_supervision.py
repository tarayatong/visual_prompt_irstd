"""Probability calibration and explicit supervision for frozen-key retrieval."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SimilarityCalibration(nn.Module):
    """Learn a positive affine logit map, preserving similarity ordering."""
    def __init__(self, center=0., scale=1.):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(1/max(float(scale), 1e-4))))
        self.bias = nn.Parameter(torch.tensor(-float(center)/max(float(scale), 1e-4)))

    def forward(self, score):
        return self.log_scale.exp()*score+self.bias


def balanced_mask_supervision(logits, target):
    target = target.bool()
    assert target.any() and (~target).any()
    bce = .5*(F.softplus(-logits[target]).mean()+F.softplus(logits[~target]).mean())
    probability = logits.sigmoid()
    intersection = probability[target].sum()
    union = probability.sum()+target.sum()-intersection
    soft_iou = 1-(intersection+1e-6)/(union+1e-6)
    return bce+soft_iou, bce, soft_iou


def fused_probability(obj_logits, sim_logits):
    return .5*(obj_logits.sigmoid()+sim_logits.sigmoid())


def fused_classification_loss(obj_logits, sim_logits, target):
    # Stable BCE of the arithmetic mean of the two probabilities.
    log_p = torch.logsumexp(torch.stack((F.logsigmoid(obj_logits), F.logsigmoid(sim_logits))), dim=0)-math.log(2)
    log_n = torch.logsumexp(torch.stack((F.logsigmoid(-obj_logits), F.logsigmoid(-sim_logits))), dim=0)-math.log(2)
    return -(target*log_p+(1-target)*log_n).mean()
