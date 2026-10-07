"""Sparse-target losses. Proposed weights require validation, not test tuning."""
import torch
from torch import nn
from torch.nn import functional as F


LOSS_DEFAULTS = dict(loss_mode='bce_dice', dice_weight=1.0,
                     background_weight=0.25, hard_weight=0.5, hard_negative_ratio=5.0, hard_negative_min=64,
                     hard_fn_fraction=0.25,
                     boundary_background_weight=1.0)


class SegmentationCriterion(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        cfg = {**LOSS_DEFAULTS, **kwargs}
        if set(cfg) != set(LOSS_DEFAULTS):
            raise ValueError('Unknown loss setting')
        self.cfg = cfg
        if cfg['loss_mode'] not in ('bce_dice', 'balanced_hard'):
            raise ValueError('Unknown loss mode')
        if not 0 <= cfg['boundary_background_weight'] <= 1:
            raise ValueError('Boundary weight must be in [0,1]')
        for key in ('background_weight', 'hard_weight', 'hard_negative_ratio',
                    'hard_fn_fraction', 'dice_weight'):
            if cfg[key] < 0:
                raise ValueError(f'{key} must be nonnegative')
        if cfg['hard_negative_min'] < 1 or cfg['hard_fn_fraction'] <= 0:
            raise ValueError('Invalid hard-negative settings')

    def forward(self, logits, masks):
        # Explicit FP32 even inside autocast; log-sigmoid avoids log(0).
        logits, y = logits.float(), masks.float()
        if logits.shape != y.shape or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError('Loss expects equally shaped Bx1xHxW tensors')
        p = logits.sigmoid()
        bce_map = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
        dims = (1, 2, 3)
        dice = (1 - (2 * (p*y).sum(dims) + 1) / (p.sum(dims) + y.sum(dims) + 1)).mean()
        parts = {'bce': bce_map.mean(), 'dice': dice}
        c = self.cfg
        if c['loss_mode'] == 'bce_dice':
            return parts['bce'] + dice, parts

        # Only soften background immediately outside GT. Never erode tiny GT.
        ring = (F.max_pool2d(y, 3, stride=1, padding=1) > 0) & (y < 0.5)
        reliability = torch.where(ring, c['boundary_background_weight'], 1.0)
        foreground, background, hard_fp, hard_fn = [], [], [], []
        for i in range(len(y)):
            pos = y[i] > 0.5
            neg = ~pos
            zero = logits[i].sum() * 0
            npos, nneg = int(pos.sum()), int(neg.sum())
            foreground.append(bce_map[i][pos].mean() if npos else zero)
            bg_losses = (bce_map[i] * reliability[i])[neg]
            background.append(bg_losses.mean() if nneg else zero)
            if nneg:
                k = min(nneg, max(c['hard_negative_min'], int(c['hard_negative_ratio'] * npos)))
                # Selection is detached; selected loss values remain differentiable.
                scores = p[i].detach()[neg]
                chosen = scores.topk(k).indices
                hard_fp.append(bg_losses[chosen].mean())
            else:
                hard_fp.append(zero)
            if npos:
                k_fn = min(npos, max(1, int(c['hard_fn_fraction'] * npos + 0.999999)))
                fn_scores = (1 - p[i].detach())[pos]
                fn_losses = bce_map[i][pos]
                hard_fn.append(fn_losses[fn_scores.topk(k_fn).indices].mean())
            else:
                hard_fn.append(zero)
        parts.update(foreground=torch.stack(foreground).mean(),
                     background=torch.stack(background).mean(),
                     hard_fp=torch.stack(hard_fp).mean(),
                     hard_fn=torch.stack(hard_fn).mean(),
                     hard=0.5 * (torch.stack(hard_fp).mean() + torch.stack(hard_fn).mean()))
        loss = (parts['foreground'] + c['background_weight']*parts['background']
                + c['hard_weight']*parts['hard']
                + c['dice_weight']*dice)
        return loss, parts
