"""Memory-only residuals for the existing SAM2 Tiny/SANSA decoder path."""
import types

import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.semantic_memory import RecallMemoryReader


class TinyMemoryResidual(nn.Module):
    """Keep original memory attention intact; isolate three memory readouts.

    Keys and similarity maps use the same frozen reference coordinates as the
    previous radial+memory experiment. Semantic tokens have no spatial RoPE.
    Negative values are read separately, never pooled together with positives.
    """

    def __init__(self, mode, keys, positive_count):
        super().__init__()
        if mode not in ('h0', 'c', 'b'):
            raise ValueError(mode)
        self.mode = mode
        self.maps = None
        self.register_buffer('keys', keys.detach().float().clone())
        self.positive_end = 2 + positive_count
        assert 0 < self.positive_end < len(keys)
        self.reader = RecallMemoryReader(len(keys))
        if mode == 'h0':
            self.output = nn.Conv2d(1, 32, 1, bias=False)
        elif mode == 'c':
            self.norm = nn.GroupNorm(1, 256)
            self.reduce = nn.Conv2d(256, 32, 1)
            self.output = nn.Conv2d(32, 256, 1, bias=False)
        else:
            self.query = nn.Linear(256, 64, bias=False)
            self.key = nn.Linear(keys.shape[1], 64, bias=False)
            self.value = nn.Linear(keys.shape[1], 64, bias=False)
            self.output = nn.Conv2d(128, 256, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, main, high_res):
        if self.maps is None:
            raise RuntimeError('Prepare query similarity maps before decoding')
        maps = self.maps.to(device=main.device, dtype=main.dtype)
        batch, count, height, width = maps.shape
        score = self.reader(maps.permute(0, 2, 3, 1).reshape(-1, count))
        score = score.reshape(batch, 1, height, width)
        if self.mode == 'h0':
            score = F.interpolate(score, high_res[0].shape[-2:], mode='bilinear', align_corners=False)
            high_res[0] = high_res[0] + self.output(score)
            return main, high_res
        gate = F.interpolate(score, main.shape[-2:], mode='bilinear', align_corners=False).sigmoid()
        if self.mode == 'c':
            values = F.gelu(self.reduce(self.norm(main)))
        else:
            queries = main.flatten(2).transpose(1, 2)
            q = F.normalize(self.query(queries), dim=-1)
            k = F.normalize(self.key(self.keys), dim=-1)
            v = self.value(self.keys)
            logits = q @ k.T / 0.2
            cut = self.positive_end
            positive = logits[..., :cut].softmax(-1) @ v[:cut]
            negative = logits[..., cut:].softmax(-1) @ v[cut:]
            values = torch.cat((positive, negative), -1).transpose(1, 2)
            values = values.reshape(batch, 128, *main.shape[-2:])
        return main + gate * self.output(values), high_res


def install_memory_residual(model, fusion):
    """Patch only query decoding; supports/native spatial memory are unchanged."""
    if hasattr(model, 'memory_fusion'):
        raise ValueError('Memory residual already installed')
    model.add_module('memory_fusion', fusion)

    def decode(self, backbone_out, idx, memory_idx, memory_bank):
        current = backbone_out.get_current_feats(idx)
        positions = backbone_out.get_current_pos_embeds(idx)
        high_res = backbone_out.get_high_res_features(current)
        main = self.sam._prepare_memory_conditioned_features(
            frame_idx=memory_idx, current_vision_feats=current[-1:],
            current_vision_pos_embeds=positions[-1:],
            feat_sizes=backbone_out.feat_sizes[-1:], num_frames=memory_idx+1,
            memory_bank=memory_bank)
        main, high_res = self.memory_fusion(main, high_res)
        return self.sam._forward_sam_heads(
            backbone_features=main, high_res_features=high_res,
            multimask_output=memory_idx > 0)

    model._compute_decoder_out_w_mem = types.MethodType(decode, model)
    return model
