"""Prototype memory built from the SAME support/query backbone forward.

No reference encoder, no radial branch, no persistent train-time feature cache.
Support keys are detached and rebuilt per episode in current adapter coordinates.
"""
import types

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.core.semantic_direction import (
    SemanticDirectionConfig, edge_mask_from_image, bright_flat_mask_from_target_mean,
    pool_feature_token, mean_token, build_score_token,
)
from visual_prompt.models.semantic_memory import RecallMemoryReader


def support_regions(image, mask, grid=64):
    """Image/GT geometry only; precomputed on allocated train images, on CPU."""
    config = SemanticDirectionConfig()
    def coverage(binary, expand=False):
        binary = binary.astype(np.float32)
        if expand:
            radius = int(np.ceil(max(binary.shape)/grid))
            binary = cv2.dilate(binary, np.ones((2*radius+1, 2*radius+1), np.uint8))
        return torch.from_numpy(cv2.resize(binary, (grid, grid), interpolation=cv2.INTER_AREA) > 0)

    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    instances = []
    for index in range(1, count):
        selected = coverage(labels == index, expand=True)
        if selected.any():
            instances.append(selected)
    if not instances:
        raise ValueError('Support image must contain a target')
    edge = edge_mask_from_image(image, config.edge_method, config.edge_quantile)
    excluded = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    bright = bright_flat_mask_from_target_mean(image, mask) & ~mask.astype(bool)
    negatives = []
    for binary in (edge & ~excluded, bright):
        selected = coverage(binary)
        negatives.append(selected)
    return dict(instances=instances, negatives=negatives)


def descriptor(model, backbone, index):
    """Reuse SAM2's H0/H1/F features and align them on the main feature grid."""
    current = backbone.get_current_feats(index)
    high = backbone.get_high_res_features(current)
    size = backbone.feat_sizes[-1]
    main = current[-1].permute(1, 2, 0).reshape(1, 256, *size)
    main = main + model.sam.no_mem_embed.reshape(1, 256, 1, 1)
    layers = [F.interpolate(x.float(), size, mode='bilinear', align_corners=False) for x in high]
    return F.normalize(torch.cat([*layers, main.float()], 1), dim=1)


@torch.no_grad()
def build_episode_keys(model, backbone, indices, regions):
    positives, image_tokens = [], []
    negatives = {'high_freq': [], 'bright_flat': []}
    for index, geometry in zip(indices, regions):
        feature = descriptor(model, backbone, index)[0].permute(1, 2, 0)
        local = [pool_feature_token(feature, region.to(feature.device), 'mean_max', True)
                 for region in geometry['instances']]
        positives.extend(local)
        image_tokens.append(mean_token(local))
        for name, region in zip(negatives, geometry['negatives']):
            token = pool_feature_token(feature, region.to(feature.device), 'mean', True)
            if token is not None:
                negatives[name].append(token)
    config = SemanticDirectionConfig()
    corrected, _ = build_score_token(image_tokens, negatives, config)
    positive_keys = [build_score_token([p], negatives, config)[0] for p in positives[:8]]
    negative_keys = [p for values in negatives.values() for p in values][:9]
    # Fixed semantic slots: raw/corrected mean, up to8 instances, up to9 negatives.
    # Missing slots are zero, not duplicated prototypes or invented examples.
    keys = corrected.new_zeros(19, 352)
    keys[0] = mean_token(image_tokens)[0]
    keys[1] = corrected[0]
    for slot, token in enumerate(positive_keys, 2):
        keys[slot] = token[0]
    for slot, token in enumerate(negative_keys, 10):
        keys[slot] = token[0]
    return keys.detach()


class SharedMemoryGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.reader = RecallMemoryReader(19)
        self.norm = nn.GroupNorm(1, 256)
        self.reduce = nn.Conv2d(256, 32, 1)
        self.output = nn.Conv2d(32, 256, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, main, query_descriptor, keys):
        maps = torch.einsum('kc,bchw->bkhw', keys.float(), query_descriptor.float())
        batch, channels, height, width = maps.shape
        score = self.reader(maps.permute(0, 2, 3, 1).reshape(-1, channels))
        gate = score.reshape(batch, 1, height, width).sigmoid()
        delta = self.output(F.gelu(self.reduce(self.norm(main))))
        return main + gate * delta


def install_shared_memory(model):
    model.add_module('shared_memory', SharedMemoryGate().to(model.device))

    def decode(self, backbone, idx, memory_idx, memory_bank):
        current = backbone.get_current_feats(idx)
        position = backbone.get_current_pos_embeds(idx)
        high = backbone.get_high_res_features(current)
        main = self.sam._prepare_memory_conditioned_features(
            frame_idx=memory_idx, current_vision_feats=current[-1:],
            current_vision_pos_embeds=position[-1:], feat_sizes=backbone.feat_sizes[-1:],
            num_frames=memory_idx+1, memory_bank=memory_bank)
        start = idx-memory_idx
        episode = idx // (memory_idx+1)
        keys = build_episode_keys(self, backbone, range(start, idx), self.support_geometry[episode])
        main = self.shared_memory(main, descriptor(self, backbone, idx), keys)
        return self.sam._forward_sam_heads(backbone_features=main, high_res_features=high,
                                          multimask_output=memory_idx > 0)

    model._compute_decoder_out_w_mem = types.MethodType(decode, model)
    return model
