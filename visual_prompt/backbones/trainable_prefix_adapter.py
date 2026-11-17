"""Hiera prefixes with an optional partial last stage and an autograd entry point.

The inference API keeps the frozen predictor contract. Training calls ``encode``
directly; it must not pass through no_grad feature helpers.
"""
import torch
from torch import nn
from torch.nn import functional as F
from visual_prompt.backbones.truncated_adapter import TruncatedSimilarityPredictor


class TrainablePrefixPredictor(TruncatedSimilarityPredictor):
    def __init__(self, checkpoint, config, blocks, device='cuda:0', image_size=512):
        # Start with all stages needed by the requested block boundary.
        super().__init__(checkpoint, config, stages=3 if blocks > 3 else 2,
                         mobile=False, device=device, image_size=image_size)
        enc = self.model.image_encoder
        if not 1 <= blocks <= len(enc.blocks):
            raise ValueError(blocks)
        enc.blocks = nn.ModuleList(list(enc.blocks)[:blocks])
        enc.stage_ends = sorted(set([i for i in enc.stage_ends if i < blocks] + [blocks-1]))
        enc.channel_list = [enc.blocks[i].dim_out for i in reversed(enc.stage_ends)]
        self.sim_feature_layers = [f'block{i}' for i in enc.stage_ends]

    def encode(self, images):
        x = self.transform.forward_batch(images).to(self.device)
        return self.model.image_encoder(x)

    def aligned(self, images):
        features = self.encode(images)
        shape = features[0].shape[-2:]
        return torch.cat([f if f.shape[-2:] == shape else
                          F.interpolate(f, shape, mode='bilinear', align_corners=False)
                          for f in features], 1)

    @torch.no_grad()
    def set_image_batch(self, images):
        self.original_size = images[0].shape[:2]
        self.input_size = (self.image_size, self.image_size)
        outputs = self.encode(images)
        self._features = dict(zip(self.sim_feature_layers, outputs, strict=True))
        self.features = outputs[-1]
