"""Replace SAM2's dense support memory with the user's19 prototype slots.

Reuse the SAME support/query encoder and pretrained native memory attention.
No spatial memory encoder, dense tokens, object pointers or extra Reader branch.
"""
import types

import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.tiny_shared_memory import build_episode_keys


class RemovedSpatialMemoryEncoder(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError('Dense spatial memory encoding is disabled in prototype-only run')


class PrototypeMemoryInterface(nn.Module):
    def __init__(self, share_foreground_type=False, *, fixed=False, num_types=3, value_gate=False):
        super().__init__()
        self.fixed = fixed
        self.share_foreground_type = share_foreground_type
        if not fixed:
            self.norm = nn.LayerNorm(352)
            self.projection = nn.Linear(352, 64, bias=False)
            self.output_norm = nn.LayerNorm(64)
            # Raw target / corrected target / negative semantic identities.
            self.types = nn.Embedding(num_types, 64) if num_types else None
            if self.types is not None:
                nn.init.zeros_(self.types.weight)
        self.last_count = None
        self.value_gate = None
        if value_gate:
            assert not fixed and num_types > 0
            self.query_norm = nn.LayerNorm(256)
            self.value_gate = nn.Sequential(nn.Linear(384, 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.zeros_(self.value_gate[-1].weight)
            nn.init.zeros_(self.value_gate[-1].bias)

    def prepare(self, keys):
        """Encode semantic slots or an explicitly tagged sequence of grid tokens."""
        explicit_tags = None
        if isinstance(keys, tuple):
            keys, explicit_tags = keys
        if explicit_tags is None:
            assert keys.shape == (19, 352) or (self.fixed and keys.shape == (19, 64))
        else:
            assert keys.shape[1:] == (352,) and explicit_tags.shape == (keys.shape[0],)
        valid = keys.float().norm(dim=-1) > 0
        vectors = keys[valid]
        if self.fixed:
            if vectors.shape[-1] == 352:
                vectors = F.adaptive_avg_pool1d(vectors.float()[None], 64)[0]
                vectors = F.layer_norm(vectors, (64,)).to(keys.dtype)
            memory = vectors[:, None, :]
            position = torch.zeros_like(memory)
        else:
            memory = self.output_norm(self.projection(self.norm(vectors)))[:, None, :]
            if self.types is None:
                position = torch.zeros_like(memory)
            else:
                if explicit_tags is None:
                    indices = torch.arange(19, device=keys.device)
                    explicit_tags = torch.where(indices == 0, 0, torch.where(indices < 10, 1, 2))
                    if self.share_foreground_type:
                        explicit_tags = explicit_tags.masked_fill(explicit_tags == 1, 0)
                position = self.types(explicit_tags[valid])[:, None, :]
        self.last_count = int(memory.shape[0])
        minimum_count = 1 if explicit_tags is not None else 2
        assert minimum_count <= self.last_count <= keys.shape[0]
        return memory, position

    def decode(self, model, backbone, index, encoded):
        current = backbone.get_current_feats(index)
        position = backbone.get_current_pos_embeds(index)
        high = backbone.get_high_res_features(current)
        memory, memory_position = encoded
        # Reuse the existing non-spatial-token exclusion mechanism: ALL memory
        # tokens are semantic prototypes, so none get grid-based key RoPE.
        attention = model.sam.memory_attention
        arguments = dict(curr=current[-1:], curr_pos=position[-1:], memory=memory,
                         memory_pos=memory_position, num_obj_ptr_tokens=memory.shape[0])
        if self.value_gate is None:
            sequence = attention(**arguments)
        else:
            query = self.query_norm(current[-1].mean(dim=0, keepdim=True))
            gate_input = torch.cat((memory, query.expand(memory.shape[0], -1, -1),
                                    memory_position), dim=-1)
            gains = (2 * self.value_gate(gate_input).sigmoid()).transpose(0, 1)

            def gate_projected_values(module, inputs, projected):
                assert projected.shape[:2] == gains.shape[:2]
                return projected * gains.to(projected.dtype)

            handles = []
            try:
                for layer in attention.layers:
                    handles.append(layer.cross_attn_image.v_proj.register_forward_hook(gate_projected_values))
                sequence = attention(**arguments)
            finally:
                for handle in handles:
                    handle.remove()
        main = sequence.permute(1, 2, 0).reshape(1, 256, *backbone.feat_sizes[-1])
        return model.sam._forward_sam_heads(
            backbone_features=main, high_res_features=high, multimask_output=True)


def install_prototype_memory(
    model, key_builder=build_episode_keys, share_foreground_type=False, *, fixed=False, num_types=3,
    value_gate=False,
):
    removed = sum(parameter.numel() for parameter in model.sam.memory_encoder.parameters())
    model.sam.memory_encoder = RemovedSpatialMemoryEncoder()
    interface = PrototypeMemoryInterface(share_foreground_type=share_foreground_type,
                                         fixed=fixed, num_types=num_types, value_gate=value_gate)
    model.add_module('prototype_memory', interface.to(model.device))
    model.removed_spatial_encoder_parameters = removed

    def forward(self, samples, prompt_dict):
        shots = prompt_dict['shots']
        processed, batch, frames, sizes = self._preprocess_visual_features(samples, self.sam.image_size)
        assert frames == shots+1
        backbone = self._forward_backbone(processed, sizes)
        masks = []
        for episode in range(batch):
            start = episode*frames
            query = start+shots
            precomputed = getattr(self, 'episode_memory_keys', None)
            keys = (precomputed[episode] if precomputed is not None else
                    key_builder(self, backbone, range(start, query), self.support_geometry[episode]))
            encoded = self.prototype_memory.prepare(keys)
            decoded = self.prototype_memory.decode(self, backbone, query, encoded)
            masks.append(decoded.masks)
        logits = torch.cat(masks, dim=0)
        logits = F.interpolate(logits, sizes[0], mode='bilinear', align_corners=False)[:, 0]
        return {'pred_masks': logits}

    model.forward = types.MethodType(forward, model)
    return model
