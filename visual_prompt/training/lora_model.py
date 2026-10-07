"""Frozen MobileSAM with trainable PatchMerging and independent K/V LoRA."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from visual_prompt.backbones.mobile_sam.builder import build_mobile_sam
from visual_prompt.backbones.mobile_sam.modeling.tiny_vit_sam import (
    Attention as EncoderAttention, PatchMerging,
)
from visual_prompt.backbones.mobile_sam.modeling.transformer import Attention as DecoderAttention


class LowRank(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.a = nn.Linear(in_dim, rank, bias=False)
        self.b = nn.Linear(rank, out_dim, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.b(self.a(x)) * self.scale


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.lora = LowRank(base.in_features, base.out_features, rank, alpha)

    def forward(self, x):
        return self.base(x) + self.lora(x)


class KVQKV(nn.Module):
    """TinyViT packs [q,k,v] *within each head*, not in three flat slices."""
    def __init__(self, attention, rank, alpha):
        super().__init__()
        self.base = attention.qkv
        self.heads = attention.num_heads
        self.key_dim = attention.key_dim
        self.value_dim = attention.d
        self.lora_k = LowRank(self.base.in_features, attention.nh_kd, rank, alpha)
        self.lora_v = LowRank(self.base.in_features, attention.dh, rank, alpha)

    def forward(self, x):
        shape = (*x.shape[:-1], self.heads)
        packed = self.base(x).reshape(*shape, 2 * self.key_dim + self.value_dim)
        q, k, v = packed.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        k = k + self.lora_k(x).reshape(*shape, self.key_dim)
        v = v + self.lora_v(x).reshape(*shape, self.value_dim)
        return torch.cat([q, k, v], dim=-1).flatten(-2)


class MobileSAMKVLoRA(nn.Module):
    def __init__(self, checkpoint_path, rank=8, alpha=8.0, gradient_checkpointing=True,
                 train_mask_output=True):
        super().__init__()
        self.sam = build_mobile_sam(checkpoint=checkpoint_path)
        self.gradient_checkpointing = gradient_checkpointing
        self.sam.requires_grad_(False)
        for module in list(self.sam.image_encoder.modules()):
            if isinstance(module, PatchMerging):
                module.requires_grad_(True)
            elif isinstance(module, EncoderAttention):
                module.qkv = KVQKV(module, rank, alpha)
        for module in list(self.sam.mask_decoder.modules()):
            if isinstance(module, DecoderAttention):
                module.k_proj = LoRALinear(module.k_proj, rank, alpha)
                module.v_proj = LoRALinear(module.v_proj, rank, alpha)
        self.train_mask_output = train_mask_output
        if train_mask_output:
            for module in self.mask_output_modules():
                module.requires_grad_(True)
        self.train()

    def mask_output_modules(self):
        decoder = self.sam.mask_decoder
        return (decoder.output_upscaling, decoder.output_hypernetworks_mlps, decoder.mask_tokens)

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen BN statistics and any inference attention caches frozen.
        self.sam.eval()
        if mode:
            if self.train_mask_output:
                for module in self.mask_output_modules():
                    module.train()
            for module in self.sam.image_encoder.modules():
                if isinstance(module, PatchMerging):
                    module.train()
        return self

    def parameter_counts(self):
        counts = dict(patch_merging=0, encoder_lora=0, decoder_lora=0, decoder_mask_output=0)
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if '.downsample.' in name:
                counts['patch_merging'] += p.numel()
            elif 'image_encoder' in name and 'lora_' in name:
                counts['encoder_lora'] += p.numel()
            elif 'mask_decoder' in name and '.lora.' in name:
                counts['decoder_lora'] += p.numel()
            elif any(name.startswith('sam.mask_decoder.' + part + '.') for part in
                     ('output_upscaling', 'output_hypernetworks_mlps', 'mask_tokens')):
                counts['decoder_mask_output'] += p.numel()
            else:
                raise RuntimeError(f"Unexpected trainable parameter: {name}")
        counts['trainable'] = sum(counts.values())
        counts['total_with_lora'] = sum(p.numel() for p in self.parameters())
        return counts

    def encode(self, image):
        encoder = self.sam.image_encoder
        x = encoder.patch_embed(image)
        for layer in encoder.layers:
            for block in layer.blocks:
                if self.training and self.gradient_checkpointing and x.requires_grad:
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            # Trainable BN is outside recomputed blocks: update exactly once.
            if layer.downsample is not None:
                x = layer.downsample(x)
        b, _, c = x.shape
        return encoder.neck(x.reshape(b, 64, 64, c).permute(0, 3, 1, 2))

    def forward(self, images):
        """Square RGB [B,3,H,W] in [0,1]; returns logits at input resolution."""
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[-2] != images.shape[-1]:
            raise ValueError("Expected square RGB BCHW input")
        output_size = images.shape[-2:]
        x = F.interpolate(images, (1024, 1024), mode='bilinear', align_corners=False) * 255.0
        x = (x - self.sam.pixel_mean) / self.sam.pixel_std
        embeddings = self.encode(x)
        sparse, dense = self.sam.prompt_encoder(points=None, boxes=None, masks=None)
        pe = self.sam.prompt_encoder.get_dense_pe()
        # Original decoder broadcasts one image to multiple prompts. Decode each
        # image separately to avoid accidental B*B expansion for batch_size > 1.
        logits = []
        for embedding in embeddings.split(1):
            mask, _ = self.sam.mask_decoder(
                image_embeddings=embedding, image_pe=pe,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            logits.append(mask)
        return F.interpolate(torch.cat(logits), output_size, mode='bilinear', align_corners=False)


def segmentation_loss(logits, masks):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, masks.float())
    probs = logits.sigmoid()
    dims = (1, 2, 3)
    dice = 1 - ((2 * (probs * masks).sum(dims) + 1) /
                (probs.sum(dims) + masks.sum(dims) + 1)).mean()
    return bce + dice
