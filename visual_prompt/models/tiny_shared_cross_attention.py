"""Shared-encoder prototype cross-attention after native SAM2 memory attention."""
import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.models.semantic_memory import RecallMemoryReader
from visual_prompt.models.tiny_shared_memory import install_shared_memory


class SharedPrototypeCrossAttention(nn.Module):
    """Query pixels read positive/negative prototype values in separate pools.

    Preserve the previous Reader gate and insertion point. Change the residual
    source from a query-feature bottleneck to retrieved prototype value vectors.
    Empty slots are masked. An entirely empty negative pool returns exact zeros.
    Semantic prototypes have no artificial spatial positions or spatial RoPE.
    """
    def __init__(self):
        super().__init__()
        self.reader = RecallMemoryReader(19)
        self.query_norm = nn.LayerNorm(256)
        self.key_norm = nn.LayerNorm(352)
        self.query = nn.Linear(256, 64, bias=False)
        self.key = nn.Linear(352, 64, bias=False)
        self.value = nn.Linear(352, 64, bias=False)
        self.output = nn.Conv2d(128, 256, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    @staticmethod
    def read_pool(logits, values, valid):
        # Compute softmax in FP32, explicitly exclude padded prototype slots.
        weights = logits.float().masked_fill(~valid[None, None, :], -1e4).softmax(-1)
        weights = weights * valid[None, None, :]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return weights.to(values.dtype) @ values

    def forward(self, main, query_descriptor, keys):
        batch, _, height, width = main.shape
        maps = torch.einsum('kc,bchw->bkhw', keys.float(), query_descriptor.float())
        score = self.reader(maps.permute(0, 2, 3, 1).reshape(-1, 19))
        gate = score.reshape(batch, 1, height, width).sigmoid()
        queries = self.query_norm(main.flatten(2).transpose(1, 2))
        normalized_keys = self.key_norm(keys)
        q = F.normalize(self.query(queries), dim=-1)
        k = F.normalize(self.key(normalized_keys), dim=-1)
        values = self.value(normalized_keys)
        logits = q @ k.T / .2
        valid = keys.float().norm(dim=-1) > 0
        positive = self.read_pool(logits[..., :10], values[:10], valid[:10])
        negative = self.read_pool(logits[..., 10:], values[10:], valid[10:])
        retrieved = torch.cat([positive, negative], -1).transpose(1, 2)
        retrieved = retrieved.reshape(batch, 128, height, width)
        return main + gate * self.output(retrieved)


def install_shared_cross_attention(model):
    # Reuse the exact episode/key/descriptor plumbing of the completed c run.
    install_shared_memory(model)
    model.shared_memory = SharedPrototypeCrossAttention().to(model.device)
    return model
