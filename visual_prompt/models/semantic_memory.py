"""Optional dense memory reader; frozen support keys, trainable retrieval residual."""
import torch
from torch import nn


class RecallMemoryReader(nn.Module):
    def __init__(self, channels, hidden=16, residual_bound=.25):
        super().__init__()
        self.read = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.read[-1].weight)
        nn.init.zeros_(self.read[-1].bias)
        self.residual_bound = residual_bound

    def forward(self, features):
        # Channel 1 is the instance-preserving, image-balanced corrected e0.
        return features[..., 1] + self.residual_bound * self.read(features).squeeze(-1).tanh()

    def expanded(self, channels, old_to_new):
        """Initialize an enlarged memory reader without changing old predictions.

        New key columns start at zero. Subsequent shared updates still require
        replay/retention constraints; this initializer alone is not anti-forgetting.
        """
        if len(old_to_new) != self.read[0].in_features or len(set(old_to_new)) != len(old_to_new):
            raise ValueError('Provide one distinct new position for every old channel')
        if old_to_new[:2] != [0, 1] or min(old_to_new) < 0 or max(old_to_new) >= channels:
            raise ValueError('Preserve baseline channels 0/1 and valid mapped indices')
        result = RecallMemoryReader(channels, self.read[0].out_features, self.residual_bound)
        reference = next(self.parameters())
        result.to(device=reference.device, dtype=reference.dtype)
        with torch.no_grad():
            result.read[0].weight.zero_()
            result.read[0].weight[:, old_to_new] = self.read[0].weight
            result.read[0].bias.copy_(self.read[0].bias)
            result.read[2].load_state_dict(self.read[2].state_dict())
        return result

    @torch.no_grad()
    def score_map(self, maps, chunk=32768):
        c, h, w = maps.shape
        points = maps.reshape(c, -1).T
        return torch.cat([self(points[i:i+chunk]) for i in range(0, len(points), chunk)]).reshape(h, w)
