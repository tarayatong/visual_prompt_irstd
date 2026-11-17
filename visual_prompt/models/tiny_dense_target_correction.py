"""Fixed target-direction correction in SAM2's native dense memory space."""
import torch
from torch.nn import functional as F

from visual_prompt.core.semantic_direction import (
    SemanticDirectionConfig,
    build_score_token,
    mean_token,
    pool_feature_token,
)
from visual_prompt.models.tiny_shared_memory import support_regions


def correct_dense_target_memory(bank, pairs, correction, foreground_gain):
    """Adjust foreground cells in-place in the support bank, keeping all tokens."""
    positives = []
    negatives = {'high_freq': [], 'bright_flat': []}
    foreground_masks = {}
    for index, (image, mask) in enumerate(pairs):
        memory = bank[index]['maskmem_features']
        _, _, height, width = memory.shape
        if height != width:
            raise ValueError('Expected square SAM2 memory grids')
        regions = support_regions(image.astype('uint8'), mask > 0, grid=width)
        foreground = torch.stack(regions['instances']).any(dim=0).to(memory.device)
        foreground_masks[index] = foreground
        if correction:
            feature = memory[0].float().permute(1, 2, 0)
            positives.append(pool_feature_token(feature, foreground, 'mean'))
            for tokens, region in zip(negatives.values(), regions['negatives'], strict=True):
                # A pooled negative cell may overlap a target after downsampling.
                background = region.to(memory.device) & ~foreground
                tokens.append(pool_feature_token(feature, background, 'mean'))

    direction_delta = None
    if correction:
        raw_direction = mean_token(positives)
        corrected_direction, _ = build_score_token(positives, negatives, SemanticDirectionConfig())
        direction_delta = (corrected_direction - raw_direction)[0]

    for index, foreground in foreground_masks.items():
        memory = bank[index]['maskmem_features'].clone()
        target = memory[0, :, foreground].float()
        if direction_delta is not None:
            norm = target.norm(dim=0, keepdim=True)
            moved = target + correction * norm * direction_delta[:, None]
            target = F.normalize(moved, dim=0) * norm
        memory[0, :, foreground] = ((1 + foreground_gain) * target).to(memory.dtype)
        bank[index]['maskmem_features'] = memory
