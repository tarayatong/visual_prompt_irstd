"""Token controls for the existing 19-slot, 352-to-64 memory interface."""
import zlib

import cv2
import numpy as np
import torch

from visual_prompt.core.semantic_direction import SemanticDirectionConfig, build_score_token, mean_token, pool_feature_token
from visual_prompt.models.tiny_shared_memory import build_episode_keys, descriptor, support_regions


def _radical_inverse(index):
    value, scale = 0., .5
    while index:
        value += (index & 1) * scale
        index >>= 1
        scale *= .5
    return value


@torch.no_grad()
def budget_matched_spatial(model, backbone, indices, regions):
    """Keep semantic validity and slot-type layout, but replace all vectors."""
    indices = tuple(indices)
    reference = build_episode_keys(model, backbone, indices, regions)
    valid = reference.float().norm(dim=-1) > 0
    count = int(valid.sum())
    keys = torch.zeros_like(reference)
    vectors = []
    per_image, remainder = divmod(count, len(indices))
    for i, index in enumerate(indices):
        take = per_image + int(i < remainder)
        if not take:
            continue
        grid = descriptor(model, backbone, index)[0]
        h, w = grid.shape[-2:]
        if take == 1:
            vectors.append(grid[:, h//2, w//2][None])
        else:
            # Deterministic two-dimensional low-discrepancy sampling, no
            # foreground labels or image-dependent selection of coordinates.
            x = [int((j+.5)*w/take) for j in range(take)]
            y = [int(_radical_inverse(j)*h) for j in range(take)]
            vectors.append(grid[:, y, x].transpose(0, 1))
    keys[valid] = torch.cat(vectors)
    assert torch.equal(keys.float().norm(dim=-1) > 0, valid)
    return keys


@torch.no_grad()
def no_background(model, backbone, indices, regions):
    """No negative correction and no negative attention keys."""
    positives, image_tokens = [], []
    for index, geometry in zip(indices, regions, strict=True):
        feature = descriptor(model, backbone, index)[0].permute(1, 2, 0)
        local = [pool_feature_token(feature, mask.to(feature.device), 'mean_max', True)
                 for mask in geometry['instances']]
        positives.extend(local)
        image_tokens.append(mean_token(local))
    raw = mean_token(image_tokens)
    keys = raw.new_zeros(19, 352)
    # Keep both global slots and their type embeddings. The slot that would
    # have held the corrected vector now contains its uncorrected counterpart.
    keys[0] = keys[1] = raw[0]
    for slot, token in enumerate(positives[:8], 2):
        keys[slot] = token[0]
    return keys


def foreground_sampling_regions(image, mask):
    geometry = support_regions(image, mask)
    # A cell is eligible only when it overlaps the undilated target mask.
    coverage = cv2.resize(mask.astype(np.float32), (64, 64), interpolation=cv2.INTER_AREA)
    geometry['foreground_cells'] = torch.from_numpy(np.argwhere(coverage > 0))
    return geometry


@torch.no_grad()
def foreground_sampled(model, backbone, indices, regions):
    """Sample foreground cells with the same per-episode budget as no_background."""
    count = 2 + min(8, sum(len(geometry['instances']) for geometry in regions))
    per_image, remainder = divmod(count, len(regions))
    keys = None
    start = 0
    for i, (index, geometry) in enumerate(zip(indices, regions, strict=True)):
        take = per_image + int(i < remainder)
        if not take:
            continue
        grid = descriptor(model, backbone, index)[0]
        if keys is None:
            keys = grid.new_zeros(19, grid.shape[0])
        cells = geometry['foreground_cells']
        assert len(cells) > 0
        # Even quantiles of the eligible cells. Repeat cells if the target is
        # too small to supply distinct points at the matched token budget.
        offsets = (2 * torch.arange(take) + 1) * len(cells) // (2 * take)
        y, x = cells[offsets].to(grid.device).unbind(dim=1)
        keys[start:start + take] = grid[:, y, x].transpose(0, 1)
        start += take
    assert start == count and int((keys.norm(dim=-1) > 0).sum()) == count
    return keys


@torch.no_grad()
def positive_prototypes(model, backbone, indices, regions):
    keys = no_background(model, backbone, indices, regions)
    keys[1].zero_()
    return keys


def random_sampling_regions(image, mask):
    geometry = support_regions(image, mask)
    geometry['sampling_seed'] = zlib.crc32(image.tobytes())
    return geometry


@torch.no_grad()
def random_spatial(model, backbone, indices, regions, *, seed):
    """Uniformly sample support cells; only the token count comes from semantic memory."""
    indices = tuple(indices)
    reference = build_episode_keys(model, backbone, indices, regions)
    count = int((reference.float().norm(dim=-1) > 0).sum())
    # Fix draws for each support set and seed across restarts and evaluation.
    for geometry in regions:
        seed = zlib.crc32(geometry['sampling_seed'].to_bytes(4, 'little'), seed)
    generator = torch.Generator().manual_seed(seed)
    grids = torch.cat([descriptor(model, backbone, index).flatten(2)[0].T for index in indices])
    locations = torch.randperm(len(grids), generator=generator)[:count]
    tags = torch.randint(4, (19,), generator=generator).to(grids.device)
    keys = torch.zeros_like(reference)
    keys[:count] = grids[locations.to(grids.device)]
    assert int((keys.float().norm(dim=-1) > 0).sum()) == count
    return keys, tags


def grid_triplet_regions(image, mask):
    geometry = foreground_sampling_regions(image, mask)
    foreground = torch.zeros(64, 64, dtype=torch.bool)
    y, x = geometry['foreground_cells'].T
    foreground[y, x] = True
    geometry['grid_pools'] = (
        foreground.flatten().nonzero().flatten(),
        *((region & ~foreground).flatten().nonzero().flatten() for region in geometry['negatives']),
    )
    geometry['sampling_seed'] = zlib.crc32(image.tobytes())
    return geometry


def grid_sampling_generator(regions, seed):
    for geometry in regions:
        seed = zlib.crc32(geometry['sampling_seed'].to_bytes(4, 'little'), seed)
    return torch.Generator().manual_seed(seed)


def draw_grid_cells(pool, count, generator):
    assert len(pool) > 0
    order = torch.randperm(len(pool), generator=generator)[:count]
    if len(order) < count:
        repeats = torch.randint(len(pool), (count - len(order),), generator=generator)
        order = torch.cat((order, repeats))
    return pool[order]


@torch.no_grad()
def sample_grid_memory(model, backbone, indices, regions, *, seed, balanced):
    generator = grid_sampling_generator(regions, seed)
    tags = torch.randint(4, (20,), generator=generator)
    grids = torch.cat([descriptor(model, backbone, index).flatten(2)[0].T for index in indices])
    if balanced:
        foreground = torch.cat([region['grid_pools'][0] + i * 4096
                                for i, region in enumerate(regions)])
        background = torch.ones(len(grids), dtype=torch.bool)
        background[foreground] = False
        cells = torch.cat((draw_grid_cells(foreground, 10, generator),
                           draw_grid_cells(background.nonzero().flatten(), 10, generator)))
    else:
        cells = torch.randperm(len(grids), generator=generator)[:20]
    return grids[cells.to(grids.device)], tags.to(grids.device)


@torch.no_grad()
def sample_instance_memory(model, backbone, indices, regions, *, seed, corrected):
    positives = []
    negatives = {'high_freq': [], 'bright_flat': []}
    for index, geometry in zip(indices, regions, strict=True):
        feature = descriptor(model, backbone, index)[0].permute(1, 2, 0)
        positives.extend(pool_feature_token(feature, mask.to(feature.device), 'mean_max', True)
                         for mask in geometry['instances'])
        if corrected:
            for name, cells in zip(negatives, geometry['grid_pools'][1:], strict=True):
                if len(cells):
                    flat = feature.reshape(-1, feature.shape[-1])
                    pooled = flat[cells.to(feature.device)].mean(dim=0, keepdim=True)
                    negatives[name].append(torch.nn.functional.normalize(pooled, dim=-1))
    keys = torch.cat(positives)
    tags = torch.randint(4, (len(keys),), generator=grid_sampling_generator(regions, seed))
    if corrected and any(negatives.values()):
        with torch.autocast(keys.device.type, enabled=False):
            config = SemanticDirectionConfig()
            corrected_keys = torch.cat([build_score_token([p], negatives, config)[0] for p in positives])
            keys = torch.where(corrected_keys.norm(dim=-1, keepdim=True) > 0, corrected_keys, keys)
    return keys, tags.to(keys.device)


@torch.no_grad()
def grid_triplets(model, backbone, indices, regions, *, seed, corrected, audit):
    """Use the same 20 foreground locations and type labels in both arms."""
    width = 64 * 64
    for geometry in regions:
        seed = zlib.crc32(geometry['sampling_seed'].to_bytes(4, 'little'), seed)
    generator = torch.Generator().manual_seed(seed)
    pools = [
        torch.cat([geometry['grid_pools'][kind] + i * width for i, geometry in enumerate(regions)])
        for kind in range(3)
    ]
    tags = torch.randint(4, (20,), generator=generator)
    draws = []
    for pool in pools:
        if len(pool) >= 20:
            selection = torch.randperm(len(pool), generator=generator)[:20]
        elif len(pool):
            order = torch.randperm(len(pool), generator=generator)
            repeats = torch.randint(len(pool), (20 - len(pool),), generator=generator)
            selection = torch.cat((order, repeats))
        else:
            draws.append(pool)
            continue
        draws.append(pool[selection])
    assert len(draws[0]) == 20
    stats = audit.setdefault('train' if model.training else 'eval', {})
    for label, value in zip(('episodes', 'repeated_foreground', 'missing_high_frequency', 'missing_high_value'),
                            (1, 20 - draws[0].unique().numel(), int(not len(draws[1])), int(not len(draws[2])))):
        stats[label] = stats.get(label, 0) + value
    grids = torch.cat([descriptor(model, backbone, index).flatten(2)[0].T for index in indices])
    points = grids[draws[0].to(grids.device)]
    if corrected and len(draws[1]) and len(draws[2]):
        # Batched form of the existing rank-truncated negative-subspace projection.
        with torch.autocast(grids.device.type, enabled=False):
            negatives = torch.stack([grids[draw.to(grids.device)] for draw in draws[1:]], dim=1)
            _, singular, basis = torch.linalg.svd(torch.nn.functional.normalize(negatives.float(), dim=-1),
                                                 full_matrices=False)
            active = singular > singular[:, :1] * 1e-5
            target = torch.nn.functional.normalize(points.float(), dim=-1)
            coefficients = (basis @ target.unsqueeze(-1)).squeeze(-1) * active
            residual = target - (coefficients.unsqueeze(1) @ basis).squeeze(1)
            norm = residual.norm(dim=-1, keepdim=True)
            points = torch.where(norm >= 1e-8, residual / (norm + 1e-8), points)
    return points, tags.to(grids.device)
