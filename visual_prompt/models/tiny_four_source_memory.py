"""Build fixed-budget instance and image-level prototypes from episode supports."""
import zlib
import random

import cv2
import numpy as np
import torch

from visual_prompt.core.semantic_direction import SemanticDirectionConfig, build_score_token, pool_feature_token
from visual_prompt.models.tiny_shared_memory import descriptor, support_regions

SOURCES = ('instance', 'corrected_instance', 'hard_background', 'ordinary_background')


def four_source_regions(image, mask):
    regions = support_regions(image, mask)
    foreground = torch.from_numpy(cv2.resize(mask.astype(np.float32), (64, 64),
                                             interpolation=cv2.INTER_AREA) > 0)
    high_freq, high_value = (region & ~foreground for region in regions['negatives'])
    regions['negatives'] = (high_freq, high_value)
    regions['ordinary'] = ~(foreground | high_freq | high_value)
    regions['seed'] = zlib.crc32(image.tobytes())
    return regions


def eligible_support(regions):
    return any(region.any() for region in regions['negatives']) and bool(regions['ordinary'].any())


@torch.no_grad()
def build_instance_pool_keys(model, pairs, geometry, queries, background_sets, *, seed, audit):
    """Encode sampled instances plus the original background supports in small batches."""
    plans, requested = [], {}
    for query, backgrounds in zip(queries, background_sets, strict=True):
        candidates = [(sid, i) for sid, info in geometry.items() if sid != query
                      for i in range(len(info['instances']))]
        plan = random.Random(f'{seed}/instances/{query}').sample(candidates, 20)
        plans.append(plan)
        for sid, index in plan:
            requested.setdefault(sid, set()).add(index)
        for sid in backgrounds:
            requested.setdefault(sid, set())
        audit[query or 'evaluation'] = dict(instances=plan, background_supports=list(backgrounds))
    background_ids = {sid for group in background_sets for sid in group}
    ids, instances, backgrounds = list(requested), {}, {}
    for start in range(0, len(ids), 4):
        chunk = ids[start:start + 4]
        images = torch.stack([pairs[sid][0] for sid in chunk])[None].to(model.device)
        processed, _, _, sizes = model._preprocess_visual_features(images, 1024)
        backbone = model._forward_backbone(processed, sizes)
        for i, sid in enumerate(chunk):
            region = geometry[sid]
            feature = descriptor(model, backbone, i)[0].permute(1, 2, 0)
            for index in requested[sid]:
                instances[sid, index] = pool_feature_token(
                    feature, region['instances'][index].to(feature.device), 'mean_max', True)
            if sid in background_ids:
                backgrounds[sid] = tuple(pool_feature_token(feature, mask.to(feature.device), 'mean', True)
                                         for mask in (*region['negatives'], region['ordinary']))
    results = []
    for plan, support_ids in zip(plans, background_sets, strict=True):
        vectors = [instances[item] for item in plan]
        negative = {name: [backgrounds[sid][i] for sid in support_ids if backgrounds[sid][i] is not None]
                    for i, name in enumerate(('high_freq', 'bright_flat'))}
        hard = [token for pool in negative.values() for token in pool]
        ordinary = [backgrounds[sid][2] for sid in support_ids if backgrounds[sid][2] is not None]
        assert hard and ordinary
        # Reuse the background ordering of the previous 80-token arm.
        episode_seed = seed
        original_instance_count = 0
        for sid in support_ids:
            episode_seed = zlib.crc32(geometry[sid]['seed'].to_bytes(4, 'little'), episode_seed)
            original_instance_count += len(geometry[sid]['instances'])
        generator = torch.Generator().manual_seed(episode_seed)
        torch.randperm(original_instance_count, generator=generator)
        sampled_backgrounds = []
        for pool in (hard, ordinary):
            order = torch.randperm(len(pool), generator=generator).tolist()
            sampled_backgrounds.extend(pool[order[i % len(pool)]] for i in range(20))
        with torch.autocast(vectors[0].device.type, enabled=False):
            config = SemanticDirectionConfig()
            corrected = [build_score_token([token], negative, config)[0] for token in vectors]
        keys = torch.cat([*vectors, *corrected, *sampled_backgrounds])
        assert keys.shape == (80, 352) and (keys.norm(dim=-1) > 0).all()
        results.append((keys, torch.arange(4, device=keys.device).repeat_interleave(20)))
    return results


@torch.no_grad()
def build_keys(model, backbone, indices, regions, *, seed, mode, audit, counts=None):
    instances, ordinary = [], []
    negatives = {'high_freq': [], 'bright_flat': []}
    for index, geometry in zip(indices, regions, strict=True):
        seed = zlib.crc32(geometry['seed'].to_bytes(4, 'little'), seed)
        feature = descriptor(model, backbone, index)[0].permute(1, 2, 0)
        instances.extend(pool_feature_token(feature, mask.to(feature.device), 'mean_max', True)
                         for mask in geometry['instances'])
        for tokens, mask in zip(negatives.values(), geometry['negatives'], strict=True):
            vector = pool_feature_token(feature, mask.to(feature.device), 'mean', True)
            if vector is not None:
                tokens.append(vector)
        vector = pool_feature_token(feature, geometry['ordinary'].to(feature.device), 'mean', True)
        if vector is not None:
            ordinary.append(vector)

    generator = torch.Generator().manual_seed(seed)
    if counts is None:
        counts = (5, 5, 5, 5) if mode == 'mixed' else tuple(20 if source == mode else 0 for source in SOURCES)
    hard = [token for tokens in negatives.values() for token in tokens]
    pools = (instances, instances, hard, ordinary)
    stats = audit.setdefault('train' if model.training else 'eval', {})
    values, labels = [], []
    instance_order = torch.randperm(len(instances), generator=generator)
    for index, (source, pool, count) in enumerate(zip(SOURCES, pools, counts, strict=True)):
        if not count:
            continue
        if not pool:
            raise ValueError(f'No eligible {source} prototype in episode')
        order = instance_order if index < 2 else torch.randperm(len(pool), generator=generator)
        selected = [pool[item] for item in order.repeat((count + len(pool) - 1) // len(pool))[:count].tolist()]
        if source == 'corrected_instance':
            assert hard
            # Correct distinct instances once, then reuse the corresponding vectors.
            with torch.autocast(selected[0].device.type, enabled=False):
                config = SemanticDirectionConfig()
                corrected = [build_score_token([token], negatives, config)[0] for token in instances]
            assert all(token.norm() > 0 for token in corrected)
            selected = [corrected[item] for item in order.repeat((count + len(pool) - 1) // len(pool))[:count].tolist()]
        values.extend(selected)
        labels.extend([index] * count)
        stats[source] = stats.get(source, 0) + min(count, len(pool))
    keys = torch.cat(values)
    stats['episodes'] = stats.get('episodes', 0) + 1
    assert keys.shape == (sum(counts), 352) and (keys.norm(dim=-1) > 0).all()
    return keys, torch.tensor(labels, device=keys.device)
