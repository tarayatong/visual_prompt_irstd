"""Objectness ROI descriptors from fixed128-to512 retrieval-window features."""
import numpy as np
import torch
from torch.nn import functional as F

from visual_prompt.core.coarse_window_gate import WINDOWS
from visual_prompt.models.full_image_objectness import _roi_grid
from visual_prompt.pipelines.semantic_prompt_pipeline import get_named_sim_features


def owner_windows(points):
    """Nearest fixed-grid window center; deterministic row-major tie breaking."""
    if not points:
        return np.empty(0, np.int64)
    centers = np.asarray(WINDOWS) + 63.5
    coordinates = np.asarray(points)
    owners = ((coordinates[:, None]-centers[None])**2).sum(2).argmin(1)
    for (y, x), index in zip(points, owners):
        y0, x0 = WINDOWS[int(index)]
        if not (y0 <= y < y0+128 and x0 <= x < x0+128):
            raise ValueError('Candidate outside the512 working image')
    return owners


@torch.no_grad()
def ensure_window_features(predictor, image, points, cache, batch_size=8):
    missing = sorted(set(map(int, owner_windows(points)))-set(cache))
    for start in range(0, len(missing), batch_size):
        indexes = missing[start:start+batch_size]
        predictor.set_image_batch([image[y:y+128, x:x+128] for y,x in [WINDOWS[i] for i in indexes]])
        native = get_named_sim_features(predictor, align=False)
        for j, index in enumerate(indexes):
            cache[index] = [value[j:j+1] for _, value in native]
    return len(missing)


@torch.no_grad()
def sample_window_rois(points, cache, crop_size=64, grid_size=32):
    owners = owner_windows(points)
    result = [None]*len(points)
    for index in sorted(set(map(int, owners))):
        positions = np.flatnonzero(owners == index).tolist()
        y0, x0 = WINDOWS[index]
        local = [(points[i][0]-y0, points[i][1]-x0) for i in positions]
        features = cache[index]
        grid = _roi_grid(local, (128, 128), crop_size, grid_size, features[0].device)
        sampled = [F.grid_sample(feature.float().expand(len(positions), -1, -1, -1), grid,
                    mode='bilinear', padding_mode='zeros', align_corners=True) for feature in features]
        rois = F.normalize(torch.cat(sampled, 1), dim=1)
        for j, position in enumerate(positions):
            result[position] = rois[j]
    return torch.stack(result)


@torch.no_grad()
def sample_window_points(points, cache):
    """Sample exact candidate coordinates in their owner window, without ROI pooling."""
    owners = owner_windows(points)
    if not len(owners):
        return torch.empty((0, 0))
    first = cache[int(owners[0])]
    result = torch.empty((len(points), sum(f.shape[1] for f in first)), device=first[0].device)
    for index in sorted(set(map(int, owners))):
        positions = np.flatnonzero(owners == index).tolist()
        y0, x0 = WINDOWS[index]
        grid = torch.tensor([[[[2*(points[i][1]-x0)/127-1, 2*(points[i][0]-y0)/127-1]]
                              for i in positions]], device=first[0].device)
        sampled = [F.grid_sample(feature.float(), grid, mode='bilinear', padding_mode='zeros',
                    align_corners=True)[0, :, :, 0].T for feature in cache[index]]
        result[positions] = F.normalize(torch.cat(sampled, dim=1), dim=1)
    return result
