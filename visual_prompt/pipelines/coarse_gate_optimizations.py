"""Execution optimizations preserving the coarse-gated detector's geometry."""
import math
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from visual_prompt.core.coarse_window_gate import WINDOWS
from visual_prompt.evaluation import eval_sliding_window_semantic_mask as ev
from visual_prompt.models.radial_profile import DirectionalRadialOperator, RadialProfileConfig
from visual_prompt.pipelines.semantic_prompt_pipeline import get_sim_features, get_named_sim_features


@torch.no_grad()
def gpu_score_peaks(score, k=5, radius=7):
    """Same row-major tie breaking and OpenCV disk as the reference CPU NMS."""
    work = score.float().clone()
    height, width = work.shape
    disk = np.zeros((2*radius+1, 2*radius+1), np.uint8)
    cv2.circle(disk, (radius, radius), radius, 1, thickness=-1)
    disk = torch.as_tensor(disk, device=work.device, dtype=torch.bool)
    result = []
    for _ in range(k):
        index = int(work.flatten().argmax().item())
        y, x = divmod(index, width)
        if not math.isfinite(float(work[y, x].item())):
            break
        result.append((y, x))
        y0, y1 = max(0, y-radius), min(height, y+radius+1)
        x0, x1 = max(0, x-radius), min(width, x+radius+1)
        local = disk[y0-y+radius:y1-y+radius, x0-x+radius:x1-x+radius]
        work[y0:y1, x0:x1].masked_fill_(local, -torch.inf)
    return result


@torch.no_grad()
def fine_maps(study, image, selected, batch_size=1, gpu_stitch=False, feature_cache=None):
    channels = len(study.memory['keys'])
    if gpu_stitch:
        total = torch.zeros((channels, 512, 512), device=study.device)
        canonical = torch.zeros((512, 512), device=study.device)
        weight = torch.zeros_like(canonical)
    else:
        total = np.zeros((channels, 512, 512), np.float32)
        canonical = np.zeros((512, 512), np.float32)
        weight = np.zeros((512, 512), np.float32)
    keys = study.memory['keys'].to(study.device)
    for start in range(0, len(selected), batch_size):
        indexes = selected[start:start+batch_size]
        crops = [image[y:y+128, x:x+128] for y, x in [WINDOWS[int(i)] for i in indexes]]
        study.query.set_image_batch(crops)
        if feature_cache is not None:
            native = get_named_sim_features(study.query, align=False)
            for j, index in enumerate(indexes):
                feature_cache[int(index)] = [value[j:j+1] for _, value in native]
        feature = get_sim_features(study.query).float()
        single = ev.compute_sim_batch_from_features(study.query, study.e0, feature, 256)
        raw = torch.einsum('kc,bchw->bkhw', keys, F.normalize(feature, dim=1))
        raw = F.interpolate(raw, (256, 256), mode='bilinear', align_corners=False)
        patches = study.query.postprocess_masks(raw, study.query.input_size, study.query.original_size)
        if not gpu_stitch:
            single = single.cpu().numpy(); patches = patches.cpu().numpy()
        for j, index in enumerate(indexes):
            y, x = WINDOWS[int(index)]
            canonical[y:y+128, x:x+128] += single[j]
            total[:, y:y+128, x:x+128] += patches[j]
            weight[y:y+128, x:x+128] += 1
    if gpu_stitch:
        denominator = weight.clamp_min(1)
        total /= denominator[None]; total[0] = canonical/denominator
        return total, weight > 0
    total /= np.maximum(weight, 1)[None]; total[0] = canonical/np.maximum(weight, 1)
    return torch.from_numpy(total).to(study.device), weight > 0


class GpuRadialSampler:
    """Compute the original contrast slice, retaining FP16 rounding on GPU.

    The reference computes intensity/contrast/first/second and discards three
    components. This computes contrast alone with identical geometry and validity.
    Larger query chunks reduce sampling launches; one gray upload serves all ROIs.
    """
    def __init__(self, reference, device):
        self.device = device
        self.offsets = reference.offsets
        self.op = DirectionalRadialOperator(1, RadialProfileConfig(
            fusion='concat', components=('contrast',),
            radii=(1., 2., 3., 4., 6., 8., 12., 16.), chunk_size=8192)).to(device).eval()

    @torch.no_grad()
    def sample_batch(self, image, points):
        if not points:
            return (torch.empty((0, 64, 64, 64), device=self.device),
                    torch.empty((0, 64, 64, 64), device=self.device, dtype=torch.bool))
        gray = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)/255).to(self.device)[None, None]
        centers = torch.tensor([(x, y) for y, x in points], device=self.device)
        queries = (self.offsets + centers[:, None]).reshape(1, -1, 2)
        profiles = self.op.sample_profiles(gray, queries)
        count = len(points)
        raw = profiles.values.reshape(count, 4096, 64).transpose(1, 2).reshape(count, 64, 64, 64)
        valid = profiles.valid.reshape(count, 4096, 64).transpose(1, 2).reshape(count, 64, 64, 64)
        return raw.half().float(), valid
