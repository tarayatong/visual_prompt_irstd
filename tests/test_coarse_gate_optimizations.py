import numpy as np
import torch
from visual_prompt.core.candidate_proposals import greedy_score_peaks
from visual_prompt.pipelines.coarse_gate_optimizations import gpu_score_peaks, GpuRadialSampler
from visual_prompt.pipelines.p2p_framework import RadialSampler


def test_gpu_nms_matches_reference():
    rng = np.random.default_rng(7)
    for score in [rng.random((39, 45), dtype=np.float32), np.ones((39, 45), np.float32), np.full((39, 45), -np.inf, np.float32)]:
        score[0, -1] = 2.; score[-1, 0] = 2.
        expected = [(y, x) for y, x, _ in greedy_score_peaks(score, 10, 7)]
        assert gpu_score_peaks(torch.from_numpy(score).cuda(), 10, 7) == expected


def test_gpu_radial_matches_border_and_interior():
    rng = np.random.default_rng(9)
    image = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    points = [(0, 0), (511, 511), (250, 251)]
    reference = RadialSampler(torch.device('cuda:0'))
    optimized = GpuRadialSampler(reference, torch.device('cuda:0'))
    raw, valid = optimized.sample_batch(image, points)
    for i, (y, x) in enumerate(points):
        old_raw, old_valid = reference.sample(image, y, x)
        assert torch.equal(valid[i].cpu(), old_valid)
        assert torch.equal(raw[i].cpu(), old_raw.float())
