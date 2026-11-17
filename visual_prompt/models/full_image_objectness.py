"""Candidate descriptors sampled from a single full-image SAM2 encoding."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from visual_prompt.evaluation import eval_sliding_window_semantic_mask as ev
from visual_prompt.pipelines.semantic_prompt_pipeline import get_named_sim_features


DESCRIPTOR_NAMES = ("sim", "center", "mean", "max", "diff", "hand")


def _map_stats(value: np.ndarray) -> np.ndarray:
    height, width = value.shape
    cy, cx = height // 2, width // 2
    yy, xx = np.mgrid[:height, :width]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    scale = min(height, width)
    ring = (radius >= scale / 8) & (radius <= 3 * scale / 8)
    ring_mean = float(value[ring].mean()) if ring.any() else float(value.mean())
    return np.asarray(
        [value[cy, cx], value.max(), value.mean(), value.std(), value[cy, cx] - ring_mean],
        dtype=np.float32,
    )


def _crop_with_padding(image: np.ndarray, value: np.ndarray, y: int, x: int, size: int):
    return ev.crop_with_padding(image, value, y - size // 2, x - size // 2, size)


def _roi_grid(
    points: list[tuple[int, int]], image_shape: tuple[int, int], crop_size: int,
    grid_size: int, device: torch.device,
) -> torch.Tensor:
    """Build an align-corners grid in full-image coordinates for candidate ROIs."""
    height, width = image_shape
    offset = torch.linspace(
        -(crop_size - 1) / 2, (crop_size - 1) / 2, grid_size, device=device
    )
    grids = []
    for y, x in points:
        sample_y = float(y) + offset
        sample_x = float(x) + offset
        normalized_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
        normalized_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
        gy, gx = torch.meshgrid(normalized_y, normalized_x, indexing="ij")
        grids.append(torch.stack((gx, gy), dim=-1))
    return torch.stack(grids)


@torch.no_grad()
def sample_full_image_rois(
    predictor,
    image: np.ndarray,
    points: list[tuple[int, int]],
    crop_size: int = 64,
    grid_size: int = 32,
    batch_size: int = 32,
    encode_image: bool = True,
) -> torch.Tensor:
    """Encode ``image`` once and ROI-sample all candidate points from its feature maps."""
    if not points:
        return torch.empty((0, 0, grid_size, grid_size), device=predictor.device)
    if encode_image:
        predictor.set_image(image)
    elif tuple(predictor.original_size or ()) != tuple(image.shape[:2]):
        raise ValueError('Reused features must belong to this full-image geometry')
    named_features = get_named_sim_features(predictor, align=False)
    chunks = []
    for start in range(0, len(points), batch_size):
        chunk_points = points[start : start + batch_size]
        grid = _roi_grid(
            chunk_points, image.shape[:2], crop_size, grid_size, predictor.device
        )
        sampled_layers = []
        for _, feature in named_features:
            expanded = feature.float().expand(len(chunk_points), -1, -1, -1)
            sampled_layers.append(
                F.grid_sample(
                    expanded,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
            )
        chunks.append(F.normalize(torch.cat(sampled_layers, dim=1), dim=1))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def describe_full_image_points(
    predictor,
    image: np.ndarray,
    score: np.ndarray,
    points: list[tuple[int, int]],
    crop_size: int,
    local_contrast_sigma: float,
    grid_size: int = 32,
    encode_image: bool = True,
    roi_features: torch.Tensor | None = None,
) -> dict[str, np.ndarray]:
    """Build objectness descriptors while performing only one SAM2 image encoding."""
    if not points:
        return {name: np.empty((0, 0), dtype=np.float32) for name in DESCRIPTOR_NAMES}

    if roi_features is None:
        rois = sample_full_image_rois(
            predictor, image, points, crop_size=crop_size, grid_size=grid_size,
            encode_image=encode_image,
        )
    else:
        if roi_features.ndim != 4 or roi_features.shape[0] != len(points) or roi_features.shape[-2:] != (grid_size, grid_size):
            raise ValueError('Precomputed ROIs must match the candidate count and sampling grid')
        rois = roi_features
    _, _, height, width = rois.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=rois.device),
        torch.arange(width, device=rois.device),
        indexing="ij",
    )
    radius = torch.sqrt(
        (yy - height // 2).float() ** 2 + (xx - width // 2).float() ** 2
    )
    ring = (radius >= min(height, width) / 8) & (radius <= 3 * min(height, width) / 8)
    center = rois[:, :, height // 2, width // 2]
    mean = rois.mean(dim=(-2, -1))
    maximum = rois.amax(dim=(-2, -1))
    ring_mean = rois[:, :, ring].mean(dim=-1)

    sim_values, hand_values = [], []
    blank = np.zeros(image.shape[:2], dtype=np.uint8)
    for y, x in points:
        image_crop, _ = _crop_with_padding(image, blank, y, x, crop_size)
        _, score_crop = _crop_with_padding(image, score, y, x, crop_size)
        gray = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        contrast = ev.local_contrast_score(image_crop, local_contrast_sigma)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = ev.robust_norm(cv2.magnitude(gx, gy))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        top_hat = ev.robust_norm(gray - opened)
        sim_base = _map_stats(score_crop)
        sim_values.append(np.concatenate([sim_base, [sim_base[1] - sim_base[2]]]))
        hand_values.append(
            np.concatenate([_map_stats(value) for value in (gray, contrast, gradient, top_hat)])
        )

    return {
        "sim": np.asarray(sim_values, dtype=np.float32),
        "center": center.cpu().numpy().astype(np.float32),
        "mean": mean.cpu().numpy().astype(np.float32),
        "max": maximum.cpu().numpy().astype(np.float32),
        "diff": (center - ring_mean).cpu().numpy().astype(np.float32),
        "hand": np.asarray(hand_values, dtype=np.float32),
    }


def _same_local_pool(value: torch.Tensor, kernel_size: int, mode: str) -> torch.Tensor:
    before = (kernel_size - 1) // 2
    after = kernel_size - 1 - before
    padded = F.pad(value, (before, after, before, after), mode="constant", value=0.0)
    if mode == "mean":
        return F.avg_pool2d(padded, kernel_size, stride=1)
    if mode == "max":
        return F.max_pool2d(padded, kernel_size, stride=1)
    raise ValueError(mode)


def _ring_pool(value: torch.Tensor, kernel_size: int) -> torch.Tensor:
    center = (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(
        torch.arange(kernel_size, device=value.device),
        torch.arange(kernel_size, device=value.device),
        indexing="ij",
    )
    radius = torch.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    ring = ((radius >= kernel_size / 8) & (radius <= 3 * kernel_size / 8)).float()
    ring = ring / ring.sum().clamp_min(1.0)
    channels = value.shape[1]
    weight = ring[None, None].expand(channels, 1, -1, -1)
    before = (kernel_size - 1) // 2
    after = kernel_size - 1 - before
    padded = F.pad(value, (before, after, before, after), mode="constant", value=0.0)
    return F.conv2d(padded, weight, groups=channels)


def _dense_map_stats(value: torch.Tensor, kernel_size: int) -> torch.Tensor:
    mean = _same_local_pool(value, kernel_size, "mean")
    maximum = _same_local_pool(value, kernel_size, "max")
    variance = (_same_local_pool(value.square(), kernel_size, "mean") - mean.square()).clamp_min(0)
    ring_mean = _ring_pool(value, kernel_size)
    return torch.cat((value, maximum, mean, variance.sqrt(), value - ring_mean), dim=1)


@torch.no_grad()
def dense_full_image_descriptor(
    predictor,
    image: np.ndarray,
    score: np.ndarray,
    crop_size: int,
    local_contrast_sigma: float,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Approximate the ROI descriptor convolutionally at every stride-4 feature cell."""
    predictor.set_image(image)
    named = get_named_sim_features(predictor, align=False)
    target_size = max((feature.shape[-2:] for _, feature in named), key=lambda size: size[0] * size[1])
    aligned = [
        feature.float() if feature.shape[-2:] == target_size else F.interpolate(
            feature.float(), size=target_size, mode="bilinear", align_corners=False
        )
        for _, feature in named
    ]
    feature = F.normalize(torch.cat(aligned, dim=1), dim=1)
    height, width = target_size
    scale = 0.5 * (height / image.shape[0] + width / image.shape[1])
    kernel_size = max(3, int(round(crop_size * scale)))

    feature_mean = _same_local_pool(feature, kernel_size, "mean")
    feature_max = _same_local_pool(feature, kernel_size, "max")
    feature_diff = feature - _ring_pool(feature, kernel_size)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    contrast = ev.local_contrast_score(image, local_contrast_sigma)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = ev.robust_norm(cv2.magnitude(gx, gy))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    top_hat = ev.robust_norm(gray - opened)

    def tensor_map(value: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(value).to(feature.device, dtype=torch.float32)[None, None]
        return F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)

    score_tensor = tensor_map(score.astype(np.float32))
    sim_stats = _dense_map_stats(score_tensor, kernel_size)
    sim = torch.cat((sim_stats, sim_stats[:, 1:2] - sim_stats[:, 2:3]), dim=1)
    hand = torch.cat([
        _dense_map_stats(tensor_map(value), kernel_size)
        for value in (gray, contrast, gradient, top_hat)
    ], dim=1)
    descriptor = torch.cat((sim, feature, feature_mean, feature_max, feature_diff, hand), dim=1)
    return descriptor, target_size


@torch.no_grad()
def score_dense_descriptor(
    model,
    descriptor: torch.Tensor,
    checkpoint: dict,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Append F4 prototype similarities, normalize, and score a dense descriptor map."""
    _, channels_without_proto, height, width = descriptor.shape
    feature_channels = (channels_without_proto - 26) // 4
    center = descriptor[:, 6 : 6 + feature_channels]
    mean_start = 6 + feature_channels
    mean = descriptor[:, mean_start : mean_start + feature_channels]
    target = torch.as_tensor(
        checkpoint["prototype_target"], device=descriptor.device, dtype=torch.float32
    )
    background = torch.as_tensor(
        checkpoint["prototype_background"], device=descriptor.device, dtype=torch.float32
    )
    center_norm = F.normalize(center, dim=1)
    mean_norm = F.normalize(mean, dim=1)
    tc = torch.einsum("bchw,c->bhw", center_norm, target)[:, None]
    tm = torch.einsum("bchw,c->bhw", mean_norm, target)[:, None]
    bc = torch.einsum("bchw,c->bhw", center_norm, background)[:, None]
    bm = torch.einsum("bchw,c->bhw", mean_norm, background)[:, None]
    full = torch.cat((descriptor, tc, tm, bc, bm, tc - bc, tm - bm), dim=1)
    flat = full.permute(0, 2, 3, 1).reshape(-1, full.shape[1])
    mean_value = torch.as_tensor(checkpoint["mean"], device=flat.device, dtype=torch.float32)
    std_value = torch.as_tensor(checkpoint["std"], device=flat.device, dtype=torch.float32)
    outputs = []
    for start in range(0, len(flat), batch_size):
        outputs.append(model((flat[start : start + batch_size] - mean_value) / std_value))
    return torch.cat(outputs).reshape(height, width)
