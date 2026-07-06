import argparse
import csv
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from visual_prompt.core.image_utils import normalize_map, overlay_mask
from visual_prompt.core.mask_utils import binary_metrics, component_at_seed, support_area_prior, top_area_mask
from visual_prompt.core.metrics import PD_FAmeter, mIoUmeter
from visual_prompt.data import ImageFolder
from visual_prompt.data.irstd import RawIRSTDDataset, build_raw_support_paths, dilated_hit, distance_to_gt, raw_irstd_paths
from visual_prompt.core.semantic_direction import SemanticDirectionConfig, parse_optional_float, parse_optional_int
from visual_prompt.backbones.mobile_sam import DEFAULT_CHECKPOINT, SamPredictor, sam_model_registry
from visual_prompt.backbones.sam2_adapter import (
    DEFAULT_SAM2_BPLUS_CHECKPOINT,
    DEFAULT_SAM2_BPLUS_CONFIG,
    Sam2SimilarityPredictor,
    build_sam2_bplus,
)
from visual_prompt.pipelines.semantic_prompt_pipeline import (
    compute_sim_from_features,
    get_named_sim_features,
    get_sim_features,
    load_rgb_and_mask,
    register_sim_feature_hooks,
    seed_everything,
)

DEFAULT_SUPPORT_IDS = "XDU514,XDU213,XDU168,XDU445,XDU347"


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def crop_with_padding(image, mask, y0, x0, size):
    h, w = image.shape[:2]
    y1 = y0 + size
    x1 = x0 + size
    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_y1 = min(h, y1)
    src_x1 = min(w, x1)
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0

    image_crop = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    mask_crop = np.zeros((size, size), dtype=mask.dtype)
    image_crop[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = image[
        src_y0:src_y1, src_x0:src_x1
    ]
    mask_crop[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = mask[src_y0:src_y1, src_x0:src_x1]

    if src_y0 > y0 or src_x0 > x0 or src_y1 < y1 or src_x1 < x1:
        valid = np.zeros((size, size), dtype=np.uint8)
        valid[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = 1
        image_crop = cv2.inpaint(image_crop, 1 - valid, 3, cv2.INPAINT_TELEA)
    return image_crop, mask_crop


def crop_support_target(image, mask, window_size):
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        cy, cx = image.shape[0] // 2, image.shape[1] // 2
    else:
        cy = int(round(float(ys.mean())))
        cx = int(round(float(xs.mean())))
    y0 = cy - window_size // 2
    x0 = cx - window_size // 2
    return crop_with_padding(image, mask, y0, x0, window_size)


def parse_float_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    value = str(value).strip()
    if not value or value.lower() in {"none", "null", "off"}:
        return []
    return [float(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]


def safe_linear_contrast(image, requested_scale):
    if requested_scale <= 0:
        raise ValueError(f"Support value augmentation scale must be positive, got {requested_scale}")
    image_f = image.astype(np.float32)
    if abs(requested_scale - 1.0) <= 1e-8:
        return image.copy(), 1.0

    lo = float(image_f.min())
    hi = float(image_f.max())
    if hi <= lo:
        return image.copy(), 1.0

    pivot = 0.5 * (lo + hi)
    effective_scale = float(requested_scale)
    if requested_scale > 1.0:
        max_scale = np.inf
        if pivot > lo:
            max_scale = min(max_scale, pivot / (pivot - lo))
        if hi > pivot:
            max_scale = min(max_scale, (255.0 - pivot) / (hi - pivot))
        if np.isfinite(max_scale):
            effective_scale = min(effective_scale, max(1.0, max_scale * (1.0 - 1e-6)))

    transformed = pivot + effective_scale * (image_f - pivot)
    if float(transformed.min()) < -1e-3 or float(transformed.max()) > 255.0 + 1e-3:
        raise RuntimeError(
            f"Unsafe support contrast transform: requested={requested_scale} "
            f"effective={effective_scale} range=({transformed.min()}, {transformed.max()})"
        )
    return np.rint(transformed).astype(np.uint8), effective_scale


def append_value_aug_items(items, image_id, image, mask, scales):
    items.append(
        {
            "image_id": image_id,
            "variant_id": "orig",
            "image": image,
            "mask": mask,
            "requested_scale": 1.0,
            "effective_scale": 1.0,
        }
    )
    for scale in scales:
        if abs(scale - 1.0) <= 1e-8:
            continue
        aug_image, effective_scale = safe_linear_contrast(image, scale)
        items.append(
            {
                "image_id": image_id,
                "variant_id": f"scale_{scale:g}",
                "image": aug_image,
                "mask": mask,
                "requested_scale": float(scale),
                "effective_scale": float(effective_scale),
            }
        )


def build_support_value_items(support_paths, image_size, value_aug_scales):
    scales = parse_float_list(value_aug_scales)
    items = []
    for image_path, mask_path in support_paths:
        image, mask, _ = load_rgb_and_mask(image_path, mask_path, (image_size, image_size))
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        append_value_aug_items(items, image_id, image, mask, scales)
    return items


def build_support_crop_value_items(support_paths, image_size, window_size, value_aug_scales):
    scales = parse_float_list(value_aug_scales)
    items = []
    for image_path, mask_path in support_paths:
        image, mask, _ = load_rgb_and_mask(image_path, mask_path, (image_size, image_size))
        crop_image, crop_mask = crop_support_target(image, mask, window_size)
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        append_value_aug_items(items, image_id, crop_image, crop_mask, scales)
    return items


def summarize_support_value_items(items):
    aug_items = [item for item in items if item["variant_id"] != "orig"]
    if not aug_items:
        return "none"
    parts = []
    for item in aug_items:
        parts.append(f"{item['image_id']}:{item['requested_scale']:g}->{item['effective_scale']:.4g}")
    return ";".join(parts)


def sliding_starts(length, window_size, stride):
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    last = length - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def maybe_normalize_window(score, mode):
    if mode == "none":
        return score
    if mode == "percentile":
        return normalize_map(score)
    if mode == "zscore":
        return (score - float(score.mean())) / (float(score.std()) + 1e-6)
    raise ValueError(f"Unknown window normalization mode: {mode}")


def reduce_window_score(score, mode, topk_fraction):
    flat = score.reshape(-1)
    if mode == "mean":
        return float(flat.mean())
    if mode == "max":
        return float(flat.max())
    if mode == "topk_mean":
        k = int(max(1, round(flat.size * topk_fraction)))
        k = min(k, flat.size)
        idx = np.argpartition(flat, -k)[-k:]
        return float(flat[idx].mean())
    raise ValueError(f"Unknown window reduce mode: {mode}")


def score_stats(score, mask):
    top_idx = int(np.argmax(score.reshape(-1)))
    top_y, top_x = np.unravel_index(top_idx, score.shape)
    hit = bool(mask[top_y, top_x] > 0)
    hit_dilate3 = bool(hit or dilated_hit(mask, top_y, top_x, radius=3))
    return {
        "top_x": int(top_x),
        "top_y": int(top_y),
        "top_score": float(score[top_y, top_x]),
        "top_hit": hit,
        "top_hit_dilate3": hit_dilate3,
        "top_dist_to_gt": distance_to_gt(mask, top_y, top_x),
    }


def save_result_panel(output_dir, sample_id, image, gt, score, pred, max_side=900):
    os.makedirs(output_dir, exist_ok=True)
    score_norm = normalize_map(score)
    heat = cv2.applyColorMap((score_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.55 * image + 0.45 * heat, 0, 255).astype(np.uint8)
    gt_overlay = overlay_mask(image, gt, color=(0, 255, 80), alpha=0.45)
    pred_overlay = overlay_mask(image, pred, color=(255, 60, 60), alpha=0.45)
    panel = np.concatenate([gt_overlay, overlay, pred_overlay], axis=1)
    if panel.shape[1] > max_side:
        scale = max_side / panel.shape[1]
        panel = cv2.resize(panel, (int(panel.shape[1] * scale), int(panel.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(output_dir, f"{sample_id}_sliding_panel.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def save_iter_debug_panel(
    output_dir,
    sample_id,
    image,
    gt,
    initial_score,
    refined_score,
    candidates,
    pred,
    suppression=None,
    protection=None,
    max_side=1600,
):
    os.makedirs(output_dir, exist_ok=True)

    def heat_overlay(score):
        score_norm = normalize_map(score)
        heat = cv2.applyColorMap((score_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        return np.clip(0.55 * image + 0.45 * heat, 0, 255).astype(np.uint8)

    if suppression is None:
        suppression = np.zeros(gt.shape, dtype=np.float32)
    if protection is None:
        protection = np.zeros(gt.shape, dtype=np.float32)
    suppression_heat = cv2.applyColorMap((minmax_norm(suppression) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    suppression_overlay = np.clip(0.60 * image + 0.40 * cv2.cvtColor(suppression_heat, cv2.COLOR_BGR2RGB), 0, 255).astype(
        np.uint8
    )
    protection_heat = cv2.applyColorMap((minmax_norm(protection) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    candidate_overlay = np.clip(0.60 * image + 0.40 * cv2.cvtColor(protection_heat, cv2.COLOR_BGR2RGB), 0, 255).astype(
        np.uint8
    )
    for item in candidates[:12]:
        cv2.circle(candidate_overlay, (int(item["seed_x"]), int(item["seed_y"])), 3, (255, 255, 255), 1)

    panels = [
        overlay_mask(image, gt, color=(0, 255, 80), alpha=0.45),
        heat_overlay(initial_score),
        suppression_overlay,
        candidate_overlay,
        heat_overlay(refined_score),
        overlay_mask(image, pred, color=(255, 60, 60), alpha=0.45),
    ]
    panel = np.concatenate(panels, axis=1)
    if panel.shape[1] > max_side:
        scale = max_side / panel.shape[1]
        panel = cv2.resize(panel, (int(panel.shape[1] * scale), int(panel.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(output_dir, f"{sample_id}_iter_debug.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def save_iter_stage_panel(output_dir, sample_id, image, stage_scores, stage_masks, max_side=1800):
    if not stage_scores:
        return
    os.makedirs(output_dir, exist_ok=True)
    panels = []
    for name in stage_scores:
        score_norm = minmax_norm(stage_scores[name])
        heat = cv2.applyColorMap((score_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        score_panel = np.clip(0.55 * image + 0.45 * heat, 0, 255).astype(np.uint8)
        cv2.putText(score_panel, f"{name} score", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        panels.append(score_panel)
        mask = stage_masks.get(name, np.zeros(score_norm.shape, dtype=np.float32))
        mask_heat = cv2.applyColorMap((minmax_norm(mask) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        mask_heat = cv2.cvtColor(mask_heat, cv2.COLOR_BGR2RGB)
        mask_panel = np.clip(0.60 * image + 0.40 * mask_heat, 0, 255).astype(np.uint8)
        cv2.putText(mask_panel, f"{name} soft", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        panels.append(mask_panel)
    panel = np.concatenate(panels, axis=1)
    if panel.shape[1] > max_side:
        scale = max_side / panel.shape[1]
        panel = cv2.resize(panel, (int(panel.shape[1] * scale), int(panel.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(output_dir, f"{sample_id}_iter_stages.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def titled_panel(panel, title, title_height=20):
    canvas = np.zeros((panel.shape[0] + title_height, panel.shape[1], 3), dtype=np.uint8)
    canvas[title_height:] = panel
    cv2.putText(canvas, title, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)
    return canvas


def score_heat_panel(image, score, title):
    heat = cv2.applyColorMap((minmax_norm(score) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    panel = np.clip(0.55 * image + 0.45 * heat, 0, 255).astype(np.uint8)
    return titled_panel(panel, title)


def mask_heat_panel(image, mask, title):
    heat = cv2.applyColorMap((minmax_norm(mask) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    panel = np.clip(0.62 * image + 0.38 * heat, 0, 255).astype(np.uint8)
    return titled_panel(panel, title)


def binary_overlay_panel(image, mask, title, color=(255, 60, 60)):
    return titled_panel(overlay_mask(image, mask, color=color, alpha=0.45), title)


def anchor_map_panel(shape, branches, title):
    panel = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    for rank, branch in enumerate(branches):
        y, x = int(branch["seed_y"]), int(branch["seed_x"])
        value = 255 - min(rank, 5) * 28
        panel[max(0, y - 1) : min(shape[0], y + 2), x : min(shape[1], x + 1)] = (value, value, value)
        panel[y : min(shape[0], y + 1), max(0, x - 1) : min(shape[1], x + 2)] = (value, value, value)
    return titled_panel(panel, title)


def save_branch_flow_panels(output_dir, sample_id, image, initial_score, refined_score, branches, support_area, args, max_side=2200):
    if not branches:
        return
    os.makedirs(output_dir, exist_ok=True)
    ranked = sorted(branches, key=lambda item: item.get("branch_score", 0.0), reverse=True)
    overview_panels = [
        score_heat_panel(image, initial_score, "initial fused"),
        anchor_map_panel(initial_score.shape, ranked, "anchors only"),
        score_heat_panel(image, refined_score, "refined fused"),
    ]
    overview = np.concatenate(overview_panels, axis=1)
    if overview.shape[1] > max_side:
        scale = max_side / overview.shape[1]
        overview = cv2.resize(overview, (int(overview.shape[1] * scale), int(overview.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(output_dir, f"{sample_id}_branch_overview.png"), cv2.cvtColor(overview, cv2.COLOR_RGB2BGR))

    for rank, branch in enumerate(ranked[: args.ctf_debug_max_branches]):
        stats = score_stats(branch["score"], np.zeros(branch["score"].shape, dtype=np.uint8))
        _, mask_info = extract_final_mask(branch["score"], image, support_area, stats, args)
        panels = [
            score_heat_panel(
                image,
                branch["score"],
                (
                    f"b{rank} fused s={branch.get('branch_score', 0.0):.2f} "
                    f"q={branch.get('mask_quality', 0.0):.2f} c={branch.get('layer_consistency', 0.0):.2f}"
                ),
            ),
            binary_overlay_panel(image, mask_info["pred"], f"b{rank} final area x{mask_info['area_scale']:.2g}"),
        ]
        for layer_name in branch["stage_scores"]:
            panels.append(score_heat_panel(image, branch["stage_scores"][layer_name], f"{layer_name} score"))
            panels.append(mask_heat_panel(image, branch["stage_masks"][layer_name], f"{layer_name} soft"))
        row = np.concatenate(panels, axis=1)
        if row.shape[1] > max_side:
            scale = max_side / row.shape[1]
            row = cv2.resize(row, (int(row.shape[1] * scale), int(row.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(output_dir, f"{sample_id}_branch_{rank:02d}_flow.png"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))


def save_final_masks_panel(output_dir, sample_id, image, final_mask_info, max_side=1600):
    kept = final_mask_info.get("kept_masks", [])
    if not kept:
        return
    os.makedirs(output_dir, exist_ok=True)
    panels = []
    for index, item in enumerate(kept[:8]):
        title = f"m{index} q={item['quality']:.2f} x{item['area_scale']:g}"
        panels.append(binary_overlay_panel(image, item["pred"], title))
    panels.append(binary_overlay_panel(image, final_mask_info["pred"], f"union n={len(kept)}", color=(255, 60, 60)))
    panel = np.concatenate(panels, axis=1)
    if panel.shape[1] > max_side:
        scale = max_side / panel.shape[1]
        panel = cv2.resize(panel, (int(panel.shape[1] * scale), int(panel.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(output_dir, f"{sample_id}_final_masks.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def save_layer_score_panel(output_dir, sample_id, image, layer_debug, max_side=2200):
    if not layer_debug:
        return
    os.makedirs(output_dir, exist_ok=True)
    panels = []
    for name, item in layer_debug.items():
        native = np.squeeze(item["native"])
        post = np.squeeze(item["post"])
        native_vis = cv2.resize(minmax_norm(native), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        panels.append(mask_heat_panel(image, native_vis, f"{name} native {native.shape[0]}x{native.shape[1]}"))
        panels.append(score_heat_panel(image, post, f"{name} post"))
    panel = np.concatenate(panels, axis=1)
    if panel.shape[1] > max_side:
        scale = max_side / panel.shape[1]
        panel = cv2.resize(panel, (int(panel.shape[1] * scale), int(panel.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(output_dir, f"{sample_id}_layer_scores.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def minmax_norm(score):
    score = score.astype(np.float32)
    lo, hi = float(score.min()), float(score.max())
    if hi <= lo:
        return np.zeros_like(score, dtype=np.float32)
    return ((score - lo) / (hi - lo)).astype(np.float32)


def robust_norm(score, low=1.0, high=99.5):
    score = score.astype(np.float32)
    lo, hi = np.percentile(score, [low, high])
    if hi <= lo:
        return minmax_norm(score)
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def local_contrast_score(image, sigma):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return robust_norm(gray - local_mean, low=50.0, high=99.7)


def build_mask_score(image, sim_score, args):
    if args.mask_score_mode == "sim":
        return sim_score.astype(np.float32)
    sim_norm = minmax_norm(sim_score)
    contrast = local_contrast_score(image, args.local_contrast_sigma)
    if args.mask_score_mode == "sim_times_contrast":
        return sim_norm * contrast
    if args.mask_score_mode == "sim_plus_contrast":
        return (args.mask_sim_weight * sim_norm + (1.0 - args.mask_sim_weight) * contrast).astype(np.float32)
    raise ValueError(f"Unknown mask_score_mode: {args.mask_score_mode}")


def component_at_anchor(score, area, grow_factor, anchor_y, anchor_x, seed_radius):
    candidate = top_area_mask(score, max(1, int(round(area * grow_factor))))
    _, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    seed_y, seed_x = int(anchor_y), int(anchor_x)
    comp_id = labels[seed_y, seed_x]
    if comp_id <= 0:
        h, w = score.shape
        y0 = max(0, seed_y - seed_radius)
        y1 = min(h, seed_y + seed_radius + 1)
        x0 = max(0, seed_x - seed_radius)
        x1 = min(w, seed_x + seed_radius + 1)
        local_labels = labels[y0:y1, x0:x1]
        local_score = np.where(local_labels > 0, score[y0:y1, x0:x1], -np.inf)
        if np.isfinite(local_score).any():
            py, px = np.unravel_index(int(np.argmax(local_score.reshape(-1))), local_score.shape)
            seed_y, seed_x = int(y0 + py), int(x0 + px)
            comp_id = labels[seed_y, seed_x]
        if comp_id <= 0:
            return np.zeros_like(score, dtype=np.uint8), seed_y, seed_x, "no_anchor_component"
    comp = labels == comp_id
    if int(comp.sum()) <= area:
        return comp.astype(np.uint8), seed_y, seed_x, "anchor_component"
    clipped = np.where(comp, score, -np.inf)
    return top_area_mask(clipped, area), seed_y, seed_x, "anchor_component_clipped"


def local_maxima_points(score, topk, min_distance):
    score_f = score.astype(np.float32)
    if score_f.size == 0:
        return []
    kernel = 2 * int(max(1, min_distance)) + 1
    dilated = cv2.dilate(score_f, np.ones((kernel, kernel), dtype=np.uint8))
    peaks = np.logical_and(score_f >= dilated - 1e-8, np.isfinite(score_f))
    ys, xs = np.nonzero(peaks)
    if len(xs) == 0:
        y, x = np.unravel_index(int(np.argmax(score_f.reshape(-1))), score_f.shape)
        return [(int(y), int(x), float(score_f[y, x]))]
    order = np.argsort(score_f[ys, xs])[::-1]
    selected = []
    min_dist_sq = float(min_distance * min_distance)
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if any((y - sy) ** 2 + (x - sx) ** 2 < min_dist_sq for sy, sx, _ in selected):
            continue
        selected.append((y, x, float(score_f[y, x])))
        if len(selected) >= topk:
            break
    return selected


def mask_ring(mask, radius):
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
    return np.logical_and(dilated, mask <= 0)


def candidate_confidence(mask, score, contrast, support_area, args):
    pixels = int(mask.sum())
    if pixels <= 0:
        return None
    area_min = max(1.0, float(support_area) * args.iter_area_min_factor)
    area_max = max(area_min, float(support_area) * args.iter_area_max_factor)
    if pixels < area_min or pixels > area_max:
        return None

    ring = mask_ring(mask, args.iter_ring_radius)
    inside_score = float(score[mask > 0].mean())
    ring_score = float(score[ring].mean()) if ring.any() else float(score.mean())
    inside_contrast = float(contrast[mask > 0].mean())
    ring_contrast = float(contrast[ring].mean()) if ring.any() else float(contrast.mean())
    area_score = float(np.exp(-abs(np.log((pixels + 1e-6) / (float(support_area) + 1e-6)))))
    gap = inside_score - ring_score
    contrast_gap = inside_contrast - ring_contrast
    conf = gap + args.iter_area_weight * area_score + args.iter_contrast_weight * contrast_gap
    return {
        "confidence": float(conf),
        "score_gap": float(gap),
        "contrast_gap": float(contrast_gap),
        "area_score": float(area_score),
        "pixels": pixels,
    }


def audit_candidate_mask(mask, score, contrast, support_area, args):
    pixels = int(mask.sum())
    if pixels <= 0:
        return None, {
            "reason": "empty",
            "violation": 1.0,
            "pixels": pixels,
        }

    ring = mask_ring(mask, args.iter_ring_radius)
    inside_score = float(score[mask > 0].mean())
    ring_score = float(score[ring].mean()) if ring.any() else float(score.mean())
    inside_contrast = float(contrast[mask > 0].mean())
    ring_contrast = float(contrast[ring].mean()) if ring.any() else float(contrast.mean())
    area_min = max(1.0, float(support_area) * args.clean_area_min_factor)
    area_max = max(area_min, float(support_area) * args.clean_area_max_factor)
    area_ratio = pixels / max(float(support_area), 1.0)
    area_score = float(np.exp(-abs(np.log((pixels + 1e-6) / (float(support_area) + 1e-6)))))
    score_gap = inside_score - ring_score
    contrast_gap = inside_contrast - ring_contrast
    conf = score_gap + args.iter_area_weight * area_score + args.iter_contrast_weight * contrast_gap

    failures = []
    violation = 0.0
    if pixels < area_min:
        failures.append("too_small")
        violation += min(1.0, (area_min - pixels) / max(area_min, 1.0))
    if pixels > area_max:
        failures.append("too_large")
        violation += min(1.0, (pixels - area_max) / max(area_max, 1.0))
    if score_gap < args.clean_min_score_gap:
        failures.append("low_score_gap")
        violation += min(1.0, (args.clean_min_score_gap - score_gap) / max(abs(args.clean_min_score_gap), 1e-3))
    if contrast_gap < args.clean_min_contrast_gap:
        failures.append("low_contrast_gap")
        violation += min(
            1.0,
            (args.clean_min_contrast_gap - contrast_gap) / max(abs(args.clean_min_contrast_gap), 1e-3),
        )

    item = {
        "confidence": float(conf),
        "score_gap": float(score_gap),
        "contrast_gap": float(contrast_gap),
        "area_score": float(area_score),
        "area_ratio": float(area_ratio),
        "pixels": pixels,
    }
    if failures:
        return None, {
            **item,
            "reason": ";".join(failures),
            "violation": float(max(violation, 1e-3)),
        }
    return item, None


def nms_mask_candidates(candidates, iou_threshold, max_keep):
    kept = []
    for item in sorted(candidates, key=lambda x: x["confidence"], reverse=True):
        mask = item["mask"] > 0
        duplicate = False
        for prev in kept:
            prev_mask = prev["mask"] > 0
            inter = np.logical_and(mask, prev_mask).sum()
            union = np.logical_or(mask, prev_mask).sum()
            iou = inter / union if union else 0.0
            if iou >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
        if len(kept) >= max_keep:
            break
    return kept


def plausible_mask_candidates(score, image, support_area, args):
    contrast = local_contrast_score(image, args.local_contrast_sigma)
    mask_score = build_mask_score(image, score, args)
    peaks = local_maxima_points(score, args.iter_topk_points, args.iter_peak_min_distance)
    candidates = []
    for seed_y, seed_x, seed_score in peaks:
        for area_scale in parse_float_list(args.iter_area_scales):
            area = int(max(1, round(support_area * area_scale)))
            for grow_factor in parse_float_list(args.iter_grow_factors):
                mask, anchor_y, anchor_x, seed_mode = component_at_anchor(
                    mask_score,
                    area,
                    grow_factor,
                    seed_y,
                    seed_x,
                    args.seed_radius,
                )
                stats = candidate_confidence(mask, score, contrast, support_area, args)
                if stats is None or stats["confidence"] < args.iter_min_confidence:
                    continue
                candidates.append(
                    {
                        **stats,
                        "mask": mask.astype(np.uint8),
                        "seed_y": int(seed_y),
                        "seed_x": int(seed_x),
                        "anchor_y": int(anchor_y),
                        "anchor_x": int(anchor_x),
                        "seed_score": float(seed_score),
                        "seed_mode": seed_mode,
                        "area_scale": float(area_scale),
                        "grow_factor": float(grow_factor),
                    }
                )
    return nms_mask_candidates(candidates, args.iter_nms_iou, args.iter_max_masks)


def candidate_audit_maps(score, image, support_area, args):
    contrast = local_contrast_score(image, args.local_contrast_sigma)
    mask_score = build_mask_score(image, score, args)
    peaks = local_maxima_points(score, args.iter_topk_points, args.iter_peak_min_distance)
    plausible = []
    invalid = []
    for seed_y, seed_x, seed_score in peaks:
        for area_scale in parse_float_list(args.iter_area_scales):
            area = int(max(1, round(support_area * area_scale)))
            for grow_factor in parse_float_list(args.iter_grow_factors):
                mask, anchor_y, anchor_x, seed_mode = component_at_anchor(
                    mask_score,
                    area,
                    grow_factor,
                    seed_y,
                    seed_x,
                    args.seed_radius,
                )
                valid_stats, invalid_stats = audit_candidate_mask(mask, score, contrast, support_area, args)
                base = {
                    "mask": mask.astype(np.uint8),
                    "seed_y": int(seed_y),
                    "seed_x": int(seed_x),
                    "anchor_y": int(anchor_y),
                    "anchor_x": int(anchor_x),
                    "seed_score": float(seed_score),
                    "seed_mode": seed_mode,
                    "area_scale": float(area_scale),
                    "grow_factor": float(grow_factor),
                }
                if valid_stats is not None and valid_stats["confidence"] >= args.iter_min_confidence:
                    plausible.append({**base, **valid_stats})
                elif invalid_stats is not None:
                    invalid.append({**base, **invalid_stats})

    plausible = nms_mask_candidates(plausible, args.iter_nms_iou, args.iter_max_masks)
    protection = np.zeros(score.shape, dtype=np.float32)
    for item in plausible:
        protection = np.maximum(protection, item["mask"].astype(np.float32))

    suppression = np.zeros(score.shape, dtype=np.float32)
    for item in invalid:
        mask = item["mask"].astype(np.float32)
        if mask.max() <= 0:
            continue
        seed_y, seed_x = item["seed_y"], item["seed_x"]
        yy, xx = np.indices(score.shape)
        dist = np.sqrt((yy - seed_y) ** 2 + (xx - seed_x) ** 2).astype(np.float32)
        dist = dist / (float(dist[mask > 0].max()) + 1e-6) if np.any(mask > 0) else dist
        if "too_large" in item.get("reason", ""):
            penalty = mask * (0.35 + 0.65 * dist)
        elif "too_small" in item.get("reason", ""):
            penalty = mask * 0.35
        else:
            penalty = mask * 0.65
        suppression += float(item.get("violation", 1.0)) * penalty
    if suppression.max() > 0:
        suppression = minmax_norm(suppression)
    if protection.max() > 0:
        protection = cv2.GaussianBlur(protection, (0, 0), sigmaX=args_safe_sigma(score.shape), sigmaY=args_safe_sigma(score.shape))
        protection = minmax_norm(protection)
    return plausible, invalid, protection.astype(np.float32), suppression.astype(np.float32)


def candidate_soft_prior(candidates, shape, temperature):
    if not candidates:
        return np.zeros(shape, dtype=np.float32), 0.0
    confs = np.array([item["confidence"] for item in candidates], dtype=np.float32)
    confs = confs - float(confs.max())
    weights = np.exp(confs / max(temperature, 1e-6))
    weights = weights / max(float(weights.sum()), 1e-6)
    prior = np.zeros(shape, dtype=np.float32)
    for weight, item in zip(weights, candidates):
        core_score = item["mask"].astype(np.float32)
        prior += float(weight) * core_score
    if prior.max() > 0:
        prior = cv2.GaussianBlur(prior, (0, 0), sigmaX=args_safe_sigma(shape), sigmaY=args_safe_sigma(shape))
        prior = minmax_norm(prior)
    set_confidence = float(np.sum(weights * np.array([item["confidence"] for item in candidates], dtype=np.float32)))
    return prior.astype(np.float32), set_confidence


def args_safe_sigma(shape):
    return max(0.75, min(shape) / 256.0)


def fuse_layer_scores(layer_score_maps, layer_weights=None):
    names = list(layer_score_maps.keys())
    if not names:
        raise RuntimeError("No layer score maps available for fusion.")
    if layer_weights is None:
        layer_weights = {name: 1.0 / len(names) for name in names}
    fused = np.zeros_like(next(iter(layer_score_maps.values())), dtype=np.float32)
    weight_sum = 0.0
    for name in names:
        weight = float(layer_weights.get(name, 0.0))
        if weight <= 0:
            continue
        fused += weight * minmax_norm(layer_score_maps[name])
        weight_sum += weight
    return fused / max(weight_sum, 1e-6)


def score_top_point(score):
    y, x = np.unravel_index(int(np.argmax(score.reshape(-1))), score.shape)
    return int(y), int(x), float(score[y, x])


def gaussian_anchor_map(shape, seed_y, seed_x, sigma, plateau_radius=0.0):
    yy, xx = np.indices(shape)
    dist = np.sqrt((yy - float(seed_y)) ** 2 + (xx - float(seed_x)) ** 2).astype(np.float32)
    dist = np.maximum(0.0, dist - float(max(0.0, plateau_radius)))
    sigma = max(float(sigma), 1e-3)
    return np.exp(-(dist * dist) / (2.0 * sigma * sigma)).astype(np.float32)


def soft_area_threshold(score, support_area, area_scale):
    flat = score.reshape(-1)
    keep = int(max(1, round(float(support_area) * float(area_scale))))
    keep = min(keep, flat.size)
    if keep >= flat.size:
        return float(flat.min())
    return float(np.partition(flat, flat.size - keep)[flat.size - keep])


def make_soft_mask(score, support_area, args):
    score_norm = minmax_norm(score)
    threshold = soft_area_threshold(score_norm, support_area, args.ctf_soft_area_scale)
    temp = max(float(args.ctf_soft_temperature), 1e-4)
    soft = 1.0 / (1.0 + np.exp(-(score_norm - threshold) / temp))
    if args.ctf_soft_blur_sigma > 0:
        soft = cv2.GaussianBlur(
            soft.astype(np.float32),
            (0, 0),
            sigmaX=args.ctf_soft_blur_sigma,
            sigmaY=args.ctf_soft_blur_sigma,
        )
    return minmax_norm(soft).astype(np.float32)


def mask_gate_from_soft(mask, args):
    gate = mask.astype(np.float32)
    if args.ctf_gate_blur_sigma > 0:
        gate = cv2.GaussianBlur(gate, (0, 0), sigmaX=args.ctf_gate_blur_sigma, sigmaY=args.ctf_gate_blur_sigma)
    if gate.max() > 0:
        gate = minmax_norm(gate)
    return (args.ctf_gate_floor + (1.0 - args.ctf_gate_floor) * gate).astype(np.float32)


def refine_score_with_gate(score, gate, suppression, args):
    score_norm = minmax_norm(score)
    suppress = minmax_norm(suppression) if suppression is not None else np.zeros_like(score_norm, dtype=np.float32)
    cleaned = score_norm * gate
    cleaned = cleaned * np.clip(1.0 - args.ctf_suppression_beta * suppress, 0.0, 1.0)
    return minmax_norm(cleaned).astype(np.float32)


def compute_layer_score_maps(predictor, image, layer_scores, args):
    maps = {}
    windows_per_image = 0
    for layer_name, token in layer_scores.items():
        layer_score, windows_per_image = compute_stitched_similarity(predictor, image, token, layer_name, args)
        maps[layer_name] = layer_score.astype(np.float32)
    return maps, windows_per_image


@torch.no_grad()
def compute_layer_debug_scores(predictor, image, layer_scores, args):
    if args.use_sliding_window:
        return {}
    set_crop_image(predictor, image, args)
    named_features = {name: feat.float() for name, feat in get_named_sim_features(predictor, align=False)}
    result = {}
    for layer_name, token in layer_scores.items():
        if layer_name not in named_features:
            continue
        sim_raw, sim_post = compute_sim_from_features(predictor, token, named_features[layer_name], sim_size=0)
        result[layer_name] = {
            "native": sim_raw.detach().cpu().numpy().astype(np.float32),
            "post": sim_post.detach().cpu().numpy().astype(np.float32),
        }
    return result


def ctf_anchor_points(score, args):
    score_norm = minmax_norm(score)
    peaks = local_maxima_points(score_norm, args.ctf_anchor_topk, args.iter_peak_min_distance)
    if not peaks:
        y, x, value = score_top_point(score_norm)
        return [{"seed_y": y, "seed_x": x, "seed_score": value}]
    best = max(value for _, _, value in peaks)
    anchors = []
    for seed_y, seed_x, value in peaks:
        if value < best * args.ctf_anchor_min_rel_score:
            continue
        if value < args.ctf_anchor_min_score:
            continue
        anchors.append({"seed_y": int(seed_y), "seed_x": int(seed_x), "seed_score": float(value)})
    if not anchors:
        seed_y, seed_x, value = peaks[0]
        anchors.append({"seed_y": int(seed_y), "seed_x": int(seed_x), "seed_score": float(value)})
    return anchors[: args.ctf_anchor_topk]


def run_ctf_anchor_branch(anchor, layer_maps, deep_to_shallow, suppression, support_area, args):
    sigma = args.ctf_deep_sigma
    if sigma <= 0:
        sigma = max(2.0, np.sqrt(float(support_area)) * args.ctf_deep_sigma_scale)
    plateau = max(0.0, np.sqrt(float(support_area)) * args.ctf_deep_plateau_scale)
    anchor_y, anchor_x = anchor["seed_y"], anchor["seed_x"]
    prev_soft = gaussian_anchor_map(layer_maps[deep_to_shallow[0]].shape, anchor_y, anchor_x, sigma=sigma, plateau_radius=plateau)

    updated_maps = {}
    stage_masks = {}
    for layer_index, layer_name in enumerate(deep_to_shallow):
        raw_score = layer_maps[layer_name]
        gate = mask_gate_from_soft(prev_soft, args)
        cleaned = refine_score_with_gate(raw_score, gate, suppression, args)
        soft = make_soft_mask(cleaned, support_area, args)

        anchor_keep = gaussian_anchor_map(
            raw_score.shape,
            anchor_y,
            anchor_x,
            sigma=max(1.0, sigma * (args.ctf_anchor_decay ** layer_index)),
            plateau_radius=max(0.0, plateau * (args.ctf_anchor_decay ** layer_index)),
        )
        soft = np.maximum(soft, args.ctf_anchor_keep * anchor_keep).astype(np.float32)
        if suppression.max() > 0:
            soft = soft * np.clip(1.0 - args.ctf_suppression_beta * minmax_norm(suppression), 0.0, 1.0)
        soft = minmax_norm(soft)

        updated_maps[layer_name] = cleaned
        stage_masks[layer_name] = soft
        prev_soft = soft

    fused = fuse_layer_scores({name: updated_maps[name] for name in layer_maps.keys() if name in updated_maps})
    return {
        **anchor,
        "score": fused.astype(np.float32),
        "stage_scores": updated_maps,
        "stage_masks": stage_masks,
    }


def branch_layer_consistency(stage_masks, support_area, args):
    masks = [minmax_norm(mask) for _, mask in stage_masks.items()]
    if len(masks) < 2:
        return 0.0
    binaries = []
    centroids = []
    for mask in masks:
        threshold = soft_area_threshold(mask, support_area, args.ctf_consistency_area_scale)
        binary = mask >= threshold
        if not binary.any():
            continue
        ys, xs = np.nonzero(binary)
        binaries.append(binary)
        centroids.append((float(ys.mean()), float(xs.mean())))
    if len(binaries) < 2:
        return 0.0

    ious = []
    for prev, cur in zip(binaries[:-1], binaries[1:]):
        inter = np.logical_and(prev, cur).sum()
        union = np.logical_or(prev, cur).sum()
        ious.append(float(inter / union) if union else 0.0)
    shifts = []
    for prev, cur in zip(centroids[:-1], centroids[1:]):
        shifts.append(float(np.sqrt((prev[0] - cur[0]) ** 2 + (prev[1] - cur[1]) ** 2)))
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_shift = float(np.mean(shifts)) if shifts else 0.0
    shift_score = float(np.exp(-mean_shift / max(args.ctf_consistency_shift_scale, 1e-6)))
    return 0.5 * mean_iou + 0.5 * shift_score


def branch_context_stats(score, seed_y, seed_x, support_area, args):
    score_norm = minmax_norm(score)
    radius = max(1.0, np.sqrt(float(support_area)))
    core = gaussian_anchor_map(
        score_norm.shape,
        seed_y,
        seed_x,
        sigma=max(1.0, radius * args.ctf_branch_core_sigma_scale),
        plateau_radius=max(0.0, radius * args.ctf_branch_core_plateau_scale),
    )
    yy, xx = np.indices(score_norm.shape)
    dist = np.sqrt((yy - float(seed_y)) ** 2 + (xx - float(seed_x)) ** 2).astype(np.float32)
    ring = np.logical_and(
        dist >= radius * args.ctf_branch_context_inner_scale,
        dist <= radius * args.ctf_branch_context_outer_scale,
    )
    context = dist <= radius * args.ctf_branch_context_outer_scale
    core_mean = float((score_norm * core).sum() / max(float(core.sum()), 1e-6))
    ring_mean = float(score_norm[ring].mean()) if ring.any() else float(score_norm.mean())
    isolation = core_mean - ring_mean
    high_pixels = int(np.logical_and(context, score_norm >= args.ctf_branch_context_high_threshold).sum())
    high_area_ratio = high_pixels / max(float(support_area), 1.0)
    clutter_penalty = max(
        0.0,
        (high_area_ratio - args.ctf_branch_context_max_area_factor)
        / max(args.ctf_branch_context_max_area_factor, 1e-6),
    )
    clutter_penalty = float(min(1.0, clutter_penalty))
    isolation_score = float(np.clip((isolation - args.ctf_branch_min_isolation) / max(1.0 - args.ctf_branch_min_isolation, 1e-6), 0.0, 1.0))
    return {
        "context_core_mean": core_mean,
        "context_ring_mean": ring_mean,
        "context_isolation": float(isolation),
        "context_isolation_score": isolation_score,
        "context_high_area_ratio": float(high_area_ratio),
        "context_clutter_penalty": clutter_penalty,
    }


def score_ctf_branch(branch, image, support_area, args):
    candidates, invalid, protection, suppression = candidate_audit_maps(branch["score"], image, support_area, args)
    if candidates:
        confidence = float(np.mean([item["confidence"] for item in candidates]))
    else:
        confidence = 0.0
    stats = score_stats(branch["score"], np.zeros_like(branch["score"], dtype=np.uint8))
    _, mask_info = extract_final_mask(branch["score"], image, support_area, stats, args)
    consistency = branch_layer_consistency(branch["stage_masks"], support_area, args)
    context = branch_context_stats(branch["score"], branch["seed_y"], branch["seed_x"], support_area, args)
    invalid_penalty = min(1.0, len(invalid) / max(1.0, float(args.iter_topk_points)))
    branch_score = (
        args.ctf_branch_confidence_weight * confidence
        + args.ctf_branch_mask_quality_weight * mask_info["quality"]
        + args.ctf_branch_consistency_weight * consistency
        + args.ctf_branch_isolation_weight * context["context_isolation_score"]
        + args.ctf_branch_score_weight * branch["seed_score"]
        - args.ctf_branch_invalid_weight * invalid_penalty
        - args.ctf_branch_clutter_weight * context["context_clutter_penalty"]
    )
    return {
        **branch,
        **context,
        "confidence": confidence,
        "branch_score_raw": float(branch_score),
        "branch_score": float(branch_score),
        "mask_quality": float(mask_info["quality"]),
        "mask_area_scale": float(mask_info["area_scale"]),
        "layer_consistency": float(consistency),
        "invalid_penalty": float(invalid_penalty),
        "candidates": candidates,
        "invalid": invalid,
        "protection": protection,
        "suppression": suppression,
    }


def normalize_branch_values(branches, key):
    values = np.array([float(item.get(key, 0.0)) for item in branches], dtype=np.float32)
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo + 1e-8:
        return np.full_like(values, 0.5, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def assign_adaptive_branch_scores(branches, args):
    if not branches:
        return branches
    if args.ctf_branch_score_mode == "raw":
        return branches
    seed = normalize_branch_values(branches, "seed_score")
    confidence = normalize_branch_values(branches, "confidence")
    mask_quality = normalize_branch_values(branches, "mask_quality")
    consistency = normalize_branch_values(branches, "layer_consistency")
    isolation = normalize_branch_values(branches, "context_isolation_score")
    invalid = normalize_branch_values(branches, "invalid_penalty")
    clutter = normalize_branch_values(branches, "context_clutter_penalty")
    for index, branch in enumerate(branches):
        score = (
            args.ctf_branch_confidence_weight * float(confidence[index])
            + args.ctf_branch_mask_quality_weight * float(mask_quality[index])
            + args.ctf_branch_consistency_weight * float(consistency[index])
            + args.ctf_branch_isolation_weight * float(isolation[index])
            + args.ctf_branch_score_weight * float(seed[index])
            - args.ctf_branch_invalid_weight * float(invalid[index])
            - args.ctf_branch_clutter_weight * float(clutter[index])
        )
        branch["branch_score"] = float(score)
        branch["branch_score_seed_norm"] = float(seed[index])
        branch["branch_score_confidence_norm"] = float(confidence[index])
        branch["branch_score_mask_quality_norm"] = float(mask_quality[index])
        branch["branch_score_consistency_norm"] = float(consistency[index])
        branch["branch_score_isolation_norm"] = float(isolation[index])
        branch["branch_score_invalid_norm"] = float(invalid[index])
        branch["branch_score_clutter_norm"] = float(clutter[index])
    return branches


def fuse_ctf_branches(branches, args):
    if not branches:
        return None, []
    branches = assign_adaptive_branch_scores(branches, args)
    sorted_branches = sorted(
        branches,
        key=lambda item: (item["branch_score"], item["confidence"], item["seed_score"]),
        reverse=True,
    )
    best_score = float(sorted_branches[0]["branch_score"])
    branches = [
        item
        for item in sorted_branches
        if item["branch_score"] >= args.ctf_branch_min_keep_score
        and item["branch_score"] + args.ctf_branch_keep_margin >= best_score
        and item.get("context_isolation", 0.0) >= args.ctf_branch_hard_min_isolation
        and item.get("context_high_area_ratio", 0.0) <= args.ctf_branch_hard_max_area_ratio
    ]
    if not branches:
        branches = sorted_branches[:1]
    branches = branches[: args.ctf_beam_size]
    confs = np.array([item["branch_score"] for item in branches], dtype=np.float32)
    confs = confs - float(confs.max())
    weights = np.exp(confs / max(args.ctf_branch_temperature, 1e-6))
    weights = weights / max(float(weights.sum()), 1e-6)

    fused = np.zeros_like(branches[0]["score"], dtype=np.float32)
    for weight, branch in zip(weights, branches):
        branch_score = minmax_norm(branch["score"])
        if args.ctf_branch_fuse == "mean":
            fused += float(weight) * branch_score
        elif args.ctf_branch_fuse == "max":
            fused = np.maximum(fused, float(weight) * branch_score)
        else:
            raise ValueError(f"Unknown ctf_branch_fuse: {args.ctf_branch_fuse}")
    return minmax_norm(fused).astype(np.float32), branches


def combine_stage_maps(branches, key):
    if not branches:
        return {}
    names = list(branches[0][key].keys())
    combined = {}
    for name in names:
        stacked = [minmax_norm(branch[key][name]) for branch in branches if name in branch[key]]
        if stacked:
            combined[name] = np.max(np.stack(stacked, axis=0), axis=0).astype(np.float32)
    return combined


def proxy_extract_mask(score, image, support_area, args):
    stats = score_stats(score, np.zeros_like(score, dtype=np.uint8))
    mask_score = build_mask_score(image, score, args)
    if args.mask_mode == "top_area":
        return top_area_mask(mask_score, support_area)
    if args.mask_score_mode != "sim":
        pred, _, _, _ = component_at_anchor(
            mask_score,
            support_area,
            args.component_grow_factor,
            stats["top_y"],
            stats["top_x"],
            args.seed_radius,
        )
        return pred
    pred, _, _, _ = component_at_seed(
        score,
        support_area,
        args.component_grow_factor,
        stats["top_y"],
        stats["top_x"],
        args.seed_radius,
    )
    return pred


def mask_quality(mask, score, image, support_area, args):
    pixels = int(mask.sum())
    if pixels <= 0:
        return -np.inf, {
            "pixels": pixels,
            "score_gap": float("-inf"),
            "contrast_gap": float("-inf"),
            "area_score": 0.0,
        }
    contrast = local_contrast_score(image, args.local_contrast_sigma)
    ring = mask_ring(mask, args.iter_ring_radius)
    inside_score = float(score[mask > 0].mean())
    ring_score = float(score[ring].mean()) if ring.any() else float(score.mean())
    inside_contrast = float(contrast[mask > 0].mean())
    ring_contrast = float(contrast[ring].mean()) if ring.any() else float(contrast.mean())
    area_score = float(np.exp(-args.final_area_penalty * abs(np.log((pixels + 1e-6) / (float(support_area) + 1e-6)))))
    score_gap = inside_score - ring_score
    contrast_gap = inside_contrast - ring_contrast
    quality = score_gap + args.final_contrast_weight * contrast_gap + args.final_area_weight * area_score
    return float(quality), {
        "pixels": pixels,
        "score_gap": float(score_gap),
        "contrast_gap": float(contrast_gap),
        "area_score": float(area_score),
    }


def extract_mask_for_area(score, mask_score, area, stats, args):
    if args.mask_mode == "top_area":
        return top_area_mask(mask_score, area), stats["top_y"], stats["top_x"], "top_area"
    if args.mask_score_mode != "sim":
        return component_at_anchor(
            mask_score,
            area,
            args.component_grow_factor,
            stats["top_y"],
            stats["top_x"],
            args.seed_radius,
        )
    return component_at_seed(
        score,
        area,
        args.component_grow_factor,
        stats["top_y"],
        stats["top_x"],
        args.seed_radius,
    )


def extract_final_mask(score, image, support_area, stats, args):
    mask_score = build_mask_score(image, score, args)
    scales = parse_float_list(args.final_area_scales)
    if not scales:
        scales = [1.0]
    best = None
    for scale in scales:
        area = int(max(1, round(float(support_area) * float(scale))))
        pred, seed_y, seed_x, seed_mode = extract_mask_for_area(score, mask_score, area, stats, args)
        quality, qstats = mask_quality(pred, score, image, support_area, args)
        item = {
            "pred": pred,
            "seed_y": seed_y,
            "seed_x": seed_x,
            "seed_mode": seed_mode,
            "area": area,
            "area_scale": float(scale),
            "quality": quality,
            **qstats,
        }
        if best is None or item["quality"] > best["quality"]:
            best = item
    return mask_score, best


def final_mask_iou(left, right):
    left = left > 0
    right = right > 0
    union = np.logical_or(left, right).sum()
    if union <= 0:
        return 0.0
    return float(np.logical_and(left, right).sum() / union)


def mask_centroid(mask):
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def mask_center_distance(left, right):
    left_centroid = mask_centroid(left)
    right_centroid = mask_centroid(right)
    if left_centroid is None or right_centroid is None:
        return np.inf
    return float(np.sqrt((left_centroid[0] - right_centroid[0]) ** 2 + (left_centroid[1] - right_centroid[1]) ** 2))


def append_final_mask_candidates(candidates, score, image, support_area, seed_y, seed_x, source, args):
    mask_score = build_mask_score(image, score, args)
    scales = parse_float_list(args.final_area_scales) or [1.0]
    stats = {
        "top_y": int(seed_y),
        "top_x": int(seed_x),
        "top_score": float(score[int(seed_y), int(seed_x)]),
    }
    for scale in scales:
        area = int(max(1, round(float(support_area) * float(scale))))
        pred, out_seed_y, out_seed_x, seed_mode = extract_mask_for_area(score, mask_score, area, stats, args)
        quality, qstats = mask_quality(pred, score, image, support_area, args)
        candidates.append(
            {
                "pred": pred.astype(np.uint8),
                "seed_y": int(out_seed_y),
                "seed_x": int(out_seed_x),
                "seed_mode": seed_mode,
                "area": area,
                "area_scale": float(scale),
                "quality": quality,
                "source": source,
                **qstats,
            }
        )


def cluster_final_mask_candidates(valid, support_area, args):
    clusters = []
    center_threshold = max(args.final_cluster_min_center_distance, np.sqrt(float(support_area)) * args.final_cluster_center_scale)
    for item in sorted(valid, key=lambda x: x["quality"], reverse=True):
        assigned = False
        for cluster in clusters:
            rep = cluster["items"][0]
            if (
                final_mask_iou(item["pred"], rep["pred"]) >= args.final_cluster_iou
                or mask_center_distance(item["pred"], rep["pred"]) <= center_threshold
            ):
                cluster["items"].append(item)
                assigned = True
                break
        if not assigned:
            clusters.append({"items": [item]})

    for cluster in clusters:
        items = sorted(cluster["items"], key=lambda x: x["quality"], reverse=True)
        qualities = [float(item["quality"]) for item in items]
        areas = [float(item["pred"].sum()) for item in items]
        centers = [mask_centroid(item["pred"]) for item in items if mask_centroid(item["pred"]) is not None]
        if len(centers) >= 2:
            center_y = float(np.mean([item[0] for item in centers]))
            center_x = float(np.mean([item[1] for item in centers]))
            spread = float(np.mean([np.sqrt((item[0] - center_y) ** 2 + (item[1] - center_x) ** 2) for item in centers]))
        else:
            spread = 0.0
        area_spread = float(np.std(areas) / max(np.mean(areas), 1e-6)) if areas else 0.0
        support_score = float(np.log1p(len(items)) / np.log1p(max(1, args.final_cluster_support_norm)))
        support_score = min(1.0, support_score)
        stability = float(np.exp(-spread / max(center_threshold, 1e-6)) * np.exp(-area_spread))
        score = (
            args.final_cluster_quality_weight * float(max(qualities))
            + args.final_cluster_mean_quality_weight * float(np.mean(qualities))
            + args.final_cluster_support_weight * support_score
            + args.final_cluster_stability_weight * stability
        )
        cluster.update(
            {
                "items": items,
                "rep": items[0],
                "score": float(score),
                "support": len(items),
                "stability": stability,
                "center_spread": spread,
                "area_spread": area_spread,
            }
        )
    return sorted(clusters, key=lambda x: x["score"], reverse=True)


def select_multi_final_masks(candidates, support_area, args):
    valid = [item for item in candidates if int(item["pred"].sum()) > 0 and item["quality"] >= args.final_multi_min_quality]
    if not valid:
        return []
    valid = sorted(valid, key=lambda item: item["quality"], reverse=True)
    if args.final_mask_mode == "auto":
        clusters = cluster_final_mask_candidates(valid, support_area, args)
        if not clusters:
            return [valid[0]]
        best_cluster = clusters[0]
        close_clusters = [
            cluster for cluster in clusters if cluster["score"] + args.final_auto_cluster_delta >= best_cluster["score"]
        ]
        if len(close_clusters) > args.final_auto_max_close_clusters:
            return [best_cluster["rep"]]
        kept_clusters = [best_cluster]
        union = best_cluster["rep"]["pred"].astype(bool)
        for cluster in clusters[1:]:
            item = cluster["rep"]
            if cluster["score"] + args.final_auto_cluster_delta < best_cluster["score"]:
                continue
            if cluster["support"] < args.final_auto_min_cluster_support:
                continue
            if cluster["stability"] < args.final_auto_min_cluster_stability:
                continue
            if any(final_mask_iou(item["pred"], prev["rep"]["pred"]) >= args.final_multi_nms_iou for prev in kept_clusters):
                continue
            next_union = np.logical_or(union, item["pred"].astype(bool))
            next_total = int(next_union.sum())
            if next_total > float(support_area) * args.final_auto_max_total_area_factor:
                continue
            kept_clusters.append(cluster)
            union = next_union
            if len(kept_clusters) >= args.final_multi_max_masks:
                break
        if len(kept_clusters) < args.final_auto_min_masks:
            return [best_cluster["rep"]]
        return [cluster["rep"] for cluster in kept_clusters]
    best_quality = float(valid[0]["quality"])
    kept = []
    for item in valid:
        if item["quality"] + args.final_multi_quality_margin < best_quality:
            continue
        if any(final_mask_iou(item["pred"], prev["pred"]) >= args.final_multi_nms_iou for prev in kept):
            continue
        kept.append(item)
        if len(kept) >= args.final_multi_max_masks:
            break
    return kept


def extract_multi_final_mask(score, image, support_area, args, branches=None, include_peaks=True):
    candidates = []
    if branches:
        for index, branch in enumerate(branches):
            append_final_mask_candidates(
                candidates,
                branch["score"],
                image,
                support_area,
                branch["seed_y"],
                branch["seed_x"],
                f"branch_{index}",
                args,
            )
    if include_peaks or not candidates:
        score_norm = minmax_norm(score)
        peaks = local_maxima_points(score_norm, args.final_mask_topk, args.iter_peak_min_distance)
        if peaks:
            best = max(value for _, _, value in peaks)
            peaks = [
                (seed_y, seed_x, value)
                for seed_y, seed_x, value in peaks
                if value >= best * args.final_mask_min_rel_score and value >= args.final_mask_min_score
            ] or peaks[:1]
        for index, (seed_y, seed_x, _value) in enumerate(peaks):
            append_final_mask_candidates(candidates, score, image, support_area, seed_y, seed_x, f"peak_{index}", args)

    kept = select_multi_final_masks(candidates, support_area, args)
    if not kept:
        stats = score_stats(score, np.zeros(score.shape, dtype=np.uint8))
        mask_score, single = extract_final_mask(score, image, support_area, stats, args)
        single["mask_count"] = 1
        single["candidate_count"] = len(candidates)
        single["kept_masks"] = [single]
        return mask_score, single

    union = np.zeros_like(kept[0]["pred"], dtype=np.uint8)
    for item in kept:
        union = np.maximum(union, item["pred"].astype(np.uint8))
    best = kept[0]
    mask_score = build_mask_score(image, score, args)
    return mask_score, {
        **best,
        "pred": union,
        "area": int(sum(item["area"] for item in kept)),
        "area_scale": ";".join(f"{item['area_scale']:g}" for item in kept),
        "quality": float(np.mean([item["quality"] for item in kept])),
        "score_gap": float(np.mean([item["score_gap"] for item in kept])),
        "contrast_gap": float(np.mean([item["contrast_gap"] for item in kept])),
        "area_score": float(np.mean([item["area_score"] for item in kept])),
        "mask_count": len(kept),
        "candidate_count": len(candidates),
        "kept_masks": kept,
    }


def resolve_final_multi_source(args):
    if args.final_multi_source != "auto":
        return args.final_multi_source
    if args.backbone == "sam2_bplus":
        return "peaks"
    return "branches"


def weighted_feature_token(feature_bchw, weight_hw):
    if weight_hw is None or float(np.max(weight_hw)) <= 0:
        return None
    feature = feature_bchw.float()
    weight = torch.from_numpy(weight_hw.astype(np.float32))[None, None].to(feature.device)
    if weight.shape[-2:] != feature.shape[-2:]:
        weight = torch.nn.functional.interpolate(weight, size=feature.shape[-2:], mode="area")
    weight = weight.clamp_min(0.0)
    denom = weight.sum(dim=(-2, -1), keepdim=False).clamp_min(1e-6)
    token = (feature * weight).sum(dim=(-2, -1)) / denom
    return token / (token.norm(dim=-1, keepdim=True) + 1e-8)


@torch.no_grad()
def refine_layer_tokens_from_masks(predictor, image, layer_scores, protection, suppression, args, set_confidence):
    set_crop_image(predictor, image, args)
    named_features = {name: feat for name, feat in get_named_sim_features(predictor, align=False)}
    refined_tokens = {}
    eta_pos = min(args.token_pos_eta_max, max(0.0, args.token_pos_eta * set_confidence))
    eta_neg = min(args.token_neg_eta_max, max(0.0, args.token_neg_eta * set_confidence))
    for layer_name, base_token in layer_scores.items():
        if layer_name not in named_features:
            continue
        feature = named_features[layer_name]
        pos_token = weighted_feature_token(feature, protection)
        neg_token = weighted_feature_token(feature, suppression * (1.0 - protection))
        token = base_token.float().clone()
        token = token / (token.norm(dim=-1, keepdim=True) + 1e-8)
        updated = (1.0 - eta_pos) * token
        if pos_token is not None:
            updated = updated + eta_pos * pos_token.to(updated.device, dtype=updated.dtype)
        if neg_token is not None and eta_neg > 0:
            updated = updated - eta_neg * neg_token.to(updated.device, dtype=updated.dtype)
        refined_tokens[layer_name] = updated / (updated.norm(dim=-1, keepdim=True) + 1e-8)
    return refined_tokens


@torch.no_grad()
def compute_iterative_refined_score(predictor, image, layer_scores, args, support_area, initial_score=None):
    if initial_score is None:
        initial_score, _ = compute_stitched_similarity(
            predictor,
            image,
            next(iter(layer_scores.values())),
            next(iter(layer_scores.keys())),
            args,
        )
    layer_maps, _ = compute_layer_score_maps(predictor, image, layer_scores, args)
    if not layer_maps:
        return initial_score.astype(np.float32), {
            "iter_accepted_rounds": 0,
            "iter_candidate_count": 0,
            "iter_invalid_count": 0,
            "iter_best_confidence": float("nan"),
            "iter_debug_candidates": [],
            "iter_debug_invalid": [],
            "iter_debug_protection": np.zeros_like(initial_score, dtype=np.float32),
            "iter_debug_suppression": np.zeros_like(initial_score, dtype=np.float32),
            "iter_stage_scores": {},
            "iter_stage_masks": {},
            "iter_debug_branches": [],
            "iter_debug_all_branches": [],
        }

    fused = initial_score.astype(np.float32)
    best_candidates = []
    best_invalid = []
    best_protection = np.zeros_like(fused, dtype=np.float32)
    best_suppression = np.zeros_like(fused, dtype=np.float32)
    best_stage_scores = {}
    best_stage_masks = {}
    best_branches = []
    best_all_branches = []
    accepted_rounds = 0
    deep_to_shallow = list(reversed(list(layer_maps.keys())))
    for _round in range(args.iter_refine_rounds):
        old_anchor_y, old_anchor_x, _ = score_top_point(fused)
        candidates, invalid, protection, suppression = candidate_audit_maps(fused, image, support_area, args)
        anchors = ctf_anchor_points(fused, args)
        branches = []
        for anchor in anchors:
            branch = run_ctf_anchor_branch(anchor, layer_maps, deep_to_shallow, suppression, support_area, args)
            branches.append(score_ctf_branch(branch, image, support_area, args))
        updated_fused, kept_branches = fuse_ctf_branches(branches, args)
        if updated_fused is None:
            break
        branch_candidates = []
        branch_invalid = []
        for branch in kept_branches:
            branch_candidates.extend(branch["candidates"])
            branch_invalid.extend(branch["invalid"])
        updated_candidates, updated_invalid, updated_protection, updated_suppression = candidate_audit_maps(
            updated_fused,
            image,
            support_area,
            args,
        )
        if branch_candidates:
            updated_candidates = nms_mask_candidates(branch_candidates + updated_candidates, args.iter_nms_iou, args.iter_max_masks)
        if branch_invalid:
            updated_invalid = branch_invalid + updated_invalid
        updated_stage_scores = combine_stage_maps(kept_branches, "stage_scores")
        updated_stage_masks = combine_stage_maps(kept_branches, "stage_masks")
        old_mean = float(np.mean([item["confidence"] for item in candidates])) if candidates else 0.0
        new_mean = float(np.mean([item["confidence"] for item in updated_candidates])) if updated_candidates else -np.inf
        old_pred_pixels = int(proxy_extract_mask(fused, image, support_area, args).sum())
        new_pred_pixels = int(proxy_extract_mask(updated_fused, image, support_area, args).sum())
        if old_pred_pixels > 0 and new_pred_pixels < old_pred_pixels * args.iter_min_pred_area_ratio:
            best_candidates = candidates
            best_invalid = invalid
            best_protection = protection
            best_suppression = suppression
            best_stage_scores = updated_stage_scores
            best_stage_masks = updated_stage_masks
            best_branches = kept_branches
            best_all_branches = branches
            break
        if old_pred_pixels > 0 and new_pred_pixels > old_pred_pixels * args.iter_max_pred_area_ratio:
            best_candidates = candidates
            best_invalid = invalid
            best_protection = protection
            best_suppression = suppression
            best_stage_scores = updated_stage_scores
            best_stage_masks = updated_stage_masks
            best_branches = kept_branches
            best_all_branches = branches
            break
        if candidates and new_mean + args.iter_accept_margin < old_mean:
            best_candidates = candidates
            best_invalid = invalid
            best_protection = protection
            best_suppression = suppression
            best_stage_scores = updated_stage_scores
            best_stage_masks = updated_stage_masks
            best_branches = kept_branches
            best_all_branches = branches
            break
        fused = updated_fused
        best_candidates = updated_candidates or candidates
        best_invalid = updated_invalid or invalid
        best_protection = updated_protection
        best_suppression = updated_suppression
        best_stage_scores = updated_stage_scores
        best_stage_masks = updated_stage_masks
        best_branches = kept_branches
        best_all_branches = branches
        accepted_rounds += 1

        new_y, new_x, _ = score_top_point(fused)
        if (new_y - old_anchor_y) ** 2 + (new_x - old_anchor_x) ** 2 <= args.ctf_stable_radius ** 2:
            if accepted_rounds >= max(1, args.ctf_min_rounds):
                break
    info = {
        "iter_accepted_rounds": accepted_rounds,
        "iter_candidate_count": len(best_candidates),
        "iter_invalid_count": len(best_invalid),
        "iter_best_confidence": max([item["confidence"] for item in best_candidates], default=float("nan")),
        "iter_anchor_count": len(best_branches),
        "iter_mean_branch_score": float(np.mean([item["branch_score"] for item in best_branches])) if best_branches else float("nan"),
        "iter_mean_branch_mask_quality": float(np.mean([item["mask_quality"] for item in best_branches])) if best_branches else float("nan"),
        "iter_mean_branch_consistency": float(np.mean([item["layer_consistency"] for item in best_branches])) if best_branches else float("nan"),
        "iter_mean_branch_invalid_penalty": float(np.mean([item["invalid_penalty"] for item in best_branches])) if best_branches else float("nan"),
        "iter_debug_candidates": best_candidates,
        "iter_debug_invalid": best_invalid,
        "iter_debug_protection": best_protection,
        "iter_debug_suppression": best_suppression,
        "iter_stage_scores": best_stage_scores,
        "iter_stage_masks": best_stage_masks,
        "iter_debug_branches": best_branches,
        "iter_debug_all_branches": best_all_branches,
    }
    return fused.astype(np.float32), info


@torch.no_grad()
def build_crop_layer_score_tokens(predictor, crop_support_items, global_support_items, config, negative_scope):
    layer_targets = {}
    layer_negs = {}
    from visual_prompt.core.semantic_direction import build_score_token, extract_support_semantic_tokens

    for item in crop_support_items:
        crop_image = item["image"]
        crop_mask = item["mask"]
        mask_3c = np.repeat(crop_mask[:, :, None] * 255, 3, axis=2).astype(np.uint8)
        mask_pad = predictor.set_image(crop_image, mask_3c)
        for layer_name, layer_feat in get_named_sim_features(predictor, align=False):
            target_token, negative_tokens = extract_support_semantic_tokens(crop_image, crop_mask, mask_pad, layer_feat, config)
            layer_targets.setdefault(layer_name, []).append(target_token)
            layer_negs.setdefault(layer_name, {"high_freq": [], "bright_flat": [], "hard_negative": []})
            if negative_scope == "crop":
                for name, token in negative_tokens.items():
                    layer_negs[layer_name].setdefault(name, []).append(token)

    if negative_scope == "global":
        for item in global_support_items:
            image = item["image"]
            mask = item["mask"]
            mask_3c = np.repeat(mask[:, :, None] * 255, 3, axis=2).astype(np.uint8)
            mask_pad = predictor.set_image(image, mask_3c)
            for layer_name, layer_feat in get_named_sim_features(predictor, align=False):
                _, negative_tokens = extract_support_semantic_tokens(image, mask, mask_pad, layer_feat, config)
                layer_negs.setdefault(layer_name, {"high_freq": [], "bright_flat": [], "hard_negative": []})
                for name, token in negative_tokens.items():
                    layer_negs[layer_name].setdefault(name, []).append(token)

    layer_scores = {}
    layer_infos = {}
    for layer_name, target_tokens in layer_targets.items():
        score_token, info = build_score_token(target_tokens, layer_negs.get(layer_name, {}), config)
        layer_scores[layer_name] = score_token
        layer_infos[layer_name] = info
    return layer_scores, layer_infos


@torch.no_grad()
def build_crop_fused_score_token(predictor, crop_support_items, global_support_items, args):
    target_tokens = []
    missing_ids = []
    negative_tokens = {"high_freq": [], "bright_flat": [], "hard_negative": []}
    from visual_prompt.core.semantic_direction import build_score_token, extract_support_semantic_tokens

    config = build_config(args)
    for item in crop_support_items:
        support_key = f"{item['image_id']}:{item['variant_id']}"
        crop_image = item["image"]
        crop_mask = item["mask"]
        mask_3c = np.repeat(crop_mask[:, :, None] * 255, 3, axis=2).astype(np.uint8)
        mask_pad = predictor.set_image(crop_image, mask_3c)
        token, negs = extract_support_semantic_tokens(crop_image, crop_mask, mask_pad, get_sim_features(predictor), config)
        if token is None:
            missing_ids.append(support_key)
        else:
            target_tokens.append(token)
        if args.support_negative_scope == "crop":
            for name, neg in negs.items():
                negative_tokens.setdefault(name, []).append(neg)

    if args.support_negative_scope == "global":
        for item in global_support_items:
            image = item["image"]
            mask = item["mask"]
            mask_3c = np.repeat(mask[:, :, None] * 255, 3, axis=2).astype(np.uint8)
            mask_pad = predictor.set_image(image, mask_3c)
            _, negs = extract_support_semantic_tokens(image, mask, mask_pad, get_sim_features(predictor), config)
            for name, neg in negs.items():
                negative_tokens.setdefault(name, []).append(neg)

    score_token, info = build_score_token(target_tokens, negative_tokens, config)
    info["valid_target_token_count"] = len(target_tokens)
    info["missing_target_token_count"] = len(missing_ids)
    info["missing_target_support_ids"] = missing_ids
    return score_token, info


@torch.no_grad()
def stitched_similarity(predictor, image, score_token, layer_name, args):
    h, w = image.shape[:2]
    stride = int(round(args.window_size * (1.0 - args.overlap)))
    stride = max(1, stride)
    y_starts = sliding_starts(h, args.window_size, stride)
    x_starts = sliding_starts(w, args.window_size, stride)
    score_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    dummy_mask = np.zeros((h, w), dtype=np.uint8)
    for y0 in y_starts:
        for x0 in x_starts:
            crop, _ = crop_with_padding(image, dummy_mask, y0, x0, args.window_size)
            set_crop_image(predictor, crop, args)
            if layer_name == "fused_all_layers":
                features = get_sim_features(predictor).float()
            else:
                named = {name: feat for name, feat in get_named_sim_features(predictor, align=False)}
                features = named[layer_name].float()
            _, sim_post = compute_sim_from_features(predictor, score_token, features, args.window_sim_size)
            patch = sim_post.detach().cpu().numpy().astype(np.float32)
            if patch.shape != (args.window_size, args.window_size):
                patch = cv2.resize(patch, (args.window_size, args.window_size), interpolation=cv2.INTER_LINEAR)
            patch = maybe_normalize_window(patch, args.window_normalize)
            if args.window_score_mode == "scalar":
                patch = np.full_like(patch, reduce_window_score(patch, args.window_reduce, args.window_topk_fraction))
            score_sum[y0 : y0 + args.window_size, x0 : x0 + args.window_size] += patch
            weight_sum[y0 : y0 + args.window_size, x0 : x0 + args.window_size] += 1.0

    return score_sum / np.maximum(weight_sum, 1e-6), len(y_starts) * len(x_starts)


def use_cuda_amp(predictor, args):
    return bool(args.use_amp and predictor.device.type == "cuda")


def image_to_model_tensor(predictor, image):
    if predictor.model.image_format != "RGB":
        image = image[..., ::-1]
    transformed = predictor.transform.apply_image(image)
    tensor = torch.as_tensor(transformed, device=predictor.device)
    return tensor.permute(2, 0, 1).contiguous()[None, :, :, :]


def set_crop_image(predictor, crop, args):
    if not use_cuda_amp(predictor, args):
        predictor.set_image(crop)
        return
    input_tensor = image_to_model_tensor(predictor, crop)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        predictor.set_torch_image(input_tensor, crop.shape[:2])


def compute_sim_batch_from_features(predictor, target_feat, cur_feat_bchw, sim_size):
    target_feat = target_feat.reshape(-1)
    target_feat = target_feat / (target_feat.norm(dim=0, keepdim=False) + 1e-8)
    feat = cur_feat_bchw
    feat = feat / (feat.norm(dim=1, keepdim=True) + 1e-8)
    sim_raw = torch.einsum("c,bchw->bhw", target_feat, feat).unsqueeze(1)
    if sim_size > 0 and sim_raw.shape[-2:] != (sim_size, sim_size):
        sim_raw = torch.nn.functional.interpolate(
            sim_raw,
            size=(sim_size, sim_size),
            mode="bilinear",
            align_corners=False,
        )
    postprocess = predictor.postprocess_masks if hasattr(predictor, "postprocess_masks") else predictor.model.postprocess_masks
    sim_post = postprocess(sim_raw, predictor.input_size, predictor.original_size)
    return sim_post[:, 0]


def encode_crop_batch(predictor, crops):
    tensors = [image_to_model_tensor(predictor, crop) for crop in crops]
    batch = torch.cat(tensors, dim=0)
    predictor.set_torch_image(batch, crops[0].shape[:2])


@torch.no_grad()
def stitched_similarity_batched(predictor, image, score_token, layer_name, args):
    h, w = image.shape[:2]
    stride = int(round(args.window_size * (1.0 - args.overlap)))
    stride = max(1, stride)
    y_starts = sliding_starts(h, args.window_size, stride)
    x_starts = sliding_starts(w, args.window_size, stride)
    windows = [(y0, x0) for y0 in y_starts for x0 in x_starts]
    score_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    batch_size = max(1, int(args.window_batch_size))

    dummy_mask = np.zeros((h, w), dtype=np.uint8)
    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start : batch_start + batch_size]
        crops = [
            crop_with_padding(image, dummy_mask, y0, x0, args.window_size)[0]
            for y0, x0 in batch_windows
        ]
        encode_crop_batch(predictor, crops)
        if layer_name == "fused_all_layers":
            features = get_sim_features(predictor).float()
        else:
            named = {name: feat for name, feat in get_named_sim_features(predictor, align=False)}
            features = named[layer_name].float()
        sim_posts = compute_sim_batch_from_features(predictor, score_token, features, args.window_sim_size)
        patches = sim_posts.detach().cpu().numpy().astype(np.float32)
        for patch, (y0, x0) in zip(patches, batch_windows):
            if patch.shape != (args.window_size, args.window_size):
                patch = cv2.resize(patch, (args.window_size, args.window_size), interpolation=cv2.INTER_LINEAR)
            patch = maybe_normalize_window(patch, args.window_normalize)
            if args.window_score_mode == "scalar":
                patch = np.full_like(patch, reduce_window_score(patch, args.window_reduce, args.window_topk_fraction))
            score_sum[y0 : y0 + args.window_size, x0 : x0 + args.window_size] += patch
            weight_sum[y0 : y0 + args.window_size, x0 : x0 + args.window_size] += 1.0

    return score_sum / np.maximum(weight_sum, 1e-6), len(windows)


@torch.no_grad()
def compute_stitched_similarity(predictor, image, score_token, layer_name, args):
    if not args.use_sliding_window:
        predictor.set_image(image)
        if layer_name == "fused_all_layers":
            features = get_sim_features(predictor).float()
        else:
            named = {name: feat for name, feat in get_named_sim_features(predictor, align=False)}
            features = named[layer_name].float()
        _, similarity = compute_sim_from_features(
            predictor,
            score_token,
            features,
            args.window_sim_size,
        )
        return similarity.detach().cpu().numpy().astype(np.float32), 1
    if args.window_batch_size <= 1:
        return stitched_similarity(predictor, image, score_token, layer_name, args)
    try:
        if use_cuda_amp(predictor, args):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return stitched_similarity_batched(predictor, image, score_token, layer_name, args)
        return stitched_similarity_batched(predictor, image, score_token, layer_name, args)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"CUDA OOM with window_batch_size={args.window_batch_size}; "
            "falling back to serial window inference for this image."
        )
        return stitched_similarity(predictor, image, score_token, layer_name, args)


def build_config(args):
    return SemanticDirectionConfig(
        use_negative_subspace=args.use_vector_correction,
        use_high_freq_negative=args.use_high_freq_negative,
        edge_method=args.edge_method,
        edge_quantile=parse_optional_float(args.edge_quantile),
        edge_exclude_dilate=args.edge_exclude_dilate,
        use_bright_negative=args.use_bright_negative,
        use_hard_negative=args.use_hard_negative,
        hard_negative_min_pixels=args.hard_negative_min_pixels,
        hard_negative_exclude_known=args.hard_negative_exclude_known,
        bright_threshold_mode=args.bright_threshold_mode,
        bright_quantile=args.bright_quantile,
        bright_flat_edge_quantile=args.bright_flat_edge_quantile,
        negative_subspace_ridge=args.negative_subspace_ridge,
        target_pool=args.target_pool,
        unit_features=args.prototype_pool_unit_features,
        target_rf_radius_scale=args.target_rf_radius_scale,
        target_rf_radius_min=args.target_rf_radius_min,
        target_rf_radius_max=args.target_rf_radius_max,
    )


def parse_id_list(value):
    if value is None:
        return []
    return [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]


def sample_id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def attach_sample_ids(dataset):
    dataset.sample_ids = [sample_id_from_path(path) for path in dataset.images]
    return dataset


def select_positive_support_paths(dataset, k_shot, support_ids=""):
    requested_ids = parse_id_list(support_ids)
    requested_set = set(requested_ids)
    selected = []
    positive_ids = []
    for image_path, mask_path in zip(dataset.images, dataset.masks):
        sample_id = sample_id_from_path(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read support mask: {mask_path}")
        if int((mask >= 127).sum()) <= 0:
            continue
        positive_ids.append(sample_id)
        if requested_ids:
            if sample_id in requested_set:
                selected.append((image_path, mask_path))
        elif len(selected) < k_shot:
            selected.append((image_path, mask_path))

    if requested_ids:
        selected_ids = {sample_id_from_path(path) for path, _ in selected}
        missing = [sample_id for sample_id in requested_ids if sample_id not in selected_ids]
        if missing:
            raise ValueError(f"Support id(s) not found as positive train samples: {','.join(missing)}")
    if len(selected) != k_shot:
        raise ValueError(f"Expected k_shot={k_shot} support ids, got {len(selected)}")
    if not positive_ids:
        raise RuntimeError("No positive support masks found in the training split.")
    return selected


def resolve_data_splits(args):
    if args.dataset:
        train_set = ImageFolder(
            args.data_path,
            args.dataset,
            istraining=True,
            base_size=args.image_size,
            crop_size=args.image_size,
            copy_paste=False,
        )
        test_set = ImageFolder(
            args.data_path,
            args.dataset,
            istraining=False,
            base_size=args.image_size,
            crop_size=args.image_size,
            copy_paste=False,
        )
        attach_sample_ids(test_set)
        support_ids = args.support_ids
        if support_ids == DEFAULT_SUPPORT_IDS:
            support_ids = ""
        support_paths = select_positive_support_paths(train_set, args.k_shot, support_ids=support_ids)
        return support_paths, test_set

    support_paths = build_raw_support_paths(args.data_root, args.k_shot, support_ids=args.support_ids)
    test_set = RawIRSTDDataset(args.data_root, args.split, args.image_size)
    return support_paths, test_set


def parse_args():
    parser = argparse.ArgumentParser("Evaluate crop-prototype sliding-window semantic similarity on IRSTD.")
    parser.add_argument("--data_root", default="dataset/IRSTD-1k", type=str)
    parser.add_argument("--data_path", default="./dataset/set_configs", type=str)
    parser.add_argument("--dataset", default="", type=str, help="Optional ImageFolder dataset name, e.g. NUAA-SIRST or NUDT-SIRST.")
    parser.add_argument("--split", default="test.txt", type=str)
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=111, type=int)
    parser.add_argument("--backbone", default="mobile_sam", choices=["mobile_sam", "sam2_bplus"])
    parser.add_argument("--sam_type", default="vit_t", choices=["vit_t"])
    parser.add_argument("--sam_ckpt", default=str(DEFAULT_CHECKPOINT), type=str)
    parser.add_argument("--sam2_config", default=DEFAULT_SAM2_BPLUS_CONFIG, type=str)
    parser.add_argument("--sam2_ckpt", default=DEFAULT_SAM2_BPLUS_CHECKPOINT, type=str)
    parser.add_argument("--sim_feature_layers", default="all", type=str)
    parser.add_argument("--score_layer", default="fused_all_layers", type=str)
    parser.add_argument("--use_vector_correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_high_freq_negative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_sliding_window", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--window_size", default=128, type=int)
    parser.add_argument("--overlap", default=0.5, type=float)
    parser.add_argument("--window_sim_size", default=256, type=int)
    parser.add_argument("--window_normalize", default="none", choices=["none", "percentile", "zscore"])
    parser.add_argument("--window_score_mode", default="dense", choices=["dense", "scalar"])
    parser.add_argument("--window_reduce", default="max", choices=["max", "mean", "topk_mean"])
    parser.add_argument("--window_topk_fraction", default=0.01, type=float)
    parser.add_argument("--window_batch_size", default=1, type=int, help="CUDA batch size for sliding-window image encoding.")
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=False, help="Use CUDA autocast for sliding-window image encoding.")
    parser.add_argument("--support_negative_scope", default="global", choices=["global", "crop"])
    parser.add_argument("--k_shot", default=5, type=int)
    parser.add_argument("--support_ids", default=DEFAULT_SUPPORT_IDS, type=str)
    parser.add_argument(
        "--support_value_aug_scales",
        default="",
        type=str,
        help=(
            "Comma-separated positive linear contrast scales for support-set token extraction. "
            "The original support image is always included; augmented images use "
            "pivot + scale * (image - pivot) with a per-image safe scale that never exceeds [0, 255]."
        ),
    )
    parser.add_argument(
        "--sample_ids",
        default="",
        type=str,
        help="Comma-separated test sample ids to evaluate; empty means use the split order.",
    )
    parser.add_argument(
        "--allow_sample_ids_outside_split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For raw IRSTD paths, allow explicit sample_ids to be loaded even when they are not listed in split.",
    )
    parser.add_argument("--max_test_samples", default=0, type=int)
    parser.add_argument("--mask_mode", default="component", choices=["component", "top_area"])
    parser.add_argument(
        "--mask_score_mode",
        default="sim",
        choices=["sim", "sim_times_contrast", "sim_plus_contrast"],
        help="Score map used for mask extraction; localization metrics still use the semantic similarity map.",
    )
    parser.add_argument("--local_contrast_sigma", default=7.0, type=float)
    parser.add_argument("--mask_sim_weight", default=0.5, type=float)
    parser.add_argument("--area_prior", default="median", choices=["median", "mean"])
    parser.add_argument("--area_scale", default=1.0, type=float)
    parser.add_argument("--final_area_scales", default="0.75,1.0,1.5,2.0,3.0", type=str)
    parser.add_argument("--final_area_weight", default=0.15, type=float)
    parser.add_argument("--final_contrast_weight", default=0.20, type=float)
    parser.add_argument("--final_area_penalty", default=0.35, type=float)
    parser.add_argument("--final_mask_mode", default="auto", choices=["single", "multi", "auto"])
    parser.add_argument("--final_multi_source", default="auto", choices=["auto", "branches", "peaks", "hybrid"])
    parser.add_argument("--final_multi_max_masks", default=4, type=int)
    parser.add_argument("--final_multi_nms_iou", default=0.30, type=float)
    parser.add_argument("--final_multi_quality_margin", default=0.12, type=float)
    parser.add_argument("--final_multi_min_quality", default=-1e9, type=float)
    parser.add_argument("--final_auto_quality_delta", default=0.10, type=float)
    parser.add_argument("--final_auto_max_total_area_factor", default=3.0, type=float)
    parser.add_argument("--final_auto_min_masks", default=2, type=int)
    parser.add_argument("--final_auto_max_valid_candidates", default=16, type=int)
    parser.add_argument("--final_auto_cluster_delta", default=0.18, type=float)
    parser.add_argument("--final_auto_min_cluster_support", default=1, type=int)
    parser.add_argument("--final_auto_min_cluster_stability", default=0.20, type=float)
    parser.add_argument("--final_auto_max_close_clusters", default=999, type=int)
    parser.add_argument("--final_cluster_iou", default=0.20, type=float)
    parser.add_argument("--final_cluster_center_scale", default=2.5, type=float)
    parser.add_argument("--final_cluster_min_center_distance", default=4.0, type=float)
    parser.add_argument("--final_cluster_support_norm", default=4, type=int)
    parser.add_argument("--final_cluster_quality_weight", default=0.45, type=float)
    parser.add_argument("--final_cluster_mean_quality_weight", default=0.20, type=float)
    parser.add_argument("--final_cluster_support_weight", default=0.20, type=float)
    parser.add_argument("--final_cluster_stability_weight", default=0.15, type=float)
    parser.add_argument("--final_mask_topk", default=6, type=int)
    parser.add_argument("--final_mask_min_rel_score", default=0.70, type=float)
    parser.add_argument("--final_mask_min_score", default=0.30, type=float)
    parser.add_argument("--component_grow_factor", default=8.0, type=float)
    parser.add_argument("--seed_radius", default=3, type=int)
    parser.add_argument("--edge_method", default="canny", choices=["sobel", "canny", "log"])
    parser.add_argument("--edge_quantile", default="none", type=str)
    parser.add_argument("--edge_exclude_dilate", default=5, type=int)
    parser.add_argument("--use_bright_negative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_hard_negative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard_negative_min_pixels", default=1, type=int)
    parser.add_argument("--hard_negative_exclude_known", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bright_threshold_mode", default="target_mean", choices=["target_mean", "global_quantile"])
    parser.add_argument("--bright_quantile", default=0.95, type=float)
    parser.add_argument("--bright_flat_edge_quantile", default=0.50, type=float)
    parser.add_argument("--negative_subspace_ridge", default=0.0, type=float)
    parser.add_argument("--target_pool", default="mean_max", choices=["mean", "max", "mean_max"])
    parser.add_argument("--prototype_pool_unit_features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target_rf_radius_scale", default=1.0, type=float)
    parser.add_argument("--target_rf_radius_min", default=0, type=int)
    parser.add_argument("--target_rf_radius_max", default=None, type=parse_optional_int)
    parser.add_argument("--use_iterative_refine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--iter_refine_rounds", default=1, type=int)
    parser.add_argument("--iter_topk_points", default=12, type=int)
    parser.add_argument("--iter_peak_min_distance", default=12, type=int)
    parser.add_argument("--iter_area_scales", default="0.5,1.0,1.5,2.0", type=str)
    parser.add_argument("--iter_grow_factors", default="4,8,12", type=str)
    parser.add_argument("--iter_area_min_factor", default=0.35, type=float)
    parser.add_argument("--iter_area_max_factor", default=3.0, type=float)
    parser.add_argument("--iter_area_weight", default=0.25, type=float)
    parser.add_argument("--iter_contrast_weight", default=0.25, type=float)
    parser.add_argument("--iter_ring_radius", default=5, type=int)
    parser.add_argument("--iter_min_confidence", default=-0.05, type=float)
    parser.add_argument("--iter_min_set_confidence", default=0.0, type=float)
    parser.add_argument("--iter_candidate_temperature", default=0.2, type=float)
    parser.add_argument("--iter_nms_iou", default=0.65, type=float)
    parser.add_argument("--iter_max_masks", default=8, type=int)
    parser.add_argument("--iter_eta", default=0.25, type=float)
    parser.add_argument("--iter_eta_max", default=0.20, type=float)
    parser.add_argument("--iter_prior_boost", default=0.35, type=float)
    parser.add_argument("--iter_accept_margin", default=0.02, type=float)
    parser.add_argument("--iter_min_pred_area_ratio", default=0.90, type=float)
    parser.add_argument("--iter_max_pred_area_ratio", default=1.80, type=float)
    parser.add_argument("--clean_area_min_factor", default=0.35, type=float)
    parser.add_argument("--clean_area_max_factor", default=3.0, type=float)
    parser.add_argument("--clean_min_score_gap", default=0.02, type=float)
    parser.add_argument("--clean_min_contrast_gap", default=-0.05, type=float)
    parser.add_argument("--clean_suppression_beta", default=0.45, type=float)
    parser.add_argument("--token_pos_eta", default=0.25, type=float)
    parser.add_argument("--token_pos_eta_max", default=0.25, type=float)
    parser.add_argument("--token_neg_eta", default=0.10, type=float)
    parser.add_argument("--token_neg_eta_max", default=0.10, type=float)
    parser.add_argument("--ctf_deep_sigma", default=0.0, type=float)
    parser.add_argument("--ctf_deep_sigma_scale", default=2.0, type=float)
    parser.add_argument("--ctf_deep_plateau_scale", default=0.75, type=float)
    parser.add_argument("--ctf_anchor_decay", default=0.75, type=float)
    parser.add_argument("--ctf_anchor_keep", default=0.25, type=float)
    parser.add_argument("--ctf_gate_floor", default=0.18, type=float)
    parser.add_argument("--ctf_gate_blur_sigma", default=1.5, type=float)
    parser.add_argument("--ctf_soft_area_scale", default=3.0, type=float)
    parser.add_argument("--ctf_soft_temperature", default=0.08, type=float)
    parser.add_argument("--ctf_soft_blur_sigma", default=0.8, type=float)
    parser.add_argument("--ctf_suppression_beta", default=0.65, type=float)
    parser.add_argument("--ctf_stable_radius", default=2, type=int)
    parser.add_argument("--ctf_min_rounds", default=1, type=int)
    parser.add_argument("--ctf_anchor_topk", default=6, type=int)
    parser.add_argument("--ctf_anchor_min_rel_score", default=0.70, type=float)
    parser.add_argument("--ctf_anchor_min_score", default=0.30, type=float)
    parser.add_argument("--ctf_beam_size", default=4, type=int)
    parser.add_argument("--ctf_branch_score_mode", default="adaptive", choices=["adaptive", "raw"])
    parser.add_argument("--ctf_branch_temperature", default=0.25, type=float)
    parser.add_argument("--ctf_branch_confidence_weight", default=0.30, type=float)
    parser.add_argument("--ctf_branch_mask_quality_weight", default=0.45, type=float)
    parser.add_argument("--ctf_branch_consistency_weight", default=0.20, type=float)
    parser.add_argument("--ctf_branch_isolation_weight", default=0.35, type=float)
    parser.add_argument("--ctf_branch_score_weight", default=0.05, type=float)
    parser.add_argument("--ctf_branch_invalid_weight", default=0.25, type=float)
    parser.add_argument("--ctf_branch_clutter_weight", default=0.45, type=float)
    parser.add_argument("--ctf_branch_keep_margin", default=0.75, type=float)
    parser.add_argument("--ctf_branch_min_keep_score", default=-1e9, type=float)
    parser.add_argument("--ctf_branch_hard_min_isolation", default=-0.05, type=float)
    parser.add_argument("--ctf_branch_hard_max_area_ratio", default=8.0, type=float)
    parser.add_argument("--ctf_branch_fuse", default="max", choices=["max", "mean"])
    parser.add_argument("--ctf_consistency_area_scale", default=2.0, type=float)
    parser.add_argument("--ctf_consistency_shift_scale", default=8.0, type=float)
    parser.add_argument("--ctf_branch_core_sigma_scale", default=1.0, type=float)
    parser.add_argument("--ctf_branch_core_plateau_scale", default=0.5, type=float)
    parser.add_argument("--ctf_branch_context_inner_scale", default=2.0, type=float)
    parser.add_argument("--ctf_branch_context_outer_scale", default=8.0, type=float)
    parser.add_argument("--ctf_branch_context_high_threshold", default=0.55, type=float)
    parser.add_argument("--ctf_branch_context_max_area_factor", default=4.0, type=float)
    parser.add_argument("--ctf_branch_min_isolation", default=0.05, type=float)
    parser.add_argument("--ctf_debug_max_branches", default=4, type=int)
    parser.add_argument("--iter_debug_visualizations", default=0, type=int)
    parser.add_argument("--layer_debug_visualizations", default=0, type=int)
    parser.add_argument("--visualization_samples", default=3, type=int)
    parser.add_argument("--output_dir", default="outputs/runs/irstd-1k/sliding-window-mask/manual_default", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.backbone == "sam2_bplus":
        sam = build_sam2_bplus(checkpoint=args.sam2_ckpt, config=args.sam2_config, device=str(device))
        predictor = Sam2SimilarityPredictor(sam)
    else:
        sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt).to(device)
        sam.eval()
        for param in sam.parameters():
            param.requires_grad_(False)
        predictor = SamPredictor(sam)
    register_sim_feature_hooks(predictor, args.sim_feature_layers)
    final_multi_source = resolve_final_multi_source(args)

    os.makedirs(args.output_dir, exist_ok=True)
    support_paths, test_set = resolve_data_splits(args)
    support_ids = [os.path.splitext(os.path.basename(path))[0] for path, _ in support_paths]
    support_full_items = build_support_value_items(support_paths, args.image_size, args.support_value_aug_scales)
    support_crop_size = args.window_size if args.use_sliding_window else args.image_size
    support_crop_items = build_support_crop_value_items(
        support_paths,
        args.image_size,
        support_crop_size,
        args.support_value_aug_scales,
    )
    support_full_value_aug_summary = summarize_support_value_items(support_full_items)
    support_crop_value_aug_summary = summarize_support_value_items(support_crop_items)
    print(f"support_ids={','.join(support_ids)}")
    print(
        f"support_crop_token_images={len(support_crop_items)} "
        f"support_global_negative_images={len(support_full_items)} "
        f"support_value_aug_scales={args.support_value_aug_scales or 'none'}"
    )
    print(
        f"use_vector_correction={args.use_vector_correction} "
        f"backbone={args.backbone} "
        f"use_sliding_window={args.use_sliding_window} "
        f"window_size={args.window_size} overlap={args.overlap:.2f} "
        f"score_layer={args.score_layer} support_negative_scope={args.support_negative_scope}"
    )

    config = build_config(args)
    layer_scores, layer_infos = build_crop_layer_score_tokens(
        predictor,
        support_crop_items,
        support_full_items,
        config,
        args.support_negative_scope,
    )
    fused_score, fused_info = build_crop_fused_score_token(predictor, support_crop_items, support_full_items, args)
    if args.score_layer == "fused_all_layers":
        score_token = fused_score
        score_info = fused_info
    else:
        if args.score_layer not in layer_scores:
            raise ValueError(f"Unknown score layer {args.score_layer}; available={','.join(layer_scores.keys())}")
        score_token = layer_scores[args.score_layer]
        score_info = layer_infos[args.score_layer]

    area = int(max(1, round(support_area_prior(support_paths, args.image_size, args.area_prior) * args.area_scale)))
    requested_sample_ids = [item.strip() for item in args.sample_ids.split(",") if item.strip()]
    if requested_sample_ids:
        sample_to_index = {sample_id: idx for idx, sample_id in enumerate(test_set.sample_ids)}
        missing_sample_ids = [sample_id for sample_id in requested_sample_ids if sample_id not in sample_to_index]
        if missing_sample_ids:
            if args.dataset or not args.allow_sample_ids_outside_split:
                raise ValueError(f"Unknown sample ids: {','.join(missing_sample_ids)}")
            for sample_id in missing_sample_ids:
                image_path, mask_path = raw_irstd_paths(args.data_root, sample_id)
                if not os.path.exists(image_path) or not os.path.exists(mask_path):
                    raise ValueError(f"Unknown sample id: {sample_id}")
                test_set.sample_ids.append(sample_id)
                test_set.images.append(image_path)
                test_set.masks.append(mask_path)
            sample_to_index = {sample_id: idx for idx, sample_id in enumerate(test_set.sample_ids)}
        eval_indices = [sample_to_index[sample_id] for sample_id in requested_sample_ids]
    else:
        total_split = min(args.max_test_samples, len(test_set)) if args.max_test_samples > 0 else len(test_set)
        eval_indices = list(range(total_split))
    total = len(eval_indices)
    miou_meter = mIoUmeter()
    pdfa_meter = PD_FAmeter()
    detail_rows = []
    branch_rows = []
    per_image_ious = []
    per_image_f1s = []
    top_hits = 0
    top_hits_dilate3 = 0
    distances = []
    windows_per_image = 0
    iter_rounds = []
    iter_candidate_counts = []
    iter_invalid_counts = []
    iter_best_confidences = []

    for output_index, index in enumerate(tqdm(eval_indices, total=total)):
        sample_id = test_set.sample_ids[index]
        image_path = test_set.images[index]
        mask_path = test_set.masks[index]
        image, gt, _ = load_rgb_and_mask(image_path, mask_path, (args.image_size, args.image_size))
        iter_info = {
            "iter_accepted_rounds": 0,
            "iter_candidate_count": 0,
            "iter_invalid_count": 0,
            "iter_best_confidence": float("nan"),
            "iter_debug_candidates": [],
            "iter_debug_suppression": np.zeros((args.image_size, args.image_size), dtype=np.float32),
            "iter_debug_protection": np.zeros((args.image_size, args.image_size), dtype=np.float32),
            "iter_stage_scores": {},
            "iter_stage_masks": {},
            "iter_debug_branches": [],
            "iter_debug_all_branches": [],
            "iter_anchor_count": 0,
            "iter_mean_branch_score": float("nan"),
            "iter_mean_branch_mask_quality": float("nan"),
            "iter_mean_branch_consistency": float("nan"),
            "iter_mean_branch_invalid_penalty": float("nan"),
        }
        score, windows_per_image = compute_stitched_similarity(predictor, image, score_token, args.score_layer, args)
        initial_score = score.copy()
        if args.use_iterative_refine:
            score, iter_info = compute_iterative_refined_score(
                predictor,
                image,
                layer_scores,
                args,
                area,
                initial_score=score,
            )
        iter_rounds.append(iter_info["iter_accepted_rounds"])
        iter_candidate_counts.append(iter_info["iter_candidate_count"])
        iter_invalid_counts.append(iter_info["iter_invalid_count"])
        iter_best_confidences.append(iter_info["iter_best_confidence"])
        stats = score_stats(score, gt)
        top_hits += int(stats["top_hit"])
        top_hits_dilate3 += int(stats["top_hit_dilate3"])
        distances.append(stats["top_dist_to_gt"])

        if args.final_mask_mode == "multi":
            final_branches = iter_info.get("iter_debug_branches", []) if final_multi_source in {"branches", "hybrid"} else []
            if not final_branches and final_multi_source in {"branches", "hybrid"}:
                final_branches = []
            mask_score, final_mask_info = extract_multi_final_mask(
                score,
                image,
                area,
                args,
                branches=final_branches if final_branches else None,
                include_peaks=final_multi_source in {"peaks", "hybrid"},
            )
        elif args.final_mask_mode == "auto":
            final_branches = iter_info.get("iter_debug_all_branches", []) if final_multi_source in {"branches", "hybrid"} else []
            mask_score, final_mask_info = extract_multi_final_mask(
                score,
                image,
                area,
                args,
                branches=final_branches if final_branches else None,
                include_peaks=final_multi_source in {"peaks", "hybrid"},
            )
        else:
            mask_score, final_mask_info = extract_final_mask(score, image, area, stats, args)
            final_mask_info["mask_count"] = 1
            final_mask_info["candidate_count"] = 1
            final_mask_info["kept_masks"] = [final_mask_info]
        pred = final_mask_info["pred"]
        seed_y = final_mask_info["seed_y"]
        seed_x = final_mask_info["seed_x"]
        seed_mode = final_mask_info["seed_mode"]
        iou, f1, inter, pred_pixels, gt_pixels = binary_metrics(pred, gt)
        per_image_ious.append(iou)
        per_image_f1s.append(f1)

        pred_t = torch.from_numpy(pred.reshape(1, 1, args.image_size, args.image_size)).float()
        gt_t = torch.from_numpy(gt.reshape(1, 1, args.image_size, args.image_size)).float()
        miou_meter.update(pred_t, gt_t)
        pdfa_meter.update(
            pred_t.reshape(1, args.image_size, args.image_size),
            gt_t.reshape(1, args.image_size, args.image_size),
            [args.image_size, args.image_size],
        )

        if requested_sample_ids or output_index < args.visualization_samples:
            save_result_panel(args.output_dir, sample_id, image, gt, mask_score, pred)
            save_final_masks_panel(args.output_dir, sample_id, image, final_mask_info)
        if args.use_iterative_refine and output_index < args.iter_debug_visualizations:
            save_iter_debug_panel(
                args.output_dir,
                sample_id,
                image,
                gt,
                initial_score,
                score,
                iter_info["iter_debug_candidates"],
                pred,
                suppression=iter_info["iter_debug_suppression"],
                protection=iter_info["iter_debug_protection"],
            )
            save_iter_stage_panel(
                args.output_dir,
                sample_id,
                image,
                iter_info["iter_stage_scores"],
                iter_info["iter_stage_masks"],
            )
            save_branch_flow_panels(
                args.output_dir,
                sample_id,
                image,
                initial_score,
                score,
                iter_info.get("iter_debug_branches", []),
                area,
                args,
            )
        if output_index < args.layer_debug_visualizations:
            layer_debug = compute_layer_debug_scores(predictor, image, layer_scores, args)
            save_layer_score_panel(args.output_dir, sample_id, image, layer_debug)

        kept_branch_ids = {
            (int(item["seed_y"]), int(item["seed_x"])) for item in iter_info.get("iter_debug_branches", [])
        }
        for branch_index, branch in enumerate(iter_info.get("iter_debug_all_branches", [])):
            branch_rows.append(
                {
                    "sample_id": sample_id,
                    "branch_index": branch_index,
                    "kept": (int(branch["seed_y"]), int(branch["seed_x"])) in kept_branch_ids,
                    "seed_x": int(branch["seed_x"]),
                    "seed_y": int(branch["seed_y"]),
                    "seed_score": float(branch.get("seed_score", float("nan"))),
                    "branch_score": float(branch.get("branch_score", float("nan"))),
                    "branch_score_raw": float(branch.get("branch_score_raw", float("nan"))),
                    "confidence": float(branch.get("confidence", float("nan"))),
                    "mask_quality": float(branch.get("mask_quality", float("nan"))),
                    "mask_area_scale": float(branch.get("mask_area_scale", float("nan"))),
                    "layer_consistency": float(branch.get("layer_consistency", float("nan"))),
                    "context_isolation": float(branch.get("context_isolation", float("nan"))),
                    "context_isolation_score": float(branch.get("context_isolation_score", float("nan"))),
                    "context_high_area_ratio": float(branch.get("context_high_area_ratio", float("nan"))),
                    "context_clutter_penalty": float(branch.get("context_clutter_penalty", float("nan"))),
                    "context_core_mean": float(branch.get("context_core_mean", float("nan"))),
                    "context_ring_mean": float(branch.get("context_ring_mean", float("nan"))),
                    "invalid_penalty": float(branch.get("invalid_penalty", float("nan"))),
                    "seed_norm": float(branch.get("branch_score_seed_norm", float("nan"))),
                    "confidence_norm": float(branch.get("branch_score_confidence_norm", float("nan"))),
                    "mask_quality_norm": float(branch.get("branch_score_mask_quality_norm", float("nan"))),
                    "consistency_norm": float(branch.get("branch_score_consistency_norm", float("nan"))),
                    "isolation_norm": float(branch.get("branch_score_isolation_norm", float("nan"))),
                    "invalid_norm": float(branch.get("branch_score_invalid_norm", float("nan"))),
                    "clutter_norm": float(branch.get("branch_score_clutter_norm", float("nan"))),
                }
            )

        detail_rows.append(
            {
                "sample_id": sample_id,
                "score_layer": args.score_layer,
                "mask_score_mode": args.mask_score_mode,
                "top_x": stats["top_x"],
                "top_y": stats["top_y"],
                "top_score": stats["top_score"],
                "top_hit": stats["top_hit"],
                "top_hit_dilate3": stats["top_hit_dilate3"],
                "top_dist_to_gt": stats["top_dist_to_gt"],
                "seed_x": int(seed_x),
                "seed_y": int(seed_y),
                "seed_mode": seed_mode,
                "final_area": final_mask_info["area"],
                "final_area_scale": final_mask_info["area_scale"],
                "final_mask_mode": args.final_mask_mode,
                "final_mask_count": final_mask_info.get("mask_count", 1),
                "final_candidate_count": final_mask_info.get("candidate_count", 1),
                "final_mask_quality": final_mask_info["quality"],
                "final_score_gap": final_mask_info["score_gap"],
                "final_contrast_gap": final_mask_info["contrast_gap"],
                "final_area_score": final_mask_info["area_score"],
                "iou": iou,
                "f1": f1,
                "intersection_pixels": inter,
                "pred_pixels": pred_pixels,
                "gt_pixels": gt_pixels,
                "iter_accepted_rounds": iter_info["iter_accepted_rounds"],
                "iter_candidate_count": iter_info["iter_candidate_count"],
                "iter_invalid_count": iter_info["iter_invalid_count"],
                "iter_anchor_count": iter_info.get("iter_anchor_count", 0),
                "iter_mean_branch_score": iter_info.get("iter_mean_branch_score", float("nan")),
                "iter_mean_branch_mask_quality": iter_info.get("iter_mean_branch_mask_quality", float("nan")),
                "iter_mean_branch_consistency": iter_info.get("iter_mean_branch_consistency", float("nan")),
                "iter_mean_branch_invalid_penalty": iter_info.get("iter_mean_branch_invalid_penalty", float("nan")),
                "iter_best_confidence": iter_info["iter_best_confidence"],
            }
        )

    _, miou, fscore = miou_meter.get()
    pd, fa = pdfa_meter.get()
    valid_distances = [d for d in distances if not np.isnan(d)]
    summary = {
        "support_ids": ";".join(support_ids),
        "support_crop_token_images": len(support_crop_items),
        "support_global_negative_images": len(support_full_items),
        "support_value_aug_scales": args.support_value_aug_scales,
        "support_crop_value_aug_effective": support_crop_value_aug_summary,
        "support_full_value_aug_effective": support_full_value_aug_summary,
        "score_layer": args.score_layer,
        "use_vector_correction": args.use_vector_correction,
        "use_high_freq_negative": args.use_high_freq_negative,
        "use_bright_negative": args.use_bright_negative,
        "use_sliding_window": args.use_sliding_window,
        "use_iterative_refine": args.use_iterative_refine,
        "mask_score_mode": args.mask_score_mode,
        "local_contrast_sigma": args.local_contrast_sigma,
        "mask_sim_weight": args.mask_sim_weight,
        "window_size": args.window_size,
        "overlap": args.overlap,
        "stride": int(round(args.window_size * (1.0 - args.overlap))) if args.use_sliding_window else args.image_size,
        "window_normalize": args.window_normalize,
        "window_score_mode": args.window_score_mode,
        "window_reduce": args.window_reduce,
        "window_topk_fraction": args.window_topk_fraction,
        "support_negative_scope": args.support_negative_scope,
        "windows_per_image": windows_per_image,
        "iter_refine_rounds": args.iter_refine_rounds,
        "iter_topk_points": args.iter_topk_points,
        "iter_area_scales": args.iter_area_scales,
        "iter_grow_factors": args.iter_grow_factors,
        "ctf_deep_sigma": args.ctf_deep_sigma,
        "ctf_deep_sigma_scale": args.ctf_deep_sigma_scale,
        "ctf_deep_plateau_scale": args.ctf_deep_plateau_scale,
        "ctf_soft_area_scale": args.ctf_soft_area_scale,
        "ctf_suppression_beta": args.ctf_suppression_beta,
        "ctf_anchor_topk": args.ctf_anchor_topk,
        "ctf_anchor_min_rel_score": args.ctf_anchor_min_rel_score,
        "ctf_beam_size": args.ctf_beam_size,
        "ctf_branch_score_mode": args.ctf_branch_score_mode,
        "ctf_branch_confidence_weight": args.ctf_branch_confidence_weight,
        "ctf_branch_mask_quality_weight": args.ctf_branch_mask_quality_weight,
        "ctf_branch_consistency_weight": args.ctf_branch_consistency_weight,
        "ctf_branch_isolation_weight": args.ctf_branch_isolation_weight,
        "ctf_branch_clutter_weight": args.ctf_branch_clutter_weight,
        "ctf_branch_keep_margin": args.ctf_branch_keep_margin,
        "ctf_branch_hard_max_area_ratio": args.ctf_branch_hard_max_area_ratio,
        "ctf_branch_fuse": args.ctf_branch_fuse,
        "iter_mean_accepted_rounds": float(np.mean(iter_rounds)) if iter_rounds else 0.0,
        "iter_mean_candidate_count": float(np.mean(iter_candidate_counts)) if iter_candidate_counts else 0.0,
        "iter_mean_invalid_count": float(np.mean(iter_invalid_counts)) if iter_invalid_counts else 0.0,
        "iter_mean_best_confidence": float(np.nanmean(iter_best_confidences)) if iter_best_confidences else float("nan"),
        "area": area,
        "final_area_scales": args.final_area_scales,
        "final_area_weight": args.final_area_weight,
        "final_contrast_weight": args.final_contrast_weight,
        "final_area_penalty": args.final_area_penalty,
        "final_mask_mode": args.final_mask_mode,
        "final_multi_source": args.final_multi_source,
        "final_multi_source_resolved": final_multi_source,
        "final_multi_max_masks": args.final_multi_max_masks,
        "final_multi_nms_iou": args.final_multi_nms_iou,
        "final_multi_quality_margin": args.final_multi_quality_margin,
        "test_samples": total,
        "mIoU": float(miou),
        "mIoU_percent": float(miou) * 100.0,
        "F1": float(fscore),
        "F1_percent": float(fscore) * 100.0,
        "mean_image_iou": float(np.mean(per_image_ious)) if per_image_ious else float("nan"),
        "median_image_iou": float(np.median(per_image_ious)) if per_image_ious else float("nan"),
        "mean_image_f1": float(np.mean(per_image_f1s)) if per_image_f1s else float("nan"),
        "Pd": float(pd),
        "Pd_percent": float(pd) * 100.0,
        "Fa": float(fa),
        "Fa_per_million": float(fa) * 1e6,
        "sim_top1_hit_rate": top_hits / max(1, total),
        "sim_top1_hit_percent": top_hits / max(1, total) * 100.0,
        "sim_top1_hit_dilate3_rate": top_hits_dilate3 / max(1, total),
        "sim_top1_hit_dilate3_percent": top_hits_dilate3 / max(1, total) * 100.0,
        "sim_top1_mean_dist_to_gt": float(np.mean(valid_distances)) if valid_distances else float("nan"),
        "sim_top1_median_dist_to_gt": float(np.median(valid_distances)) if valid_distances else float("nan"),
        "negative_names": ";".join(score_info["negative_names"]),
        "negative_rank": score_info["negative_rank"],
        "target_score_cos": score_info["target_score_cos"],
    }

    summary_path = os.path.join(args.output_dir, "sliding_window_semantic_mask_summary.csv")
    detail_path = os.path.join(args.output_dir, "sliding_window_semantic_mask_detail.csv")
    branch_detail_path = os.path.join(args.output_dir, "sliding_window_semantic_mask_branch_detail.csv")
    write_csv(summary_path, [summary])
    write_csv(detail_path, detail_rows)
    if branch_rows:
        write_csv(branch_detail_path, branch_rows)
    print(f"summary={summary_path}")
    print(f"detail={detail_path}")
    if branch_rows:
        print(f"branch_detail={branch_detail_path}")
    print(
        f"mIoU={summary['mIoU_percent']:.2f} F1={summary['F1_percent']:.2f} "
        f"Pd={summary['Pd_percent']:.2f} Fa={summary['Fa_per_million']:.2f}/M "
        f"simTop1={summary['sim_top1_hit_percent']:.2f} "
        f"simTop1_d3={summary['sim_top1_hit_dilate3_percent']:.2f} "
        f"meanDist={summary['sim_top1_mean_dist_to_gt']:.2f}"
    )


if __name__ == "__main__":
    main()
