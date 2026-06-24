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
from visual_prompt.data.irstd import RawIRSTDDataset, build_raw_support_paths, dilated_hit, distance_to_gt
from visual_prompt.core.semantic_direction import SemanticDirectionConfig, parse_optional_float, parse_optional_int
from visual_prompt.backbones.mobile_sam import DEFAULT_CHECKPOINT, SamPredictor, sam_model_registry
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
    sim_post = predictor.model.postprocess_masks(sim_raw, predictor.input_size, predictor.original_size)
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
    parser.add_argument("--sam_type", default="vit_t", choices=["vit_t"])
    parser.add_argument("--sam_ckpt", default=str(DEFAULT_CHECKPOINT), type=str)
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
    parser.add_argument("--visualization_samples", default=3, type=int)
    parser.add_argument("--output_dir", default="outputs/runs/irstd-1k/sliding-window-mask/manual_default", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt).to(device)
    sam.eval()
    for param in sam.parameters():
        param.requires_grad_(False)
    predictor = SamPredictor(sam)
    register_sim_feature_hooks(predictor, args.sim_feature_layers)

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
            raise ValueError(f"Unknown sample ids: {','.join(missing_sample_ids)}")
        eval_indices = [sample_to_index[sample_id] for sample_id in requested_sample_ids]
    else:
        total_split = min(args.max_test_samples, len(test_set)) if args.max_test_samples > 0 else len(test_set)
        eval_indices = list(range(total_split))
    total = len(eval_indices)
    miou_meter = mIoUmeter()
    pdfa_meter = PD_FAmeter()
    detail_rows = []
    per_image_ious = []
    per_image_f1s = []
    top_hits = 0
    top_hits_dilate3 = 0
    distances = []
    windows_per_image = 0

    for output_index, index in enumerate(tqdm(eval_indices, total=total)):
        sample_id = test_set.sample_ids[index]
        image_path = test_set.images[index]
        mask_path = test_set.masks[index]
        image, gt, _ = load_rgb_and_mask(image_path, mask_path, (args.image_size, args.image_size))
        score, windows_per_image = compute_stitched_similarity(predictor, image, score_token, args.score_layer, args)
        stats = score_stats(score, gt)
        top_hits += int(stats["top_hit"])
        top_hits_dilate3 += int(stats["top_hit_dilate3"])
        distances.append(stats["top_dist_to_gt"])

        mask_score = build_mask_score(image, score, args)
        if args.mask_mode == "top_area":
            pred = top_area_mask(mask_score, area)
            seed_y, seed_x, seed_mode = stats["top_y"], stats["top_x"], "top_area"
        elif args.mask_score_mode != "sim":
            pred, seed_y, seed_x, seed_mode = component_at_anchor(
                mask_score,
                area,
                args.component_grow_factor,
                stats["top_y"],
                stats["top_x"],
                args.seed_radius,
            )
        else:
            pred, seed_y, seed_x, seed_mode = component_at_seed(
                score,
                area,
                args.component_grow_factor,
                stats["top_y"],
                stats["top_x"],
                args.seed_radius,
            )
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
                "iou": iou,
                "f1": f1,
                "intersection_pixels": inter,
                "pred_pixels": pred_pixels,
                "gt_pixels": gt_pixels,
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
        "area": area,
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
    write_csv(summary_path, [summary])
    write_csv(detail_path, detail_rows)
    print(f"summary={summary_path}")
    print(f"detail={detail_path}")
    print(
        f"mIoU={summary['mIoU_percent']:.2f} F1={summary['F1_percent']:.2f} "
        f"Pd={summary['Pd_percent']:.2f} Fa={summary['Fa_per_million']:.2f}/M "
        f"simTop1={summary['sim_top1_hit_percent']:.2f} "
        f"simTop1_d3={summary['sim_top1_hit_dilate3_percent']:.2f} "
        f"meanDist={summary['sim_top1_mean_dist_to_gt']:.2f}"
    )


if __name__ == "__main__":
    main()
