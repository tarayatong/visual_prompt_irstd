import argparse
import csv
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from visual_prompt.backbones.sam2_adapter import (
    DEFAULT_SAM2_BPLUS_CHECKPOINT,
    DEFAULT_SAM2_BPLUS_CONFIG,
    Sam2SimilarityPredictor,
    build_sam2_bplus,
)
from visual_prompt.core.mask_utils import binary_metrics, component_at_seed, support_area_prior
from visual_prompt.core.metrics import PD_FAmeter, mIoUmeter
from visual_prompt.core.semantic_direction import SemanticDirectionConfig, build_score_token, extract_support_semantic_tokens
from visual_prompt.data.irstd import RawIRSTDDataset, build_raw_support_paths, dilated_hit, distance_to_gt
from visual_prompt.pipelines.semantic_prompt_pipeline import (
    get_named_sim_features,
    get_sim_features,
    load_rgb_and_mask,
    register_sim_feature_hooks,
    seed_everything,
)


DEFAULT_SUPPORT_IDS = "XDU514,XDU646,XDU904,XDU660,XDU347"


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    value = str(value).strip()
    if not value or value.lower() in {"none", "null", "off"}:
        return []
    return [float(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]


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


def crop_with_padding(image, mask, y0, x0, size):
    h, w = image.shape[:2]
    y1, x1 = y0 + size, x0 + size
    src_y0, src_x0 = max(0, y0), max(0, x0)
    src_y1, src_x1 = min(h, y1), min(w, x1)
    dst_y0, dst_x0 = src_y0 - y0, src_x0 - x0
    image_crop = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
    mask_crop = np.zeros((size, size), dtype=mask.dtype)
    image_crop[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = image[src_y0:src_y1, src_x0:src_x1]
    mask_crop[dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = mask[src_y0:src_y1, src_x0:src_x1]
    return image_crop, mask_crop


def crop_support_target(image, mask, window_size):
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        cy, cx = image.shape[0] // 2, image.shape[1] // 2
    else:
        cy, cx = int(round(float(ys.mean()))), int(round(float(xs.mean())))
    return crop_with_padding(image, mask, cy - window_size // 2, cx - window_size // 2, window_size)


def sliding_starts(length, window_size, stride):
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    last = length - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def score_stats(score, mask):
    top_idx = int(np.argmax(score.reshape(-1)))
    top_y, top_x = np.unravel_index(top_idx, score.shape)
    return {
        "top_x": int(top_x),
        "top_y": int(top_y),
        "top_score": float(score[top_y, top_x]),
        "top_hit": bool(mask[top_y, top_x] > 0),
        "top_hit_dilate3": bool(mask[top_y, top_x] > 0 or dilated_hit(mask, top_y, top_x, radius=3)),
        "top_dist_to_gt": distance_to_gt(mask, top_y, top_x),
    }


def local_contrast_score(image, sigma):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return robust_norm(gray - local_mean, low=50.0, high=99.7)


def build_mask_score(image, sim_score, args):
    return minmax_norm(sim_score) * local_contrast_score(image, args.local_contrast_sigma)


def mask_ring(mask, radius=5):
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
    return np.logical_and(dilated, mask <= 0)


def mask_quality(mask, score, image, support_area, args):
    pixels = int(mask.sum())
    if pixels <= 0:
        return -1e9, {"pred_pixels": 0, "area_score": 0.0, "score_gap": 0.0, "contrast_gap": 0.0}
    score_norm = minmax_norm(score)
    contrast = local_contrast_score(image, args.local_contrast_sigma)
    ring = mask_ring(mask)
    inside_score = float(score_norm[mask > 0].mean())
    ring_score = float(score_norm[ring].mean()) if ring.any() else float(score_norm.mean())
    inside_contrast = float(contrast[mask > 0].mean())
    ring_contrast = float(contrast[ring].mean()) if ring.any() else float(contrast.mean())
    area_score = float(np.exp(-abs(np.log((pixels + 1e-6) / (float(support_area) + 1e-6)))))
    quality = (
        (inside_score - ring_score)
        + args.final_contrast_weight * (inside_contrast - ring_contrast)
        + args.final_area_weight * area_score
        - args.final_area_penalty * abs(np.log((pixels + 1e-6) / (float(support_area) + 1e-6)))
    )
    return float(quality), {
        "pred_pixels": pixels,
        "area_score": area_score,
        "score_gap": inside_score - ring_score,
        "contrast_gap": inside_contrast - ring_contrast,
    }


def extract_final_mask(score, image, support_area, stats, args):
    mask_score = build_mask_score(image, score, args)
    best = None
    for scale in parse_float_list(args.final_area_scales) or [1.0]:
        area = int(max(1, round(float(support_area) * float(scale))))
        pred, seed_y, seed_x, seed_mode = component_at_seed(
            mask_score,
            area,
            args.component_grow_factor,
            int(stats["top_y"]),
            int(stats["top_x"]),
            args.seed_radius,
        )
        quality, qstats = mask_quality(pred, score, image, support_area, args)
        item = {
            "pred": pred.astype(np.uint8),
            "seed_y": int(seed_y),
            "seed_x": int(seed_x),
            "seed_mode": seed_mode,
            "area_scale": float(scale),
            "quality": quality,
            **qstats,
        }
        if best is None or item["quality"] > best["quality"]:
            best = item
    return mask_score, best


def build_config(args):
    return SemanticDirectionConfig(
        use_negative_subspace=args.use_vector_correction,
        use_high_freq_negative=True,
        edge_method=args.edge_method,
        edge_quantile=args.edge_quantile,
        edge_exclude_dilate=args.edge_exclude_dilate,
        use_bright_negative=True,
        bright_threshold_mode=args.bright_threshold_mode,
        target_rf_radius_scale=args.target_rf_radius_scale,
        target_rf_radius_min=args.target_rf_radius_min,
    )


@torch.no_grad()
def build_support_score_token(predictor, support_paths, args):
    config = build_config(args)
    target_tokens = []
    negative_tokens = {"high_freq": [], "bright_flat": []}
    for image_path, mask_path in support_paths:
        image, mask, _ = load_rgb_and_mask(image_path, mask_path, (args.image_size, args.image_size))
        crop_image, crop_mask = crop_support_target(image, mask, args.window_size)
        mask_3c = np.repeat(crop_mask[:, :, None] * 255, 3, axis=2).astype(np.uint8)
        mask_pad = predictor.set_image(crop_image, mask_3c)
        token, _ = extract_support_semantic_tokens(crop_image, crop_mask, mask_pad, get_sim_features(predictor), config)
        if token is not None:
            target_tokens.append(token)
    for image_path, mask_path in support_paths:
        image, mask, _ = load_rgb_and_mask(image_path, mask_path, (args.image_size, args.image_size))
        mask_3c = np.repeat(mask[:, :, None] * 255, 3, axis=2).astype(np.uint8)
        mask_pad = predictor.set_image(image, mask_3c)
        _, negs = extract_support_semantic_tokens(image, mask, mask_pad, get_sim_features(predictor), config)
        for name, token in negs.items():
            negative_tokens.setdefault(name, []).append(token)
    if not target_tokens:
        raise RuntimeError("No valid support target tokens were extracted.")
    score_token, info = build_score_token(target_tokens, negative_tokens, config)
    info["valid_target_token_count"] = len(target_tokens)
    return score_token, info


@torch.no_grad()
def build_support_layer_score_tokens(predictor, support_paths, args):
    config = build_config(args)
    layer_targets = {}
    layer_negatives = {}

    for image_path, mask_path in support_paths:
        image, mask, _ = load_rgb_and_mask(image_path, mask_path, (args.image_size, args.image_size))
        crop_image, crop_mask = crop_support_target(image, mask, args.window_size)
        mask_3c = np.repeat(crop_mask[:, :, None] * 255, 3, axis=2).astype(np.uint8)
        mask_pad = predictor.set_image(crop_image, mask_3c)
        for layer_name, layer_features in get_named_sim_features(predictor, align=False):
            token, _ = extract_support_semantic_tokens(crop_image, crop_mask, mask_pad, layer_features, config)
            if token is not None:
                layer_targets.setdefault(layer_name, []).append(token)
            layer_negatives.setdefault(layer_name, {"high_freq": [], "bright_flat": []})

    for image_path, mask_path in support_paths:
        image, mask, _ = load_rgb_and_mask(image_path, mask_path, (args.image_size, args.image_size))
        mask_3c = np.repeat(mask[:, :, None] * 255, 3, axis=2).astype(np.uint8)
        mask_pad = predictor.set_image(image, mask_3c)
        for layer_name, layer_features in get_named_sim_features(predictor, align=False):
            _, negs = extract_support_semantic_tokens(image, mask, mask_pad, layer_features, config)
            layer_negatives.setdefault(layer_name, {"high_freq": [], "bright_flat": []})
            for name, token in negs.items():
                layer_negatives[layer_name].setdefault(name, []).append(token)

    layer_scores = {}
    layer_infos = {}
    for layer_name, target_tokens in layer_targets.items():
        score_token, info = build_score_token(target_tokens, layer_negatives.get(layer_name, {}), config)
        info["valid_target_token_count"] = len(target_tokens)
        layer_scores[layer_name] = score_token
        layer_infos[layer_name] = info
    if not layer_scores:
        raise RuntimeError("No valid per-layer support score tokens were extracted.")
    return layer_scores, layer_infos


def fuse_layer_score_maps(layer_maps):
    fused = None
    for score in layer_maps.values():
        score_norm = minmax_norm(score)
        fused = score_norm if fused is None else fused + score_norm
    return (fused / max(1, len(layer_maps))).astype(np.float32)


def summarize_layer_infos(layer_infos):
    names = list(layer_infos.keys())
    return {
        "negative_names": [f"{name}={'|'.join(layer_infos[name].get('negative_names', []))}" for name in names],
        "negative_rank": sum(int(layer_infos[name].get("negative_rank", 0)) for name in names),
        "target_score_cos": float(np.mean([float(layer_infos[name].get("target_score_cos", 0.0)) for name in names])),
        "negative_token_count": sum(int(layer_infos[name].get("negative_token_count", 0)) for name in names),
    }
def encode_crop_batch(predictor, crops):
    if hasattr(predictor, "set_image_batch"):
        predictor.set_image_batch(crops)
        return
    predictor.predictor.set_image_batch(crops)
    predictor.original_size = crops[0].shape[:2]
    predictor.input_size = (predictor.model.image_size, predictor.model.image_size)
    predictor.features = predictor.predictor._features["image_embed"]


def compute_sim_batch_from_features(predictor, target_feat, cur_feat_bchw, sim_size):
    target_feat = target_feat.reshape(-1)
    target_feat = target_feat / (target_feat.norm(dim=0, keepdim=False) + 1e-8)
    feat = cur_feat_bchw / (cur_feat_bchw.norm(dim=1, keepdim=True) + 1e-8)
    sim_raw = torch.einsum("c,bchw->bhw", target_feat, feat).unsqueeze(1)
    if sim_raw.shape[-2:] != (sim_size, sim_size):
        sim_raw = torch.nn.functional.interpolate(sim_raw, size=(sim_size, sim_size), mode="bilinear", align_corners=False)
    return predictor.postprocess_masks(sim_raw, predictor.input_size, predictor.original_size)[:, 0]


@torch.no_grad()
def compute_stitched_similarity(predictor, image, score_token, args, layer_name=None):
    h, w = image.shape[:2]
    stride = max(1, int(round(args.window_size * (1.0 - args.overlap))))
    windows = [
        (y0, x0)
        for y0 in sliding_starts(h, args.window_size, stride)
        for x0 in sliding_starts(w, args.window_size, stride)
    ]
    score_sum = np.zeros((h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    dummy_mask = np.zeros((h, w), dtype=np.uint8)
    for start in range(0, len(windows), args.window_batch_size):
        batch_windows = windows[start : start + args.window_batch_size]
        crops = [crop_with_padding(image, dummy_mask, y0, x0, args.window_size)[0] for y0, x0 in batch_windows]
        encode_crop_batch(predictor, crops)
        if layer_name is None:
            features = get_sim_features(predictor).float()
        else:
            features_by_name = {name: feat for name, feat in get_named_sim_features(predictor, align=False)}
            features = features_by_name[layer_name].float()
        patches = compute_sim_batch_from_features(predictor, score_token, features, args.window_sim_size)
        patches = patches.detach().cpu().numpy().astype(np.float32)
        for patch, (y0, x0) in zip(patches, batch_windows):
            if patch.shape != (args.window_size, args.window_size):
                patch = cv2.resize(patch, (args.window_size, args.window_size), interpolation=cv2.INTER_LINEAR)
            score_sum[y0 : y0 + args.window_size, x0 : x0 + args.window_size] += patch
            weight_sum[y0 : y0 + args.window_size, x0 : x0 + args.window_size] += 1.0
    return score_sum / np.maximum(weight_sum, 1e-6), len(windows)


def parse_args():
    parser = argparse.ArgumentParser("Evaluate the clean SAM2 concat-feature visual-prompt baseline.")
    parser.add_argument("--data_root", default="dataset/IRSTD-1k", type=str)
    parser.add_argument("--split", default="test.txt", type=str)
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--encoder_input_size", default=512, choices=[512,1024], type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--seed", default=111, type=int)
    parser.add_argument("--sam2_config", default=DEFAULT_SAM2_BPLUS_CONFIG, type=str)
    parser.add_argument("--sam2_ckpt", default=DEFAULT_SAM2_BPLUS_CHECKPOINT, type=str)
    parser.add_argument("--sim_feature_layers", default="all", type=str)
    parser.add_argument("--score_mode", default="concat_all_layers", choices=["concat_all_layers", "fused_all_layers"])
    parser.add_argument("--feature_source", default="decoder", choices=["decoder", "encoder_fpn"])
    parser.add_argument("--support_ids", default=DEFAULT_SUPPORT_IDS, type=str)
    parser.add_argument("--k_shot", default=5, type=int)
    parser.add_argument("--window_size", default=128, type=int)
    parser.add_argument("--overlap", default=0.5, type=float)
    parser.add_argument("--window_batch_size", default=4, type=int)
    parser.add_argument("--window_sim_size", default=256, type=int)
    parser.add_argument("--use_vector_correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_contrast_sigma", default=7.0, type=float)
    parser.add_argument("--area_prior", default="median", choices=["median", "mean"])
    parser.add_argument("--area_scale", default=1.0, type=float)
    parser.add_argument("--final_area_scales", default="0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0", type=str)
    parser.add_argument("--final_area_weight", default=0.05, type=float)
    parser.add_argument("--final_contrast_weight", default=0.20, type=float)
    parser.add_argument("--final_area_penalty", default=0.15, type=float)
    parser.add_argument("--component_grow_factor", default=8.0, type=float)
    parser.add_argument("--seed_radius", default=3, type=int)
    parser.add_argument("--edge_method", default="sobel", choices=["sobel", "canny", "log"])
    parser.add_argument("--edge_quantile", default=0.90, type=float)
    parser.add_argument("--edge_exclude_dilate", default=9, type=int)
    parser.add_argument("--bright_threshold_mode", default="target_mean", choices=["target_mean", "global_quantile"])
    parser.add_argument("--target_rf_radius_scale", default=1.0, type=float)
    parser.add_argument("--target_rf_radius_min", default=0, type=int)
    parser.add_argument("--max_test_samples", default=0, type=int)
    parser.add_argument("--output_dir", default="outputs/runs/sam2/main", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sam = build_sam2_bplus(
        checkpoint=args.sam2_ckpt,
        config=args.sam2_config,
        device=str(device),
        image_size=args.encoder_input_size,
    )
    predictor = Sam2SimilarityPredictor(sam, feature_source=args.feature_source)
    register_sim_feature_hooks(predictor, args.sim_feature_layers)
    support_paths = build_raw_support_paths(args.data_root, args.k_shot, support_ids=args.support_ids)
    test_set = RawIRSTDDataset(args.data_root, args.split, args.image_size)
    if args.score_mode == "fused_all_layers":
        layer_scores, layer_infos = build_support_layer_score_tokens(predictor, support_paths, args)
        score_token = None
        score_info = summarize_layer_infos(layer_infos)
    else:
        layer_scores = None
        score_token, score_info = build_support_score_token(predictor, support_paths, args)
    support_area = int(max(1, round(support_area_prior(support_paths, args.image_size, args.area_prior) * args.area_scale)))
    print(f"backbone=sam2_bplus feature_source={args.feature_source} score_layer={args.score_mode} window_size={args.window_size} overlap={args.overlap}")
    print(f"support_ids={','.join(os.path.splitext(os.path.basename(path))[0] for path, _ in support_paths)}")
    print(f"negative_names={';'.join(score_info.get('negative_names', []))}")
    rows = []
    miou_meter = mIoUmeter()
    pdfa_meter = PD_FAmeter()
    top_hits = 0
    top_hits_dilate3 = 0
    distances = []
    total = len(test_set) if args.max_test_samples <= 0 else min(len(test_set), args.max_test_samples)
    for index in tqdm(range(total), desc="eval"):
        image_path = test_set.images[index]
        mask_path = test_set.masks[index]
        sample_id = test_set.sample_ids[index]
        image, gt, _ = load_rgb_and_mask(image_path, mask_path, (args.image_size, args.image_size))
        if args.score_mode == "fused_all_layers":
            layer_maps = {}
            windows_per_image = 0
            for cur_layer_name, layer_token in layer_scores.items():
                layer_maps[cur_layer_name], windows_per_image = compute_stitched_similarity(
                    predictor,
                    image,
                    layer_token,
                    args,
                    layer_name=cur_layer_name,
                )
            score = fuse_layer_score_maps(layer_maps)
        else:
            score, windows_per_image = compute_stitched_similarity(predictor, image, score_token, args)
        stats = score_stats(score, gt)
        top_hits += int(stats["top_hit"])
        top_hits_dilate3 += int(stats["top_hit_dilate3"])
        distances.append(stats["top_dist_to_gt"])
        mask_score, final_mask_info = extract_final_mask(score, image, support_area, stats, args)
        pred = final_mask_info["pred"]
        iou, f1, inter, pred_pixels, gt_pixels = binary_metrics(pred, gt)
        pred_t = torch.from_numpy(pred.reshape(1, 1, args.image_size, args.image_size)).float()
        gt_t = torch.from_numpy(gt.reshape(1, 1, args.image_size, args.image_size)).float()
        miou_meter.update(pred_t, gt_t)
        pdfa_meter.update(pred_t.reshape(1, args.image_size, args.image_size), gt_t.reshape(1, args.image_size, args.image_size), [args.image_size, args.image_size])
        rows.append(
            {
                "sample_id": sample_id,
                "iou": iou,
                "f1": f1,
                "inter_pixels": inter,
                "pred_pixels": pred_pixels,
                "gt_pixels": gt_pixels,
                "top_hit": int(stats["top_hit"]),
                "top_hit_dilate3": int(stats["top_hit_dilate3"]),
                "top_dist_to_gt": stats["top_dist_to_gt"],
                "top_y": stats["top_y"],
                "top_x": stats["top_x"],
                "top_score": stats["top_score"],
                "windows_per_image": windows_per_image,
                "mask_quality": final_mask_info["quality"],
                "mask_area_scale": final_mask_info["area_scale"],
                "seed_mode": final_mask_info["seed_mode"],
            }
        )
    write_csv(os.path.join(args.output_dir, "detail.csv"), rows)
    _, miou, fscore = miou_meter.get()
    pd, fa = pdfa_meter.get()
    valid_distances = [item for item in distances if np.isfinite(item)]
    summary = {
        "backbone": "sam2_bplus",
        "feature_source": args.feature_source,
        "score_layer": args.score_mode,
        "support_negative_scope": "global",
        "use_vector_correction": args.use_vector_correction,
        "use_high_freq_negative": True,
        "use_bright_negative": True,
        "window_size": args.window_size,
        "encoder_input_size": predictor.input_size[0],
        "overlap": args.overlap,
        "window_batch_size": args.window_batch_size,
        "support_area": support_area,
        "mIoU_percent": float(miou) * 100.0,
        "F1_percent": float(fscore) * 100.0,
        "Pd_percent": float(pd) * 100.0,
        "Fa_per_million": float(fa) * 1e6,
        "sim_top1_hit_percent": top_hits / max(1, total) * 100.0,
        "sim_top1_hit_dilate3_percent": top_hits_dilate3 / max(1, total) * 100.0,
        "sim_top1_mean_dist_to_gt": float(np.mean(valid_distances)) if valid_distances else float("nan"),
        "negative_names": ";".join(score_info.get("negative_names", [])),
        "negative_rank": score_info.get("negative_rank", 0),
        "target_score_cos": score_info.get("target_score_cos", float("nan")),
    }
    write_csv(os.path.join(args.output_dir, "summary.csv"), [summary])
    print(summary)


if __name__ == "__main__":
    main()
