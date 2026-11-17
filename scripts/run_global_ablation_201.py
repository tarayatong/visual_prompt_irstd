"""Evaluate the standardized global ablation on all 201 official test images."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scripts.run_phase_b_mask_head import gaussian_map, load_baseline
from scripts.run_phase_c_objectness import descriptor, map_stats
from visual_prompt.backbones.sam2_adapter import Sam2SimilarityPredictor, build_sam2_bplus
from visual_prompt.core.candidate_proposals import (
    candidate_counts,
    centroid_detection_counts,
    connected_instances,
    greedy_instance_matches,
    greedy_score_peaks,
    snapped_instance_centers,
)
from visual_prompt.core.mask_utils import binary_metrics, support_area_prior
from visual_prompt.data.irstd import RawIRSTDDataset, build_raw_support_paths
from visual_prompt.evaluation import eval_sliding_window_semantic_mask as ev
from visual_prompt.models.lightweight_heads import MaskVectorMLP, ObjectnessMLP
from visual_prompt.models.full_image_objectness import describe_full_image_points
from visual_prompt.pipelines.semantic_prompt_pipeline import (
    get_named_sim_features,
    load_rgb_and_mask,
    register_sim_feature_hooks,
    seed_everything,
)


SYSTEMS = [
    "g0_sam2_positive_only",
    "g1_negative_correction_top1",
    "g1_negative_correction_top5",
    "g2_mask_standard_top1",
    "g2_mask_standard_top5",
    "g2u_mask_oracle_gt",
    "g2c_mask_oracle_candidate",
    "g3_objectness_rule",
    "g4s_separate_train_joint_infer",
    "g4j_joint_train_joint_infer",
]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def crop_at(image: np.ndarray, value: np.ndarray, y: int, x: int, size: int):
    return ev.crop_with_padding(image, value, y - size // 2, x - size // 2, size)


@torch.no_grad()
def encode_local(predictor, image, y, x, crop_size, mask_size):
    blank = np.zeros(image.shape[:2], dtype=np.uint8)
    image_crop, _ = crop_at(image, blank, y, x, crop_size)
    predictor.set_image(image_crop)
    named = get_named_sim_features(predictor, align=False)
    parts32 = [
        F.interpolate(value.float(), size=(32, 32), mode="bilinear", align_corners=False)
        for _, value in named
    ]
    parts64 = [
        F.interpolate(value.float(), size=(mask_size, mask_size), mode="bilinear", align_corners=False)
        for _, value in named
    ]
    feature32 = F.normalize(torch.cat(parts32, dim=1), dim=1)[0]
    feature64 = F.normalize(torch.cat(parts64, dim=1), dim=1)
    return image_crop, feature32, feature64


def candidate_arrays(feature32, image_crop, score_crop, contrast_sigma):
    channels, height, width = feature32.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=feature32.device),
        torch.arange(width, device=feature32.device), indexing="ij"
    )
    radius = torch.sqrt((yy - height // 2).float() ** 2 + (xx - width // 2).float() ** 2)
    ring = (radius >= height / 8) & (radius <= 3 * height / 8)
    center = feature32[:, height // 2, width // 2]
    mean = feature32.mean(dim=(-2, -1))
    maximum = feature32.amax(dim=(-2, -1))
    ring_mean = feature32[:, ring].mean(dim=1)
    gray = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    contrast = ev.local_contrast_score(image_crop, contrast_sigma)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = ev.robust_norm(cv2.magnitude(gx, gy))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    top_hat = ev.robust_norm(gray - opened)
    sim_base = map_stats(score_crop)
    sim = np.concatenate([sim_base, [sim_base[1] - sim_base[2]]]).astype(np.float32)
    return {
        "sim": sim,
        "center": center.cpu().numpy().astype(np.float32),
        "mean": mean.cpu().numpy().astype(np.float32),
        "max": maximum.cpu().numpy().astype(np.float32),
        "diff": (center - ring_mean).cpu().numpy().astype(np.float32),
        "hand": np.concatenate([map_stats(value) for value in (gray, contrast, gradient, top_hat)]).astype(np.float32),
    }


@torch.no_grad()
def infer_mask(mask_model, feature64, image_crop, e0, gaussian, baseline, residual_scale, temperature):
    similarity = torch.einsum("bchw,c->bhw", feature64, e0)
    contrast_np = cv2.resize(
        ev.local_contrast_score(image_crop, baseline.local_contrast_sigma),
        (feature64.shape[-1], feature64.shape[-2]), interpolation=cv2.INTER_LINEAR,
    )
    contrast = torch.from_numpy(contrast_np).float().to(feature64.device)[None]
    logits = mask_model(
        feature64, e0, similarity, gaussian[None], contrast, residual_scale, temperature
    )
    return (logits[0] > 0).cpu().numpy().astype(np.uint8)


def paste_local(full: np.ndarray, local: np.ndarray, y: int, x: int):
    size = local.shape[0]
    y0, x0 = y - size // 2, x - size // 2
    src_y0, src_x0 = max(0, -y0), max(0, -x0)
    dst_y0, dst_x0 = max(0, y0), max(0, x0)
    height = min(size - src_y0, full.shape[0] - dst_y0)
    width = min(size - src_x0, full.shape[1] - dst_x0)
    if height > 0 and width > 0:
        full[dst_y0 : dst_y0 + height, dst_x0 : dst_x0 + width] |= local[
            src_y0 : src_y0 + height, src_x0 : src_x0 + width
        ]


def local_union(shape, point_masks, indices):
    result = np.zeros(shape, dtype=np.uint8)
    for index in indices:
        y, x, mask = point_masks[index]
        paste_local(result, mask, y, x)
    return result


def rule_union(score, image, points, support_area, baseline):
    result = np.zeros(score.shape, dtype=np.uint8)
    for y, x in points:
        _, info = ev.extract_final_mask(
            score, image, support_area, {"top_y": y, "top_x": x}, baseline
        )
        result |= info["pred"].astype(np.uint8)
    return result


def add_metrics(row, system, pred, gt):
    iou, f1, inter, pred_pixels, gt_pixels = binary_metrics(pred, gt)
    match05 = greedy_instance_matches(pred, gt, 0.5)
    detection = centroid_detection_counts(pred, gt)
    row.update({
        f"{system}_iou": iou,
        f"{system}_f1": f1,
        f"{system}_inter": inter,
        f"{system}_pred_pixels": pred_pixels,
        "gt_pixels": gt_pixels,
        f"{system}_tp05": match05["tp"],
        f"{system}_fp05": match05["fp"],
        f"{system}_fn05": match05["fn"],
        f"{system}_detected": detection["detected_targets"],
        f"{system}_false_pixels": detection["false_alarm_pixels"],
    })


def group_names(target_count):
    result = ["all", "single" if target_count == 1 else "multi"]
    if target_count >= 3:
        result.append("three_plus")
    return result


def summarize_systems(rows):
    summaries = []
    for system in SYSTEMS:
        grouped = defaultdict(list)
        for row in rows:
            for group in group_names(row["target_count"]):
                grouped[group].append(row)
        for group, items in grouped.items():
            inter = sum(item[f"{system}_inter"] for item in items)
            pred_pixels = sum(item[f"{system}_pred_pixels"] for item in items)
            gt_pixels = sum(item["gt_pixels"] for item in items)
            union = pred_pixels + gt_pixels - inter
            tp = sum(item[f"{system}_tp05"] for item in items)
            fp = sum(item[f"{system}_fp05"] for item in items)
            fn = sum(item[f"{system}_fn05"] for item in items)
            targets = sum(item["target_count"] for item in items)
            summaries.append({
                "system": system,
                "group": group,
                "images": len(items),
                "targets": targets,
                "global_iou_percent": 100 * inter / max(1, union),
                "mean_image_iou_percent": 100 * float(np.mean([item[f"{system}_iou"] for item in items])),
                "pixel_f1_percent": 100 * (2 * inter / max(1, pred_pixels + gt_pixels)),
                "pd_percent": 100 * sum(item[f"{system}_detected"] for item in items) / max(1, targets),
                "fa_per_million": 1e6 * (pred_pixels-inter) / max(
                    1, sum(item["image_pixels"] for item in items)
                ),
                "instance_recall_iou05_percent": 100 * tp / max(1, tp + fn),
                "instance_precision_iou05_percent": 100 * tp / max(1, tp + fp),
            })
    return summaries


def summarize_points(rows):
    result = []
    for method in sorted({row["method"] for row in rows}):
        items = [row for row in rows if row["method"] == method]
        targets = sum(row["target_count"] for row in items)
        selected = sum(row["selected"] for row in items)
        true_candidates = sum(row["true_candidates"] for row in items)
        result.append({
            "method": method,
            "images": len(items),
            "targets": targets,
            "target_recall_exact_percent": 100 * sum(row["recalled_exact"] for row in items) / max(1, targets),
            "target_recall_dilate3_percent": 100 * sum(row["recalled_dilate3"] for row in items) / max(1, targets),
            "candidate_precision_dilate3_percent": 100 * true_candidates / max(1, selected),
            "selected_per_image": selected / max(1, len(items)),
            "false_or_duplicate_per_image": sum(row["false_or_duplicate"] for row in items) / max(1, len(items)),
            "all_targets_recalled_dilate3_percent": 100 * float(np.mean([row["all_dilate3"] for row in items])),
        })
    return result


def load_models(final_dir, device, e0):
    object_ckpt = torch.load(final_dir / "artifacts" / "objectness_final.pt", map_location="cpu", weights_only=False)
    if object_ckpt.get("feature_protocol") != "full_image_encode_then_roi_sample_v1":
        raise RuntimeError(
            "Objectness checkpoint uses the retired crop-encoding protocol. "
            "Re-run Phase C and final-head training before evaluation."
        )
    object_model = ObjectnessMLP(int(object_ckpt["input_dim"]), int(object_ckpt["hidden"])).to(device)
    object_model.load_state_dict(object_ckpt["model"])
    object_model.eval()
    mask_ckpt = torch.load(final_dir / "artifacts" / "mask_final_a2.pt", map_location="cpu", weights_only=False)
    mask_model = MaskVectorMLP(
        int(mask_ckpt["channels"]), int(mask_ckpt["hidden"]), e0, "m2_residual"
    ).to(device)
    mask_model.load_state_dict(mask_ckpt["model"])
    mask_model.eval()
    return object_model, object_ckpt, mask_model, mask_ckpt


def load_joint_models(joint_dir, device, e0):
    checkpoint = torch.load(
        joint_dir / "artifacts" / "joint_final.pt", map_location="cpu", weights_only=False
    )
    if checkpoint.get("object_feature_protocol") != "full_image_encode_then_roi_sample_v1":
        raise RuntimeError(
            "Joint checkpoint uses the retired crop-encoding objectness protocol. "
            "Re-run joint-head training before evaluation."
        )
    object_model = ObjectnessMLP(
        int(checkpoint["object_input_dim"]), int(checkpoint["object_hidden"])
    ).to(device)
    object_model.load_state_dict(checkpoint["object_model"])
    object_model.eval()
    mask_model = MaskVectorMLP(
        int(checkpoint["mask_channels"]), int(checkpoint["mask_hidden"]), e0, "m2_residual"
    ).to(device)
    mask_model.load_state_dict(checkpoint["mask_model"])
    mask_model.eval()
    return object_model, mask_model, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_config", required=True)
    parser.add_argument("--final_heads_dir", required=True)
    parser.add_argument("--joint_heads_dir", required=True)
    parser.add_argument("--corrected_score_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--feature_size", type=int, default=64)
    parser.add_argument("--objectness_grid_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--nms_radius", type=int, default=7)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output_dir)
    raw_score_dir = output / "artifacts" / "scores_positive_only"
    corrected_score_dir = Path(args.corrected_score_dir)
    metrics_dir = output / "metrics"
    metadata_dir = output / "metadata"
    for path in (raw_score_dir, metrics_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    _, baseline = load_baseline(Path(args.baseline_config))
    raw_args = copy.deepcopy(baseline)
    raw_args.use_vector_correction = False
    device = torch.device(baseline.device if torch.cuda.is_available() else "cpu")
    seed_everything(baseline.seed)
    dataset = RawIRSTDDataset(baseline.data_root, baseline.split, baseline.image_size)
    total = len(dataset) if args.max_samples <= 0 else min(args.max_samples, len(dataset))
    support_paths = build_raw_support_paths(
        baseline.data_root, baseline.k_shot, support_ids=baseline.support_ids
    )
    support_area = int(max(1, round(
        support_area_prior(support_paths, baseline.image_size, baseline.area_prior) * baseline.area_scale
    )))

    sam = build_sam2_bplus(
        baseline.sam2_ckpt, baseline.sam2_config, str(device), baseline.encoder_input_size
    )
    predictor = Sam2SimilarityPredictor(sam, feature_source=baseline.feature_source)
    register_sim_feature_hooks(predictor, baseline.sim_feature_layers)
    raw_token, raw_info = ev.build_support_score_token(predictor, support_paths, raw_args)
    corrected_token, corrected_info = ev.build_support_score_token(predictor, support_paths, baseline)
    e0 = F.normalize(corrected_token.flatten().float(), dim=0).to(device)
    object_model, object_ckpt, mask_model, mask_ckpt = load_models(
        Path(args.final_heads_dir), device, e0
    )
    joint_object_model, joint_mask_model, joint_ckpt = load_joint_models(
        Path(args.joint_heads_dir), device, e0
    )
    gaussian = torch.from_numpy(gaussian_map(args.feature_size, 4.0)).float().to(device)
    proto = (object_ckpt["prototype_target"], object_ckpt["prototype_background"])
    object_mean = np.asarray(object_ckpt["mean"], dtype=np.float32)
    object_std = np.asarray(object_ckpt["std"], dtype=np.float32)
    object_threshold = float(object_ckpt["threshold"])
    joint_proto = (joint_ckpt["prototype_target"], joint_ckpt["prototype_background"])
    joint_mean = np.asarray(joint_ckpt["mean"], dtype=np.float32)
    joint_std = np.asarray(joint_ckpt["std"], dtype=np.float32)
    joint_threshold = float(joint_ckpt["threshold"])
    write_json(metadata_dir / "support_tokens.json", {
        "positive_only": raw_info,
        "negative_correction": corrected_info,
        "support_area": support_area,
        "objectness_threshold": object_threshold,
        "objectness_threshold_source": object_ckpt["threshold_source"],
        "joint_objectness_threshold": joint_threshold,
        "joint_objectness_threshold_source": joint_ckpt["threshold_source"],
    })

    image_rows, point_rows, candidate_score_rows = [], [], []
    for index in tqdm(range(total), desc="global ablation test201"):
        sample_id = dataset.sample_ids[index]
        image, gt, _ = load_rgb_and_mask(
            dataset.images[index], dataset.masks[index], (baseline.image_size, baseline.image_size)
        )
        raw_path = raw_score_dir / f"{sample_id}.npy"
        if raw_path.exists():
            raw_score = np.load(raw_path)
        else:
            raw_score, _ = ev.compute_stitched_similarity(predictor, image, raw_token, raw_args)
            np.save(raw_path, raw_score.astype(np.float32))
        corrected_path = corrected_score_dir / f"{sample_id}.npy"
        if corrected_path.exists():
            corrected_score = np.load(corrected_path)
        else:
            corrected_score, _ = ev.compute_stitched_similarity(
                predictor, image, corrected_token, baseline
            )
            np.save(corrected_path, corrected_score.astype(np.float32))

        instances = connected_instances(gt)
        raw_points = greedy_score_peaks(raw_score, args.top_k, args.nms_radius)
        corrected_points = greedy_score_peaks(corrected_score, args.top_k, args.nms_radius)
        corrected_xy = [(y, x) for y, x, _ in corrected_points]
        descriptor_parts = describe_full_image_points(
            predictor,
            image,
            corrected_score,
            corrected_xy,
            args.crop_size,
            baseline.local_contrast_sigma,
            args.objectness_grid_size,
        )
        descriptor_rows = [
            {"sample_id": sample_id, "point_index": point_index}
            for point_index in range(len(corrected_points))
        ]
        arrays = {sample_id: descriptor_parts}
        matrix = np.stack([descriptor(row, arrays, "f4", proto) for row in descriptor_rows])
        with torch.no_grad():
            object_logits = object_model(
                torch.from_numpy((matrix - object_mean) / object_std).to(device)
            )
            object_scores = torch.sigmoid(object_logits).cpu().numpy()
            object_logits = object_logits.cpu().numpy()
        selected = [i for i, value in enumerate(object_scores) if value >= object_threshold]
        joint_matrix = np.stack([
            descriptor(row, arrays, "f4", joint_proto) for row in descriptor_rows
        ])
        with torch.no_grad():
            joint_object_logits = joint_object_model(
                torch.from_numpy((joint_matrix - joint_mean) / joint_std).to(device)
            )
            joint_object_scores = torch.sigmoid(joint_object_logits).cpu().numpy()
            joint_object_logits = joint_object_logits.cpu().numpy()
        joint_selected = [
            i for i, value in enumerate(joint_object_scores) if value >= joint_threshold
        ]

        # The ablation still needs masks for every candidate for G2 and oracle rows.
        # A deployment path can run MaskHead only for ``selected`` after this gate.
        candidate_masks = []
        joint_candidate_masks = []
        for point_index, (y, x, _) in enumerate(corrected_points):
            image_crop, _, feature64 = encode_local(
                predictor, image, y, x, args.crop_size, args.feature_size
            )
            local = infer_mask(
                mask_model, feature64, image_crop, e0, gaussian, baseline,
                float(mask_ckpt["residual_scale"]), float(mask_ckpt["temperature"]),
            )
            candidate_masks.append((y, x, local))
            joint_local = infer_mask(
                joint_mask_model, feature64, image_crop, e0, gaussian, baseline,
                float(joint_ckpt["residual_scale"]), float(joint_ckpt["temperature"]),
            )
            joint_candidate_masks.append((y, x, joint_local))
        oracle_candidate_indices, seen = [], set()
        dilated_counts = candidate_counts(instances, corrected_points, tolerance=3)
        for point_index, ((y, x, similarity_score), assignment) in enumerate(
            zip(corrected_points, dilated_counts["assignments"])
        ):
            candidate_score_rows.append({
                "sample_id": sample_id,
                "rank": point_index + 1,
                "y": y,
                "x": x,
                "similarity_score": float(similarity_score),
                "assignment_dilate3": int(assignment),
                "label_dilate3": int(assignment > 0),
                "objectness_logit": float(object_logits[point_index]),
                "objectness_score": float(object_scores[point_index]),
                "objectness_threshold": object_threshold,
                "objectness_kept": int(point_index in selected),
                "joint_objectness_logit": float(joint_object_logits[point_index]),
                "joint_objectness_score": float(joint_object_scores[point_index]),
                "joint_objectness_threshold": joint_threshold,
                "joint_objectness_kept": int(point_index in joint_selected),
            })
        for point_index, assignment in enumerate(dilated_counts["assignments"]):
            if assignment > 0 and assignment not in seen:
                oracle_candidate_indices.append(point_index)
                seen.add(assignment)

        gt_point_masks = []
        for _, y, x in snapped_instance_centers(instances):
            image_crop, _, feature64 = encode_local(
                predictor, image, y, x, args.crop_size, args.feature_size
            )
            local = infer_mask(
                mask_model, feature64, image_crop, e0, gaussian, baseline,
                float(mask_ckpt["residual_scale"]), float(mask_ckpt["temperature"]),
            )
            gt_point_masks.append((y, x, local))

        raw_xy = [(y, x) for y, x, _ in raw_points]
        predictions = {
            "g0_sam2_positive_only": rule_union(
                raw_score, image, raw_xy[:1], support_area, raw_args
            ),
            "g1_negative_correction_top1": rule_union(
                corrected_score, image, corrected_xy[:1], support_area, baseline
            ),
            "g1_negative_correction_top5": rule_union(
                corrected_score, image, corrected_xy, support_area, baseline
            ),
            "g2_mask_standard_top1": local_union(gt.shape, candidate_masks, [0]),
            "g2_mask_standard_top5": local_union(
                gt.shape, candidate_masks, range(len(candidate_masks))
            ),
            "g2u_mask_oracle_gt": local_union(
                gt.shape, gt_point_masks, range(len(gt_point_masks))
            ),
            "g2c_mask_oracle_candidate": local_union(
                gt.shape, candidate_masks, oracle_candidate_indices
            ),
            "g3_objectness_rule": rule_union(
                corrected_score, image, [corrected_xy[i] for i in selected],
                support_area, baseline,
            ),
            "g4s_separate_train_joint_infer": local_union(
                gt.shape, candidate_masks, selected
            ),
            "g4j_joint_train_joint_infer": local_union(
                gt.shape, joint_candidate_masks, joint_selected
            ),
        }
        image_row = {
            "sample_id": sample_id,
            "target_count": instances.count,
            "image_pixels": int(gt.size),
            "objectness_selected": len(selected),
            "joint_objectness_selected": len(joint_selected),
        }
        for system, pred in predictions.items():
            add_metrics(image_row, system, pred, gt)
        image_rows.append(image_row)

        point_methods = {
            "positive_only_top1": raw_points[:1],
            "positive_only_top5_nms7": raw_points,
            "negative_correction_top1": corrected_points[:1],
            "negative_correction_top5_nms7": corrected_points,
            "objectness_selected": [corrected_points[i] for i in selected],
            "joint_objectness_selected": [corrected_points[i] for i in joint_selected],
        }
        for method, points in point_methods.items():
            exact = candidate_counts(instances, points, tolerance=0)
            dilated = candidate_counts(instances, points, tolerance=3)
            true_candidates = len(points) - dilated["false_candidates"]
            point_rows.append({
                "sample_id": sample_id,
                "method": method,
                "target_count": instances.count,
                "selected": len(points),
                "recalled_exact": exact["recalled_targets"],
                "recalled_dilate3": dilated["recalled_targets"],
                "true_candidates": true_candidates,
                "false_or_duplicate": len(points) - dilated["recalled_targets"],
                "all_dilate3": dilated["all_targets_recalled"],
            })

    system_summary = summarize_systems(image_rows)
    point_summary = summarize_points(point_rows)
    write_csv(metrics_dir / "image_metrics.csv", image_rows)
    write_csv(metrics_dir / "system_summary.csv", system_summary)
    write_csv(metrics_dir / "point_image_metrics.csv", point_rows)
    write_csv(metrics_dir / "point_summary.csv", point_summary)
    write_csv(metrics_dir / "top5_objectness_scores.csv", candidate_score_rows)
    all_systems = {row["system"]: row for row in system_summary if row["group"] == "all"}
    report = {
        "test_images": len(image_rows),
        "test_targets": int(sum(row["target_count"] for row in image_rows)),
        "train_images_for_heads": 50,
        "systems_all": all_systems,
        "point_summary": point_summary,
    }
    write_json(metrics_dir / "global_ablation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
