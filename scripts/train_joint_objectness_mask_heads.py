"""Jointly fine-tune ObjectnessMLP and MaskVectorMLP with differentiable gating."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scripts.run_global_ablation_201 import crop_at, encode_local
from scripts.run_phase_b_mask_head import gaussian_map, load_baseline
from scripts.run_phase_c_objectness import descriptor, load_arrays, select_threshold
from visual_prompt.backbones.sam2_adapter import Sam2SimilarityPredictor, build_sam2_bplus
from visual_prompt.core.candidate_proposals import connected_instances
from visual_prompt.data.irstd import build_raw_support_paths, raw_irstd_paths
from visual_prompt.evaluation import eval_sliding_window_semantic_mask as ev
from visual_prompt.models.lightweight_heads import MaskVectorMLP, ObjectnessMLP
from visual_prompt.pipelines.semantic_prompt_pipeline import (
    load_rgb_and_mask,
    register_sim_feature_hooks,
    seed_everything,
)


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def build_candidate_items(predictor, baseline, records, sample_ids, e0, crop_size, feature_size):
    eval_by_id = {
        sample_id: [
            row for row in records
            if row["sample_id"] == sample_id and row["source"] == "eval"
        ]
        for sample_id in sample_ids
    }
    items = []
    for sample_id in tqdm(sample_ids, desc="cache joint candidate crops"):
        image_path, mask_path = raw_irstd_paths(baseline.data_root, sample_id)
        image, gt, _ = load_rgb_and_mask(
            image_path, mask_path, (baseline.image_size, baseline.image_size)
        )
        instances = connected_instances(gt)
        for row in eval_by_id[sample_id]:
            y, x = int(row["y"]), int(row["x"])
            image_crop, _, feature = encode_local(
                predictor, image, y, x, crop_size, feature_size
            )
            assignment = int(row["assignment"])
            target_full = (instances.labels == assignment).astype(np.uint8) if assignment > 0 else np.zeros_like(gt)
            _, target = crop_at(image, target_full, y, x, crop_size)
            if target.shape != (feature_size, feature_size):
                target = cv2.resize(target, (feature_size, feature_size), interpolation=cv2.INTER_NEAREST)
            contrast = cv2.resize(
                ev.local_contrast_score(image_crop, baseline.local_contrast_sigma),
                (feature_size, feature_size), interpolation=cv2.INTER_LINEAR,
            )
            items.append({
                "row": row,
                "sample_id": sample_id,
                "assignment": assignment,
                "label": int(assignment > 0),
                "target_count": int(row["target_count"]),
                "feature": feature[0].cpu().half(),
                "target": torch.from_numpy(target.astype(np.uint8)),
                "contrast": torch.from_numpy(contrast).half(),
            })
    return items


def probability_iou_loss(probability: torch.Tensor, target: torch.Tensor, epsilon=1e-6):
    intersection = (probability * target).sum(dim=(-2, -1))
    union = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) - intersection
    return (1.0 - (intersection + epsilon) / (union + epsilon)).mean()


def balanced_indices(items, allowed_ids, rng):
    positive = [i for i, item in enumerate(items) if item["sample_id"] in allowed_ids and item["label"] == 1]
    negative = [i for i, item in enumerate(items) if item["sample_id"] in allowed_ids and item["label"] == 0]
    count = min(len(positive), len(negative))
    chosen = rng.sample(positive, count) + rng.sample(negative, count)
    rng.shuffle(chosen)
    return chosen


def stack_items(items, indices, arrays, proto, mean, std, e0, gaussian, device):
    selected = [items[i] for i in indices]
    matrix = np.stack([descriptor(item["row"], arrays, "f4", proto) for item in selected])
    matrix = (matrix - mean) / std
    feature = torch.stack([item["feature"] for item in selected]).float().to(device)
    target = torch.stack([item["target"] for item in selected]).float().to(device)
    contrast = torch.stack([item["contrast"] for item in selected]).float().to(device)
    similarity = torch.einsum("bchw,c->bhw", feature, e0)
    labels = torch.tensor([item["label"] for item in selected], dtype=torch.float32, device=device)
    return (
        torch.from_numpy(matrix).float().to(device), feature, target, contrast,
        similarity, gaussian[None].expand(len(selected), -1, -1), labels,
    )


def jointly_finetune(
    object_model, mask_model, items, allowed_ids, arrays, proto, mean, std,
    e0, gaussian, args, device, seed,
):
    parameters = list(object_model.parameters()) + list(mask_model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = random.Random(seed)
    history = []
    object_model.train()
    mask_model.train()
    for epoch in range(args.epochs):
        indices = balanced_indices(items, allowed_ids, rng)
        totals = []
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            desc, feature, target, contrast, similarity, point_map, labels = stack_items(
                items, batch_indices, arrays, proto, mean, std, e0, gaussian, device
            )
            object_logits = object_model(desc)
            mask_logits = mask_model(
                feature, e0, similarity, point_map, contrast,
                args.residual_scale, args.temperature,
            )
            object_probability = torch.sigmoid(object_logits)[:, None, None]
            mask_probability = torch.sigmoid(mask_logits)
            gated_probability = object_probability * mask_probability
            object_loss = F.binary_cross_entropy_with_logits(object_logits, labels)
            positive = labels > 0.5
            negative = ~positive
            zero = object_logits.sum() * 0.0
            mask_loss = (
                probability_iou_loss(mask_probability[positive], target[positive])
                if positive.any() else zero
            )
            joint_positive = (
                probability_iou_loss(gated_probability[positive], target[positive])
                if positive.any() else zero
            )
            joint_negative = gated_probability[negative].mean() if negative.any() else zero
            joint_loss = joint_positive + args.negative_area_weight * joint_negative
            loss = object_loss + args.mask_loss_weight * mask_loss + args.joint_loss_weight * joint_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals.append((float(loss.detach()), float(object_loss.detach()), float(mask_loss.detach()), float(joint_loss.detach())))
        history.append(np.mean(totals, axis=0).tolist())
    return history


@torch.no_grad()
def object_scores(model, items, indices, arrays, proto, mean, std, device):
    matrix = np.stack([descriptor(items[i]["row"], arrays, "f4", proto) for i in indices])
    matrix = (matrix - mean) / std
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(matrix).float().to(device))).cpu().numpy()


def load_pair(object_path, mask_path, e0, args, device):
    object_ckpt = torch.load(object_path, map_location="cpu", weights_only=False)
    if object_ckpt.get("feature_protocol") != "full_image_encode_then_roi_sample_v1":
        raise RuntimeError(
            f"{object_path} uses the retired crop-encoding objectness protocol; rerun Phase C/final training."
        )
    mask_ckpt = torch.load(mask_path, map_location="cpu", weights_only=False)
    object_model = ObjectnessMLP(int(object_ckpt["input_dim"]), args.object_hidden).to(device)
    object_model.load_state_dict(object_ckpt["model"])
    mask_model = MaskVectorMLP(int(mask_ckpt["channels"]), int(mask_ckpt.get("hidden", args.mask_hidden)), e0, mask_ckpt.get("variant", "m2_residual")).to(device)
    mask_model.load_state_dict(mask_ckpt["model"])
    return object_model, mask_model, object_ckpt, mask_ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_config", required=True)
    parser.add_argument("--phase_c_dir", required=True)
    parser.add_argument("--phase_c_refine_dir", required=True)
    parser.add_argument("--phase_e_dir", required=True)
    parser.add_argument("--separate_heads_dir", required=True)
    parser.add_argument("--folds_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_ids", required=True)
    parser.add_argument("--support_ids", required=True)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--feature_size", type=int, default=64)
    parser.add_argument("--object_hidden", type=int, default=128)
    parser.add_argument("--mask_hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mask_loss_weight", type=float, default=0.5)
    parser.add_argument("--joint_loss_weight", type=float, default=1.0)
    parser.add_argument("--negative_area_weight", type=float, default=0.5)
    parser.add_argument("--residual_scale", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    output = Path(args.output_dir)
    (output / "artifacts").mkdir(parents=True, exist_ok=True)
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    _, baseline = load_baseline(Path(args.baseline_config))
    sample_ids, support_ids = parse_list(args.sample_ids), parse_list(args.support_ids)
    device = torch.device(baseline.device if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    sam = build_sam2_bplus(
        baseline.sam2_ckpt, baseline.sam2_config, str(device), baseline.encoder_input_size
    )
    predictor = Sam2SimilarityPredictor(sam, feature_source=baseline.feature_source)
    register_sim_feature_hooks(predictor, baseline.sim_feature_layers)
    support_paths = build_raw_support_paths(
        baseline.data_root, len(support_ids), support_ids=",".join(support_ids)
    )
    e0, _ = ev.build_support_score_token(predictor, support_paths, baseline)
    e0 = F.normalize(e0.flatten().float(), dim=0).to(device)

    phase_c = Path(args.phase_c_dir)
    records = json.loads((phase_c / "metadata" / "feature_manifest.json").read_text(encoding="utf-8"))["records"]
    arrays = load_arrays(phase_c / "artifacts" / "candidate_features", sample_ids)
    items = build_candidate_items(
        predictor, baseline, records, sample_ids, e0,
        args.crop_size, args.feature_size,
    )
    del predictor, sam
    torch.cuda.empty_cache()
    gaussian = torch.from_numpy(gaussian_map(args.feature_size, 4.0)).float().to(device)

    folds = json.loads(Path(args.folds_json).read_text(encoding="utf-8"))["validation_folds"]
    oof_rows, oof_values, fold_reports = [], [], []
    non_support_ids = set(sample_ids) - set(support_ids)
    for fold, val_ids_list in enumerate(folds):
        val_ids = set(val_ids_list)
        train_ids = set(sample_ids) - val_ids
        object_path = Path(args.phase_c_refine_dir) / "artifacts" / "checkpoints" / f"e9r_f4_fold{fold}.pt"
        mask_path = Path(args.phase_e_dir) / "artifacts" / "checkpoints" / f"a2_point_jitter_fold{fold}.pt"
        object_model, mask_model, object_ckpt, _ = load_pair(object_path, mask_path, e0, args, device)
        mean = np.asarray(object_ckpt["mean"], dtype=np.float32)
        std = np.asarray(object_ckpt["std"], dtype=np.float32)
        proto = (object_ckpt["prototype_target"], object_ckpt["prototype_background"])
        history = jointly_finetune(
            object_model, mask_model, items, train_ids, arrays, proto, mean, std,
            e0, gaussian, args, device, args.seed + fold,
        )
        val_indices = [i for i, item in enumerate(items) if item["sample_id"] in val_ids]
        scores = object_scores(object_model, items, val_indices, arrays, proto, mean, std, device)
        for index, score in zip(val_indices, scores):
            item = items[index]
            oof_rows.append({
                "sample_id": item["sample_id"], "assignment": item["assignment"],
                "label": item["label"], "target_count": item["target_count"],
            })
            oof_values.append(float(score))
        fold_reports.append({"fold": fold, "train_images": len(train_ids), "val_images": len(val_ids), "final_losses": history[-1]})
        del object_model, mask_model
        torch.cuda.empty_cache()

    oof_values_np = np.asarray(oof_values, dtype=np.float32)
    threshold, oof_metric, _ = select_threshold(oof_rows, oof_values_np)

    separate_dir = Path(args.separate_heads_dir) / "artifacts"
    object_model, mask_model, object_ckpt, mask_ckpt = load_pair(
        separate_dir / "objectness_final.pt", separate_dir / "mask_final_a2.pt", e0, args, device
    )
    mean = np.asarray(object_ckpt["mean"], dtype=np.float32)
    std = np.asarray(object_ckpt["std"], dtype=np.float32)
    proto = (object_ckpt["prototype_target"], object_ckpt["prototype_background"])
    final_history = jointly_finetune(
        object_model, mask_model, items, set(sample_ids), arrays, proto, mean, std,
        e0, gaussian, args, device, args.seed + 100,
    )
    checkpoint = {
        "object_model": object_model.state_dict(),
        "mask_model": mask_model.state_dict(),
        "object_input_dim": int(object_ckpt["input_dim"]),
        "object_hidden": args.object_hidden,
        "mask_channels": int(mask_ckpt["channels"]),
        "mask_hidden": int(mask_ckpt.get("hidden", args.mask_hidden)),
        "mask_variant": mask_ckpt.get("variant", "m2_residual"),
        "mean": mean,
        "std": std,
        "prototype_target": proto[0],
        "prototype_background": proto[1],
        "object_feature_protocol": "full_image_encode_then_roi_sample_v1",
        "threshold": float(threshold),
        "threshold_source": "joint-training pooled OOF predictions from 45 non-support images",
        "residual_scale": args.residual_scale,
        "temperature": args.temperature,
        "sample_ids": sample_ids,
        "loss": {
            "object_bce": 1.0,
            "mask_iou": args.mask_loss_weight,
            "gated_joint": args.joint_loss_weight,
            "negative_area": args.negative_area_weight,
        },
    }
    torch.save(checkpoint, output / "artifacts" / "joint_final.pt")
    report = {
        "protocol": "warm-start separate heads; objectness descriptors come from one full-image SAM2 encoding; jointly optimize BCE + mask IoU + differentiable objectness-gated mask loss",
        "training_images": len(sample_ids),
        "candidate_crops": len(items),
        "positive_candidate_crops": sum(item["label"] for item in items),
        "negative_candidate_crops": sum(1 - item["label"] for item in items),
        "oof_candidates": len(oof_rows),
        "oof_threshold": float(threshold),
        "oof_recall_percent": 100 * oof_metric["recall"],
        "oof_precision_percent": 100 * oof_metric["precision"],
        "oof_f1_percent": 100 * oof_metric["f1"],
        "oof_false_candidates_per_image": oof_metric["fp_per_image"],
        "folds": fold_reports,
        "final_losses": final_history[-1],
    }
    write_json(output / "metrics" / "joint_training_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
