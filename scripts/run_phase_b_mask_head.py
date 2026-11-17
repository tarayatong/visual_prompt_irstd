"""Run E4-E5 cross-validation for the oracle-point lightweight mask head."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from visual_prompt.backbones.sam2_adapter import Sam2SimilarityPredictor, build_sam2_bplus
from visual_prompt.core.candidate_proposals import connected_instances, snapped_instance_centers
from visual_prompt.core.mask_utils import binary_metrics
from visual_prompt.data.irstd import build_raw_support_paths, raw_irstd_paths
from visual_prompt.evaluation import eval_sliding_window_semantic_mask as ev
from visual_prompt.models.lightweight_heads import MaskVectorMLP, soft_iou_loss
from visual_prompt.pipelines.semantic_prompt_pipeline import get_named_sim_features, load_rgb_and_mask, register_sim_feature_hooks, seed_everything


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in parse_list(value)]


def load_baseline(path: Path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        args = ev.parse_args()
    finally:
        sys.argv = old_argv
    for section in ("args", "flags"):
        for key, value in config.get(section, {}).items():
            setattr(args, key, value)
    return config, args


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


def crop_at(image: np.ndarray, mask: np.ndarray, y: int, x: int, size: int):
    return ev.crop_with_padding(image, mask, y - size // 2, x - size // 2, size)


def gaussian_map(size: int, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    center = size // 2
    return np.exp(-((yy - center) ** 2 + (xx - center) ** 2) / (2 * sigma * sigma)).astype(np.float32)


def feature_signature(config: dict, support_ids: list[str], sample_ids: list[str], crop_size: int, feature_size: int) -> str:
    payload = {
        "config": config,
        "support_ids": support_ids,
        "sample_ids": sample_ids,
        "crop_size": crop_size,
        "feature_size": feature_size,
        "version": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def build_feature_cache(predictor, args, sample_ids, crop_size, feature_size, cache_dir):
    records = []
    for sample_id in tqdm(sample_ids, desc="cache local oracle-point features"):
        image_path, mask_path = raw_irstd_paths(args.data_root, sample_id)
        image, gt, _ = load_rgb_and_mask(image_path, mask_path, (args.image_size, args.image_size))
        instances = connected_instances(gt)
        for component_id, y, x in snapped_instance_centers(instances):
            instance_mask = (instances.labels == component_id).astype(np.uint8)
            image_crop, mask_crop = crop_at(image, instance_mask, y, x, crop_size)
            predictor.set_image(image_crop)
            named = get_named_sim_features(predictor, align=False)
            parts = [F.interpolate(value.float(), size=(feature_size, feature_size), mode="bilinear", align_corners=False) for _, value in named]
            feature = F.normalize(torch.cat(parts, dim=1), dim=1)[0].cpu().half().numpy()
            resized_mask = cv2.resize(mask_crop, (feature_size, feature_size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            contrast = cv2.resize(ev.local_contrast_score(image_crop, args.local_contrast_sigma), (feature_size, feature_size), interpolation=cv2.INTER_LINEAR).astype(np.float16)
            key = f"{sample_id}_c{component_id}"
            np.savez(
                cache_dir / f"{key}.npz",
                feature=feature,
                mask=resized_mask,
                contrast=contrast,
            )
            records.append({
                "key": key,
                "sample_id": sample_id,
                "component_id": component_id,
                "point_y": y,
                "point_x": x,
                "target_count": instances.count,
                "gt_area_full": int(instance_mask.sum()),
                "gt_area_crop": int(mask_crop.sum()),
                "crop_truncated": int(mask_crop.sum() < instance_mask.sum()),
            })
    return records


def load_records(cache_dir: Path, records: list[dict], initial_vector: torch.Tensor, gaussian: torch.Tensor):
    loaded = []
    e0 = F.normalize(initial_vector.float().flatten(), dim=0)
    for record in records:
        data = np.load(cache_dir / f"{record['key']}.npz")
        feature = torch.from_numpy(data["feature"].astype(np.float32))
        mask = torch.from_numpy(data["mask"].astype(np.float32))
        contrast = torch.from_numpy(data["contrast"].astype(np.float32))
        similarity = torch.einsum("chw,c->hw", feature, e0)
        loaded.append({
            **record,
            "feature": feature,
            "mask": mask,
            "contrast": contrast,
            "similarity": similarity,
            "gaussian": gaussian,
        })
    return loaded


def make_folds(records: list[dict], support_ids: set[str], fold_count: int, seed: int):
    image_meta = {}
    for record in records:
        meta = image_meta.setdefault(record["sample_id"], {"target_count": record["target_count"], "area": 0})
        meta["area"] += record["gt_area_full"]
    validation_ids = [sample_id for sample_id in image_meta if sample_id not in support_ids]
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for sample_id in validation_ids:
        count = image_meta[sample_id]["target_count"]
        buckets[min(count, 3)].append(sample_id)
    folds = [[] for _ in range(fold_count)]
    for bucket in buckets.values():
        bucket.sort(key=lambda item: image_meta[item]["area"])
        rng.shuffle(bucket)
        for index, sample_id in enumerate(bucket):
            folds[index % fold_count].append(sample_id)
    return [sorted(fold) for fold in folds]


def batches(items, batch_size, rng):
    order = list(range(len(items)))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [items[index] for index in order[start : start + batch_size]]


def stack_batch(batch, device):
    return tuple(
        torch.stack([item[name] for item in batch]).to(device)
        for name in ("feature", "mask", "similarity", "gaussian", "contrast")
    )


@torch.no_grad()
def evaluate_model(model, items, e0, args, variant, device, fold, seed):
    rows = []
    model.eval()
    for item in items:
        feature, mask, similarity, gaussian, contrast = stack_batch([item], device)
        logits = model(feature, e0, similarity, gaussian, contrast, args.residual_scale, args.temperature)
        pred = (logits[0] > 0).cpu().numpy().astype(np.uint8)
        gt = mask[0].cpu().numpy().astype(np.uint8)
        iou, f1, inter, pred_pixels, gt_pixels = binary_metrics(pred, gt)
        rows.append({
            "variant": variant,
            "fold": fold,
            "seed": seed,
            "sample_id": item["sample_id"],
            "component_id": item["component_id"],
            "target_count": item["target_count"],
            "iou": iou,
            "f1": f1,
            "inter": inter,
            "pred_pixels": pred_pixels,
            "gt_pixels": gt_pixels,
        })
    return rows


def tune_fixed_threshold(train_items):
    values = torch.cat([item["similarity"].flatten() for item in train_items])
    low, high = torch.quantile(values, torch.tensor([0.01, 0.99])).tolist()
    thresholds = torch.linspace(low, high, 161)
    best = (-1.0, 0.0)
    for threshold in thresholds:
        scores = []
        for item in train_items:
            pred = (item["similarity"] > threshold).numpy().astype(np.uint8)
            scores.append(binary_metrics(pred, item["mask"].numpy().astype(np.uint8))[0])
        mean_iou = float(np.mean(scores))
        if mean_iou > best[0]:
            best = (mean_iou, float(threshold))
    return best[1]


def evaluate_fixed(items, threshold, fold):
    rows = []
    for item in items:
        pred = (item["similarity"] > threshold).numpy().astype(np.uint8)
        gt = item["mask"].numpy().astype(np.uint8)
        iou, f1, inter, pred_pixels, gt_pixels = binary_metrics(pred, gt)
        rows.append({
            "variant": "m0_fixed",
            "fold": fold,
            "seed": 0,
            "sample_id": item["sample_id"],
            "component_id": item["component_id"],
            "target_count": item["target_count"],
            "iou": iou,
            "f1": f1,
            "inter": inter,
            "pred_pixels": pred_pixels,
            "gt_pixels": gt_pixels,
            "threshold": threshold,
        })
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["fold"], row["seed"])].append(row)
    fold_rows = []
    for (variant, fold, seed), items in grouped.items():
        fold_rows.append({
            "variant": variant,
            "fold": fold,
            "seed": seed,
            "instances": len(items),
            "mean_instance_iou_percent": 100 * float(np.mean([item["iou"] for item in items])),
            "mean_instance_f1_percent": 100 * float(np.mean([item["f1"] for item in items])),
        })
    variant_rows = []
    for variant in sorted({row["variant"] for row in fold_rows}):
        values = [row["mean_instance_iou_percent"] for row in fold_rows if row["variant"] == variant]
        f1s = [row["mean_instance_f1_percent"] for row in fold_rows if row["variant"] == variant]
        variant_rows.append({
            "variant": variant,
            "fold_seed_runs": len(values),
            "mean_instance_iou_percent": float(np.mean(values)),
            "std_fold_seed_iou_percent": float(np.std(values)),
            "mean_instance_f1_percent": float(np.mean(f1s)),
        })
    return fold_rows, variant_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_ids", required=True)
    parser.add_argument("--support_ids", required=True)
    parser.add_argument("--variants", default="m0_fixed,m1_direct,m2_residual,m6_spatial")
    parser.add_argument("--seeds", default="111")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--feature_size", type=int, default=64)
    parser.add_argument("--gaussian_sigma", type=float, default=4.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--residual_scale", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    cache_dir = output / "artifacts" / "local_features"
    checkpoint_dir = output / "artifacts" / "checkpoints"
    metrics_dir = output / "metrics"
    metadata_dir = output / "metadata"
    for path in (cache_dir, checkpoint_dir, metrics_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    config, baseline = load_baseline(Path(args.baseline_config))
    sample_ids, support_ids = parse_list(args.sample_ids), parse_list(args.support_ids)
    missing_support = set(support_ids) - set(sample_ids)
    if missing_support:
        raise ValueError(f"Support IDs missing from 50 samples: {sorted(missing_support)}")
    signature = feature_signature(config, support_ids, sample_ids, args.crop_size, args.feature_size)
    manifest_path = metadata_dir / "feature_manifest.json"
    old_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    cache_valid = old_manifest.get("signature") == signature and not args.force_recompute

    device = torch.device(baseline.device if torch.cuda.is_available() else "cpu")
    sam = build_sam2_bplus(baseline.sam2_ckpt, baseline.sam2_config, str(device), baseline.encoder_input_size)
    predictor = Sam2SimilarityPredictor(sam, feature_source=baseline.feature_source)
    register_sim_feature_hooks(predictor, baseline.sim_feature_layers)
    support_paths = build_raw_support_paths(baseline.data_root, len(support_ids), support_ids=",".join(support_ids))
    e0, e0_info = ev.build_support_score_token(predictor, support_paths, baseline)
    e0 = F.normalize(e0.flatten().float(), dim=0).to(device)
    write_json(metadata_dir / "initial_vector_info.json", e0_info)

    if cache_valid:
        records = old_manifest["records"]
    else:
        records = build_feature_cache(predictor, baseline, sample_ids, args.crop_size, args.feature_size, cache_dir)
        write_json(manifest_path, {"signature": signature, "records": records})
    del predictor, sam
    torch.cuda.empty_cache()

    gaussian = torch.from_numpy(gaussian_map(args.feature_size, args.gaussian_sigma))
    items = load_records(cache_dir, records, e0.cpu(), gaussian)
    folds = make_folds(records, set(support_ids), args.folds, seed=111)
    write_json(metadata_dir / "folds.json", {"fixed_support_train_only": support_ids, "validation_folds": folds})

    variants = parse_list(args.variants)
    seeds = parse_ints(args.seeds)
    all_rows, training_rows = [], []
    channels = items[0]["feature"].shape[0]
    for fold, val_ids in enumerate(folds):
        val_ids_set = set(val_ids)
        train_items = [item for item in items if item["sample_id"] not in val_ids_set]
        val_items = [item for item in items if item["sample_id"] in val_ids_set]
        if "m0_fixed" in variants:
            threshold = tune_fixed_threshold(train_items)
            all_rows.extend(evaluate_fixed(val_items, threshold, fold))
        for variant in [name for name in variants if name != "m0_fixed"]:
            for seed in seeds:
                seed_everything(seed + fold)
                model = MaskVectorMLP(channels, args.hidden, e0, variant).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                rng = random.Random(seed + fold)
                final_loss = None
                for epoch in range(args.epochs):
                    model.train()
                    losses = []
                    for batch in batches(train_items, args.batch_size, rng):
                        feature, mask, similarity, point_map, contrast = stack_batch(batch, device)
                        logits = model(feature, e0, similarity, point_map, contrast, args.residual_scale, args.temperature)
                        loss = soft_iou_loss(logits, mask)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        losses.append(float(loss.detach().cpu()))
                    final_loss = float(np.mean(losses))
                checkpoint = checkpoint_dir / f"{variant}_fold{fold}_seed{seed}.pt"
                torch.save({"model": model.state_dict(), "variant": variant, "fold": fold, "seed": seed, "channels": channels}, checkpoint)
                all_rows.extend(evaluate_model(model, val_items, e0, args, variant, device, fold, seed))
                training_rows.append({"variant": variant, "fold": fold, "seed": seed, "epochs": args.epochs, "final_train_loss": final_loss})
                write_csv(metrics_dir / "instance_results.csv", all_rows)
                write_csv(metrics_dir / "training_summary.csv", training_rows)

    fold_rows, variant_rows = summarize(all_rows)
    best = max(variant_rows, key=lambda row: row["mean_instance_iou_percent"])
    report = {
        "images": len(sample_ids),
        "instances": len(items),
        "fixed_support_images": len(support_ids),
        "cross_validation_images": len(sample_ids) - len(support_ids),
        "rule_mask_reference_iou_percent": 58.64114088195973,
        "gate_b_percent": 63.64114088195973,
        "best_variant": best,
        "gate_b_passed": best["mean_instance_iou_percent"] >= 63.64114088195973,
    }
    write_csv(metrics_dir / "fold_summary.csv", fold_rows)
    write_csv(metrics_dir / "variant_summary.csv", variant_rows)
    write_json(metrics_dir / "phase_b_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
