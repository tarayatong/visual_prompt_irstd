"""Build shared candidate features or reproduce the retained F4 classifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from visual_prompt.core.candidate_proposals import connected_instances, greedy_score_peaks, point_instance_id, snapped_instance_centers
from visual_prompt.data.irstd import build_raw_support_paths, raw_irstd_paths
from visual_prompt.evaluation import eval_sliding_window_semantic_mask as ev
from visual_prompt.models.full_image_objectness import describe_full_image_points
from visual_prompt.models.lightweight_heads import ObjectnessMLP
from visual_prompt.pipelines.semantic_prompt_pipeline import load_rgb_and_mask, register_sim_feature_hooks, seed_everything


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


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


def cache_signature(config: dict, baseline, sample_ids: list[str], support_ids: list[str], args) -> str:
    checkpoint = Path(baseline.sam2_ckpt)
    stat = checkpoint.stat()
    payload = {
        "config": config,
        "sample_ids": sample_ids,
        "support_ids": support_ids,
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "crop_size": args.crop_size,
        "hard_pool": args.hard_pool,
        "feature_protocol": "full_image_encode_then_roi_sample_v1",
        "objectness_grid_size": args.objectness_grid_size,
        "version": 3,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def map_stats(value: np.ndarray) -> np.ndarray:
    size = value.shape[0]
    center = size // 2
    yy, xx = np.mgrid[:size, :size]
    radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    ring = (radius >= size / 8) & (radius <= 3 * size / 8)
    ring_mean = float(value[ring].mean()) if ring.any() else float(value.mean())
    return np.asarray([
        value[center, center], value.max(), value.mean(), value.std(),
        value[center, center] - ring_mean,
    ], dtype=np.float32)


def crop_at(image: np.ndarray, mask: np.ndarray, y: int, x: int, size: int):
    return ev.crop_with_padding(image, mask, y - size // 2, x - size // 2, size)


def point_key(y: int, x: int) -> str:
    return f"{int(y)}_{int(x)}"


def collect_points(image: np.ndarray, gt: np.ndarray, score: np.ndarray, rng: random.Random, args):
    instances = connected_instances(gt)
    point_sources: dict[tuple[int, int], set[str]] = defaultdict(set)
    rows = []

    def add(source: str, y: int, x: int, assignment: int | None = None):
        y = int(np.clip(y, 0, gt.shape[0] - 1))
        x = int(np.clip(x, 0, gt.shape[1] - 1))
        assignment = point_instance_id(instances.labels, y, x, tolerance=3) if assignment is None else assignment
        point_sources[(y, x)].add(source)
        rows.append({"source": source, "y": y, "x": x, "assignment": int(assignment), "label": int(assignment > 0)})

    for component_id, y, x in snapped_instance_centers(instances):
        add("gt", y, x, component_id)
        ys, xs = np.nonzero(instances.labels == component_id)
        choices = list(range(len(ys)))
        rng.shuffle(choices)
        used = 0
        for index in choices:
            jy, jx = int(ys[index]), int(xs[index])
            if abs(jy - y) + abs(jx - x) == 0:
                continue
            add("jitter", jy, jx, component_id)
            used += 1
            if used >= args.jitters_per_target:
                break

        ring_added = 0
        for radius in (5, 8, 12, 16):
            for angle in range(0, 360, 45):
                ry = int(round(y + radius * math.sin(math.radians(angle))))
                rx = int(round(x + radius * math.cos(math.radians(angle))))
                if 0 <= ry < gt.shape[0] and 0 <= rx < gt.shape[1] and point_instance_id(instances.labels, ry, rx, 3) == 0:
                    add("ring", ry, rx, 0)
                    ring_added += 1
                    if ring_added >= args.ring_per_target:
                        break
            if ring_added >= args.ring_per_target:
                break

    hard_points = greedy_score_peaks(score, args.hard_pool, radius=3)
    for y, x, _ in hard_points:
        assignment = point_instance_id(instances.labels, y, x, tolerance=3)
        if assignment == 0:
            add("hard", y, x, 0)

    for y, x, _ in greedy_score_peaks(score, 5, radius=7):
        add("eval", y, x)

    attempts = 0
    random_added = 0
    while random_added < args.random_per_image and attempts < 5000:
        attempts += 1
        y, x = rng.randrange(gt.shape[0]), rng.randrange(gt.shape[1])
        if point_instance_id(instances.labels, y, x, tolerance=8) == 0:
            add("random", y, x, 0)
            random_added += 1

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    edge_added = 0
    for y, x, _ in greedy_score_peaks(gradient, max(40, args.edge_per_image * 4), radius=7):
        if point_instance_id(instances.labels, y, x, tolerance=3) == 0:
            add("edge", y, x, 0)
            edge_added += 1
            if edge_added >= args.edge_per_image:
                break
    return instances, point_sources, rows


@torch.no_grad()
def describe_points(predictor, image, score, points, crop_size, local_contrast_sigma):
    return describe_full_image_points(
        predictor,
        image,
        score,
        points,
        crop_size,
        local_contrast_sigma,
        grid_size=32,
    )


@torch.no_grad()
def build_cache(predictor, score_token, baseline, sample_ids, cache_dir, args):
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for image_index, sample_id in enumerate(tqdm(sample_ids, desc="cache objectness candidates")):
        image_path, mask_path = raw_irstd_paths(baseline.data_root, sample_id)
        image, gt, _ = load_rgb_and_mask(image_path, mask_path, (baseline.image_size, baseline.image_size))
        score_path = cache_dir / f"{sample_id}_score.npy"
        if score_path.exists() and not args.force_recompute:
            score = np.load(score_path)
        else:
            score, _ = ev.compute_stitched_similarity(predictor, image, score_token, baseline)
            np.save(score_path, score.astype(np.float32))
        rng = random.Random(args.seed * 1009 + image_index)
        instances, point_sources, source_rows = collect_points(image, gt, score, rng, args)
        points = sorted(point_sources)
        point_to_index = {point: index for index, point in enumerate(points)}
        desc = describe_full_image_points(
            predictor, image, score, points, args.crop_size,
            baseline.local_contrast_sigma, args.objectness_grid_size,
        )
        np.savez_compressed(cache_dir / f"{sample_id}_desc.npz", points=np.asarray(points, dtype=np.int16), **desc)
        for row in source_rows:
            row.update({
                "sample_id": sample_id,
                "point_index": point_to_index[(row["y"], row["x"])],
                "target_count": instances.count,
            })
            records.append(row)
    return records


def load_arrays(cache_dir: Path, sample_ids: list[str]):
    result = {}
    for sample_id in sample_ids:
        data = np.load(cache_dir / f"{sample_id}_desc.npz")
        result[sample_id] = {name: data[name].astype(np.float32) for name in ("sim", "center", "mean", "max", "diff", "hand")}
    return result


def make_folds(records, support_ids: set[str], fold_count: int, seed: int):
    meta = {}
    for row in records:
        meta[row["sample_id"]] = row["target_count"]
    ids = [sample_id for sample_id in meta if sample_id not in support_ids]
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for sample_id in ids:
        buckets[min(meta[sample_id], 3)].append(sample_id)
    folds = [[] for _ in range(fold_count)]
    for bucket in buckets.values():
        rng.shuffle(bucket)
        for index, sample_id in enumerate(bucket):
            folds[index % fold_count].append(sample_id)
    return [sorted(fold) for fold in folds]


def normalized_rows(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-8)


def prototypes(records, arrays, train_ids):
    positives, negatives = [], []
    for row in records:
        if row["sample_id"] not in train_ids:
            continue
        center = arrays[row["sample_id"]]["center"][row["point_index"]]
        if row["source"] == "gt":
            positives.append(center)
        elif row["source"] == "hard" and row["label"] == 0:
            negatives.append(center)
    target = np.mean(positives, axis=0)
    background = np.mean(negatives, axis=0)
    return target / max(np.linalg.norm(target), 1e-8), background / max(np.linalg.norm(background), 1e-8)


def descriptor(row, arrays, variant, proto):
    item = arrays[row["sample_id"]]
    index = row["point_index"]
    parts = [item["sim"][index]]
    if variant in {"f1", "f2", "f3", "f4"}:
        parts += [item["center"][index], item["mean"][index], item["max"][index]]
    if variant in {"f2", "f3", "f4"}:
        parts.append(item["diff"][index])
    if variant in {"f3", "f4"}:
        parts.append(item["hand"][index])
    if variant == "f4":
        target, background = proto
        center = item["center"][index]
        mean = item["mean"][index]
        center = center / max(np.linalg.norm(center), 1e-8)
        mean = mean / max(np.linalg.norm(mean), 1e-8)
        tc, tm = float(center @ target), float(mean @ target)
        bc, bm = float(center @ background), float(mean @ background)
        parts.append(np.asarray([tc, tm, bc, bm, tc - bc, tm - bm], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


REGIMES = {
    "o0": ({"gt"}, {"random"}),
    "o1": ({"gt"}, {"hard"}),
    "o2": ({"gt"}, {"hard", "random"}),
    "o3": ({"gt", "jitter"}, {"hard", "random", "ring", "edge"}),
    "o4": ({"gt", "jitter"}, {"hard"}),
    "o5": ({"gt", "jitter"}, {"hard", "ring", "edge"}),
}


def sample_epoch(rows, regime, sampling, rng):
    positive_sources, negative_sources = REGIMES[regime]
    positives = [row for row in rows if row["source"] in positive_sources and row["label"] == 1]
    negatives = [row for row in rows if row["source"] in negative_sources and row["label"] == 0]

    def draw(pool, count):
        if not pool:
            return []
        return [pool[rng.randrange(len(pool))] for _ in range(count)]

    n = len(positives)
    chosen = list(positives)
    if sampling == "stratified":
        hard = [row for row in negatives if row["source"] != "random"]
        random_bg = [row for row in negatives if row["source"] == "random"]
        chosen += draw(hard, n) + draw(random_bg, n)
    else:
        chosen += draw(negatives, int(round(n * float(sampling))))
    rng.shuffle(chosen)
    return chosen


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def detection_metrics(rows, scores, threshold):
    grouped = defaultdict(list)
    for row, score in zip(rows, scores):
        grouped[row["sample_id"]].append((row, float(score)))
    tp = fp = fn = 0
    for items in grouped.values():
        target_count = items[0][0]["target_count"]
        selected = [row for row, score in items if score >= threshold]
        hits = {row["assignment"] for row in selected if row["assignment"] > 0}
        tp += len(hits)
        fp += len(selected) - len(hits)
        fn += target_count - len(hits)
    images = len(grouped)
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "recall": recall, "precision": precision, "f1": f1, "fp_per_image": fp / max(1, images)}


def select_threshold(rows, scores):
    values = np.unique(scores)
    thresholds = np.r_[np.inf, values[::-1], -np.inf]
    best = None
    sweep = []
    for threshold in thresholds:
        metric = detection_metrics(rows, scores, float(threshold))
        sweep.append((float(threshold), metric))
        key = (metric["f1"], metric["recall"], -metric["fp_per_image"])
        if best is None or key > best[0]:
            best = (key, float(threshold), metric)
    return best[1], best[2], sweep


def recall_at_fppi(sweep, limit):
    eligible = [metric["recall"] for _, metric in sweep if metric["fp_per_image"] <= limit + 1e-12]
    return max(eligible, default=0.0)


def build_matrix(rows, arrays, variant, proto):
    return np.stack([descriptor(row, arrays, variant, proto) for row in rows])


def score_model(model, rows, arrays, variant, proto, mean, std, device):
    if not rows:
        return np.empty(0, dtype=np.float32)
    matrix = (build_matrix(rows, arrays, variant, proto) - mean) / std
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(matrix).to(device))).cpu().numpy()
    return scores


def run_fold(experiment, fold, train_ids, val_ids, records, arrays, args, device, checkpoint_dir):
    regime, variant, sampling = experiment["regime"], experiment["descriptor"], experiment["sampling"]
    train_pool = [row for row in records if row["sample_id"] in train_ids and row["source"] != "eval"]
    train_eval = [row for row in records if row["sample_id"] in train_ids and row["source"] == "eval"]
    val_eval = [row for row in records if row["sample_id"] in val_ids and row["source"] == "eval"]
    proto = prototypes(records, arrays, train_ids)
    allowed_pos, allowed_neg = REGIMES[regime]
    fit_rows = [row for row in train_pool if (row["label"] == 1 and row["source"] in allowed_pos) or (row["label"] == 0 and row["source"] in allowed_neg)]
    fit_matrix = build_matrix(fit_rows, arrays, variant, proto)
    mean = fit_matrix.mean(axis=0, keepdims=True)
    std = np.maximum(fit_matrix.std(axis=0, keepdims=True), 1e-5)
    model = ObjectnessMLP(fit_matrix.shape[1], args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = random.Random(args.seed + fold * 10007 + experiment["order"] * 1000003)
    final_loss = 0.0
    for _ in range(args.epochs):
        epoch_rows = sample_epoch(train_pool, regime, sampling, rng)
        rng.shuffle(epoch_rows)
        losses = []
        for start in range(0, len(epoch_rows), args.batch_size):
            batch = epoch_rows[start : start + args.batch_size]
            matrix = (build_matrix(batch, arrays, variant, proto) - mean) / std
            labels = np.asarray([row["label"] for row in batch], dtype=np.float32)
            logits = model(torch.from_numpy(matrix).to(device))
            loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(labels).to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses))

    train_scores = score_model(model, train_eval, arrays, variant, proto, mean, std, device)
    threshold, train_metric, _ = select_threshold(train_eval, train_scores)
    val_scores = score_model(model, val_eval, arrays, variant, proto, mean, std, device)
    val_metric = detection_metrics(val_eval, val_scores, threshold)
    _, _, val_sweep = select_threshold(val_eval, val_scores)
    labels = np.asarray([row["label"] for row in val_eval], dtype=np.int64)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "input_dim": fit_matrix.shape[1], "mean": mean, "std": std,
        "threshold": threshold, "prototype_target": proto[0], "prototype_background": proto[1],
        "experiment": experiment, "fold": fold,
        "feature_protocol": "full_image_encode_then_roi_sample_v1",
    }, checkpoint_dir / f"{experiment['id']}_fold{fold}.pt")
    result = {
        "experiment": experiment["id"], "stage": experiment["stage"], "regime": regime,
        "descriptor": variant, "sampling": sampling, "fold": fold, "seed": args.seed,
        "train_images": len(train_ids), "val_images": len(val_ids), "val_candidates": len(val_eval),
        "input_dim": fit_matrix.shape[1], "threshold": threshold, "final_train_loss": final_loss,
        "candidate_ap_percent": 100 * average_precision(labels, val_scores),
        "target_recall_percent": 100 * val_metric["recall"],
        "precision_percent": 100 * val_metric["precision"], "target_f1_percent": 100 * val_metric["f1"],
        "false_candidates_per_image": val_metric["fp_per_image"],
        "recall_at_0_5_fppi_percent": 100 * recall_at_fppi(val_sweep, 0.5),
        "recall_at_1_0_fppi_percent": 100 * recall_at_fppi(val_sweep, 1.0),
        "train_target_recall_percent": 100 * train_metric["recall"],
    }
    candidate_rows = [{
        "experiment": experiment["id"], "fold": fold, "sample_id": row["sample_id"],
        "y": row["y"], "x": row["x"], "assignment": row["assignment"], "label": row["label"],
        "score": float(score), "selected": int(score >= threshold),
    } for row, score in zip(val_eval, val_scores)]
    return result, candidate_rows


def summarize(rows):
    result = []
    for experiment in sorted({row["experiment"] for row in rows}):
        items = [row for row in rows if row["experiment"] == experiment]
        first = items[0]
        result.append({
            "experiment": experiment, "stage": first["stage"], "regime": first["regime"],
            "descriptor": first["descriptor"], "sampling": first["sampling"], "folds": len(items),
            "input_dim": first["input_dim"],
            "candidate_ap_percent": float(np.mean([row["candidate_ap_percent"] for row in items])),
            "std_ap_percent": float(np.std([row["candidate_ap_percent"] for row in items])),
            "target_recall_percent": float(np.mean([row["target_recall_percent"] for row in items])),
            "precision_percent": float(np.mean([row["precision_percent"] for row in items])),
            "target_f1_percent": float(np.mean([row["target_f1_percent"] for row in items])),
            "false_candidates_per_image": float(np.mean([row["false_candidates_per_image"] for row in items])),
            "recall_at_0_5_fppi_percent": float(np.mean([row["recall_at_0_5_fppi_percent"] for row in items])),
            "recall_at_1_0_fppi_percent": float(np.mean([row["recall_at_1_0_fppi_percent"] for row in items])),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_ids", required=True)
    parser.add_argument("--support_ids", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--crop_size", type=int, default=64)
    parser.add_argument("--objectness_grid_size", type=int, default=32)
    parser.add_argument("--hard_pool", type=int, default=50)
    parser.add_argument("--random_per_image", type=int, default=20)
    parser.add_argument("--edge_per_image", type=int, default=10)
    parser.add_argument("--ring_per_target", type=int, default=4)
    parser.add_argument("--jitters_per_target", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--suite", choices=("standard", "refine"), default="refine")
    parser.add_argument("--cache_only", action="store_true", help="Build shared features without retired search experiments")
    parser.add_argument("--cache_source", default="")
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()
    if args.suite == 'standard' and not args.cache_only:
        parser.error('The standard search is retired. Use --cache_only or --suite refine for retained F4.')

    output = Path(args.output_dir)
    cache_root = Path(args.cache_source) if args.cache_source else output
    cache_dir = cache_root / "artifacts" / "candidate_features"
    checkpoint_dir = output / "artifacts" / "checkpoints"
    metrics_dir = output / "metrics"
    metadata_dir = output / "metadata"
    for path in (cache_dir, checkpoint_dir, metrics_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)

    config, baseline = load_baseline(Path(args.baseline_config))
    sample_ids, support_ids = parse_list(args.sample_ids), parse_list(args.support_ids)
    signature = cache_signature(config, baseline, sample_ids, support_ids, args)
    manifest_path = cache_root / "metadata" / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    cache_valid = manifest.get("signature") == signature and not args.force_recompute

    device = torch.device(baseline.device if torch.cuda.is_available() else "cpu")
    if not cache_valid:
        seed_everything(args.seed)
        sam = build_sam2_bplus(baseline.sam2_ckpt, baseline.sam2_config, str(device), baseline.encoder_input_size)
        predictor = Sam2SimilarityPredictor(sam, feature_source=baseline.feature_source)
        register_sim_feature_hooks(predictor, baseline.sim_feature_layers)
        support_paths = build_raw_support_paths(baseline.data_root, len(support_ids), support_ids=",".join(support_ids))
        score_token, score_info = ev.build_support_score_token(predictor, support_paths, baseline)
        write_json(metadata_dir / "score_token_info.json", score_info)
        records = build_cache(predictor, score_token, baseline, sample_ids, cache_dir, args)
        write_json(manifest_path, {
            "signature": signature,
            "feature_protocol": "full_image_encode_then_roi_sample_v1",
            "records": records,
        })
        del predictor, sam
        torch.cuda.empty_cache()
    else:
        records = manifest["records"]

    if args.cache_only:
        print(json.dumps({'status': 'shared_cache_ready', 'images': len(sample_ids), 'trained': False}))
        return
    arrays = load_arrays(cache_dir, sample_ids)
    folds = make_folds(records, set(support_ids), args.folds, args.seed)
    write_json(metadata_dir / "folds.json", {"fixed_support_train_only": support_ids, "validation_folds": folds})
    all_ids = set(sample_ids)
    experiments = [{"id": "e9r_f4", "stage": "E9R", "regime": "o1", "descriptor": "f4", "sampling": "1"}]
    for order, experiment in enumerate(experiments):
        experiment["order"] = order

    fold_rows, candidate_rows = [], []
    for experiment in tqdm(experiments, desc="E8-E10 experiments"):
        for fold, val_ids_list in enumerate(folds):
            val_ids = set(val_ids_list)
            train_ids = all_ids - val_ids
            result, candidates = run_fold(experiment, fold, train_ids, val_ids, records, arrays, args, device, checkpoint_dir)
            fold_rows.append(result)
            candidate_rows.extend(candidates)

    summary = summarize(fold_rows)
    write_csv(metrics_dir / "fold_results.csv", fold_rows)
    write_csv(metrics_dir / "experiment_summary.csv", summary)
    write_csv(metrics_dir / "candidate_predictions.csv", candidate_rows)
    best_by_stage = {}
    for stage in sorted({row["stage"] for row in summary}):
        choices = [row for row in summary if row["stage"] == stage]
        best_by_stage[stage] = max(choices, key=lambda row: (row["candidate_ap_percent"], row["recall_at_1_0_fppi_percent"], row["target_f1_percent"]))
    report = {
        "images": len(sample_ids), "fixed_support_images": len(support_ids),
        "cross_validation_images": len(sample_ids) - len(support_ids),
        "feature_protocol": "full image -> frozen SAM2 encoder -> candidate ROI sampling -> ObjectnessMLP",
        "loss_policy": "BCE is computed only on sampled positive and negative candidate points",
        "evaluation_policy": "Top-5 + radius-7 NMS; point is positive within 3 pixels of a GT instance",
        "best_by_stage": best_by_stage,
    }
    write_json(metrics_dir / "phase_c_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
