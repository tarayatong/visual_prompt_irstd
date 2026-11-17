"""Fresh-budget frozen SAM2 inference and two-MLP training, without old caches."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scripts.prepare_comparison import DEFAULT_OUTPUT, write_json
from scripts.run_phase_b_mask_head import gaussian_map, load_baseline
from scripts.run_phase_c_objectness import descriptor
from scripts.run_global_ablation_201 import crop_at, encode_local, paste_local, rule_union
from scripts.train_joint_objectness_mask_heads import probability_iou_loss
from visual_prompt.backbones.sam2_adapter import Sam2SimilarityPredictor, build_sam2_bplus
from visual_prompt.core.candidate_proposals import connected_instances, greedy_score_peaks, point_instance_id, snapped_instance_centers
from visual_prompt.evaluation import eval_sliding_window_semantic_mask as ev
from visual_prompt.evaluation.comparison import image_metrics, summarize
from visual_prompt.models.full_image_objectness import describe_full_image_points
from visual_prompt.models.lightweight_heads import ObjectnessMLP, MaskVectorMLP
from visual_prompt.pipelines.semantic_prompt_pipeline import register_sim_feature_hooks, seed_everything, load_rgb_and_mask


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_working(record, size):
    return load_rgb_and_mask(record["image"], record["mask"], (size, size))[:2]


def get_score(predictor, image, token, baseline, path):
    if path.exists():
        return np.load(path)
    score, _ = ev.compute_stitched_similarity(predictor, image, token, baseline)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, score.astype(np.float32))
    return score


@torch.no_grad()
def cache_training(manifest, predictor, token, baseline, directory):
    directory.mkdir(parents=True, exist_ok=True)
    e0 = F.normalize(token.flatten().float(), dim=0)
    for image_index, sid in enumerate(tqdm(manifest["train_ids"], desc="cache train50")):
        path = directory / f"{sid}.pt"
        if path.exists():
            continue
        image, gt = read_working(manifest["records"][sid], baseline.image_size)
        score = get_score(predictor, image, token, baseline, directory / "scores" / f"{sid}.npy")
        instances = connected_instances(gt)
        points = [(y, x) for y, x, _ in greedy_score_peaks(score, 5, 7)]
        points += [(y, x) for _, y, x in snapped_instance_centers(instances)]
        rng = random.Random(manifest["seed"] + image_index)
        for _ in range(5):
            points.append((rng.randrange(gt.shape[0]), rng.randrange(gt.shape[1])))
        points = list(dict.fromkeys(points))
        parts = describe_full_image_points(predictor, image, score, points, 64, baseline.local_contrast_sigma, 32)
        items = []
        for index, (y, x) in enumerate(points):
            assignment = point_instance_id(instances.labels, y, x, 3)
            # Leave an ignore band around targets for negative supervision.
            if assignment == 0 and point_instance_id(instances.labels, y, x, 8):
                continue
            crop, _, feature = encode_local(predictor, image, y, x, 64, 64)
            target = (instances.labels == assignment).astype(np.uint8) if assignment else np.zeros_like(gt)
            _, target_crop = crop_at(image, target, y, x, 64)
            contrast = ev.local_contrast_score(crop, baseline.local_contrast_sigma)
            items.append(dict(sample_id=sid, label=int(assignment > 0), feature=feature[0].cpu().half(),
                              target=torch.from_numpy(target_crop), contrast=torch.from_numpy(contrast).half(),
                              parts={k: v[index] for k, v in parts.items()}))
        torch.save(items, path)


def matrix_for(items, proto):
    arrays = {str(i): {k: np.asarray(v)[None] for k, v in item["parts"].items()} for i, item in enumerate(items)}
    return np.stack([descriptor(dict(sample_id=str(i), point_index=0), arrays, "f4", proto) for i in range(len(items))])


def train_heads(manifest, token, baseline, cache, output, warm_epochs, joint_epochs):
    checkpoint_path = output / "checkpoints/joint.pt"
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (saved.get("manifest_sha256") != manifest["manifest_sha256"]
                or saved.get("warm_epochs") != warm_epochs or saved.get("joint_epochs") != joint_epochs):
            raise ValueError("Refusing to reuse a joint checkpoint with different supervision or training schedule")
        return saved
    items = []
    for sid in manifest["train_ids"]:
        items.extend(torch.load(cache / f"{sid}.pt", map_location="cpu", weights_only=False))
    labels = np.asarray([item["label"] for item in items])
    if not labels.any() or labels.all():
        raise ValueError("Need positive and negative candidates for joint training")
    prototypes = []
    for label in (1, 0):
        v = np.mean([item["parts"]["center"] for item in items if item["label"] == label], axis=0)
        prototypes.append(v / max(np.linalg.norm(v), 1e-8))
    matrix = matrix_for(items, tuple(prototypes))
    mean, std = matrix.mean(0), np.maximum(matrix.std(0), 1e-5)
    matrix = (matrix - mean) / std
    device = torch.device(baseline.device)
    e0 = F.normalize(token.flatten().float(), dim=0).to(device)
    seed_everything(manifest["seed"])
    obj = ObjectnessMLP(matrix.shape[1], 128).to(device)
    mask = MaskVectorMLP(len(e0), 128, e0, "m2_residual").to(device)
    gaussian = torch.from_numpy(gaussian_map(64, 4)).to(device)
    history = []
    rng = random.Random(manifest["seed"])
    pos, neg = np.flatnonzero(labels == 1).tolist(), np.flatnonzero(labels == 0).tolist()
    for stage, epochs, lr in (("fresh_head_warmup", warm_epochs, 1e-3), ("joint", joint_epochs, 1e-4)):
        optimizer = torch.optim.AdamW(list(obj.parameters()) + list(mask.parameters()), lr=lr, weight_decay=1e-4)
        for epoch in tqdm(range(epochs), desc=stage):
            count = min(len(pos), len(neg))
            selected = rng.sample(pos, count) + rng.sample(neg, count)
            rng.shuffle(selected)
            losses = []
            for start in range(0, len(selected), 4):
                indices = selected[start:start + 4]
                batch = [items[i] for i in indices]
                features = torch.stack([x["feature"] for x in batch]).float().to(device)
                target = torch.stack([x["target"] for x in batch]).float().to(device)
                contrast = torch.stack([x["contrast"] for x in batch]).float().to(device)
                y = torch.as_tensor(labels[indices], dtype=torch.float32, device=device)
                logits = obj(torch.from_numpy(matrix[indices]).to(device))
                sim = torch.einsum("bchw,c->bhw", features, e0)
                mlogits = mask(features, e0, sim, gaussian[None].expand(len(batch), -1, -1), contrast, .25, .1)
                pmask = torch.sigmoid(mlogits)
                positive = y > .5
                zero = logits.sum() * 0
                loss_mask = probability_iou_loss(pmask[positive], target[positive]) if positive.any() else zero
                loss = F.binary_cross_entropy_with_logits(logits, y) + .5 * loss_mask
                if stage == "joint":
                    gated = torch.sigmoid(logits)[:, None, None] * pmask
                    loss += probability_iou_loss(gated[positive], target[positive]) if positive.any() else zero
                    loss += .5 * gated[~positive].mean() if (~positive).any() else zero
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            history.append(dict(stage=stage, epoch=epoch + 1, loss=float(np.mean(losses))))
    checkpoint = dict(object_model=obj.cpu().state_dict(), mask_model=mask.cpu().state_dict(),
                      input_dim=matrix.shape[1], channels=len(e0), mean=mean, std=std, prototypes=prototypes,
                      threshold=.5, train_ids=manifest["train_ids"], support_ids=manifest["support_ids"],
                      manifest_sha256=manifest["manifest_sha256"], warm_epochs=warm_epochs, joint_epochs=joint_epochs,
                      protocol="fresh warmup on train50 then joint; fixed thresholds; no historical checkpoint",
                      object_feature_protocol="full_image_encode_then_roi_sample_v1")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    write_json(output / "training_history.json", history)
    return checkpoint


@torch.no_grad()
def evaluate(manifest, predictor, token, baseline, root, method, checkpoint=None, limit=0):
    method_dir = root / method
    pred_dir = method_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(baseline.device)
    e0 = F.normalize(token.flatten().float(), dim=0).to(device)
    support_areas = []
    for sid in manifest["support_ids"]:
        _, gt = read_working(manifest["records"][sid], baseline.image_size)
        support_areas.append(int(gt.sum()))
    support_area = max(1, round(float(np.median(support_areas))))
    if checkpoint:
        obj = ObjectnessMLP(checkpoint["input_dim"], 128).to(device).eval()
        obj.load_state_dict(checkpoint["object_model"])
        mask = MaskVectorMLP(checkpoint["channels"], 128, e0, "m2_residual").to(device).eval()
        mask.load_state_dict(checkpoint["mask_model"])
        gaussian = torch.from_numpy(gaussian_map(64, 4)).float().to(device)[None]
    ids = manifest["test_ids"][:limit or None]
    rows = []
    started = time.perf_counter()
    for sid in tqdm(ids, desc=method):
        record = manifest["records"][sid]
        prediction_path = pred_dir / f"{sid}.npy"
        if prediction_path.exists():
            pred = np.load(prediction_path)
        else:
            image, _ = read_working(record, baseline.image_size)
            score = get_score(predictor, image, token, baseline, root / "shared/test_scores" / f"{sid}.npy")
            if checkpoint is None:
                points = greedy_score_peaks(score, 1, 7)
                working_pred = rule_union(score, image, [(y, x) for y, x, _ in points], support_area, baseline).astype(np.float32)
            else:
                points = [(y, x) for y, x, _ in greedy_score_peaks(score, 5, 7)]
                parts = describe_full_image_points(predictor, image, score, points, 64, baseline.local_contrast_sigma, 32)
                matrix = matrix_for([dict(parts={k: v[i] for k, v in parts.items()}) for i in range(len(points))], checkpoint["prototypes"])
                probs = torch.sigmoid(obj(torch.from_numpy((matrix - checkpoint["mean"]) / checkpoint["std"]).to(device)))
                working_pred = np.zeros(score.shape, dtype=np.uint8)
                for i, (y, x) in enumerate(points):
                    if float(probs[i]) < .5:
                        continue
                    crop, _, feature = encode_local(predictor, image, y, x, 64, 64)
                    contrast = torch.from_numpy(ev.local_contrast_score(crop, baseline.local_contrast_sigma)).to(device)[None]
                    sim = torch.einsum("bchw,c->bhw", feature, e0)
                    logits = mask(feature, e0, sim, gaussian, contrast, .25, .1)
                    paste_local(working_pred, (logits[0] > 0).cpu().numpy().astype(np.uint8), y, x)
                working_pred = working_pred.astype(np.float32)
            h, w = record["size_hw"]
            pred = cv2.resize(working_pred, (w, h), interpolation=cv2.INTER_NEAREST)
            np.save(prediction_path, pred)
        h, w = record["size_hw"]
        gt = (np.asarray(Image.open(record["mask"]).convert("L").resize((w, h), Image.Resampling.NEAREST)) > 0).astype(np.uint8)
        rows.append(image_metrics(pred, gt, sid))
    result = summarize(rows)
    result.update(method=method, dataset=manifest["dataset"], complete_test=len(ids) == len(manifest["test_ids"]),
                  allocated_annotation_budget=50, actual_annotation_budget=50 if checkpoint else 5,
                  support_shots=5, threshold=.5, elapsed_seconds=time.perf_counter() - started,
                  manifest_sha256=manifest["manifest_sha256"], prediction_format="binary float32 original-resolution npy",
                  inference="top5 object gate + mask logits>0" if checkpoint else "existing frozen corrected top1 rule mask")
    save_csv(method_dir / "metrics/per_image.csv", rows)
    write_json(method_dir / "metrics/summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", default=["IRSTD-1k", "NUDT-SIRST", "SIRST"])
    parser.add_argument("--stage", choices=["frozen", "joint", "all"], default="all")
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--warm-epochs", type=int, default=100)
    parser.add_argument("--joint-epochs", type=int, default=30)
    args = parser.parse_args()
    for name in args.datasets:
        root = args.root / name
        manifest = json.loads((root / "protocol/manifest.json").read_text(encoding="utf-8"))
        _, baseline = load_baseline(Path("experiments/configs/sam2/main.yaml"))
        baseline.support_ids = ",".join(manifest["support_ids"])
        baseline.seed = manifest["seed"]
        # RTX 3080 10GB: identical windows, smaller batches avoid WDDM paging.
        baseline.window_batch_size = 1
        seed_everything(baseline.seed)
        sam = build_sam2_bplus(baseline.sam2_ckpt, baseline.sam2_config, baseline.device, baseline.encoder_input_size)
        predictor = Sam2SimilarityPredictor(sam, feature_source=baseline.feature_source)
        register_sim_feature_hooks(predictor, baseline.sim_feature_layers)
        pairs = [(manifest["records"][sid]["image"], manifest["records"][sid]["mask"]) for sid in manifest["support_ids"]]
        token, info = ev.build_support_score_token(predictor, pairs, baseline)
        write_json(root / "shared/support_token_info.json", info)
        write_json(root / "shared/resolved_baseline.json", vars(baseline))
        if args.stage in ("all", "frozen"):
            evaluate(manifest, predictor, token, baseline, root, "ours_frozen", limit=args.max_test)
        if args.stage in ("all", "joint"):
            cache = root / "ours_joint/cache"
            cache_training(manifest, predictor, token, baseline, cache)
            checkpoint = train_heads(manifest, token, baseline, cache, root / "ours_joint", args.warm_epochs, args.joint_epochs)
            evaluate(manifest, predictor, token, baseline, root, "ours_joint", checkpoint, args.max_test)
        del predictor, sam
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
