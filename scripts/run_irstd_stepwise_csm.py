"""IRSTD-1k train-50 stepwise comparison and matched CSM controls."""

import argparse
import csv
import os
from pathlib import Path
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from scripts import run_recent_a_comparison as base
from scripts import run_tiny_ablation_combined as combined
from scripts.eval_tiny_fixed_memory_radial import make_predictor
from scripts.run_tiny_ablation_three_seeds import FIELDS, folder_for, read, sha, write
from visual_prompt.core.semantic_direction import pool_feature_token
from visual_prompt.evaluation.comparison import image_metrics, summarize
from visual_prompt.models.tiny_memory_token_ablation import random_sampling_regions
from visual_prompt.models.tiny_shared_memory import build_episode_keys, descriptor, support_regions

DATASET = "IRSTD-1k"
SEED = 113
OUT = ROOT / "outputs/irstd_stepwise_csm_20261115"
FULL = ROOT / "outputs/tiny_radial_detail_irstd_seed113/only_256/combined"
METRICS = "epoch_0120/metrics/summary.json"
CONTROLS = ("random_equal_budget", "raw_target_instances")
MATCH_FIELDS = (
    "dataset", "seed", "split_seed", "epochs", "training_horizon",
    "manifest_sha256", "initialization_sha256", "source_commit",
    "train_annotation_ids", "support_annotation_ids", "batch_size",
    "precision", "optimizer", "lr", "threshold", "input_size",
    "backbone_input_size", "train_support_shots", "support_shots",
)


@torch.no_grad()
def random_equal_budget(model, backbone, indices, regions):
    indices = tuple(indices)
    reference = build_episode_keys(model, backbone, indices, regions)
    count = int((reference.float().norm(dim=-1) > 0).sum())
    seed = SEED
    for geometry in regions:
        seed = zlib.crc32(geometry["sampling_seed"].to_bytes(4, "little"), seed)
    generator = torch.Generator().manual_seed(seed)
    grid = torch.cat([descriptor(model, backbone, i)[0].flatten(1).T for i in indices])
    draw = torch.randperm(len(grid), generator=generator)[:count]
    tags = torch.randint(3, (len(reference),), generator=generator).to(grid.device)
    keys = torch.zeros_like(reference)
    keys[:count] = grid[draw.to(grid.device)]
    assert int((keys.float().norm(dim=-1) > 0).sum()) == count
    return keys, tags


@torch.no_grad()
def raw_target_instances(model, backbone, indices, regions):
    indices = tuple(indices)
    keys = build_episode_keys(model, backbone, indices, regions)
    valid = keys.float().norm(dim=-1) > 0
    slots = torch.where(valid[2:10])[0] + 2
    instance_vectors = []
    for index, geometry in zip(indices, regions, strict=True):
        feature = descriptor(model, backbone, index)[0].permute(1, 2, 0)
        for mask in geometry["instances"]:
            instance_vectors.append(pool_feature_token(feature, mask.to(feature.device), "mean_max", True)[0])
            if len(instance_vectors) == 8:
                break
        if len(instance_vectors) == 8:
            break
    assert len(instance_vectors) == len(slots)
    keys[slots] = torch.stack(instance_vectors)
    assert torch.equal(keys.float().norm(dim=-1) > 0, valid)
    return keys


def train_control(mode, smoke):
    builder, regions, description = {
        "random_equal_budget": (random_equal_budget, random_sampling_regions,
                                "Equal valid token count; uniform grid cells; three randomly assigned learnable types"),
        "raw_target_instances": (raw_target_instances, support_regions,
                                 "Raw instance tokens replace corrected instances; global and background slots and type IDs fixed"),
    }[mode]
    combined.OUT = OUT / ("smoke" if smoke else "trained") / mode
    combined.RUN_SEED = SEED
    combined.worker(argparse.Namespace(
        dataset=DATASET, variant="combined", smoke=smoke,
        key_builder=builder, region_builder=regions,
        radial_options={"active_stages": (1,)},
        radial_metadata=dict(
            radial_control="only_256", radial_stage_sizes=[256], radial_stage_channels=[32],
            radial_descriptor="center_minus_neighbor", radial_gate="adaptive",
            radial_sampling_points_per_query=64 * 256**2,
            memory_control=mode, prototype_slots=description,
            fusion="19-slot interface -> frozen native attention -> RCR at 256",
            matched_reference=(FULL / METRICS).relative_to(ROOT).as_posix(),
        ),
        extra_sources=[Path(__file__).resolve(), ROOT / "visual_prompt/models/tiny_memory_token_ablation.py"],
        cached_support_mean_error_limit=3e-5,
    ))


def query_only(model, manifest):
    """Use the same SAM decoder/ranking as the memory arms with null prompts."""
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]

    def predict(sid):
        with Image.open(manifest["records"][sid]["image"]) as source:
            image = np.array(source.convert("RGB").resize((640, 640), Image.Resampling.BILINEAR), dtype=np.float32)
        query = (torch.from_numpy(image.transpose(2, 0, 1) / 255) - mean) / std
        x, _, _, sizes = model._preprocess_visual_features(query[None, None].to(model.device), model.sam.image_size)
        backbone = model._forward_backbone(x, sizes)
        features = backbone.get_current_feats(0)
        query_features = (features[-1] + model.sam.no_mem_embed).permute(1, 2, 0).reshape(1, 256, 64, 64)
        decoded = model.sam._forward_sam_heads(
            backbone_features=query_features,
            high_res_features=backbone.get_high_res_features(features),
            multimask_output=True,
        )
        return F.interpolate(decoded.masks.float(), (640, 640), mode="bilinear", align_corners=False)[0, 0].sigmoid()

    return predict


def verify_predictions(folder, epoch, manifest):
    result = read(folder / f"epoch_{epoch:04d}/metrics/summary.json")
    predictions = folder / f"epoch_{epoch:04d}/predictions"
    ids = manifest["test_ids"]
    assert {path.stem for path in predictions.glob("*.png")} == set(ids)
    counts = []
    for sid in ids:
        record = manifest["records"][sid]
        h, w = record["size_hw"]
        with Image.open(predictions / f"{sid}.png") as prediction, Image.open(record["mask"]) as mask:
            gt = np.asarray(mask.convert("L").resize((w, h), Image.Resampling.NEAREST)) > 0
            counts.append(image_metrics(np.asarray(prediction) > 0, gt, sid))
    for key, value in summarize(counts).items():
        assert result[key] == value, (folder, key)
    return result


@torch.inference_mode()
def frozen_arms():
    base.seed_all()
    manifest = base.manifest_for(DATASET)
    model, *_, identity = base.setup_sansa(manifest, torch.device("cuda"))
    # The frozen comparison runs the identity-initialized adapters used before
    # fine-tuning, with the native memory/readout switches shared by C.
    model.eval().requires_grad_(False)
    for mode in ("sam2_query_only", "sam2_native_support"):
        dest = OUT / "frozen" / mode
        if (dest / "epoch_0000/metrics/summary.json").exists():
            verify_predictions(dest, 0, manifest)
            continue
        native = mode == "sam2_native_support"
        predictor = make_predictor(model, manifest, "native", False) if native else query_only(model, manifest)
        config = dict(method=mode, dataset=DATASET, seed=SEED, train_count=0, training_horizon=0,
                      manifest_sha256=manifest["manifest_sha256"],
                      initialization_sha256=identity["initialization_sha256"],
                      support_annotation_ids=manifest["support_ids"] if native else [],
                      query_prompts="none", selection="highest predicted IoU among three SAM masks",
                      parameters_trainable=0, threshold=.5)
        base.evaluate(model, predictor, manifest, dest, config, 0, identity["initialization_sha256"])
        verify_predictions(dest, 0, manifest)


def collect():
    manifest = base.manifest_for(DATASET)
    ids = manifest["test_ids"]
    sources = {
        "A SAM2 query only": (OUT / "frozen/sam2_query_only", 0),
        "B SAM2 + native memory": (OUT / "frozen/sam2_native_support", 0),
        "C Native memory + adapter": (folder_for(SEED, "native"), 120),
        "D Full CoRe-IR": (FULL, 120),
        "Random equal-budget memory": (OUT / "trained/random_equal_budget/combined", 120),
        "Raw target instances": (OUT / "trained/raw_target_instances/combined", 120),
    }
    reference = read(FULL / METRICS)
    rows = []
    for name, (folder, epoch) in sources.items():
        path = folder / f"epoch_{epoch:04d}/metrics/summary.json"
        if not path.exists():
            continue
        result = verify_predictions(folder, epoch, manifest)
        assert result["images"] == len(ids) and result["manifest_sha256"] == reference["manifest_sha256"]
        assert result["initialization_sha256"] == reference["initialization_sha256"]
        if epoch:
            assert all(result[key] == reference[key] for key in MATCH_FIELDS), name
            assert sha(folder / "checkpoints/epoch_0120.pt") == result["checkpoint_sha256"]
            assert result["all_120_epochs_verified"] and result["predictions_recount_verified"]
        if name.startswith(("D ", "Random", "Raw")):
            assert result["radial_stage_sizes"] == [256]
            assert result["parameters_trainable"] == reference["parameters_trainable"]
        rows.append(dict(method=name, train_images=50 if epoch else 0,
                         support_images=len(result["support_annotation_ids"]), epochs=epoch,
                         trainable_parameters=result["parameters_trainable"],
                         **{key: result[key] for key in FIELDS},
                         summary=path.relative_to(ROOT).as_posix()))
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "results.csv"
    temp = csv_path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, csv_path)
    write(OUT / "audit.json", dict(status="complete" if len(rows) == len(sources) else "partial",
                                 dataset=DATASET, seed=SEED, test_images=len(ids),
                                 matched_fields=MATCH_FIELDS, verified_results=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("frozen", "collect", *CONTROLS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.mode == "frozen":
        frozen_arms()
    elif args.mode == "collect":
        collect()
    else:
        train_control(args.mode, args.smoke)
