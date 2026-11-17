"""Evaluate fixed and independently resampled five-shot supports on frozen CoRe-IR weights."""

import argparse
import ast
import csv
import random
import statistics
import subprocess
import sys

import torch

from scripts.run_core_ir_common_protocol import (OUT, RETAINED_CORE, SUPPORT_DRAW_SEEDS,
    base, memory_model, memory_predictor, read, support_draw)
from scripts.run_tiny_ablation_combined import restore
from scripts.run_motivation_current_ours import save_csv

SEED_RUN = base.ROOT / "outputs/core_ir_final_three_seeds_20261116"
TRAIN_SEEDS = (111, 112, 113)
TEST_SUMMARY = "epoch_0120/metrics/summary.json"


def prior_sweeps():
    source = SEED_RUN / "runs.csv"
    if not source.is_file():
        return {}
    with source.open(encoding="utf-8-sig", newline="") as stream:
        return {(r["dataset"], int(r["seed"]), int(r["support_draw_seed"])): r
                for r in csv.DictReader(stream) if r.get("support_draw_seed")}


def summarize_sweeps():
    scores = {key: float(row["IoU_percent"]) for key, row in prior_sweeps().items()}
    for path in (OUT / "support").glob("*/seed*/support*/epoch_0120/metrics/summary.json"):
        row = read(path)
        dataset = path.relative_to(OUT / "support").parts[0]
        if row.get("dataset", dataset) != dataset:
            raise ValueError(f"Unexpected dataset in {path}")
        scores[dataset, row["seed"], row["support_draw_seed"]] = row["IoU_percent"]
    grouped = {}
    for (dataset, seed, _), value in scores.items():
        grouped.setdefault((dataset, seed), []).append(value)
    result = []
    for (dataset, seed), values in sorted(grouped.items()):
        if len(values) == 3:
            result.append(dict(dataset=dataset, training_seed=seed, support_draws=len(values),
                mean_IoU=statistics.mean(values), sd_IoU=statistics.stdev(values),
                min_IoU=min(values), max_IoU=max(values), range_pp=max(values) - min(values)))
    if result:
        save_csv(OUT / "support/by_training_seed.csv", result)
    if scores:
        save_csv(OUT / "support/per_draw.csv", [
            dict(dataset=d, training_seed=s, support_draw_seed=k, IoU=value)
            for (d, s, k), value in sorted(scores.items())
        ])
    complete = []
    for dataset in base.DATASETS:
        means = [row["mean_IoU"] for row in result if row["dataset"] == dataset]
        if len(means) == len(TRAIN_SEEDS):
            complete.append(dict(dataset=dataset, mean_IoU=statistics.mean(means),
                                 training_seed_sd_IoU=statistics.stdev(means)))
    if complete:
        save_csv(OUT / "support/by_dataset.csv", complete)
    base.write_json(OUT / "support/sweep_status.json", dict(completed_sweeps=len(scores), expected_sweeps=27))


def open_model(dataset, seed, checkpoint, m):
    base.seed_all()
    model, _ = memory_model("Ours", m)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = weights["config"]
    if (weights["epoch"] != 120 or config["seed"] != seed or config["radial_stage_sizes"] != [256] or
        config["manifest_sha256"] != m["manifest_sha256"]):
        raise ValueError(f"Support pool differs from checkpoint protocol for {dataset}")
    restore(model, weights["model"])
    model.eval()
    return model


def find_checkpoint(dataset, seed):
    retained = RETAINED_CORE.get((dataset, seed))
    epoch = "checkpoints/epoch_0120.pt"
    if retained:
        checkpoint = base.ROOT / "outputs" / retained / epoch
        if checkpoint.is_file():
            return checkpoint
    stem = f"training/{dataset}/seed{seed}/combined/{epoch}"
    fixed = SEED_RUN / "fixed_support" / stem
    legacy = SEED_RUN / stem
    if legacy.is_file():
        return legacy
    if not fixed.is_file():
        subprocess.run([sys.executable, "-B", "-u", "-m", "scripts.run_core_ir_final_three_seeds",
                        "--dataset", dataset, "--seed", str(seed)], cwd=base.ROOT, check=True)
    if not fixed.is_file() and legacy.is_file():
        return legacy
    return fixed


def random_sweeps(dataset, m):
    previously = prior_sweeps()
    for seed in TRAIN_SEEDS:
        checkpoint = find_checkpoint(dataset, seed)
        sha = base.sha(checkpoint)
        model = None
        for draw_seed in SUPPORT_DRAW_SEEDS:
            folder = OUT / "support" / dataset / f"seed{seed}" / f"support{draw_seed}"
            ids = support_draw(m, draw_seed)
            old = previously.get((dataset, seed, draw_seed))
            if (old and old["checkpoint_sha256"] == sha and ast.literal_eval(old["support_ids"]) == ids and
                (base.ROOT / old["summary"]).is_file()):
                continue
            existing = folder / TEST_SUMMARY
            if existing.is_file():
                saved = read(existing)
                if saved["checkpoint_sha256"] == sha and saved["support_ids"] == ids:
                    continue
                raise ValueError(f"Support sweep protocol changed for {folder}")
            if model is None:
                model = open_model(dataset, seed, checkpoint, m)
            view = dict(m, support_ids=ids)
            settings = dict(method="Ours", dataset=dataset, seed=seed, training_horizon=120, support_ids=ids,
                            support_draw_seed=draw_seed, manifest_sha256=m["manifest_sha256"])
            with torch.no_grad():
                predict = memory_predictor("Ours", model, view)
                base.evaluate(model, predict, view, folder, settings, 120, sha)
                del predict
        del model
    summarize_sweeps()


def run(dataset, smoke):
    seed, _ = next((seed, path) for (name, seed), path in RETAINED_CORE.items() if name == dataset)
    checkpoint = find_checkpoint(dataset, seed)
    m = dict(base.manifest_for(dataset), working_size=640)
    model = open_model(dataset, seed, checkpoint, m)
    if smoke:
        m = dict(m, test_ids=m["train_ids"][:1])
    rng = random.Random(f"{dataset}:20261117:per-query")
    draws = {sid: rng.sample(sorted(item for item in m["train_ids"] if item != sid), 5)
             for sid in m["test_ids"]}
    fixed = m["support_ids"]
    if not smoke:
        base.write_json(OUT / "support" / dataset / "per_query_draws.json", draws)
    predict = None
    for condition in ("fixed", "per_query"):
        folder = OUT / "support" / dataset / condition
        if not smoke and (folder / TEST_SUMMARY).exists():
            continue
        with torch.no_grad():
            if condition == "fixed":
                predict = memory_predictor("Ours", model, m)
            else:
                def predict(sid):
                    return memory_predictor("Ours", model, dict(m, support_ids=draws[sid]))(sid)
            if smoke:
                value = predict(m["test_ids"][0])
                assert value.ndim == 2 and torch.isfinite(value).all()
                continue
            settings = dict(method="Ours", dataset=dataset, training_horizon=120, seed=seed, support_shots=5,
                            condition=condition, manifest_sha256=m["manifest_sha256"],
                            support_ids=fixed if condition == "fixed" else None)
            base.evaluate(model, predict, m, folder, settings, 120, base.sha(checkpoint))
    if not smoke:
        scores = {condition: read(OUT / "support" / dataset / condition / TEST_SUMMARY)
                  for condition in ("fixed", "per_query")}
        base.write_json(OUT / "support" / dataset / "comparison.json", dict(
            dataset=dataset, seed=seed, fixed_IoU=scores["fixed"]["IoU_percent"],
            per_query_IoU=scores["per_query"]["IoU_percent"],
            delta_pp=scores["per_query"]["IoU_percent"] - scores["fixed"]["IoU_percent"]))
        del predict, model
        random_sweeps(dataset, m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=base.DATASETS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.dataset:
        run(args.dataset, args.smoke)
    else:
        summarize_sweeps()
