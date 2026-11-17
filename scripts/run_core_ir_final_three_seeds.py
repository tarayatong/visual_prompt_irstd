"""Evaluate final-decoder CoRe-IR across training seeds with fixed supports."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_recent_a_comparison as base

LEGACY_OUT = ROOT / "outputs/core_ir_final_three_seeds_20261116"
OUT = LEGACY_OUT / "fixed_support"
SEEDS = dict.fromkeys(("SIRST", "IRSTD-1k", "NUDT-SIRST"), (111, 112, 113))
ARCHIVED = {
    ("SIRST", 111): "tiny_radial_detail_other_datasets_seed111/SIRST/only_256/combined",
    ("IRSTD-1k", 113): "tiny_radial_detail_irstd_seed113/only_256/combined",
    ("NUDT-SIRST", 111): "tiny_radial_detail_other_datasets_seed111/NUDT-SIRST/only_256/combined",
}
METRICS = ("IoU_percent", "nIoU_percent", "F1_percent", "Pd_percent", "Fa_per_million")
EPOCH = Path("epoch_0120")
SUMMARY = EPOCH / "metrics/summary.json"
METADATA = dict(
    radial_control="final_decoder_output",
    radial_placement="post-upsampling and skip fusion, before mask-token prediction",
    radial_source="final upsampled query decoder features",
    radial_stage_sizes=[256],
    radial_stage_channels=[32],
    radial_descriptor="center_minus_neighbor",
    radial_gate="adaptive",
    radial_sampling_points_per_query=64 * 256**2,
    fusion="19 prototype slots -> frozen memory attention -> final decoder RCR -> mask prediction",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path):
    return path.relative_to(ROOT).as_posix()


def train_folder(dataset, seed):
    return OUT / "training" / dataset / f"seed{seed}" / "combined"


def eval_folder(dataset, seed):
    return OUT / "evaluation" / dataset / f"seed{seed}"


def fixed_supports(manifest):
    ids = manifest["support_ids"]
    assert len(ids) == len(set(ids)) == 5 and set(ids) <= set(manifest["train_ids"])
    assert not set(manifest["train_ids"]) & set(manifest["test_ids"])
    return ids


def protocol():
    domains = {}
    for dataset, seeds in SEEDS.items():
        manifest = base.manifest_for(dataset)
        domains[dataset] = dict(
            training_seeds=seeds,
            train_ids=manifest["train_ids"],
            test_size=len(manifest["test_ids"]),
            manifest_sha256=manifest["manifest_sha256"],
            inference_support_ids=fixed_supports(manifest),
        )
    data = dict(
        model="SAM2 Hiera Tiny + 19-slot Compact Semantic Memory + one Radial Contrast Refinement block",
        **METADATA,
        backbone_weight=relative(base.RUN / "weights/sam2_hiera_tiny.pt"),
        initialization_sha256=base.sha(base.RUN / "weights/sam2_hiera_tiny.pt"),
        epochs=120, optimizer_steps=1560, batch_size=4,
        train_support_rule="one distinct random support from the 50 labeled training images for every query episode",
        inference_support_rule="use the five original manifest supports for all three training seeds",
        aggregation="one complete test per training seed; report mean and sample SD across the three training seeds",
        working_image_size=640, encoder_image_size=1024,
        retained_training={f"{dataset}/seed{seed}": f"outputs/{folder}" for (dataset, seed), folder in ARCHIVED.items()},
        datasets=domains,
    )
    path = OUT / "protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert read(path) == json.loads(json.dumps(data)), "Training/evaluation protocol differs from existing run"
    else:
        base.write_json(path, data)
    return data


def check_training(data, manifest, seed, initialization):
    assert data["dataset"] == manifest["dataset"] and data["seed"] == seed
    assert data["manifest_sha256"] == manifest["manifest_sha256"]
    assert data["train_annotation_ids"] == manifest["train_ids"]
    assert data["initialization_sha256"] == initialization
    assert data["batch_size"] == 4 and data["optimizer_steps_total"] == 1560
    assert data["train_support_shots"] == 1 and data["support_shots"] == 5
    assert data["input_size"] == 640 and data["backbone_input_size"] == 1024
    assert data["radial_stage_sizes"] == [256] and data["radial_gate"] == "adaptive"
    assert data["prototype_slots"] == "2 global,8 corrected instances,9 negative; missing slots zero"
    assert data["parameters_trainable"] == 1153804


def run_training(dataset, seed, support_ids):
    from scripts import run_tiny_ablation_combined as trainer

    trainer.OUT = train_folder(dataset, seed).parent
    trainer.RUN_SEED = seed
    for smoke in (True, False):
        if (train_folder(dataset, seed) / "preflight.json").exists() and smoke:
            continue
        trainer.worker(argparse.Namespace(
            dataset=dataset, variant="combined", smoke=smoke,
            support_ids=support_ids, radial_options={"active_stages": (1,)},
            radial_metadata=METADATA,
            extra_sources=[Path(__file__).resolve()],
            cached_support_mean_error_limit=3e-5,
        ))


def check_sweep(path, manifest, support_ids, checkpoint_sha, seed):
    from visual_prompt.evaluation.comparison import summarize

    row = read(path)
    assert row["checkpoint_sha256"] == checkpoint_sha and row["seed"] == seed, path
    assert row["support_annotation_ids"] == support_ids and row["complete_test"], path
    assert row["prediction_coverage_verified"] and row["images"] == len(manifest["test_ids"]), path
    assert {p.stem for p in (path.parents[1] / "predictions").glob("*.png")} == set(manifest["test_ids"]), path
    with (path.parent / "per_image.csv").open(encoding="utf-8-sig", newline="") as stream:
        counts = [
            {key: (value if key == "sample_id" else float(value) if key == "image_iou" else int(value))
             for key, value in record.items()}
            for record in csv.DictReader(stream)
        ]
    assert [record["sample_id"] for record in counts] == manifest["test_ids"], path
    recomputed = summarize(counts)
    if any(row[key] != value for key, value in recomputed.items()):
        row.update(recomputed)
        base.write_json(path, row)
    return row


def run_model(dataset, seed):
    import torch
    from scripts.eval_tiny_fixed_memory_radial import make_predictor
    from scripts.run_tiny_ablation_combined import restore
    from visual_prompt.models.tiny_prototype_feature_radial import install_prototype_feature_radial
    from visual_prompt.models.tiny_prototype_memory_replacement import install_prototype_memory

    plan = protocol()
    initialization = plan["initialization_sha256"]
    manifest = base.manifest_for(dataset)
    support_ids = fixed_supports(manifest)
    archived = ARCHIVED.get((dataset, seed))
    archived_folder = ROOT / "outputs" / archived if archived else None
    prior_train = LEGACY_OUT / "training" / dataset / f"seed{seed}" / "combined"
    if archived_folder and (archived_folder / "checkpoints/epoch_0120.pt").is_file():
        folder = archived_folder
        source = "retained_final_configuration"
    elif (prior_train / "checkpoints/epoch_0120.pt").is_file():
        folder = prior_train
        source = "completed_before_support_protocol_change"
    else:
        folder = train_folder(dataset, seed)
        source = "trained_with_fixed_inference_protocol"
        if not (folder / SUMMARY).exists():
            run_training(dataset, seed, support_ids)

    checkpoint = folder / "checkpoints/epoch_0120.pt"
    checkpoint_sha = base.sha(checkpoint)
    saved_config = read(folder / "config.json")
    check_training(saved_config, manifest, seed, initialization)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint_data["epoch"] == 120 and checkpoint_data["config"] == saved_config
    assert [row["epoch"] for row in checkpoint_data["history"]] == list(range(1, 121))
    output = eval_folder(dataset, seed)
    if saved_config["support_annotation_ids"] == support_ids and (folder / SUMMARY).is_file():
        result_path = folder / SUMMARY
    else:
        result_path = output / SUMMARY
        if not result_path.exists():
            base.seed_all()
            torch.set_num_threads(2)
            model, *unused, identity = base.setup_sansa(manifest, torch.device("cuda"))
            del unused
            assert identity["initialization_sha256"] == initialization
            install_prototype_memory(model)
            install_prototype_feature_radial(model, active_stages=(1,))
            restore(model, checkpoint_data["model"])
            model.eval()
            config = dict(saved_config, **METADATA, support_annotation_ids=support_ids,
                          inference_support_protocol="original_manifest_fixed_five")
            with torch.no_grad():
                predict = make_predictor(model, manifest, "learned", True)
                base.evaluate(model, predict, manifest, output, config, 120, checkpoint_sha)

    result = check_sweep(result_path, manifest, support_ids, checkpoint_sha, seed)
    base.write_json(output / "record.json", dict(
        dataset=dataset, seed=seed, support_ids=support_ids,
        checkpoint=relative(checkpoint), checkpoint_sha256=checkpoint_sha,
        training_source=source,
        summary=relative(result_path), counts=relative(result_path.parent / "per_image.csv"),
        predictions=relative(result_path.parents[1] / "predictions"),
        **{key: result[key] for key in METRICS},
    ))


def report():
    sweeps = [read(path) for path in sorted((OUT / "evaluation").glob("*/*/record.json"))]
    dataset_results = {}
    for dataset, seeds in SEEDS.items():
        means = {}
        for seed in seeds:
            rows = [r for r in sweeps if (r["dataset"], r["seed"]) == (dataset, seed)]
            if rows:
                row, = rows
                means[str(seed)] = {key: row[key] for key in METRICS}
        complete = len(means) == len(seeds)
        dataset_results[dataset] = {
            "per_training_seed": means,
            "mean": {key: statistics.mean(row[key] for row in means.values()) for key in METRICS} if complete else None,
            "sample_std": {key: statistics.stdev(row[key] for row in means.values()) for key in METRICS} if complete else None,
        }
    base.write_json(OUT / "summary.json", dict(completed_sweeps=len(sweeps), expected_sweeps=9, datasets=dataset_results))
    if sweeps:
        path = OUT / "runs.csv"
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(sweeps[0]))
            writer.writeheader()
            writer.writerows(sweeps)
        tmp.replace(path)


def queue():
    protocol()
    script = Path(__file__).resolve()
    jobs = list(ARCHIVED)
    jobs.extend((dataset, seed) for dataset, seeds in SEEDS.items() for seed in seeds if (dataset, seed) not in ARCHIVED)
    for dataset, seed in jobs:
        if (eval_folder(dataset, seed) / "record.json").exists():
            continue
        job = dict(dataset=dataset, seed=seed)
        base.write_json(OUT / "status.json", dict(status="running", **job))
        command = [sys.executable, "-u", str(script), "--dataset", dataset, "--seed", str(seed)]
        with (OUT / f"{dataset}_seed{seed}.log").open("a", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        report()
        if result.returncode:
            base.write_json(OUT / "status.json", dict(status="failed", **job, exit_code=result.returncode))
            raise SystemExit(result.returncode)
    report()
    assert read(OUT / "summary.json")["completed_sweeps"] == 9
    base.write_json(OUT / "status.json", dict(status="complete", completed_sweeps=9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SEEDS)
    parser.add_argument("--seed", type=int, choices=(111, 112, 113))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(protocol(), indent=2))
    elif args.dataset and args.seed:
        run_model(args.dataset, args.seed)
    elif args.dataset or args.seed:
        parser.error("--dataset and --seed are required for an individual run")
    else:
        queue()
