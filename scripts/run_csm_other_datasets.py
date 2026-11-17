"""Repeat the full-model CSM controls on SIRST and NUDT-SIRST."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts import run_irstd_stepwise_csm as controls

ROOT = controls.ROOT
OUT = ROOT / "outputs/csm_controls_other_datasets"
REFERENCE = ROOT / "outputs/tiny_radial_detail_other_datasets_seed111"
DATASETS = ("SIRST", "NUDT-SIRST")
SEED = 111
ARMS = ("full", *controls.CONTROLS)
COUNT_FIELDS = ("tp", "fp", "fn", "pixels", "targets", "detected")


def configure(dataset):
    controls.DATASET = dataset
    controls.SEED = SEED
    controls.OUT = OUT / dataset
    controls.FULL = REFERENCE / dataset / "only_256/combined"


def evaluate_rows(dataset):
    configure(dataset)
    manifest = controls.base.manifest_for(dataset)
    reference = controls.read(controls.FULL / controls.METRICS)
    assert reference["seed"] == SEED
    assert reference["train_annotation_ids"] == manifest["train_ids"]
    assert reference["support_annotation_ids"] == manifest["support_ids"]
    expected = controls.read(controls.FULL / "preflight.json")["valid_memory_tokens"]
    rows, counts = [], {}
    for arm in ARMS:
        folder = controls.FULL if arm == "full" else controls.OUT / "trained" / arm / "combined"
        path = folder / controls.METRICS
        status = folder / "status.json"
        if not status.is_file() or controls.read(status)["status"] != "complete":
            continue
        result = controls.verify_predictions(folder, 120, manifest)
        assert result["dataset"] == dataset and result["images"] == len(manifest["test_ids"])
        assert all(result[k] == reference[k] for k in controls.MATCH_FIELDS), (dataset, arm)
        assert result["radial_stage_sizes"] == [256]
        assert result["parameters_trainable"] == reference["parameters_trainable"]
        assert controls.sha(folder / "checkpoints/epoch_0120.pt") == result["checkpoint_sha256"]
        assert result["all_120_epochs_verified"] and result["predictions_recount_verified"]
        if arm != "full":
            assert result["memory_control"] == arm
            preflight = controls.read(controls.OUT / "smoke" / arm / "combined/preflight.json")
            assert preflight["valid_memory_tokens"] == expected

        with path.with_name("per_image.csv").open(encoding="utf-8", newline="") as stream:
            predictions = list(csv.DictReader(stream))
        assert [row["sample_id"] for row in predictions] == manifest["test_ids"]
        counts[arm] = np.array([[int(row[key]) for key in COUNT_FIELDS] for row in predictions])
        rows.append(dict(dataset=dataset, seed=SEED, arm=arm, train_images=50, support_images=5,
                         test_images=len(predictions), valid_support_tokens=expected,
                         **{key: result[key] for key in controls.FIELDS},
                         summary=path.relative_to(ROOT).as_posix(),
                         checkpoint_sha256=result["checkpoint_sha256"]))
    return rows, counts


def metrics(counts):
    tp, fp, fn, pixels, targets, detected = counts.T
    return 100 * tp / (tp + fp + fn), 1e6 * fp / pixels, 100 * detected / targets


def paired_differences(dataset, counts):
    reference = counts["full"]
    draw = np.random.default_rng(SEED).integers(len(reference), size=(10000, len(reference)))
    reference_boot = metrics(reference[draw].sum(axis=1))
    for arm in controls.CONTROLS:
        if arm not in counts:
            continue
        control = counts[arm]
        point = np.subtract(metrics(reference.sum(axis=0)), metrics(control.sum(axis=0)))
        boot = np.subtract(reference_boot, metrics(control[draw].sum(axis=1)))
        for name, value, samples in zip(("IoU_pp", "FA_per_million", "Pd_pp"), point, boot, strict=True):
            lower, upper = np.quantile(samples, [.025, .975])
            yield dict(dataset=dataset, control=arm, metric=name, full_minus_control=value,
                       lower_95=lower, upper_95=upper)


def write_csv(path, rows):
    if rows:
        temp = path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)


def collect():
    rows, paired = [], []
    for dataset in DATASETS:
        recorded, counts = evaluate_rows(dataset)
        rows.extend(recorded)
        paired.extend(paired_differences(dataset, counts))
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "results.csv", rows)
    write_csv(OUT / "paired_image_bootstrap.csv", paired)
    controls.write(OUT / "audit.json", dict(status="complete" if len(rows) == 6 else "partial",
                                          matched_fields=controls.MATCH_FIELDS, verified_results=rows))


def queue():
    collect()
    for dataset in DATASETS:
        configure(dataset)
        for arm in controls.CONTROLS:
            status = controls.OUT / "trained" / arm / "combined/status.json"
            if status.is_file() and controls.read(status)["status"] == "complete":
                continue
            for phase, options in (("preflight", ["--smoke"]), ("training", [])):
                path = controls.OUT / f"{arm}_{phase}.log"
                path.parent.mkdir(parents=True, exist_ok=True)
                if phase == "preflight" and (
                        controls.OUT / "smoke" / arm / "combined/preflight.json").is_file():
                    continue
                print(f"{dataset} | {arm} | {phase}", flush=True)
                with path.open("a", encoding="utf-8") as log:
                    subprocess.run([sys.executable, "-u", "-m", "scripts.run_csm_other_datasets",
                                    "--dataset", dataset, "--arm", arm, *options],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            collect()
            configure(dataset)
    assert controls.read(OUT / "audit.json")["status"] == "complete"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--arm", choices=controls.CONTROLS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect()
    elif args.dataset and args.arm:
        configure(args.dataset)
        controls.train_control(args.arm, args.smoke)
    elif args.dataset or args.arm or args.smoke:
        parser.error("--dataset and --arm are required for a single run")
    else:
        queue()
