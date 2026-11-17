"""Render the fixed CoRe-IR qualitative cases with archived or fresh masks."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_core_ir_final_three_seeds import (
    ARCHIVED, OUT, base, check_training, eval_folder, fixed_supports, read,
)

REFERENCE = ROOT / "outputs/top8_analysis_20261105"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def record_path(path):
    return (path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).as_posix()


def rerun_best(selected, output):
    import torch

    from scripts.eval_tiny_fixed_memory_radial import make_predictor
    from scripts.run_tiny_ablation_combined import restore
    from visual_prompt.models.tiny_prototype_feature_radial import install_prototype_feature_radial
    from visual_prompt.models.tiny_prototype_memory_replacement import install_prototype_memory

    summary = read(OUT / "summary.json")
    assert summary["completed_sweeps"] == summary["expected_sweeps"] == 9
    plan = read(OUT / "protocol.json")
    runs = {}
    for dataset, scores in summary["datasets"].items():
        seed, best = max(scores["per_training_seed"].items(), key=lambda item: item[1]["IoU_percent"])
        run = read(eval_folder(dataset, int(seed)) / "record.json")
        assert run["IoU_percent"] == best["IoU_percent"]
        manifest = base.manifest_for(dataset)
        assert run["support_ids"] == fixed_supports(manifest) == plan["datasets"][dataset]["inference_support_ids"]
        ids = [sid for domain, sid in selected if domain == dataset]
        assert ids and set(ids) <= set(manifest["test_ids"])

        path = ROOT / run["checkpoint"]
        digest = base.sha(path)
        assert digest == run["checkpoint_sha256"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["epoch"] == 120
        check_training(checkpoint["config"], manifest, int(seed), plan["initialization_sha256"])
        base.seed_all()
        torch.set_num_threads(2)
        model, *unused, identity = base.setup_sansa(manifest, torch.device("cuda"))
        del unused
        assert identity["initialization_sha256"] == plan["initialization_sha256"]
        install_prototype_memory(model)
        install_prototype_feature_radial(model, active_stages=(1,))
        restore(model, checkpoint["model"])
        model.eval()

        folder = output / "inference" / dataset
        config = dict(method="CoRe-IR", dataset=dataset, seed=int(seed), training_horizon=120,
                      evaluation_split="qualitative", support_ids=run["support_ids"])
        with torch.no_grad():
            predict = make_predictor(model, manifest, "learned", True)
            base.evaluate(model, predict, dict(manifest, test_ids=ids), folder, config, 120, digest)
        results = folder / "epoch_0120"
        predictions = results / "predictions"
        archive_matches = {}
        for sid in ids:
            archived = ROOT / run["predictions"] / f"{sid}.png"
            with Image.open(predictions / f"{sid}.png") as fresh, Image.open(archived) as old:
                archive_matches[sid] = np.array_equal(np.asarray(fresh), np.asarray(old))
        runs[dataset] = dict(
            run,
            archived_predictions=run["predictions"],
            predictions=record_path(predictions),
            counts=record_path(results / "metrics/per_image.csv"),
            qualitative_summary=record_path(results / "metrics/summary.json"),
            pixel_identical_to_archive=archive_matches,
        )
        del model, predict, checkpoint
        torch.cuda.empty_cache()
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun-best", action="store_true",
                        help="Rerun the selected cases from each dataset's highest-IoU final checkpoint.")
    parser.add_argument("--reference", type=Path, default=REFERENCE,
                        help="Directory containing the baseline masks and per-image records.")
    parser.add_argument("--output", type=Path,
                        help="Directory for the figure and its provenance.")
    args = parser.parse_args()
    from scripts import core_ir_qualitative_plot as renderer

    output = args.output or OUT / ("qualitative_best" if args.rerun_best else "qualitative")
    output.mkdir(parents=True, exist_ok=True)
    if args.rerun_best:
        runs = rerun_best(renderer.SELECTED, output.resolve())
    else:
        runs = {dataset: read(eval_folder(dataset, seed) / "record.json") for dataset, seed in ARCHIVED}
    samples, provenance = [], []
    for dataset, sid in renderer.SELECTED:
        candidates = read_csv(args.reference / "bad_samples" / dataset / "all_per_image.csv")
        references = {row["method"]: row for row in candidates if row["sample_id"] == sid}
        image_file, mask_file = (references["Ours"][key] for key in ("image_path", "mask_path"))
        with Image.open(image_file) as picture, Image.open(mask_file) as mask:
            image = np.asarray(picture.convert("L"))
            gt = np.asarray(mask.convert("L")) > 0
        targets = renderer.components(gt)
        run = runs[dataset]
        counts = next(row for row in read_csv(ROOT / run["counts"]) if row["sample_id"] == sid)
        records, predictions = {}, {}
        for method in renderer.METHODS:
            is_ours = method == "Ours"
            if is_ours:
                path, expected = ROOT / run["predictions"] / f"{sid}.png", counts
            else:
                path = args.reference.resolve() / "qualitative_full/predictions" / dataset / method / f"{sid}.png"
                expected = references[method]
            with Image.open(path) as prediction:
                pred = np.asarray(prediction.convert("L")) >= 128
            metric = renderer.image_metrics(pred, gt, sid)
            assert all(metric[key] == int(expected[key]) for key in ("tp", "fp", "fn")), (dataset, method, sid)
            matched, false = renderer.match(gt, pred)
            predictions[method] = pred
            records[method] = dict(components=renderer.components(pred), matched=matched, false=false)
            provenance.append(dict(
                dataset=dataset, sample_id=sid, method="CoRe-IR" if is_ours else method,
                prediction=record_path(path), metrics=metric,
                run=run if is_ours else None,
            ))
        regions = {f"T{i+1}": dict(x=t["x"], y=t["y"], target=i) for i, t in enumerate(targets)}
        false_components = [
            record["components"][index] for record in records.values() for index in record["false"]
            if all(np.hypot(record["components"][index]["x"] - t["x"],
                            record["components"][index]["y"] - t["y"]) > 24 for t in targets)
        ]
        if false_components:
            false = max(false_components, key=lambda item: item["area"])
            regions["F"] = dict(x=false["x"], y=false["y"], target=None)
        fraction = .30 if sid == "001289" else .36 if len(regions) <= 2 else .31
        for region in regions.values():
            region["size"] = fraction * max(gt.shape) / renderer.ZOOM
        samples.append(dict(dataset=dataset, sid=sid, image=image, gt=gt, targets=targets,
                            preds=predictions, records=records, regions=regions))
    crops = renderer.render(samples, output / "qualitative_comparison.png")
    (output / "provenance.json").write_text(
        json.dumps(dict(predictions=provenance, crops=crops), indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {len(samples)} fixed cases from {len(provenance)} verified predictions.")


if __name__ == "__main__":
    main()
