"""Replace CSM with SAM2 dense support memory and a per-location MLP."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts import run_csm_other_datasets as previous
from scripts import run_irstd_stepwise_csm as controls
from scripts import run_tiny_ablation_radial as radial


ROOT = controls.ROOT
OUT = ROOT / 'outputs/csm_dense_memory_controls'
METRICS = controls.METRICS
ARM = 'dense_mlp'
DATASETS = {'IRSTD-1k': 113, 'SIRST': 111, 'NUDT-SIRST': 111}
REFERENCE_FIELDS = (*controls.MATCH_FIELDS, 'radial_stage_sizes', 'radial_stage_channels',
                    'radial_branch_parameters', 'radial_radii_256grid', 'radial_directions',
                    'radial_residual_scale', 'radial_descriptor', 'radial_gate',
                    'optimizer_steps_per_epoch', 'optimizer_steps_total')


def references(dataset):
    if dataset == 'IRSTD-1k':
        return (
            ROOT / 'outputs/tiny_radial_detail_irstd_seed113/only_256/combined',
            ROOT / 'outputs/irstd_stepwise_csm_20261115/trained/raw_target_instances/combined',
        )
    return (
        previous.REFERENCE / dataset / 'only_256/combined',
        previous.OUT / dataset / 'trained/raw_target_instances/combined',
    )


def worker(dataset, smoke):
    full, _ = references(dataset)
    radial.OUT = OUT / dataset
    radial.RUN_SEED = DATASETS[dataset]
    radial.worker(argparse.Namespace(
        dataset=dataset, variant=ARM, smoke=smoke, dense_mlp=True, active_stages=(1,),
        experiment_source=Path(__file__).resolve(), full_reference=(full / METRICS).relative_to(ROOT).as_posix(),
    ))


def completed(folder):
    status = folder / 'status.json'
    return status.exists() and controls.read(status)['status'] == 'complete'


def paired(dataset, signature, counts):
    full = counts['full']
    indices = np.random.default_rng(DATASETS[dataset]).integers(len(full), size=(10000, len(full)))
    full_boot = previous.metrics(full[indices].sum(axis=1))
    for arm, control in counts.items():
        if arm == 'full':
            continue
        points = np.subtract(previous.metrics(full.sum(axis=0)), previous.metrics(control.sum(axis=0)))
        draws = np.subtract(full_boot, previous.metrics(control[indices].sum(axis=1)))
        for metric, point, samples in zip(('IoU_pp', 'FA_per_million', 'Pd_pp'), points, draws, strict=True):
            lower, upper = np.quantile(samples, [.025, .975])
            yield dict(dataset=dataset, seed=signature['seed'], control=arm, metric=metric,
                       full_minus_control=point, lower_95=lower, upper_95=upper)


def collect():
    rows, differences = [], []
    for dataset, seed in DATASETS.items():
        full, raw = references(dataset)
        reference = controls.read(full / METRICS)
        manifest = controls.base.manifest_for(dataset)
        assert reference['seed'] == seed and reference['radial_stage_sizes'] == [256]
        assert reference['train_annotation_ids'] == manifest['train_ids']
        assert reference['support_annotation_ids'] == manifest['support_ids']
        counts = {}
        for arm, folder in (('full', full), (ARM, OUT / dataset / ARM), ('raw_target_instances', raw)):
            if arm == ARM and not completed(folder):
                continue
            path = folder / METRICS
            result = controls.read(path)
            assert all(result[field] == reference[field] for field in REFERENCE_FIELDS), (dataset, arm)
            assert result['all_120_epochs_verified'] and result['predictions_recount_verified']
            if arm == ARM:
                preflight = controls.read(folder / 'preflight.json')
                assert preflight['dense_mlp_gradients_verified'] and preflight['actual_batch_size'] == 4
                assert preflight['dense_spatial_tokens'] == result['dense_spatial_tokens'] == 20480
                assert result['dense_memory_mlp_parameters'] == 22656
                assert result['native_dense_memory'] and not result['prototype_memory']
                assert controls.sha(folder / 'checkpoints/epoch_0120.pt') == result['checkpoint_sha256']
            elif arm == 'raw_target_instances':
                assert result['memory_control'] == arm
            with path.with_name('per_image.csv').open(encoding='utf-8', newline='') as stream:
                predictions = list(csv.DictReader(stream))
            assert [item['sample_id'] for item in predictions] == manifest['test_ids']
            counts[arm] = np.array([[int(item[key]) for key in previous.COUNT_FIELDS] for item in predictions])
            rows.append(dict(dataset=dataset, seed=seed, arm=arm,
                             train_images=len(manifest['train_ids']), support_images=len(manifest['support_ids']),
                             test_images=len(predictions), trainable_parameters=result['parameters_trainable'],
                             spatial_memory_tokens=result.get('dense_spatial_tokens'),
                             **{key: result[key] for key in controls.FIELDS},
                             summary=path.relative_to(ROOT).as_posix(), checkpoint_sha256=result['checkpoint_sha256']))
        differences.extend(paired(dataset, reference, counts))
    OUT.mkdir(parents=True, exist_ok=True)
    previous.write_csv(OUT / 'results.csv', rows)
    previous.write_csv(OUT / 'paired_image_bootstrap.csv', differences)
    controls.write(OUT / 'audit.json', dict(status='complete' if len(rows) == 9 else 'partial',
                                          matched_fields=REFERENCE_FIELDS, verified_results=rows))


def queue():
    collect()
    for dataset in DATASETS:
        folder = OUT / dataset / ARM
        if completed(folder):
            continue
        folder.mkdir(parents=True, exist_ok=True)
        for phase, options in (('preflight', ['--smoke']), ('training', [])):
            if phase == 'preflight' and (folder / 'preflight.json').exists():
                continue
            print(f'{dataset} | {phase}', flush=True)
            with (folder / f'{phase}.log').open('a', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-u', '-m', 'scripts.run_csm_dense_memory',
                                '--dataset', dataset, *options], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        collect()
    assert controls.read(OUT / 'audit.json')['status'] == 'complete'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    if args.collect:
        collect()
    elif args.dataset:
        worker(args.dataset, args.smoke)
    elif args.smoke:
        parser.error('--smoke requires --dataset')
    else:
        queue()
