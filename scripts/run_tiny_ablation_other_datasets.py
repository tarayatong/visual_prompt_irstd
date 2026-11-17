"""Run the cross-dataset ablation, prioritizing both datasets' combined models."""
import argparse
import math
import statistics
import subprocess
import sys
import time

from run_tiny_ablation_three_seeds import FIELDS, MODES, ROOT, SEEDS, read, sha, write

OUT = ROOT / 'outputs/tiny_ablation_other_datasets_20261111'
DATASETS = {'SIRST': 86, 'NUDT-SIRST': 664}
PARAMETERS = {'native': (39931013, 1117380), 'memory': (38569957, 1140932),
              'radial': (39950453, 1136820), 'combined': (38589397, 1160372)}


def manifest_for(dataset):
    m = read(ROOT / 'outputs/comparison_50shot_5support' / dataset / 'protocol/manifest.json')
    assert len(set(m['train_ids'])) == len(m['train_ids']) == 50
    assert len(set(m['support_ids'])) == len(m['support_ids']) == 5
    assert set(m['support_ids']) <= set(m['train_ids'])
    assert len(set(m['test_ids'])) == len(m['test_ids']) == DATASETS[dataset]
    assert not set(m['train_ids']) & set(m['test_ids'])
    return m


def verified(dataset, seed, method, manifest):
    folder = OUT / f'seed{seed}' / dataset / MODES[method]
    path = folder / 'epoch_0120/metrics/summary.json'
    if not path.exists():
        return None
    result = read(path)
    if not result.get('predictions_recount_verified') or not result.get('all_120_epochs_verified'):
        return None
    assert (result['dataset'], result['seed'], result['variant']) == (dataset, seed, MODES[method])
    assert result['batch_size'] == 4 and result['epochs'] == result['training_horizon'] == 120
    assert result['manifest_sha256'] == manifest['manifest_sha256']
    assert result['train_annotation_ids'] == manifest['train_ids']
    assert result['support_annotation_ids'] == manifest['support_ids']
    assert result['complete_test'] and result['prediction_coverage_verified']
    assert result['precision'] == 'BF16 training; FP32 evaluation'
    assert result['radial'] == (method in ('radial', 'combined'))
    assert (result['pipeline_parameters_total'], result['parameters_trainable']) == PARAMETERS[method]
    assert sha(folder / 'checkpoints/epoch_0120.pt') == result['checkpoint_sha256']
    history = read(folder / 'training_history.json')
    assert [row['epoch'] for row in history] == list(range(1, 121))
    assert all(row['unique_supervised_images'] == 50 for row in history)
    assert history[-1]['optimizer_updates'] == 1560
    assert all(math.isfinite(result[key]) for key in FIELDS)
    return {key: result[key] for key in FIELDS}


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_std=statistics.stdev(values) if len(values) > 1 else None)


def summarize(manifests, cells, expected):
    datasets = {}
    for dataset, manifest in manifests.items():
        rows = {str(seed): {method: cells[(dataset, seed, method)] for method in MODES
                            if (dataset, seed, method) in cells} for seed in SEEDS}
        aggregate = {method: {key: stats([row[method][key] for row in rows.values() if method in row])
                              for key in FIELDS} for method in MODES}
        paired = {}
        for a, b in (('memory', 'native'), ('radial', 'native'), ('combined', 'native'), ('combined', 'memory')):
            deltas = {seed: {key: row[a][key] - row[b][key] for key in FIELDS}
                      for seed, row in rows.items() if a in row and b in row}
            paired[f'{a}_minus_{b}'] = dict(by_seed=deltas, IoU_percent=stats(
                [row['IoU_percent'] for row in deltas.values()]))
        datasets[dataset] = dict(test_images=DATASETS[dataset], support_ids=manifest['support_ids'],
                                 manifest_sha256=manifest['manifest_sha256'], by_seed=rows,
                                 aggregate=aggregate, paired_differences=paired)
    write(OUT / 'summary.json', dict(status='complete' if len(cells) == expected else 'partial',
                                    completed_cells=len(cells), expected_cells=expected, seeds=SEEDS,
                                    split_seed=111, train_images=50, support_images=5,
                                    batch_size=4, epochs=120, datasets=datasets))


def wait_gpu(job):
    while True:
        line = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True).splitlines()[0]
        memory, utilization = [int(item.strip()) for item in line.split(',')]
        if memory < 1800 and utilization < 20:
            return
        write(OUT / 'queue_status.json', dict(status='waiting_for_gpu', **job,
                                            gpu_memory_MiB=memory, gpu_utilization_percent=utilization))
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    followup = parser.add_mutually_exclusive_group()
    followup.add_argument('--repeat-sirst112', action='store_true')
    followup.add_argument('--extra-sirst-seed114', action='store_true')
    args = parser.parse_args()
    manifests = {dataset: manifest_for(dataset) for dataset in DATASETS}
    # Complete both datasets' combination jobs before any of the other arms.
    planned = [(dataset, seed, 'combined') for seed in SEEDS for dataset in DATASETS]
    planned += [(dataset, seed, method) for dataset in DATASETS for seed in SEEDS
                for method in MODES if method != 'combined']
    cells = {}
    for dataset, seed, method in planned:
        result = verified(dataset, seed, method, manifests[dataset])
        if result is not None:
            cells[(dataset, seed, method)] = result
    summarize(manifests, cells, len(planned))
    if args.report_only:
        return
    missing = [item for item in planned if item not in cells]
    write(OUT / 'plan.json', dict(jobs=[dict(dataset=d, seed=s, method=m) for d, s, m in missing],
                                  combination_priority=True))
    log_dir = OUT / 'logs'
    log_dir.mkdir(exist_ok=True)
    failures = []
    repeat_seed = 114 if args.extra_sirst_seed114 else 112 if args.repeat_sirst112 else None
    repeat_result = OUT / f'repeats/sirst_seed{repeat_seed}_combined/comparison.json'
    if repeat_seed is not None and not repeat_result.exists():
        job = dict(dataset='SIRST', seed=repeat_seed, method='combined', followup=True,
                   completed_cells=len(cells), expected_cells=len(planned))
        for phase in ('preflight', 'training'):
            wait_gpu(job)
            write(OUT / 'queue_status.json', dict(status=phase, **job, failed_jobs=failures))
            command = [sys.executable, '-u', str(ROOT / 'scripts/run_tiny_ablation_sirst112_repeat.py'),
                       '--seed', str(repeat_seed)]
            if phase == 'preflight':
                command.append('--smoke')
            log_path = log_dir / f'SIRST_{repeat_seed}_combined_repeat_{phase}.log'
            with log_path.open('a', encoding='utf-8') as log:
                code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if code:
                failures.append(dict(**job, phase=phase, exit_code=code, log=str(log_path)))
                print(f'SIRST seed{repeat_seed} combined follow-up: {phase} failed, see {log_path}', flush=True)
                break
            print(f'SIRST seed{repeat_seed} combined follow-up: {phase} passed', flush=True)
    for index, (dataset, seed, method) in enumerate(missing, 1):
        job = dict(dataset=dataset, seed=seed, method=method, job_index=index, jobs_total=len(missing),
                   completed_cells=len(cells), expected_cells=len(planned))
        for phase in ('preflight', 'training'):
            wait_gpu(job)
            write(OUT / 'queue_status.json', dict(status=phase, **job, failed_jobs=failures))
            command = [sys.executable, '-u', str(ROOT / f'scripts/run_tiny_ablation_{method}.py'),
                       '--dataset', dataset, '--seed', str(seed)]
            if phase == 'preflight':
                command.append('--smoke')
            log_path = log_dir / f'{dataset}_{seed}_{method}_{phase}.log'
            with log_path.open('a', encoding='utf-8') as log:
                code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if code:
                failures.append(dict(**job, phase=phase, exit_code=code, log=str(log_path)))
                print(f'{dataset} seed{seed} {method}: {phase} failed, see {log_path}', flush=True)
                break
            print(f'{dataset} seed{seed} {method}: {phase} passed', flush=True)
        else:
            result = verified(dataset, seed, method, manifests[dataset])
            if result is None:
                failures.append(dict(**job, phase='verification', error='Missing verified result'))
            else:
                cells[(dataset, seed, method)] = result
                summarize(manifests, cells, len(planned))
    write(OUT / 'queue_status.json', dict(status='complete' if not failures else 'partial_failure',
                                        completed_cells=len(cells), expected_cells=len(planned),
                                        failed_jobs=failures))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
