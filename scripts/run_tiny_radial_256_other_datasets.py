"""Run the 256-only sampling and gate controls on SIRST and NUDT-SIRST."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

import run_tiny_ablation_other_datasets as cross
import run_tiny_radial_detail_irstd as radial
from run_tiny_ablation_three_seeds import FIELDS, ROOT, read, write
from run_tiny_radial_256_irstd import CONTROLS
from run_tiny_radial_detail_other_datasets import OUT as PREVIOUS, reference_path, wait_gpu

OUT = ROOT / 'outputs/tiny_radial_256_other_datasets_seed111'
SCRIPT = Path(__file__).resolve()
SEED = 111
ROWS = {
    'only_128': ('128', 'Difference', 'Adaptive'),
    'dual_scale': ('128+256', 'Difference', 'Adaptive'),
    'neighbor_values': ('256', 'Pixel value', 'Adaptive'),
    'fixed_gate_05': ('256', 'Difference', 'Fixed (0.5)'),
    'only_256': ('256', 'Difference', 'Adaptive'),
}


def baseline_path(dataset):
    return PREVIOUS / dataset / 'only_256/combined' / radial.METRICS


def worker(dataset, mode, smoke):
    import run_tiny_ablation_combined as runner

    runner.OUT = OUT / dataset / mode
    runner.RUN_SEED = SEED
    radial.CONTROLS = CONTROLS
    metadata = radial.settings(mode)
    metadata.update(
        fusion='19 prototype slots -> frozen native attention -> radial residual at 256',
        radial_detail_reference=baseline_path(dataset).relative_to(ROOT).as_posix(),
    )
    runner.worker(argparse.Namespace(
        dataset=dataset, variant='combined', smoke=smoke,
        radial_options=CONTROLS[mode], radial_metadata=metadata,
        extra_sources=[SCRIPT, Path(radial.__file__).resolve(),
                       ROOT / 'scripts/run_tiny_radial_256_irstd.py'],
    ))


def existing_rows(dataset):
    dual_path = reference_path(dataset)
    dual = read(dual_path)
    assert cross.verified(dataset, SEED, 'combined', cross.manifest_for(dataset)) is not None
    radial.OUT = PREVIOUS / dataset
    rows = {'dual_scale': dict(metrics={key: dual[key] for key in FIELDS},
                               summary=dual_path.relative_to(ROOT).as_posix())}
    for mode in ('only_128', 'only_256'):
        rows[mode] = radial.verified(mode, dual)
        assert rows[mode] is not None
    return rows


def verify_control(dataset, mode):
    radial.OUT = OUT / dataset
    path = baseline_path(dataset)
    row = radial.verified(mode, read(path))
    if row is not None:
        assert read(ROOT / row['summary'])['radial_detail_reference'] == path.relative_to(ROOT).as_posix()
    return row


def report(datasets, failures):
    lines = ['# Radial ablation on SIRST and NUDT-SIRST', '',
             'Training seed 111, 50 training images, five support images, and epoch 120 evaluation.']
    for dataset, rows in datasets.items():
        lines.extend(['', f'## {dataset}', '',
                      '| Stage(s) | Sampling | Gate | IoU ↑ | nIoU ↑ | F1 ↑ | Pd ↑ | FA ↓ |',
                      '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |'])
        for mode, labels in ROWS.items():
            result = rows.get(mode)
            values = ([f'{result["metrics"][field]:.2f}' for field in FIELDS]
                      if result else ['Pending', '—', '—', '—', '—'])
            if not result and any(item['dataset'] == dataset and item['mode'] == mode for item in failures):
                values[0] = 'Failed'
            lines.append('| ' + ' | '.join((*labels, *values)) + ' |')
    temporary = OUT / 'comparison.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(OUT / 'comparison.md')
    write(OUT / 'comparison.json', dict(seed=SEED, datasets=datasets, row_order=list(ROWS), failures=failures))


def queue(not_before, plan_only):
    datasets = {dataset: existing_rows(dataset) for dataset in cross.DATASETS}
    radial.CONTROLS = CONTROLS
    for dataset, rows in datasets.items():
        for mode in CONTROLS:
            result = verify_control(dataset, mode)
            if result is not None:
                rows[mode] = result
    jobs = [(dataset, mode) for dataset, rows in datasets.items() for mode in CONTROLS if mode not in rows]
    if plan_only:
        for dataset, rows in datasets.items():
            print(f'{dataset}: verified {list(rows)}')
        print(f'{len(jobs)} remaining runs: {jobs}')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    status = OUT / 'queue_status.json'
    failures = []
    report(datasets, failures)
    if not_before is not None and not_before.timestamp() > time.time():
        write(status, dict(status='scheduled', not_before=not_before.isoformat(), pid=os.getpid(),
                           seed=SEED, remaining_jobs=jobs))
        while time.time() < not_before.timestamp():
            time.sleep(max(0, min(60, not_before.timestamp() - time.time())))
    for dataset, mode in jobs:
        for phase in ('preflight', 'training'):
            job = dict(dataset=dataset, mode=mode, phase=phase, seed=SEED, pid=os.getpid())
            wait_gpu(status, job)
            log = OUT / f'{dataset}_{mode}_{phase}.log'
            write(status, dict(status=phase, **job, log=log.name))
            command = [sys.executable, '-u', str(SCRIPT), '--dataset', dataset, '--mode', mode]
            if phase == 'preflight':
                command.append('--smoke')
            with log.open('a', encoding='utf-8') as stream:
                code = subprocess.call(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if code:
                failures.append(dict(**job, exit_code=code, log=log.name))
                break
        else:
            result = verify_control(dataset, mode)
            assert result is not None
            datasets[dataset][mode] = result
        report(datasets, failures)
    completed = [(dataset, mode) for dataset, rows in datasets.items() for mode in CONTROLS if mode in rows]
    write(status, dict(status='partial_failure' if failures else 'complete',
                       completed=completed, failures=failures, seed=SEED))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--not-before', type=datetime.fromisoformat)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--dataset', choices=cross.DATASETS)
    parser.add_argument('--mode', choices=CONTROLS)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.dataset and args.mode:
        worker(args.dataset, args.mode, args.smoke)
    elif args.dataset or args.mode or args.smoke:
        parser.error('--dataset and --mode are required for an individual run')
    else:
        queue(args.not_before, args.plan_only)
