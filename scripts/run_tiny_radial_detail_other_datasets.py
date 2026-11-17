"""Run the four IRSTD radial controls on SIRST and NUDT-SIRST."""
import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

import run_tiny_ablation_other_datasets as cross
import run_tiny_radial_detail_irstd as radial
from run_tiny_ablation_three_seeds import FIELDS, ROOT, read, write

OUT = ROOT / 'outputs/tiny_radial_detail_other_datasets_seed111'
SEED = 111
SCRIPT = Path(__file__).resolve()


def reference_path(dataset):
    return cross.OUT / f'seed{SEED}' / dataset / 'combined' / radial.METRICS


def worker(dataset, mode, smoke):
    import run_tiny_ablation_combined as runner

    runner.OUT = OUT / dataset / mode
    runner.RUN_SEED = SEED
    metadata = radial.settings(mode)
    metadata.update(
        fusion=f'19 prototype slots -> frozen native attention -> radial residuals at {metadata["radial_stage_sizes"]}',
        radial_detail_reference=reference_path(dataset).relative_to(ROOT).as_posix(),
    )
    runner.worker(argparse.Namespace(
        dataset=dataset, variant='combined', smoke=smoke,
        radial_options=radial.CONTROLS[mode], radial_metadata=metadata,
        extra_sources=[SCRIPT, Path(radial.__file__).resolve()],
    ))


def verify_control(dataset, mode, reference):
    radial.OUT = OUT / dataset
    result = radial.verified(mode, reference)
    if result is not None:
        assert read(ROOT / result['summary'])['radial_detail_reference'] == (
            reference_path(dataset).relative_to(ROOT).as_posix()
        )
    return result


def report(references, completed, failures):
    rows = []
    for dataset, reference in references.items():
        rows.append(dict(
            dataset=dataset, seed=SEED, mode='dual_scale_reference',
            **{key: reference[key] for key in FIELDS}, IoU_delta_pp=0.0,
            summary=reference_path(dataset).relative_to(ROOT).as_posix(),
        ))
        for mode, result in completed[dataset].items():
            rows.append(dict(
                dataset=dataset, seed=SEED, mode=mode, **result['metrics'],
                IoU_delta_pp=result['delta_from_full']['IoU_percent'],
                summary=result['summary'],
            ))
    path = OUT / 'comparison.csv'
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    write(OUT / 'comparison.json', dict(
        completed=sum(map(len, completed.values())), expected=len(references)*len(radial.CONTROLS),
        seed=SEED, split_seed=111, epochs=120, batch_size=4, rows=rows, failures=failures,
    ))


def wait_gpu(status, job):
    # Leave the already queued IRSTD runs time to acquire the GPU between phases.
    consecutive_idle = 0
    while consecutive_idle < 4:
        line = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            text=True,
        ).splitlines()[0]
        memory, utilization = (int(value.strip()) for value in line.split(','))
        consecutive_idle = consecutive_idle + 1 if memory < 1800 and utilization < 20 else 0
        if consecutive_idle < 4:
            write(status, dict(status='waiting_for_gpu', **job,
                               gpu_memory_MiB=memory, gpu_utilization_percent=utilization))
            time.sleep(15)


def queue(not_before, plan_only):
    references = {}
    for dataset in cross.DATASETS:
        assert cross.verified(dataset, SEED, 'combined', cross.manifest_for(dataset)) is not None
        references[dataset] = read(reference_path(dataset))
    completed = {
        dataset: {mode: row for mode in radial.CONTROLS
                  if (row := verify_control(dataset, mode, ref)) is not None}
        for dataset, ref in references.items()
    }
    jobs = [(dataset, mode) for dataset in references for mode in radial.CONTROLS
            if mode not in completed[dataset]]
    if plan_only:
        for dataset, reference in references.items():
            print(f"{dataset} seed {SEED}: baseline IoU {reference['IoU_percent']:.4f}")
        print(f'{len(jobs)} remaining runs: {jobs}')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    failures = []
    report(references, completed, failures)
    status = OUT / 'queue_status.json'
    if not_before is not None and not_before.timestamp() > time.time():
        write(status, dict(status='scheduled', not_before=not_before.isoformat(), pid=os.getpid(),
                           seed=SEED, remaining_jobs=jobs))
        while time.time() < not_before.timestamp():
            time.sleep(max(0, min(60, not_before.timestamp() - time.time())))
    for dataset, mode in jobs:
        for phase in ('preflight', 'training'):
            job = dict(dataset=dataset, mode=mode, phase=phase, seed=SEED,
                       completed=sum(map(len, completed.values())), pid=os.getpid())
            wait_gpu(status, job)
            log = OUT / f'{dataset}_{mode}_{phase}.log'
            write(status, dict(status=phase, **job, log=log.name))
            command = [sys.executable, '-u', str(SCRIPT), '--dataset', dataset, '--mode', mode]
            if phase == 'preflight':
                command.append('--smoke')
            with log.open('a', encoding='utf-8') as stream:
                code = subprocess.call(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if code:
                failures.append(dict(**job, log=log.name, exit_code=code))
                break
        else:
            result = verify_control(dataset, mode, references[dataset])
            if result is None:
                failures.append(dict(dataset=dataset, mode=mode, phase='verification'))
            else:
                completed[dataset][mode] = result
        report(references, completed, failures)
    write(status, dict(status='partial_failure' if failures else 'complete', seed=SEED,
                       completed=sum(map(len, completed.values())), failures=failures))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--not-before', type=datetime.fromisoformat)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--dataset', choices=cross.DATASETS)
    parser.add_argument('--mode', choices=radial.CONTROLS)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.dataset and args.mode:
        worker(args.dataset, args.mode, args.smoke)
    elif args.dataset or args.mode or args.smoke:
        parser.error('--dataset and --mode are required for an individual run')
    else:
        queue(args.not_before, args.plan_only)
