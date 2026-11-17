"""IRSTD-1k seed 113 radial controls with the 19-token memory held fixed."""
import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

from run_tiny_ablation_three_seeds import FIELDS, ROOT, folder_for, read, sha, verify, write

OUT = ROOT / 'outputs/tiny_radial_detail_irstd_seed113'
SEED = 113
METRICS = 'epoch_0120/metrics/summary.json'
REFERENCE = folder_for(SEED, 'combined') / METRICS
CONTROLS = {
    'only_128': dict(active_stages=(0,)),
    'only_256': dict(active_stages=(1,)),
    'neighbor_values': dict(neighbor_values=True),
    'fixed_gate_05': dict(fixed_gate=True),
}


def settings(mode):
    options = CONTROLS[mode]
    stages = options.get('active_stages', (0, 1))
    grids = [(128, 256)[index] for index in stages]
    return dict(
        radial_control=mode,
        radial_stage_sizes=grids,
        radial_stage_channels=[(64, 32)[index] for index in stages],
        radial_descriptor='neighbor_values' if options.get('neighbor_values') else 'center_minus_neighbor',
        radial_gate=.5 if options.get('fixed_gate') else 'adaptive',
        radial_sampling_points_per_query=64*sum(grid**2 for grid in grids),
    )


def worker(mode, smoke):
    import run_tiny_ablation_combined as runner

    runner.OUT = OUT / mode
    runner.RUN_SEED = SEED
    metadata = settings(mode)
    grids = metadata['radial_stage_sizes']
    metadata.update(
        fusion=f'19 prototype slots -> frozen native attention -> radial residuals at {grids}',
        matched_reference=REFERENCE.relative_to(ROOT).as_posix(),
        baseline_note='Matched IRSTD-1k seed113 combined reference.',
    )
    runner.worker(argparse.Namespace(
        dataset='IRSTD-1k', variant='combined', smoke=smoke,
        radial_options=CONTROLS[mode], radial_metadata=metadata,
        extra_sources=[Path(__file__).resolve()],
    ))


def verified(mode, reference):
    folder = OUT / mode / 'combined'
    status = folder / 'status.json'
    if not status.exists() or read(status)['status'] != 'complete':
        return None
    path = folder / METRICS
    row = read(path)
    for key, value in settings(mode).items():
        assert row[key] == value, (mode, key)
    for key in (
        'dataset', 'seed', 'split_seed', 'batch_size', 'epochs', 'manifest_sha256',
        'support_annotation_ids', 'train_annotation_ids', 'initialization_sha256',
        'optimizer', 'lr', 'precision', 'threshold', 'prototype_slots',
        'radial_radii_128grid', 'radial_radii_256grid', 'radial_residual_scale',
        'radial_compressed_channels', 'radial_descriptor_channels',
    ):
        assert row[key] == reference[key], (mode, key)
    assert row['all_120_epochs_verified'] and row['predictions_recount_verified']
    assert row['complete_test'] and row['prediction_coverage_verified']
    assert row['checkpoint_sha256'] == sha(folder / 'checkpoints/epoch_0120.pt')
    assert all(math.isfinite(row[key]) for key in FIELDS)
    assert read(folder / 'preflight.json')['active_radial_heads_have_finite_nonzero_gradients']
    return dict(
        **settings(mode),
        metrics={key: row[key] for key in FIELDS},
        delta_from_full={key: row[key]-reference[key] for key in FIELDS},
        parameters_trainable=row['parameters_trainable'],
        summary=path.relative_to(ROOT).as_posix(),
    )


def report(reference, results):
    write(OUT / 'comparison.json', dict(
        status='complete' if len(results) == len(CONTROLS) else 'partial',
        dataset='IRSTD-1k', seed=SEED, split_seed=111, epochs=120, batch_size=4,
        reference=dict(
            radial_stage_sizes=[128, 256], radial_descriptor='center_minus_neighbor',
            radial_gate='adaptive', radial_sampling_points_per_query=64*(128**2+256**2),
            metrics={key: reference[key] for key in FIELDS},
            parameters_trainable=reference['parameters_trainable'],
            summary=REFERENCE.relative_to(ROOT).as_posix(),
        ),
        controls=results,
    ))


def queue():
    assert verify(SEED, 'combined') is not None
    reference = read(REFERENCE)
    results = {mode: row for mode in CONTROLS if (row := verified(mode, reference)) is not None}
    report(reference, results)
    status = OUT / 'queue_status.json'
    for mode in CONTROLS:
        if mode in results:
            continue
        for phase in ('preflight', 'training'):
            job = dict(mode=mode, phase=phase, seed=SEED, completed_modes=list(results))
            idle_checks = 0
            while idle_checks < 3:
                line = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                     '--format=csv,noheader,nounits'], text=True,
                ).splitlines()[0]
                memory, utilization = (int(value.strip()) for value in line.split(','))
                idle_checks = idle_checks + 1 if memory < 1800 and utilization < 20 else 0
                if idle_checks == 3:
                    break
                write(status, dict(status='waiting_for_gpu', **job,
                                   gpu_memory_MiB=memory, gpu_utilization_percent=utilization))
                time.sleep(15)
            log = OUT / f'{mode}_{phase}.log'
            write(status, dict(status=phase, **job, log=log.name))
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--mode', mode]
            if phase == 'preflight':
                cmd.append('--smoke')
            with log.open('a', encoding='utf-8') as stream:
                code = subprocess.call(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if code:
                write(status, dict(status='failed', **job, log=log.name, exit_code=code))
                raise SystemExit(code)
            print(f'{mode}: {phase} complete', flush=True)
        results[mode] = verified(mode, reference)
        assert results[mode] is not None
        report(reference, results)
    write(status, dict(status='complete', seed=SEED, completed_modes=list(results)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=CONTROLS)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.mode:
        worker(args.mode, args.smoke)
    elif args.smoke:
        parser.error('--smoke requires --mode')
    else:
        queue()
