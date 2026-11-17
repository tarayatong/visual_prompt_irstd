"""Complete the four-arm x three-training-seed ablation without rerunning cells.

Fixed train50/support5/test201 allocation. Seeds111,112,113 affect training only.
Reuse five verified cells; run seven missing cells with unchanged layer code.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/tiny_ablation_three_seeds_20261110'
SEEDS = (111, 112, 113)
MODES = {'native':'native', 'memory':'tokens', 'radial':'feature_decoder_gated', 'combined':'combined'}
FIELDS = ('IoU_percent', 'nIoU_percent', 'F1_percent', 'Pd_percent', 'Fa_per_million')
EXISTING = {
    (111,'native'): 'outputs/tiny_shared_memory_batch4_20261110/IRSTD-1k/native',
    (111,'memory'): 'outputs/tiny_prototype_only_memory_20261110/IRSTD-1k/tokens',
    (111,'radial'): 'outputs/tiny_decoder_feature_radial_gated_20261110/IRSTD-1k/feature_decoder_gated',
    (111,'combined'): 'outputs/tiny_prototype_feature_radial_20261110/IRSTD-1k/combined',
    (112,'memory'): 'outputs/tiny_prototype_only_memory_seed112_20261110/IRSTD-1k/tokens',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def folder_for(seed, method):
    if (seed,method) in EXISTING:
        return ROOT/EXISTING[(seed,method)]
    return OUT/f'seed{seed}'/'IRSTD-1k'/MODES[method]


def verify(seed, method):
    folder = folder_for(seed, method)
    summary_path = folder/'epoch_0120/metrics/summary.json'
    if not summary_path.exists():
        return None
    result = read(summary_path)
    assert result['seed']==seed and result['epochs']==120
    assert result['batch_size']==4 and result['training_horizon']==120
    assert result['predictions_recount_verified'] and result['all_120_epochs_verified']
    assert result['complete_test'] and result['prediction_coverage_verified']
    reference = read(ROOT/EXISTING[(111,method)]/'epoch_0120/metrics/summary.json')
    for key in ('manifest_sha256','train_annotation_ids','initialization_sha256',
                'parameters_total','parameters_trainable','input_size','backbone_input_size',
                'train_support_shots','support_shots','precision','lr','optimizer'):
        assert result[key]==reference[key],(seed,method,key)
    checkpoint = folder/'checkpoints/epoch_0120.pt'
    assert checkpoint.exists() and sha(checkpoint)==result['checkpoint_sha256']
    history = read(folder/'training_history.json')
    assert [row['epoch'] for row in history]==list(range(1,121))
    assert all(row['unique_supervised_images']==50 for row in history)
    assert history[-1]['optimizer_updates']==1560
    assert all(math.isfinite(result[key]) for key in FIELDS)
    return dict(seed=seed, method=method, reused=(seed,method) in EXISTING,
                metrics={key:result[key] for key in FIELDS},
                parameters_total=result['pipeline_parameters_total'],
                parameters_trainable=result['parameters_trainable'],
                summary=summary_path.relative_to(ROOT).as_posix(),
                checkpoint_sha256=result['checkpoint_sha256'], verified=True)


def summarize():
    cells = []
    lookup = {}
    for seed in SEEDS:
        for method in MODES:
            value = verify(seed,method)
            if value:
                cells.append(value)
                lookup[(seed,method)] = value['metrics']
    aggregate = {}
    for method in MODES:
        available = [lookup[(seed,method)] for seed in SEEDS if (seed,method) in lookup]
        metrics = {}
        for key in FIELDS:
            values = [row[key] for row in available]
            metrics[key] = dict(n=len(values), mean=statistics.mean(values) if values else None,
                                sample_std=statistics.stdev(values) if len(values)>1 else None)
        aggregate[method] = metrics
    paired = {}
    contrasts = {'memory_minus_native':('memory','native'),
                 'radial_minus_native':('radial','native'),
                 'combined_minus_native':('combined','native'),
                 'combined_minus_memory':('combined','memory')}
    for name,(left,right) in contrasts.items():
        by_seed = {str(seed):{key:lookup[(seed,left)][key]-lookup[(seed,right)][key] for key in FIELDS}
                   for seed in SEEDS if (seed,left) in lookup and (seed,right) in lookup}
        values = [row['IoU_percent'] for row in by_seed.values()]
        paired[name] = dict(by_seed=by_seed, n=len(values),
                            IoU_mean=statistics.mean(values) if values else None,
                            IoU_sample_std=statistics.stdev(values) if len(values)>1 else None)
    result = dict(status='complete' if len(cells)==12 else 'partial',
                  completed_cells=len(cells), expected_cells=12, seeds=list(SEEDS),
                  split_seed=111, fixed_train50_support5_test201=True,
                  batch_size=4, epochs=120, precision='BF16 training; FP32 evaluation',
                  note='Sample standard deviation uses ddof=1; no checkpoint or seed selection by test metrics',
                  cells=cells, aggregate=aggregate, paired_differences=paired)
    write(OUT/'summary.json',result)
    return result


def wait_gpu(job):
    while True:
        line = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'],text=True).splitlines()[0]
        memory, utilization = [int(value.strip()) for value in line.split(',')]
        if memory < 1800 and utilization < 20:
            return
        write(OUT/'queue_status.json',dict(status='waiting_for_gpu', job=job,
                                          gpu_memory_MiB=memory,gpu_utilization_percent=utilization))
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-only',action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    initial = summarize()
    if args.report_only:
        print(json.dumps(initial),flush=True)
        return
    jobs = [(seed,method) for seed in SEEDS for method in MODES if verify(seed,method) is None]
    plan = dict(seeds=list(SEEDS), methods=MODES, split_seed=111, batch_size=4, epochs=120,
                initially_verified=initial['completed_cells'],
                missing_jobs=[dict(seed=seed,method=method) for seed,method in jobs],
                canonical_architecture_unchanged=True, preserve_existing_results=True)
    write(OUT/'plan.json',plan)
    print(f"Verified {initial['completed_cells']}/12 cells; {len(jobs)} runs remain",flush=True)
    for index,(seed,method) in enumerate(jobs):
        if verify(seed,method):
            continue
        for smoke in (True,False):
            job = dict(seed=seed,method=method,phase='preflight' if smoke else 'training',
                       job_index=index+1,jobs_total=len(jobs))
            wait_gpu(job)
            write(OUT/'queue_status.json',dict(status='preflight' if smoke else 'running',**job))
            worker = ROOT/f'scripts/run_tiny_ablation_{method}.py'
            command = [sys.executable,'-u',str(worker),'--seed',str(seed)]
            if smoke:
                command.append('--smoke')
            log_dir = OUT/'logs'
            log_dir.mkdir(exist_ok=True)
            name = f'{seed}_{method}_{"preflight" if smoke else "train"}'
            with (log_dir/(name+'.stdout.log')).open('a',encoding='utf-8') as stdout, \
                 (log_dir/(name+'.stderr.log')).open('a',encoding='utf-8') as stderr:
                code = subprocess.call(command,cwd=ROOT,stdout=stdout,stderr=stderr)
            if code:
                write(OUT/'queue_status.json',dict(status='failed',exit_code=code,**job,
                    stderr=(log_dir/(name+'.stderr.log')).relative_to(ROOT).as_posix()))
                raise SystemExit(code)
            print(f'Completed {method} seed{seed} {job["phase"]}',flush=True)
        assert verify(seed,method)
        current = summarize()
        print(f"Verified {current['completed_cells']}/12 cells",flush=True)
    final = summarize()
    assert final['completed_cells']==12
    write(OUT/'queue_status.json',dict(status='complete',completed_cells=12,seeds=list(SEEDS)))
    print(json.dumps(final['aggregate']),flush=True)


if __name__=='__main__':
    main()
