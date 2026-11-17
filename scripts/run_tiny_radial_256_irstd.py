"""Repeat the IRSTD sampling and gate controls at decoder resolution 256."""
import argparse
from pathlib import Path
import subprocess
import sys

import run_tiny_radial_detail_irstd as radial
from run_tiny_ablation_three_seeds import FIELDS, ROOT, read, verify, write
from run_tiny_radial_detail_other_datasets import wait_gpu

OUT = ROOT / 'outputs/tiny_radial_256_irstd_seed113'
SCRIPT = Path(__file__).resolve()
REFERENCE = radial.OUT / 'only_256/combined' / radial.METRICS
CONTROLS = {
    mode: {**radial.CONTROLS[mode], 'active_stages': (1,)}
    for mode in ('neighbor_values', 'fixed_gate_05')
}
ROWS = {
    'dual_scale': ('128＋256', '中心–周围差分', '自适应'),
    'only_128': ('只接 128', '中心–周围差分', '自适应'),
    'only_256': ('**只接 256（基准）**', '中心–周围差分', '自适应'),
    'neighbor_values': ('只接 256', '周围采样值', '自适应'),
    'fixed_gate_05': ('只接 256', '中心–周围差分', '固定 0.5'),
}


def worker(mode, smoke):
    import run_tiny_ablation_combined as runner

    runner.OUT = OUT / mode
    runner.RUN_SEED = radial.SEED
    radial.CONTROLS = CONTROLS
    metadata = radial.settings(mode)
    metadata.update(
        fusion='19 prototype slots -> frozen native attention -> radial residual at 256',
        matched_reference=REFERENCE.relative_to(ROOT).as_posix(),
        baseline_note='Matched IRSTD-1k seed113 256-only difference and adaptive gate reference.',
    )
    runner.worker(argparse.Namespace(
        dataset='IRSTD-1k', variant='combined', smoke=smoke,
        radial_options=CONTROLS[mode], radial_metadata=metadata,
        extra_sources=[SCRIPT, Path(radial.__file__).resolve()],
    ))


def report(results, failures):
    reference_iou = results['only_256']['metrics']['IoU_percent']
    lines = [
        '# IRSTD-1k 径向模块细节消融',
        '',
        '训练种子 113，train50/support5/test201，保留 19-token 记忆和 0.1 残差系数。',
        '统一评估第 120 轮。ΔIoU 为相对「只接 256＋中心–周围差分＋自适应门控」的差值，单位为百分点。',
        '',
        '| 接入尺度 | 采样特征 | 门控 | IoU ↑ | ΔIoU | nIoU ↑ | F1 ↑ | Pd ↑ | Fa ↓ |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    rows = {}
    for mode, labels in ROWS.items():
        result = results.get(mode)
        if result is None:
            status = '待重跑' if mode not in failures else '运行失败'
            values = [status, '—', '—', '—', '—', '—']
            rows[mode] = dict(status=status)
        else:
            metrics = result['metrics']
            delta = metrics['IoU_percent'] - reference_iou
            iou = f"{metrics['IoU_percent']:.2f}"
            values = [f'**{iou}**' if mode == 'only_256' else iou,
                      '0.00' if mode == 'only_256' else f'{delta:+.2f}',
                      *(f'{metrics[key]:.2f}' for key in FIELDS[1:])]
            rows[mode] = dict(metrics=metrics, IoU_delta_pp=delta, summary=result['summary'])
        lines.append('| ' + ' | '.join((*labels, *values)) + ' |')
    temporary = OUT / 'comparison.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(OUT / 'comparison.md')
    write(OUT / 'comparison.json', dict(dataset='IRSTD-1k', seed=radial.SEED,
                                        reference='only_256', rows=rows, failures=failures))


def queue():
    assert verify(radial.SEED, 'combined') is not None
    dual = read(radial.REFERENCE)
    results = {'dual_scale': dict(metrics={key: dual[key] for key in FIELDS},
                                  summary=radial.REFERENCE.relative_to(ROOT).as_posix())}
    for mode in ('only_128', 'only_256'):
        results[mode] = radial.verified(mode, dual)
        assert results[mode] is not None
    reference = read(REFERENCE)
    radial.OUT, radial.CONTROLS = OUT, CONTROLS
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in CONTROLS:
        result = radial.verified(mode, reference)
        if result is not None:
            results[mode] = result
    failures = {}
    status = OUT / 'queue_status.json'
    report(results, failures)
    for mode in CONTROLS:
        if mode in results:
            continue
        for phase in ('preflight', 'training'):
            job = dict(dataset='IRSTD-1k', mode=mode, phase=phase, seed=radial.SEED)
            wait_gpu(status, job)
            log = OUT / f'{mode}_{phase}.log'
            write(status, dict(status=phase, **job, log=log.name))
            command = [sys.executable, '-u', str(SCRIPT), '--mode', mode]
            if phase == 'preflight':
                command.append('--smoke')
            with log.open('a', encoding='utf-8') as stream:
                code = subprocess.call(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if code:
                failures[mode] = dict(phase=phase, exit_code=code, log=log.name)
                break
        else:
            result = radial.verified(mode, reference)
            assert result is not None
            results[mode] = result
        report(results, failures)
    write(status, dict(status='partial_failure' if failures else 'complete',
                       completed_modes=[mode for mode in CONTROLS if mode in results],
                       failures=failures))
    if failures:
        raise SystemExit(1)


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
