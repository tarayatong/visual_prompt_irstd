"""Sequential radial experiment queue with audited stage gates and persistent state."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from scripts.run_global_ablation_201 import write_json, write_csv
from scripts.run_objectness_positive_jitter import load, read_csv
from scripts.audit_radial_content_wave import counter_summary
from visual_prompt.core.process_identity import process_identity

ROOT = Path('outputs/studies/objectness_mask')
HOME = ROOT/'radial_autonomous_search_20261029'
AMPLITUDE = ROOT/'radial_independent_search_20261029'
CONTENT = ROOT/'radial_content_search_20261029'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def alive(record):
    try:
        actual = process_identity(record['pid'])
        return actual is not None and actual['create_time'] == record['create_time'] and actual['executable'].lower() == record['executable'].lower()
    except KeyError:
        return False


def base_catalog():
    out = []
    for root, names in [(AMPLITUDE, ['A0', 'A1', 'A2', 'A3', 'A4']), (CONTENT, ['B1', 'B2', 'B3'])]:
        for head in ('obj', 'mask'):
            for name in names:
                out.append(dict(id=f'historical_{head}_{name}', head=head, seed=111, variant=name,
                                summary=str(root/head/name/'summary.json'), output=str(root),
                                components='all' if name.startswith('A') else {'B1': 'contrast', 'B2': 'contrast_first', 'B3': 'contrast_first_second'}[name],
                                geometry={}, architecture=None, status='complete', imported=True))
    return out


def task_metric(candidate):
    return read(candidate['summary'])['task_metrics']


def rank(candidate, baseline):
    t = task_metric(candidate)
    if candidate['head'] == 'obj':
        return (t['recall_at_bg_0.25'], t['recall_at_bg_0.5'], t['loose']['precision'], -read(candidate['summary'])['parameters'])
    b = task_metric(baseline)
    image = t['image_summary'][0]; original = b['image_summary'][0]
    eligible = (t['negative_empty_percent'] >= b['negative_empty_percent']-2 and
                t['positive_empty_percent'] <= b['positive_empty_percent']+2 and
                image['miou_percent'] >= original['miou_percent']-.5 and
                image['pd_percent'] >= original['pd_percent']-100/70 and
                image['fa_per_million'] <= max(1., original['fa_per_million'])*1.1)
    return (int(eligible), t['positive_ROI_global_IoU'], image['miou_percent'], t['negative_empty_percent'], -t['negative_FP_mean'])


def best(catalog, head, radial_only=False, projected_only=False):
    baseline = next(c for c in catalog if c['id'] == f'historical_{head}_A0')
    pool = [c for c in catalog if c['head'] == head and c['seed'] == 111 and c['status'] == 'complete']
    if radial_only:
        pool = [c for c in pool if c['variant'] not in ('A0', 'A4')]
    if projected_only:
        pool = [c for c in pool if (c.get('architecture') or {}).get('projection') == 'nonlinear']
    return max(pool, key=lambda c: rank(c, baseline))


def add_job(state, stage, label, head, variant, seed=111, parent=None, geometry=None, architecture=None, components=None):
    job_id = f'{stage}_{head}_{label}_s{seed}'
    if any(j['id'] == job_id for j in state['jobs']):
        return
    geometry = geometry if geometry is not None else (parent or {}).get('geometry', {})
    architecture = architecture if architecture is not None else (parent or {}).get('architecture')
    components = components or (parent or {}).get('components', 'all' if head == 'obj' else 'contrast')
    output = HOME/'runs'/job_id
    state['jobs'].append(dict(id=job_id, stage=stage, label=label, head=head, variant=variant, seed=seed,
                              parent_id=(parent or {}).get('id'), geometry=geometry, architecture=architecture,
                              components=components, output=str(output), summary=str(output/head/variant/'summary.json'), status='pending'))


def populate(state, stage):
    catalog = state['catalog']
    if stage == 'confirmation':
        for seed in (222, 333):
            for variant in ('A0', 'B1'):
                add_job(state, stage, variant, 'mask', variant, seed,
                        components='all' if variant == 'A0' else 'contrast')
    elif stage == 'geometry':
        for head in ('obj', 'mask'):
            parent = next(c for c in catalog if c['id'] == f'historical_{head}_{"A1" if head == "obj" else "B1"}')
            for variant, geometry in [('C1', dict(radii=[1,2,3,4,6,8,12,16])), ('C2', dict(opposite_pairs=True))]:
                add_job(state, stage, variant, head, variant, parent=parent, geometry=geometry)
            if head == 'obj':
                add_job(state, stage, 'C3', head, 'C3', parent=parent, geometry=dict(smooth_sigma=.6))
    elif stage == 'readout':
        for head in ('obj', 'mask'):
            parent = best(catalog, head, radial_only=True)
            for mode, name in [('nonlinear', 'D1'), ('linear', 'D1_linear')]:
                add_job(state, stage, name, head, name, parent=parent,
                        architecture=dict(projection=mode, projected_channels=32))
            if head == 'mask':
                for mode, name in [('candidate', 'D2'), ('zero_control', 'D2_zero')]:
                    add_job(state, stage, name, head, name, parent=parent,
                            architecture=dict(projection='nonlinear', projected_channels=32, condition=mode))
    elif stage == 'supervision':
        for head in ('obj', 'mask'):
            projected = best(catalog, head, radial_only=True, projected_only=True)
            if head == 'obj':
                spec = dict(projected['architecture'], auxiliary='radial_classification', aux_weight=.1)
                add_job(state, stage, 'E1', head, 'E1', parent=projected, architecture=spec)
            else:
                main = best(catalog, head, radial_only=True)
                for name, kind, parent in [('E2', 'radial_mask', projected), ('E3', 'hard_negative', main),
                                           ('E4_1', 'edge', projected), ('E4_3', 'edge', projected)]:
                    spec = dict(parent.get('architecture') or {}, auxiliary=kind, aux_weight=.1)
                    if kind == 'edge':
                        spec['thickness'] = int(name[-1])
                    add_job(state, stage, name, head, name, parent=parent, architecture=spec)
    elif stage == 'final_confirmation':
        state['provisional_best'] = {}
        for head in ('obj', 'mask'):
            winner = best(catalog, head)
            state['provisional_best'][head] = winner['id']
            # Baseline always participates; repeats of completed identical candidates are reused below.
            baseline = next(c for c in catalog if c['id'] == f'historical_{head}_A0')
            candidates = [baseline]+([winner] if winner['id'] != baseline['id'] else [])
            for parent in candidates:
                for seed in (222, 333):
                    add_job(state, stage, parent['id'], head, parent['variant'], seed, parent=parent)
    else:
        raise ValueError(stage)


def persist(state):
    state['updated_at'] = now()
    path = HOME/'queue.json'; temporary = HOME/'queue.tmp.json'
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def command(job):
    args = [sys.executable, '-m', 'scripts.run_radial_independent_search', '--variants', job['variant'], '--heads', job['head'],
            '--seed', str(job['seed']), '--output_dir', job['output'], '--amplitude-parent', str(AMPLITUDE), '--components', job['components']]
    geometry = job.get('geometry', {})
    if geometry:
        signature = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()[:12]
        args.extend(['--cache-root', str(HOME/'cache'/signature)])
        if geometry.get('radii'):
            args.extend(['--radii', ','.join(map(str, geometry['radii']))])
        if geometry.get('opposite_pairs'):
            args.append('--opposite-pairs')
        if geometry.get('smooth_sigma'):
            args.extend(['--smooth-sigma', str(geometry['smooth_sigma'])])
    else:
        args.extend(['--cache-root', str(AMPLITUDE)])
    if job.get('architecture') is not None:
        path = HOME/'job_configs'/f'{job["id"]}.json'; write_json(path, job['architecture'])
        args.extend(['--architecture-json', str(path)])
    return args


def audit_job(job):
    d = read(job['summary']); output = Path(job['output'])
    v = read(output/'validation.json')
    assert v['models'] == 5 and v['official_test_read'] is False and v['joint_inference'] is False
    assert d['OOF_images'] == 45
    all_ids = []
    for fold in range(5):
        ck = load(output/job['head']/job['variant']/'artifacts'/f'fold{fold}.pt')
        assert ck['seed'] == job['seed']
        assert not set(ck['sample_ids']) & set(ck['validation_ids'])
        all_ids.extend(ck['validation_ids'])
        assert np.isfinite(np.asarray(ck['radial_mean'])).all() and np.isfinite(np.asarray(ck['radial_std'])).all()
        ref = load(AMPLITUDE/job['head']/'A0/artifacts'/f'fold{fold}.pt')
        hash_key = 'sampling_sha256' if job['head'] == 'mask' else 'schedule_sha256'
        assert ck[hash_key] == ref[hash_key]
    assert len(all_ids) == len(set(all_ids)) == 45
    if job['head'] == 'mask':
        rows = read_csv(output/'mask'/job['variant']/'metrics/OOF_images.csv')
        rois = read_csv(output/'mask'/job['variant']/'metrics/OOF_ROIs.csv')
        assert len(rows) == 45 and sum(int(r['candidate']) for r in rois) == 225
        for kind in d['task_metrics']['image_summary']:
            values = counter_summary(rows, kind['system'])
            for key, actual in values.items():
                assert abs(actual-kind[key]) < 1e-8
    else:
        rows = read_csv(output/'obj'/job['variant']/'metrics/OOF_predictions.csv')
        assert len(rows) == 225
        t = d['task_metrics']['fixed09']; assert abs(t['recall']-t['tp']/70) < 1e-12
    audit = dict(passed=True, time=now(), job_id=job['id'], folds=5, OOF_images=45, official_test_used=False,
                 paired_sampling=True, independent_task=True, raw_counter_check=True)
    write_json(output/'stage_audit.json', audit)
    return audit


def stage_review(state, stage):
    jobs = [j for j in state['jobs'] if j['stage'] == stage]
    for job in jobs:
        assert job['status'] == 'complete' and read(Path(job['output'])/'stage_audit.json')['passed']
    lines = []
    for job in jobs:
        d = read(job['summary']); t = d['task_metrics']; z = d['diagnostics_fold_mean']
        baseline = next(c for c in state['catalog'] if c['id'] == f'historical_{job["head"]}_A0')
        parent = next((c for c in state['catalog'] if c['id'] == job.get('parent_id')), baseline)
        b = task_metric(baseline); p = task_metric(parent)
        primary = t['recall_at_bg_0.25']*100 if job['head']=='obj' else t['positive_ROI_global_IoU']
        secondary = t['recall_at_bg_0.5']*100 if job['head']=='obj' else t['negative_empty_percent']
        baseline_primary = b['recall_at_bg_0.25']*100 if job['head']=='obj' else b['positive_ROI_global_IoU']
        parent_primary = p['recall_at_bg_0.25']*100 if job['head']=='obj' else p['positive_ROI_global_IoU']
        lines.append(dict(id=job['id'], head=job['head'], seed=job['seed'], J=z['Fisher_J'], Delta_m=z['margin_gain'],
                          primary=primary, secondary=secondary, delta_vs_A0=primary-baseline_primary,
                          delta_vs_parent=primary-parent_primary))
        d['Chinese_conclusion'] = [f'独立主指标={primary:.2f}，相对A0变化{primary-baseline_primary:+.2f}个百分点，相对父设置变化{primary-parent_primary:+.2f}个百分点。',
                                  f'独立次指标={secondary:.2f}；J与Delta_m详见中间量，不用中间量替代任务指标，跨种子比较另行记录。',
                                  '完成五折与配对采样/计数核验；只保留OOF证据，不读取201测试图，不进行联合推理搜索。']
        write_json(Path(job['summary']), d)
    selected = {head: best(state['catalog'], head)['id'] for head in ('obj', 'mask')}
    value = dict(stage=stage, completed_jobs=len(jobs), completed_models=5*len(jobs), independently_selected=selected,
                 results=lines, official_test_used=False, joint_inference=False,
                 Chinese_conclusion=['本阶段各配置五折与采样/计数审计通过，独立指标及J、Delta_m见结果表。',
                                     '选择仅依据OOF任务指标；中间量用于解释，原基线始终保留，单种子优胜仍需复验。',
                                     '本阶段不读取201张测试图，不运行联合推理；下一阶段使用分别选定的径向父设置。'])
    value['reused_jobs'] = sum(bool(j.get('reused_from')) for j in jobs)
    value['newly_fitted_models'] = 5*(len(jobs)-value['reused_jobs'])
    if stage == 'confirmation':
        differences = []
        for seed in (111, 222, 333):
            baseline = next(c for c in state['catalog'] if c['head']=='mask' and c['variant']=='A0' and c['seed']==seed)
            trial = next(c for c in state['catalog'] if c['head']=='mask' and c['variant']=='B1' and c['seed']==seed)
            a, b = task_metric(baseline), task_metric(trial)
            differences.append(dict(seed=seed, positive_ROI_IoU_delta=b['positive_ROI_global_IoU']-a['positive_ROI_global_IoU'],
                                    negative_empty_delta=b['negative_empty_percent']-a['negative_empty_percent'],
                                    Top5_mIoU_delta=b['image_summary'][0]['miou_percent']-a['image_summary'][0]['miou_percent']))
        value['paired_mask_seed_differences'] = differences
        value['B1_confirmed_on_all_three_seeds'] = all(r['Top5_mIoU_delta'] > 0 and r['positive_ROI_IoU_delta'] >= 0 for r in differences)
        value['Chinese_conclusion'] = [f'Mask B1与A0完成3种子配对，Top5 mIoU平均变化{np.mean([r["Top5_mIoU_delta"] for r in differences]):+.2f}个百分点。',
                                      f'3种子均同时改善局部IoU和全图IoU：{value["B1_confirmed_on_all_three_seeds"]}；未确认时仍可有界探索几何原因，但不将B1认定为稳定新基线。',
                                      '下一步只推进独立几何消融，最终选择保留A0与种子方差控制。']
    write_json(HOME/'reviews'/f'{stage}.json', value); write_csv(HOME/'reviews'/f'{stage}.csv', lines)
    state['completed_stages'].append(stage); persist(state)
    print(json.dumps(dict(event='stage_complete', stage=stage, selected=selected), ensure_ascii=False), flush=True)


def main():
    HOME.mkdir(parents=True, exist_ok=True); lock = HOME/'controller.lock'
    if lock.exists():
        old = read(lock)
        if alive(old):
            raise RuntimeError(f'Existing live controller {old["pid"]}; refusing duplicate')
        lock.unlink()
    identity = dict(process_identity(os.getpid()), started_at=now())
    with lock.open('x', encoding='utf-8') as f:
        json.dump(identity, f)
    write_json(HOME/'controller_identity.json', identity)
    state_path = HOME/'queue.json'
    state = read(state_path) if state_path.exists() else dict(jobs=[], catalog=base_catalog(), completed_stages=[],
                                                           status='running', official_test_access=False, joint_inference=False)
    for job in state['jobs']:
        if job.get('status') == 'running' and job.get('pid'):
            try:
                child = process_identity(job['pid'])
                if child and child['create_time'] == job['process_create_time']:
                    raise RuntimeError(f'Existing live orphan worker {job["pid"]}; inspect/adopt before restarting queue')
            except ProcessLookupError:
                pass
    stages = ['confirmation', 'geometry', 'readout', 'supervision', 'final_confirmation']
    try:
        for stage in stages:
            if stage in state['completed_stages']:
                continue
            state['stage'] = stage; state['status'] = 'running'; populate(state, stage); persist(state)
            for job in [j for j in state['jobs'] if j['stage'] == stage and j['status'] != 'complete']:
                # Reuse a fully audited identical configuration/seed rather than fitting it again.
                reusable = next((c for c in state['catalog'] if c['status']=='complete' and c['head']==job['head'] and
                                 c['seed']==job['seed'] and c['variant']==job['variant'] and c['geometry']==job['geometry'] and
                                 c['components']==job['components'] and c['architecture']==job['architecture']), None)
                if reusable:
                    job.update(output=reusable['output'], summary=reusable['summary'], reused_from=reusable['id'])
                    audit_job(job); job['status']='complete'; persist(state); continue
                job['status'] = 'running'; job['started_at'] = now(); args = command(job)
                state['active_job'] = job['id']; persist(state)
                logs = HOME/'logs'; logs.mkdir(exist_ok=True)
                with (logs/f'{job["id"]}.stdout.log').open('a', encoding='utf-8') as stdout, (logs/f'{job["id"]}.stderr.log').open('a', encoding='utf-8') as stderr:
                    child = subprocess.Popen(args, cwd=Path.cwd(), stdout=stdout, stderr=stderr,
                                             env=dict(os.environ, PYTHONIOENCODING='utf-8'))
                    job['pid'] = child.pid; job['process_create_time'] = process_identity(child.pid)['create_time']; persist(state)
                    while child.poll() is None:
                        state['heartbeat_at'] = now(); persist(state); time.sleep(5)
                    if child.returncode:
                        job['status'] = 'failed'; job['returncode'] = child.returncode
                        state['status'] = 'needs_repair'; persist(state)
                        raise RuntimeError(f'Job {job["id"]} failed; see stderr log; no next-stage selection performed')
                audit_job(job); job['status'] = 'complete'; job['ended_at'] = now()
                state['catalog'].append(dict(job)); state['active_job'] = None; persist(state)
            stage_review(state, stage)
        state['status'] = 'awaiting_final_robust_selection_and_test_implementation'
        state['active_job'] = None; persist(state)
        print('All preregistered independent stages fitted/audited; heartbeat must review multi-seed winners, optional justified combinations, then lock final independent tests.', flush=True)
    except Exception:
        state['error'] = traceback.format_exc(); state['status'] = 'needs_repair'; persist(state)
        raise
    finally:
        if lock.exists() and read(lock).get('pid') == os.getpid():
            lock.unlink()


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args(); main()
