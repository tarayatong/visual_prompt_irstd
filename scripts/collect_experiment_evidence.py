"""Build paper-facing evidence from saved per-image counts, without inference."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs/studies/objectness_mask'
OUT = ROOT / 'experiments/evidence'
SCORES = ('IoU', 'nIoU', 'F1', 'Pd', 'Fa')


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows, system=None):
    def total(key):
        return sum(float(r[key]) for r in rows)
    if system is not None:
        tp = total(system + '_inter')
        fp = total(system + '_pred_pixels') - tp
        fn = total('gt_pixels') - tp
        pixels, targets = total('image_pixels'), total('target_count')
        detected = total(system + '_detected')
        niou = mean(float(r[system + '_iou']) for r in rows)
    else:
        tp, fp, fn = (total(k) for k in ('tp', 'fp', 'fn'))
        pixels, targets, detected = (total(k) for k in ('pixels', 'targets', 'detected'))
        niou = mean(float(r['image_iou']) for r in rows)
    assert pixels > 0 and targets > 0 and min(tp, fp, fn) >= 0
    return dict(IoU=100*tp/(tp+fp+fn), nIoU=100*niou,
                F1=200*tp/(2*tp+fp+fn), Pd=100*detected/targets,
                Fa=1e6*fp/pixels)


def check(actual, expected):
    for key in SCORES:
        if not math.isclose(actual[key], float(expected[key]), rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError(f'{key}: counts={actual[key]}, summary={expected[key]}')


def table(rows, columns):
    def value(row, key):
        v = row[key]
        return f'{v:.3f}' if isinstance(v, float) else str(v)
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
                      '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
                     ['| ' + ' | '.join(value(r, k) for k in columns) + ' |' for r in rows])


def main():
    evidence, audits = [], []
    groups = [
        ('global_ablation_test201', 'system_summary.csv', 'image_metrics.csv',
         dict(IoU='global_iou_percent', nIoU='mean_image_iou_percent', F1='pixel_f1_percent', Pd='pd_percent', Fa='fa_per_million')),
        ('mask_without_objectness_20261027', 'summary.csv', 'image_metrics.csv', None),
        ('mask_center_ablation_20261026', 'summary.csv', 'image_metrics.csv', None),
        ('loose_objectness_oof_20261027', 'test_summary.csv', 'test_image_metrics.csv', None),
    ]
    for group, summary_name, detail_name, mapping in groups:
        folder = STUDY / group / 'metrics'
        detail_path = folder / detail_name
        detail = read(detail_path)
        assert len(detail) == len({r['sample_id'] for r in detail}) == 201
        mapping = mapping or dict(IoU='miou_percent', nIoU='niou_percent', F1='f1_percent', Pd='pd_percent', Fa='fa_per_million')
        for row in read(folder / summary_name):
            if row.get('group', 'all') != 'all':
                continue
            actual = metrics(detail, row['system'])
            check(actual, {key: row[field] for key, field in mapping.items()})
            evidence.append(dict(experiment=group, system=row['system'], images=201,
                                 **actual, source=str(detail_path.relative_to(ROOT)).replace('\\', '/')))
        audits.append(dict(source=str(detail_path.relative_to(ROOT)).replace('\\', '/'),
                           sha256=hashlib.sha256(detail_path.read_bytes()).hexdigest(),
                           images=201, check='all summary rows agree with full-image counts'))
    save(OUT / 'module_results.csv', evidence)

    folds = read(STUDY / 'phase_b_mask_head/metrics/fold_summary.csv')
    mask_cv = []
    for variant in ('m0_fixed', 'm1_direct', 'm2_residual'):
        values = [float(r['mean_instance_iou_percent']) for r in folds if r['variant'] == variant]
        assert len(values) == 5
        mask_cv.append(dict(variant=variant, folds=5, mean_instance_IoU=mean(values), std_across_folds=pstdev(values)))
    save(OUT / 'mask_adaptation_cv.csv', mask_cv)

    mobile_root = ROOT / 'outputs/runs/mobile/weighted_negative_directions'
    mobile = []
    for row in read(mobile_root / 'summary.csv'):
        detail_path = mobile_root / (row['method']+'_detail.csv')
        detail = read(detail_path)
        assert len(detail) == len({r['sample_id'] for r in detail}) == 201
        tp = sum(float(r['inter_pixels']) for r in detail)
        pp = sum(float(r['pred_pixels']) for r in detail)
        gp = sum(float(r['gt_pixels']) for r in detail)
        actual = dict(IoU=100*tp/(pp+gp-tp), nIoU=100*mean(float(r['iou']) for r in detail), F1=200*tp/(pp+gp))
        for key, field in dict(IoU='mIoU_percent', nIoU='mean_image_IoU_percent', F1='F1_percent').items():
            assert math.isclose(actual[key], float(row[field]), abs_tol=1e-7)
        mobile.append(dict(method=row['method'], backbone='MobileSAM', images=201, **actual,
                           source=str(detail_path.relative_to(ROOT)).replace('\\', '/')))
        audits.append(dict(source=str(detail_path.relative_to(ROOT)).replace('\\', '/'), sha256=hashlib.sha256(detail_path.read_bytes()).hexdigest(),
                           images=201, check='IoU/nIoU/F1 verified; legacy Pd/Fa deliberately excluded'))
    save(OUT / 'mobile_memory_reference.csv', mobile)

    # Preserve all competitors, including low scores and unavailable methods.
    comparison_root = ROOT / 'outputs/comparison_50shot_5support'
    comparison, refreshed_rows = [], []
    corrected_fa = 0
    for row in read(comparison_root / 'full16/comparison_summary.csv'):
        dataset, method = row['dataset'], row['method']
        folder = comparison_root / ('' if method in ('ours_frozen', 'ours_joint', 'HERA') else 'full16') / dataset / method
        summary_path = folder / 'metrics/summary.json'
        detail_path = folder / 'metrics/per_image.csv'
        result = dict(dataset=dataset, method=method, annotation_budget=row['actual_annotation_budget'],
                      status=row['status'], epochs=row['training_epochs'], schedule_status=row['schedule_status'],
                      review='未完成，不进入结论',
                      IoU='', nIoU='', F1='', Pd='', Fa='', source=str(summary_path.relative_to(ROOT)).replace('\\', '/'))
        if row['status'] == 'complete':
            summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
            detail = read(detail_path)
            manifest = json.loads((comparison_root / dataset / 'protocol/manifest.json').read_text(encoding='utf-8-sig'))
            assert summary['complete_test'] and summary['manifest_sha256'] == manifest['manifest_sha256']
            assert len(detail) == len(manifest['test_ids']) == len({r['sample_id'] for r in detail})
            assert {r['sample_id'] for r in detail} == set(manifest['test_ids'])
            actual = metrics(detail)
            check(actual, {key: summary[field] for key, field in dict(IoU='IoU_percent', nIoU='nIoU_percent', F1='F1_percent', Pd='Pd_percent', Fa='Fa_per_million').items()})
            if not math.isclose(float(row['Fa_per_million']), actual['Fa'], abs_tol=1e-7):
                corrected_fa += 1
            result.update(actual)
            result['review'] = '单次设置参考，正式比较待协议对齐'
            if actual['IoU'] < 1 or actual['Pd'] == 0:
                result['review'] = '低值/退化输出，复现待核验，不据此证明优势'
            elif row['schedule_status'] and 'cap_' in row['schedule_status']:
                result['review'] = '训练日程待复核，不作已收敛比较'
            row.update({field: actual[key] for key, field in dict(IoU='IoU_percent', nIoU='nIoU_percent', F1='F1_percent', Pd='Pd_percent', Fa='Fa_per_million').items()})
            audits.append(dict(source=str(detail_path.relative_to(ROOT)).replace('\\', '/'),
                               sha256=hashlib.sha256(detail_path.read_bytes()).hexdigest(),
                               images=len(detail), check='counts, summary, unique IDs and split agree'))
        comparison.append(result)
        refreshed_rows.append(row)
    fss_path = comparison_root / 'fss50_v2/comparison_summary.json'
    fss_rows = json.loads(fss_path.read_text(encoding='utf-8-sig'))
    for row in fss_rows:
        folder = comparison_root / ('full16' if row['labels'] == 5 else 'fss50_v2') / row['dataset'] / row['method']
        detail_path, summary_path = folder / 'metrics/per_image.csv', folder / 'metrics/summary.json'
        if not row.get('complete_test') or not detail_path.exists():
            continue
        actual = metrics(read(detail_path))
        row.update({field: actual[key] for key, field in dict(IoU='IoU_percent', nIoU='nIoU_percent', F1='F1_percent', Pd='Pd_percent', Fa='Fa_per_million').items()})
        if row['labels'] != 50:
            continue
        summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
        manifest = json.loads((comparison_root / row['dataset'] / 'protocol/manifest.json').read_text(encoding='utf-8-sig'))
        detail = read(detail_path)
        assert summary['complete_test'] and summary['manifest_sha256'] == manifest['manifest_sha256']
        assert len(detail) == len(manifest['test_ids']) == len({r['sample_id'] for r in detail})
        assert {r['sample_id'] for r in detail} == set(manifest['test_ids'])
        check(actual, {key: summary[field] for key, field in dict(IoU='IoU_percent', nIoU='nIoU_percent', F1='F1_percent', Pd='Pd_percent', Fa='Fa_per_million').items()})
        comparison.append(dict(dataset=row['dataset'], method=row['method']+'_50label', annotation_budget=50,
                               status='complete', epochs=summary.get('epochs', ''), schedule_status=summary.get('schedule_status', ''),
                               review='50张适配已完成；单次设置，非纯样本数消融',
                               **actual, source=str(summary_path.relative_to(ROOT)).replace('\\', '/')))
        audits.append(dict(source=str(detail_path.relative_to(ROOT)).replace('\\', '/'), sha256=hashlib.sha256(detail_path.read_bytes()).hexdigest(),
                           images=len(detail), check='50-label adaptation counts, summary and split agree'))
    fss_path.write_text(json.dumps(fss_rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (comparison_root / 'fss50_v2/comparison_results.md').write_text('# 50张监督协议修正结果\n\n当前IFA三数据集完成；MPA暂停，不能当作完成。5→50还改变了目标域微调和支持推理，不能当作纯样本数消融。完整已核验表见 experiments/evidence/comparison_single_dataset.csv。\n', encoding='utf-8')
    save(OUT / 'comparison_single_dataset.csv', comparison)
    # Repair the stale public aggregate too, so there is one metric definition.
    save(comparison_root / 'full16/comparison_summary.csv', refreshed_rows)
    (comparison_root / 'full16/comparison_summary.json').write_text(json.dumps(refreshed_rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    comparison_markdown = '# 三数据集单域少样本对比\n\n从逐图统计核验，Fa统一为全图背景FP/M；训练日程及复现偏差见同目录 protocol_deviations.md。未完成项留空，不记为零。数值完整不代表复现有效，逐行review限制必须保留。\n\n' + table(comparison, ['dataset', 'method', 'annotation_budget', 'status', 'review', *SCORES]) + '\n'
    (comparison_root / 'full16/comparison_results.md').write_text(comparison_markdown, encoding='utf-8')
    base_path = comparison_root / 'comparison_summary.csv'
    base_rows = read(base_path)
    for row in base_rows:
        path = comparison_root / row['dataset'] / row['method'] / 'metrics/per_image.csv'
        if row['status'] == 'complete' and path.exists():
            actual = metrics(read(path))
            row.update({field: actual[key] for key, field in dict(IoU='IoU_percent', nIoU='nIoU_percent', F1='F1_percent', Pd='Pd_percent', Fa='Fa_per_million').items()})
    save(base_path, base_rows)
    (comparison_root / 'comparison_summary.json').write_text(json.dumps(base_rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (comparison_root / 'comparison_results.md').write_text('# 首批结果\n\n最新完整核验表见 [全部方法](full16/comparison_results.md)。首批HDNet为800轮，完整表采用后续已记录训练日程，两批不可混用。\n', encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'verification.json').write_text(json.dumps(dict(
        date='2026-10-27', checks=audits, module_rows=len(evidence), comparison_rows=len(comparison),
        stale_aggregate_fa_rows=corrected_fa,
        limitation='Single split/seed; CV standard deviation is across folds, not independent seeds. No new inference.'), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    def select(group, systems):
        return [dict(system=r['system'], **{k: r[k] for k in SCORES}) for r in evidence
                if r['experiment'] == group and r['system'] in systems]
    lines = ['# 历史可用结果与观点对应', '',
        '本页保留早期配对实验与统计快照。当前正式结果、61.08后的研究分支和待做实验统一见[实验总览](experiments.md)；正式对比以[三数据集主表](comparison_three_datasets.md)为准。', '',
        '本页由 `python -m scripts.collect_experiment_evidence` 从逐图计数重算。IoU/nIoU/F1/Pd为%，Fa为全图背景FP/M；所有数值保留不利对照。完整来源、SHA256与校验见 experiments/evidence/verification.json。', '',
        '## A1：负语义矫正已有正向证据', '',
        table(select('global_ablation_test201', ['g0_sam2_positive_only', 'g1_negative_correction_top1']), ['system', *SCORES]), '',
        '同一201图测试、Top1和规则Mask，IoU由23.84升至35.13。支持“这一支持集负语义矫正有效”；尚未隔离正/负记忆容量、采样、两类负语义及随机负样本，不能据此证明完整记忆机制或跨域泛化。', '',
        '同一观点的MobileSAM补充对照（独立骨干，不能与SAM2直接合并成增量消融）：', '',
        table(mobile, ['method', 'IoU', 'nIoU', 'F1']), '',
        '仅正目标原型IoU12.64，稳定高频/高亮负语义40.26，支持侧加权优化40.85。加权优化幅度小且nIoU下降，保留为备选；旧Pd/Fa未在此核验，不进入正式结果。骨干变化不是跨数据集泛化证据。', '',
        '## A2：少量参数的Mask适配已有证据', '',
        table(mask_cv, ['variant', 'folds', 'mean_instance_IoU', 'std_across_folds']), '',
        '固定语义向量m0对照与直接预测m1、残差矢量m2：训练内五折、GT中心局部实例IoU；非全图IoU，标准差是折间差异。m2优于固定向量；m1与m2接近，不能声称残差结构显著更优。', '',
        '## A3：候选筛选与Mask是互补的', '',
        table(select('global_ablation_test201', ['g1_negative_correction_top5', 'g2_mask_standard_top5', 'g3_objectness_rule', 'g4s_separate_train_joint_infer', 'g4j_joint_train_joint_infer', 'g2u_mask_oracle_gt']), ['system', *SCORES]), '',
        '旧版统一链路：仅候选+规则Mask、加学习Mask、仅加筛选、两者串联及warm-start联合微调。GT中心73.13仅是诊断参考。支持两模块组成链路的价值，不与新拒绝Mask/残差Obj混成一个单因素消融；标签预算也由仅支持变为50张，不能归因为结构的唯一贡献。', '',
        '## A4：拒绝监督有效，但存在分割代价', '',
        table(select('mask_without_objectness_20261027', ['positive_only_no_obj_top5', 'with_rejection_no_obj_top5', 'positive_only_residual_obj09_nms16', 'with_rejection_residual_obj09_nms16', 'positive_only_gt_center', 'with_rejection_gt_center']), ['system', *SCORES]), '',
        '相同正样本与更新次数，新增错误候选空Mask监督，无Obj时IoU21.81→56.63、Fa835.46→110.99；GT中心IoU72.51→66.20。正例对照必须保留，不能删除低分对照后只报改善。', '',
        '## A5：工作点选择影响整条链路', '',
        table(select('loose_objectness_oof_20261027', ['fixed09', 'OOF_loose']), ['system', *SCORES]), '',
        '仅Obj阈值变化：0.9与训练内OOF召回规则锁定的0.360282719。Mask/权重不变；支持召回—虚警取舍，不支持显著性或跨域泛化。', '',
        '## 暂留：中心Mask卷积/边缘结构', '',
        table(select('mask_center_ablation_20261026', ['plain_gt_center', 'conv_gt_center', 'edge3_gt_center', 'plain_fixed_old_obj', 'edge3_fixed_old_obj']), ['system', *SCORES]), '',
        '中心IoU无一致改善，串联edge3比本轮plain高1.53个百分点；作为待复验备选保留，不列核心贡献。无增强Obj残差及F4+CNN拼接也仅是单种子候选，来源见保留登记。', '',
        '## C1：三个数据集的单域少样本对比', '',
        '完整方法表见 experiments/evidence/comparison_single_dataset.csv。下面展示本方法、HDNet及已完成的IFA正式50张适配；不挑选删除其余方法。', '',
        table([r for r in comparison if r['method'] in ('ours_frozen', 'ours_joint', 'HDNet', 'IFA_50label')], ['dataset', 'method', 'annotation_budget', 'review', *SCORES]), '',
        '各数据集独立训练/适配，属于单域结果。ours_frozen实际5张标注，ours_joint和HDNet实际50张；不当作等监督消融。ours_joint不是当前59.26版模型。本方法未在所有数据集优于HDNet。外部方法的训练日程和适配偏差仍需复现审查，低分不自动构成优越性证明。', '',
        '特别标记：full16的HDNet/NUDT-SIRST IoU约0.004%，属于明显退化输出，不能支持优越性结论。首批800轮同方法IoU约70.89%，属于不同日程的历史参考，也不能挑选替换500轮最终结果后隐藏协议差异。先保留两批记录供修复/核查，正式对比状态待定。', '',
        'IFA正式50张适配已完成三个数据集，单独标为IFA_50label。MPA的50张版本已暂停，保留可续训状态，尚无正式完整结果。5→50同时改变训练及推理方式，不作纯标注数量消融。', '',
        '## 尚无证据', '',
        '跨数据集迁移矩阵、完整记忆采样机制、容量效率、支持集稳定性、e0双用途的独立消融，以及第二模块新增结构：均待做，设置待定。', '']
    (ROOT / 'docs/experiment_results.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(module_rows=len(evidence), comparison_rows=len(comparison),
                          checked_sources=len(audits), stale_aggregate_fa_rows=corrected_fa)))


if __name__ == '__main__':
    main()
