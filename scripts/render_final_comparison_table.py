"""Render the comparison including the supplemental MPCNet comparison without launching any experiments."""
import csv
import json
from pathlib import Path
from scripts.comparison_method_metadata import METADATA, NEW_METHODS, SUPPLEMENTAL_METHODS, metadata
from scripts.collect_experiment_evidence import ROOT, read, metrics, check
from scripts.report_recent_a_comparison import verified_summary, METRICS
from scripts.run_recent_a_comparison import RUN, manifest_for
from scripts.comparison_adaptation_table import apply_replacements

DATASETS=('SIRST','IRSTD-1k','NUDT-SIRST')
FIELDS=('IoU','nIoU','F1','Pd','Fa')


def render(legacy, lookup, overrides, comparison_overrides, corrected_ready):
    before={(r['dataset'],r['method']):(r['values'],r['source']) for r in legacy}
    rows=[]
    for original in legacy:
        row=dict(original)
        row['method']='IFA' if row['method']=='IFA_50label' else row['method']
        row['origin']='preserved_existing_result'
        rows.append(row)
    for method in ('Matcher','HERA','PI-CLIP','DSV-LFS'):
        for dataset in DATASETS:
            old=lookup.get((dataset,method),{})
            row=dict(method=method,dataset=dataset,**metadata(method),labels=old.get('annotation_budget'),
                epochs=old.get('epochs'),values=None,status='pending_weights',origin='preserved_existing_result',
                source=old.get('source'),review=old.get('review',''))
            if old.get('status')=='complete':
                path=ROOT/old['source'];detail=read(path.parent/'per_image.csv')
                value=metrics(detail);check(value,old)
                assert len(detail)==len({x['sample_id'] for x in detail})==len(manifest_for(dataset)['test_ids'])
                assert {x['sample_id'] for x in detail}==set(manifest_for(dataset)['test_ids'])
                row.update(values=value,status='replication_unverified')
            rows.append(row)
    for method in NEW_METHODS + SUPPLEMENTAL_METHODS:
        for dataset in DATASETS:
            result_root=ROOT/'outputs/comparison_mpcnet_20261105' if method=='MPCNet' else RUN
            path=result_root/dataset/method/'epoch_0120/metrics/summary.json'
            row=dict(method=method,dataset=dataset,**metadata(method),labels=50,epochs=120,
                values=None,status='pending',origin='new_method_epoch120',source=path.relative_to(ROOT).as_posix())
            if path.exists():
                summary=verified_summary(path,manifest_for(dataset),120)
                detail=read(path.parent/'per_image.csv')
                value={k:summary[f] for k,f in zip(FIELDS,METRICS)}
                positive=sum(int(x['tp'])+int(x['fp']) for x in detail)
                total=sum(int(x['pixels']) for x in detail)
                quality=('all_background' if positive==0 else 'all_foreground' if positive==total
                         else 'near_zero_IoU' if value['IoU']<1 else None)
                row.update(values=value,status='replication_unverified' if quality else 'verified',
                    metric_counts_verified=True,quality_review=quality,
                    training_horizon=summary['training_horizon'],
                    reused_existing_epoch120=summary.get('reused_existing_epoch120',False),
                    variant=summary.get('variant'),deviations=summary.get('deviations'))
            rows.append(row)
    replacements=apply_replacements(rows)
    index={(r['dataset'],r['method']):r for r in rows}
    assert len(index)==len(rows)==len(METADATA)*3
    methods=[name for category in ('FSS','IRSTD')
        for name in sorted((n for n,v in METADATA.items() if v['category']==category),key=lambda n:METADATA[n]['year'])]
    assert len(methods)==len(set(methods))==len(METADATA)-1
    methods.append('Ours')
    def cell(row,field,latex=False):
        if row['values'] is None:return '--' if latex else '待权重' if row['status']=='pending_weights' else '待完成'
        text=f'{row["values"][field]:.2f}'
        if row['status']=='replication_unverified':text+=r'$^{\dagger}$' if latex else '†'
        if row.get('test_based_display_selection') and field=='IoU':text+=r'$^{\ddagger}$' if latex else '‡'
        if row.get('adaptation_marker'):
            mark=row['adaptation_marker']
            text+=(r'$^{'+mark+'}$') if latex else f'<sup>{mark}</sup>'
        return text
    def collapse(values):
        return values[0] if len(set(values))==1 else '/'.join(values)
    def columns(method,latex=False):
        info=metadata(method);records=[index[d,method] for d in DATASETS]
        labels=collapse([str(r.get('labels') or '—') for r in records])
        epochs='分域历史配置' if method=='Ours' else collapse([str(r.get('epochs') or '—') for r in records])
        return [method,info['category'],info['venue'],str(info['year'] or '—'),labels,epochs,
            *[cell(r,f,latex) for r in records for f in FIELDS]]
    header=['方法','类别','期刊／会议','年份','标注张数 S/I/N','轮数／更新 S/I/N',
        *[f'{d} {f}'+('↓' if f=='Fa' else '↑') for d in DATASETS for f in FIELDS]]
    md=['# 三数据集对比大表','',
        '最终范围：**24个外部方法（FSS10、IRSTD14），Ours另列**。按类别、发表年份排列，每个方法仅占一行。新增7方法仍仅汇报第120轮的21组结果；按用户要求，已完成任务适配中IoU高于当前表的8组结果替换为同配置五指标，并以A1–A8角标标记。其余已有成绩保留。另按最新要求仅补MPCNet（IEEE TGRS2026）三域train50、第120轮结果；未完成记为待完成，不重跑已有方法。','',
        '新增方法：PConv+SD、SANSA、PAHNet、SAM-SPL、SFDTNet、IOVarNet、MLP-Net。21组完整测试及逐图计数核验均已完成；其中PConv+SD、SFDTNet、IOVarNet的9组结果存在退化或接近零分，保留原始数值并标记†。','',
        'IoU、nIoU、F1、Pd单位为%，Fa为误检像素/百万像素；测试集依次为SIRST86、IRSTD-1k201、NUDT-SIRST664张，原图GT>0。F1按全测试集累积TP/FP/FN计算。标注张数及日程按S/I/N顺序填写；u表示优化器更新次数，q表示每个查询任务的适配轮数，不等同于遍历全训练集的epoch；“—”表示免训练或未记录，待权重不记零。历史、适配和新增组的预算、日程、结构存在差异，因此不进行统一最优/次优排名。','',
        '| '+' | '.join(header)+' |','| '+' | '.join(['---']*3+['---:']*18)+' |']
    for method in methods:md.append('| '+' | '.join(columns(method))+' |')
    if replacements:
        md += ['', '**适配替换角标（应用于对应数据集的整组五指标）：**T＝训练日程，L＝损失，P＝支持表示，S＝输入尺度；R＝报告原标记为“—”，保留原设置的复跑参考差异，不宣称适配模块带来收益。',
            '替换标准为全图IoU改善，不要求每个指标都改善，绝不逐指标拼接。适配配置/检查点先由40/10验证划分选型，再锁定全50训练或完整测试；但本次决定哪些结果替换、及同方法多个已测版本中取最高IoU，使用了已观察的测试成绩。这是按用户要求的事后展示选择，不能称为未经测试选择的统一比较。','',
            '| 角标 | 方法 | 数据集 | 适配类型 | 原IoU→替换IoU (%) | 版本、预算与来源 |',
            '| --- | --- | --- | --- | ---: | --- |']
        for choice in replacements:
            mark=choice['marker'];tags=','.join(choice['adaptation_tags']) or 'R'
            scores=f'{choice["previous_result"]["values"]["IoU"]:.2f} → {choice["values"]["IoU"]:.2f}'
            budget=f'B={choice["annotation_budget"]}，K={choice["support_shots"]}，{choice["schedule_display"]}'
            md.append('| '+ ' | '.join([f'<sup>{mark}</sup>',choice['method'],choice['dataset'],tags,scores,
                f'{choice["version"]}；{budget}；[报告](../{choice["source"]})'])+' |')
        md += ['', 'IFA/IRSTD由原部署K=50改为K=5；ABCDFSS、TVGTANet被替换的组计入本轮验证选型的B=50，部署仍K=5，因此原5张参考与本轮差值不是单因素适配收益。ISNet采用Adam-edge版5.06%，仍明显低分；RPCANet采用已观测IoU最高的普通balanced版17.37%，hard-bg版14.36%保留在来源报告中。',
            '[适配报告及负结果](comparison_task_adaptation_results.md)；[替换记录与原始五指标](../experiments/evidence/comparison_adaptation_table_overrides.json)；[替换前完整大表](../experiments/evidence/comparison_table_history/20261105_before_task_adaptation/comparison_three_datasets.md)。']
    md += ['', '† 原始成绩保留，但复现退化或设置仍待核验，不能作为可靠优劣结论。Matcher、HERA的既有参考分数也保留此标记；PI-CLIP、DSV-LFS仍待完整任务权重，不另启重跑。',
        '‡ NUDT的SCTransNet、ILNet继续采用此前用户指定的125轮结果；该选择发生在观察历史测试成绩之后。原500轮成绩和覆盖记录保留在[历史覆盖证据](../experiments/evidence/comparison_table_overrides.json)，本次未替换其设置。','',
        '新增组均为train50、seed111。PConv+SD/IRSTD-1k直接复用此前保存的第120轮检查点，其学习率原计划终点为500；其他新增组终点为120。这里汇报的全部新增成绩均来自第120轮，没有继续500轮训练。',
        'MPCNet补充组：作者网络、三头SoftIoULoss、Kaiming初始化及Adam lr0.002；固定train50/seed111，batch4、resize256、翻转/转置，余弦终点120。相对作者默认batch8、正样本裁剪320、1000轮有所调整；不使用每轮测试选型。只在第120轮完整评测并核验后填表。作者来源：https://github.com/Wangtao-Bao/MPCNet；DOI：10.1109/TGRS.2026.3653023。','',
        '实现差异：PConv+SD为作者插件/SD损失接入MSHNet；SANSA为tiny骨干、1支持训练/5支持推理；PAHNet保留Pascal源域预训练并在目标train50适配；SAM-SPL为官方spl_t；IOVarNet记录了PDE零范数反向数值修正。详细设置随各结果config及summary保存，不能冒充论文原生全监督结果。','',
        '**Ours继续保留此前分域历史最佳观测：**SIRST为Top5＋可学习径向gain；IRSTD为Top5＋径向64→8自由聚合第5轮；NUDT为实例语义记忆＋适配Obj＋M1 Mask、K6。各域五指标来自同一配置，但这些配置根据历史测试成绩分别选择，不能表述为统一结构/预算、测试前预注册的三域结果。来源及先前设置见[Ours覆盖记录](../experiments/evidence/ours_table_overrides.json)。','',
        '历史说明：MSHNet三域训练轮数为1175/1200/1200；HDNet原为500/1200/500，其NUDT适配组现按1000u显示；其他适配替换也按真实更新数或每查询轮数单列。未替换组的历史轮数保留。此前SIRST坐标修正、IFA稀疏掩码池化问题、旧结果的有效性审计均继续保留，见[坐标修正](nuaa_coordinate_corrected.md)、outputs/comparison_failure_audit_20261101.json。',
        '发表年份按正式卷期：DNANet为TIP2023（2022在线）；MLP-Net为TGRS2025（DOI含2024）。论文链接见[方法与发表信息](comparison_experiment_plan.md)。','',
        '[完整120轮五指标](../outputs/comparison_recent_a_20261104/results_120.md)；[逐项数值与来源](../experiments/evidence/comparison_three_datasets.json)；[LaTeX表](../experiments/tables/comparison_three_datasets.tex)。','',
        '重建：`python -m scripts.build_three_dataset_table`（仅核验与排表，不运行实验）。']
    tex=[r'\begin{table*}[!t]',r'\centering',
        r'\caption{Preserved local results for24 external methods and Ours. New methods report epoch120 only. Training budgets differ; no pooled ranking is applied.}',
        r'\label{tab:comparison_three_datasets}',r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{llllll|rrrrr|rrrrr|rrrrr}',r'\hline',
        'Method & Type & Venue & Year & Labels & Epochs (S/I/N) & '+
        ' & '.join(f'{d} {f}' for d in DATASETS for f in FIELDS)+r' \\',r'\hline']
    previous=None
    for method in methods:
        if metadata(method)['category']!=previous:
            if previous is not None:tex.append(r'\hline')
            previous=metadata(method)['category']
        cols=columns(method,True)
        if method=='Ours':cols[2]='This work';cols[5]='Domain-specific historical'
        cols=[s.replace('—','--') for s in cols]
        tex.append(' & '.join(cols)+r' \\')
    tex += [r'\hline',r'\end{tabular}}',
        r'\par\footnotesize $\dagger$: raw scores retained; replication under review, including degenerate outputs. $\ddagger$: retained post-hoc NUDT epoch125 display for SCTransNet/ILNet. Missing task weights are shown as --. New epoch120 results use fixed train50/seed111; PConv+SD on IRSTD reuses its existing epoch120 checkpoint from an originally500-epoch schedule. SANSA uses tiny; PConv+SD is integrated into MSHNet. MPCNet uses the author network/loss with train50, batch4, resize256 and a120-epoch cosine schedule; unfinished results are --. Ours retains retrospectively selected domain-specific configurations, not a uniform preregistered protocol.',r'\end{table*}']
    if replacements:
        notes=[]
        for choice in replacements:
            tags=','.join(choice['adaptation_tags']) or 'R'
            notes.append(r'$^{'+choice['marker']+'}$ '+choice['method']+'/'+choice['dataset']+
                f': {tags}, B={choice["annotation_budget"]}, K={choice["support_shots"]}, {choice["schedule_display"]}; '+
                f'IoU {choice["previous_result"]["values"]["IoU"]:.2f} to {choice["values"]["IoU"]:.2f}.')
        tex[-1:-1]=[r'\par\footnotesize '+ ' '.join(notes)+
            ' All five metrics in each marked block come from one adapted configuration. T: schedule; L: loss; P: support representation; S: input scale; R: repeat retaining original settings. u: optimizer updates; q: per-query adaptation epochs. Replacement blocks and versions were selected retrospectively by observed test IoU; these are not native or unbiased validation-only comparisons. IFA changes deployment K50 to K5; adapted FSS blocks count B50 including validation selection.']
    evidence=ROOT/'experiments/evidence/comparison_three_datasets.json'
    old=json.loads(evidence.read_text(encoding='utf-8')) if evidence.exists() else {}
    # Keep the legacy interface and its exact raw numbers for existing consumers.
    old.update(results=legacy,new_method_results=[r for r in rows if r['origin']=='new_method_epoch120'],
        final_table_results=rows,method_order=methods,method_metadata=METADATA,
        external_method_count=24,external_category_counts=dict(FSS=10,IRSTD=14),
        adaptation_replacement_record='experiments/evidence/comparison_adaptation_table_overrides.json',
        adaptation_replacement_count=len(replacements),
        report_new_epochs=[120],ranking='none: mixed historical budgets/schedules; quality flags retained')
    assert before=={(r['dataset'],r['method']):(r['values'],r['source']) for r in legacy}
    (ROOT/'docs/comparison_three_datasets.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    (ROOT/'experiments/tables/comparison_three_datasets.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    evidence.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (ROOT/'experiments/tables/comparison_three_datasets.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(columns(m) for m in methods)
    print(json.dumps(dict(external_methods=24,ours_rows=1,new_groups=24,
        new_quality_flagged=sum(r['status']=='replication_unverified' for r in old['new_method_results']),
        legacy_groups_preserved=len(legacy),adaptation_replacements=len(replacements),training_started=False)))
