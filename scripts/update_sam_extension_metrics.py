"""Verify/recount new epoch120 results and incrementally update existing tables."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import math
import subprocess
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_top8_comparison import read,save,manifest
from scripts.report_recent_a_comparison import verified_summary,METRICS
from scripts.train_sam_extension_train50 import RUN,METHODS,DATASETS
from visual_prompt.evaluation.comparison import image_metrics,summarize
import numpy as np
from PIL import Image

FIELDS=('IoU','nIoU','F1','Pd','Fa')
METADATA={'IRPruneDet':('AAAI',2024),'IR-SAM2':('Remote Sensing',2026),'IRSAM':('ECCV',2024),'SAMamba':('Information Fusion',2025)}


def audit(path,m):
    result=verified_summary(path,m,120)
    assert result['train_annotation_ids']==m['train_ids'] and len(set(result['train_annotation_ids']))==50
    assert result['seed']==111 and not result['target_task_pretraining_used']
    history=read(path.parents[2]/'training_history.json')
    assert [r['epoch'] for r in history]==list(range(1,121))
    assert all(r['unique_supervised_images']==50 for r in history)
    cp=path.parents[2]/'checkpoints/epoch_0120.pt'
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==result['checkpoint_sha256']
    rows=[]
    for sid in m['test_ids']:
        with Image.open(path.parents[1]/'predictions'/f'{sid}.png') as im:pred=np.array(im.convert('L'))>0
        assert pred.shape==tuple(m['records'][sid]['size_hw']),sid
        with Image.open(m['records'][sid]['mask']) as im:gt=np.array(im.convert('L').resize((pred.shape[1],pred.shape[0]),Image.Resampling.NEAREST))>0
        rows.append(image_metrics(pred,gt,sid))
    recomputed=summarize(rows)
    for field in METRICS:assert math.isclose(result[field],recomputed[field],rel_tol=1e-8,abs_tol=1e-8),(field,path)
    config=read(path.parents[2]/'config.json')
    result.update(checkpoint=cp.relative_to(ROOT).as_posix(),
        initialization_checkpoint=config.get('checkpoint'),initialization_checkpoint_sha256=config.get('checkpoint_sha256'),
        initialization_weights_status=config['weights_status'],weights_status='project train50 seed111 epoch120 task weights',
        train50_replication_complete=True)
    if result['IoU_percent']<1:
        result['performance_warning']='IoU<1%: task adaptation degenerated; metric counts verified, not successful reproduction of paper performance'
        if result['predictions']==0:result['performance_warning']+='; all predictions empty, Fa=0 is not improved detection'
    save(path,result)
    return result


def main():
    RUN.mkdir(parents=True,exist_ok=True);rows=[]
    for method in METHODS:
        for ds in DATASETS:
            path=RUN/ds/method/'epoch_0120/metrics/summary.json'
            row=dict(method=method,dataset=ds,category='IRSTD',venue=METADATA[method][0],year=METADATA[method][1],labels=50,epochs=120,
                     source=path.relative_to(ROOT).as_posix(),values=None,status='pending',table_adaptation_star=True)
            if path.exists():
                result=audit(path,manifest(ds))
                row.update(values=dict(zip(FIELDS,(result[k] for k in METRICS))),status='verified',
                    checkpoint=str((path.parents[2]/'checkpoints/epoch_0120.pt').relative_to(ROOT)),
                    checkpoint_sha256=result['checkpoint_sha256'],metric_counts_verified=True,predictions_recount_verified=True,
                    variant=result['variant'],input_size=result['input_size'],backend=result.get('backend','native'),training_horizon=120)
                if result.get('performance_warning'):row['performance_warning']=result['performance_warning']
            rows.append(row)
    save(RUN/'results.json',rows)
    with (RUN/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        extra=['input_size','backend','variant','performance_warning','checkpoint','checkpoint_sha256']
        writer=csv.DictWriter(f,fieldnames=['method','dataset','status','labels','epochs',*FIELDS,'source',*extra])
        writer.writeheader()
        for r in rows:writer.writerow({k:r[k] for k in ['method','dataset','status','labels','epochs','source']}|{k:(r['values'] or {}).get(k,'') for k in FIELDS}|{k:r.get(k,'') for k in extra})
    header='| 方法 | 数据集 | 状态 | IoU↑ | nIoU↑ | F1↑ | Pd↑ | Fa/M↓ |'
    lines=['## 新增红外方法三数据集 train50 / epoch120 指标','',
        '用户确认：每域固定50张、seed111、训练120轮；只评测第120轮。完整测试并从保存二值预测重计五指标后才填分。',
        'SAMamba 使用256输入与 PyTorch 参考 scan；IRSAM是当前公开新版；IRPruneDet仅复用发布紧凑拓扑，全部权重和BN统计重置，未复现剪枝搜索阶段。未加载作者目标域任务权重。','',header,'|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        vals=['待完成' if r['values'] is None else f'{r["values"][k]:.2f}'+('†' if r.get('performance_warning') else '') for k in FIELDS]
        lines.append('| '+' | '.join([r['method'],r['dataset'],r['status'],*vals])+' |')
    lines+=['','测试清单分别为 SIRST86 / IRSTD-1k201 / NUDT-SIRST664 张。阈值0.5；概率图双线性恢复原尺寸；Pd采用质心距离<3像素的一对一匹配；Fa为全图误检像素/百万像素。',
            '† 表示 IoU<1%的退化结果：逐图计数已核验，但该设置未成功复现论文性能。全空预测的 Fa=0 不能解读为检测性能更好。',
            '', '[配置、训练历史与原始逐图计数](../comparison_sam_train50_20261109/)；[机器可读结果](../comparison_sam_train50_20261109/results.json)。']
    (ROOT/'outputs/top8_analysis_20261105/sam_train50_metrics.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Preserve every pre-existing row verbatim, including concurrently supplied MPCNet.
    mdpath=ROOT/'docs/comparison_three_datasets.md';md=mdpath.read_text(encoding='utf-8').splitlines()
    old=[s for s in md if s.startswith('| ') and s.split('|')[1].strip() not in METHODS]
    md=[s for s in md if not (s.startswith('| ') and s.split('|')[1].strip() in METHODS)]
    verified_count=sum(r['status']=='verified' for r in rows)
    supplement_intro=f'本轮另增 IRPruneDet、IRSAM、IR-SAM2、SAMamba 的三域 train50/epoch120 结果；已完成并核验 {verified_count}/12 组，原有结果保留。'
    md=[s for s in md if not s.startswith('本轮另增 IRPruneDet、IRSAM、IR-SAM2、SAMamba 的三域 train50/epoch120 结果；')]
    if md and md[0].startswith('# '):md[1:1]=['',supplement_intro]
    pos=next(i for i,s in enumerate(md) if s.startswith('| Ours |'))
    def cells(method):
        info=METADATA[method];out=[method,'IRSTD',info[0],str(info[1]),'50','120']
        for ds in DATASETS:
            r=next(r for r in rows if r['method']==method and r['dataset']==ds)
            out += ['待完成' if r['values'] is None else f'{r["values"][k]:.2f}*'+('†' if r.get('performance_warning') else '') for k in FIELDS]
        return out
    md[pos:pos]=['| '+' | '.join(cells(method))+' |' for method in METHODS]
    md=[s for s in md if not s.startswith('SAM新增方法补充：')]
    md+=['','SAM新增方法补充：IRPruneDet、IRSAM、IR-SAM2、SAMamba 按用户确认的三域train50/seed111/epoch120增量补充；*为明确披露的任务适配，非论文原生全监督成绩。†为IoU<1%的退化结果，计数已核验，不代表复现了论文性能；空预测的Fa=0不能解读为更优。仅完整测试并重计预测后填分；训练中留“待完成”。SAMamba输入256与参考scan；IRSAM当前公开新版；IRPruneDet为公开紧凑拓扑随机初始化重训。保留其他方法原始行。增量更新：`python -m scripts.update_sam_extension_metrics`。[原始记录](../outputs/comparison_sam_train50_20261109/results.json)。']
    assert old==[s for s in md if s.startswith('| ') and s.split('|')[1].strip() not in METHODS]
    md=[s for i,s in enumerate(md) if s or i==0 or md[i-1]]
    mdpath.write_text('\n'.join(md)+'\n',encoding='utf-8')
    csvpath=ROOT/'experiments/tables/comparison_three_datasets.csv'
    with csvpath.open(encoding='utf-8-sig',newline='') as f:table=list(csv.reader(f))
    original=[r for r in table if r[0] not in METHODS];table=[r for r in table if r[0] not in METHODS]
    pos=next(i for i,r in enumerate(table) if r[0]=='Ours');table[pos:pos]=[cells(method) for method in METHODS]
    assert original==[r for r in table if r[0] not in METHODS]
    with csvpath.open('w',encoding='utf-8-sig',newline='') as f:csv.writer(f).writerows(table)
    indexpath=ROOT/'docs/comparison_retention.md'
    index=indexpath.read_text(encoding='utf-8')
    begin='<!-- SAM_TRAIN50_BEGIN -->';end='<!-- SAM_TRAIN50_END -->'
    block=[begin,'## 新增红外方法 train50（2026-11-09）','',
           '每域50张、seed111、第120轮；配置和权重逐组唯一。初始化及实现差异见[执行协议](sam_train50_protocol_20261109.md)。','',
           '| 方法 | 数据集 | 状态 | IoU | 配置 | 第120轮权重 |','|---|---|---|---:|---|---|']
    for r in rows:
        cfg=RUN/r['dataset']/r['method']/'config.json'
        config_link=f'[config](../{cfg.relative_to(ROOT).as_posix()})' if cfg.exists() else '待启动'
        weight_link=f'[checkpoint](../{r["checkpoint"]})' if r['values'] is not None else '待完成'
        iou=f'{r["values"]["IoU"]:.2f}' if r['values'] is not None else '—'
        block.append('| '+' | '.join([r['method'],r['dataset'],r['status'],iou,config_link,weight_link])+' |')
    block += [end]
    if begin in index:
        first=index.index(begin);last=index.index(end,first)+len(end)
        index=index[:first]+'\n'.join(block)+index[last:]
    else:index=index.rstrip()+'\n\n'+'\n'.join(block)+'\n'
    indexpath.write_text(index,encoding='utf-8')
    texpath=ROOT/'experiments/tables/comparison_three_datasets.tex'
    tex=texpath.read_text(encoding='utf-8').splitlines();original=[s for s in tex if ' & ' in s and s.split(' & ')[0] not in METHODS]
    tex=[s.replace('Bold, underlined, and italic values denote the best, second-best, and third-best results, respectively.',
                   'Emphasis denotes ranks in the original retained comparison; supplemental train50 rows are unranked.') for s in tex]
    tex=[s for s in tex if s.split(' & ')[0] not in METHODS]
    pos=next(i for i,s in enumerate(tex) if s.startswith('Ours &'))
    texrows=[]
    def tex_cell(s):
        if s=='待完成':return '--'
        if s.endswith('*†'):return s[:-2]+r'$^{*\dagger}$'
        if s.endswith('*'):return s[:-1]+r'$^{*}$'
        if s.endswith('†'):return s[:-1]+r'$^{\dagger}$'
        return s
    for method in METHODS:
        c=cells(method);c=[c[0],METADATA[method][0]+"'"+str(METADATA[method][1])[-2:],*c[6:]]
        c=[tex_cell(s) for s in c]
        texrows.append(' & '.join(c)+r' \\')
    tex[pos:pos]=texrows
    tex=[s for s in tex if not s.startswith(r'\par\footnotesize SAM train50:')]
    note=(r'\par\footnotesize SAM train50: seed111, epoch120 only. IRSAM: current public release; '
          r'SAMamba:256 input and PyTorch reference scan; IRPruneDet: published compact topology with reset weights. '
          r'$^{\dagger}$: IoU below1\%; empty predictions with Fa=0 do not mean better detection.')
    tex.insert(len(tex)-1,note)
    assert original==[s for s in tex if ' & ' in s and s.split(' & ')[0] not in METHODS]
    texpath.write_text('\n'.join(tex)+'\n',encoding='utf-8')
    evidence_path=ROOT/'experiments/evidence/comparison_three_datasets.json';evidence=read(evidence_path)
    retained=[r for r in evidence['final_table_results'] if r['method'] not in METHODS]
    evidence['final_table_results']=retained+rows
    order=[m for m in evidence['method_order'] if m not in (*METHODS,'Ours')]
    evidence['method_order']=order+list(METHODS)+['Ours']
    evidence['sam_train50_supplemental_results']=rows
    info=evidence.setdefault('method_metadata',{})
    for method,(venue,year) in METADATA.items():info[method]=dict(category='IRSTD',venue=venue,year=year)
    evidence['sam_train50_scope']='Four new public architectures, each dataset train50 seed111 epoch120 only; initialization/architecture/backend deviations disclosed per result'
    external=[r for r in evidence['final_table_results'] if r['method']!='Ours']
    evidence['external_method_count']=len({r['method'] for r in external})
    evidence['external_category_counts']={c:len({r['method'] for r in external if r['category']==c}) for c in ('FSS','IRSTD')}
    save(evidence_path,evidence)
    # The serial queue invokes this while its training child has exited, so the
    # benchmark does not share this queue's GPU training load. Busy GPU -> defer.
    for method in METHODS:
        r=next(r for r in rows if r['method']==method and r['dataset']=='IRSTD-1k')
        if r['status']!='verified':continue
        path=ROOT/'outputs/top8_analysis_20261105/complexity'/f'{method}.json'
        measured=read(path) if path.exists() else {}
        if measured.get('profile_epoch')==120 and measured.get('checkpoint_sha256')==r['checkpoint_sha256'] and measured.get('fps'):continue
        benchmark=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/profile_sam_train50_complexity.py'),'--method',method],cwd=ROOT,capture_output=True,text=True)
        save(RUN/'trained_complexity'/f'{method}_benchmark_status.json',dict(exit_code=benchmark.returncode,stdout=benchmark.stdout[-2000:],stderr=benchmark.stderr[-2000:]))
    from scripts.analyze_top8_comparison import report
    report()
    print(json.dumps(dict(total=12,verified=sum(r['status']=='verified' for r in rows))),flush=True)

if __name__=='__main__':main()
