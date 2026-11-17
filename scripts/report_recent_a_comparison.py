"""Verify sufficient statistics and export independent120/500 result tables."""
import csv
import json
import math
from pathlib import Path
from scripts.run_recent_a_comparison import RUN,ROOT,METHODS,DATASETS,RETAIN,manifest_for
from scripts.prepare_comparison import write_json
from visual_prompt.evaluation.comparison import summarize
from scripts.comparison_method_metadata import metadata

METRICS=('IoU_percent','nIoU_percent','F1_percent','Pd_percent','Fa_per_million')

def verified_summary(path,m,epoch):
    result=json.loads(path.read_text(encoding='utf-8'))
    if not (result.get('complete_test') and result.get('epochs')==epoch and result.get('manifest_sha256')==m['manifest_sha256']
            and result.get('actual_annotation_budget')==50):raise ValueError('Summary protocol mismatch')
    with (path.parent/'per_image.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=len(m['test_ids']) or len({r['sample_id'] for r in rows})!=len(rows) or {r['sample_id'] for r in rows}!=set(m['test_ids']):
        raise ValueError('Test ID coverage mismatch')
    for row in rows:
        for k,v in row.items():
            if k!='sample_id':row[k]=float(v) if k=='image_iou' else int(v)
    recomputed=summarize(rows)
    for key in METRICS:
        if not math.isclose(result[key],recomputed[key],rel_tol=1e-8,abs_tol=1e-8):raise ValueError(f'Metric mismatch {key}')
    return result

def main():
    rows=[]
    for method in METHODS:
        for dataset in DATASETS:
            m=manifest_for(dataset)
            for epoch in RETAIN:
                path=RUN/dataset/method/f'epoch_{epoch:04d}'/'metrics/summary.json'
                row=dict(method=method,**metadata(method),dataset=dataset,epochs=epoch,status='pending',**{k:None for k in METRICS},source=str(path))
                if path.exists():
                    try:
                        result=verified_summary(path,m,epoch)
                        row.update(status='complete',**{k:result[k] for k in METRICS})
                    except Exception as e:row.update(status='verification_failed',error=str(e))
                rows.append(row)
    write_json(RUN/'comparison_summary.json',rows)
    evidence=ROOT/'experiments/evidence/recent_a_2025_2026';evidence.mkdir(parents=True,exist_ok=True)
    for epoch in RETAIN:
        subset=[r for r in rows if r['epochs']==epoch]
        for folder in (RUN,evidence):
            with (folder/f'results_{epoch}.csv').open('w',encoding='utf-8-sig',newline='') as f:
                keys=['method','category','venue','year','dataset','epochs','status',*METRICS,'source','error']
                writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(subset)
        lines=[f'# 固定第{epoch}轮结果','', '空值表示未完成；完整测试和逐图原始计数核验通过后才填分。','',
            '| 方法 | 数据集 | 状态 | IoU | nIoU | F1 | Pd | Fa/M |','|---|---|---|---:|---:|---:|---:|---:|']
        for row in subset:
            scores=['—' if row[k] is None else f'{row[k]:.4f}' for k in METRICS]
            lines.append('| '+' | '.join([row['method'],row['dataset'],row['status'],*scores])+' |')
        (RUN/f'results_{epoch}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(total=len(rows),complete=sum(r['status']=='complete' for r in rows),failed=sum(r['status']=='verification_failed' for r in rows))))

if __name__=='__main__':main()
