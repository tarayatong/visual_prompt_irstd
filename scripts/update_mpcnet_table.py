"""Incrementally fill MPCNet; retain every other serialized result and table row."""
import csv
import io
import json
from pathlib import Path
from scripts.comparison_method_metadata import METADATA, metadata
from scripts.report_recent_a_comparison import verified_summary, METRICS
from scripts.run_recent_a_comparison import manifest_for

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/comparison_mpcnet_20261105'
DATASETS=('SIRST','IRSTD-1k','NUDT-SIRST')
FIELDS=('IoU','nIoU','F1','Pd','Fa')


def main():
    path=ROOT/'experiments/evidence/comparison_three_datasets.json'
    evidence=json.loads(path.read_text(encoding='utf-8'))
    retained=[r for r in evidence['final_table_results'] if r['method']!='MPCNet']
    # The user may reduce the retained comparison scope while MPCNet trains.
    assert len(retained)==3*len({r['method'] for r in retained})
    previous_methods={r['method'] for r in retained if r['method']!='Ours'}
    external_methods=previous_methods|{'MPCNet'}
    category_counts={category:len({r['method'] for r in retained if r['method']!='Ours' and r['category']==category})+(category=='IRSTD') for category in ('FSS','IRSTD')}
    rows=[]
    for dataset in DATASETS:
        source=RUN/dataset/'MPCNet/epoch_0120/metrics/summary.json'
        row=dict(method='MPCNet',dataset=dataset,**metadata('MPCNet'),labels=50,epochs=120,
                 values=None,status='pending',origin='supplemental_mpcnet_epoch120',
                 source=source.relative_to(ROOT).as_posix())
        if source.exists():
            result=verified_summary(source,manifest_for(dataset),120)
            values={f:result[k] for f,k in zip(FIELDS,METRICS)}
            row.update(values=values,status='replication_unverified' if values['IoU']<1 else 'verified',
                       metric_counts_verified=True,training_horizon=result['training_horizon'],
                       variant=result['variant'],deviations=result['deviations'])
        rows.append(row)
    def cells(latex=False):
        result=['MPCNet','IRSTD','IEEE TGRS','2026','50','120']
        for r in rows:
            for f in FIELDS:
                value='--' if latex else '待完成'
                if r['values'] is not None:
                    value=f'{r["values"][f]:.2f}'
                    if r['status']=='replication_unverified':value+=r'$^{\dagger}$' if latex else '†'
                result.append(value)
        return result
    texpath=ROOT/'experiments/tables/comparison_three_datasets.tex'
    tex=texpath.read_text(encoding='utf-8').splitlines()
    old_tex_rows=[line for line in tex if ' & ' in line and not line.startswith('MPCNet &')]
    tex=[line for line in tex if not line.startswith('MPCNet &')]
    pos=next(i for i,line in enumerate(tex) if line.startswith('Ours &'))
    if tex[pos-1]==r'\hline':pos-=1
    texcells=cells(True)
    if any(r'\begin{tabular}{ll|rrrrr|rrrrr|rrrrr}' in line for line in tex):
        texcells=[texcells[0],"TGRS'26",*texcells[6:]]
    tex.insert(pos,' & '.join(texcells)+r' \\')
    tex=[line.replace('for23 external methods','for24 external methods') for line in tex]
    note=(r'\par\footnotesize MPCNet (IEEE TGRS2026): author network and three-head SoftIoULoss; '
          'fixed train50/seed111, batch4, resize256, Adam lr0.002 and120-epoch cosine schedule. '
          'Native defaults use batch8, crop320 and1000 epochs. Evaluate epoch120 only; pending results are --. '
          f'Existing{len(previous_methods)} methods and all adaptation markers are preserved.')
    tex=[line for line in tex if not line.startswith(r'\par\footnotesize MPCNet (')]
    tex.insert(len(tex)-1,note)
    assert old_tex_rows==[line for line in tex if ' & ' in line and not line.startswith('MPCNet &')]
    texpath.write_text('\n'.join(tex)+'\n',encoding='utf-8')
    mdpath=ROOT/'docs/comparison_three_datasets.md'
    md=mdpath.read_text(encoding='utf-8').splitlines()
    old_md_rows=[line for line in md if line.startswith('| ') and not line.startswith('| MPCNet |')]
    md=[line for line in md if not line.startswith('| MPCNet |')]
    pos=next(i for i,line in enumerate(md) if line.startswith('| Ours |'))
    md.insert(pos,'| '+' | '.join(cells())+' |')
    md=[line.replace('23个外部方法（FSS10、IRSTD13）','24个外部方法（FSS10、IRSTD14）') for line in md]
    md=[line for line in md if not line.startswith('MPCNet补充：')]
    md += ['', f'MPCNet补充：仅新增本方法三域train50、seed111、第120轮结果，已有{len(previous_methods)}方法及适配块逐项保留。作者网络、三头SoftIoULoss、Kaiming初始化、Adam lr0.002；batch4、resize256、翻转/转置、余弦终点120。相对作者默认batch8、正样本裁剪320、1000轮有调整；不使用每轮测试选型。仅完整测试及逐图计数核验后填分。历史原始审计路径部分缺失，本次只保留现表已记录结果，不声称重新核验旧结果。[作者代码](https://github.com/Wangtao-Bao/MPCNet)，DOI：10.1109/TGRS.2026.3653023。增量重建：`python -m scripts.update_mpcnet_table`。']
    assert old_md_rows==[line for line in md if line.startswith('| ') and not line.startswith('| MPCNet |')]
    mdpath.write_text('\n'.join(md)+'\n',encoding='utf-8')
    csvpath=ROOT/'experiments/tables/comparison_three_datasets.csv'
    with csvpath.open(encoding='utf-8-sig',newline='') as stream: table=list(csv.reader(stream))
    old_csv=[r for r in table if r[0]!='MPCNet']
    table=[r for r in table if r[0]!='MPCNet']
    pos=next(i for i,r in enumerate(table) if r[0]=='Ours')
    csvrow=cells();csvrow=['待完成' if c=='待完成' else c for c in csvrow]
    table.insert(pos,csvrow)
    assert old_csv==[r for r in table if r[0]!='MPCNet']
    with csvpath.open('w',encoding='utf-8-sig',newline='') as stream:csv.writer(stream).writerows(table)
    evidence.update(final_table_results=retained+rows, supplemental_method_results=rows,
        method_order=[m for m in evidence['method_order'] if m not in ('MPCNet','Ours')]+['MPCNet','Ours'],
        method_metadata=METADATA,external_method_count=len(external_methods),external_category_counts=category_counts,
        supplemental_scope='MPCNet only; fixed train50 epoch120; all previous rows retained verbatim')
    assert retained==[r for r in evidence['final_table_results'] if r['method']!='MPCNet']
    path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(external_methods=len(external_methods),previous_groups_preserved=len(retained),
                         mpcnet_complete=sum(r['values'] is not None for r in rows),mpcnet_total=3)))


if __name__=='__main__': main()
