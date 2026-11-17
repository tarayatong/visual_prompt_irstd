"""Render only the user's retained table; retain MPCNet's independently updated row."""
import csv
import hashlib
import json
from pathlib import Path
from scripts.comparison_method_metadata import metadata,METADATA
from scripts.retain_comparison_user_table import ROOT,OUT,DATASETS,METHODS,FIELDS,read,write
from visual_prompt.evaluation.comparison import summarize


def main():
 plan=read(OUT/'retention_plan.json');rows=[]
 for original in plan['selected_rows']:
  row={**metadata(original['method']),**original}
  if row['values'] is None:row.update(labels=5,epochs='—',source=None)
  rows.append(row)
 expected={'SIRST':(68.64,75.39,81.41,93.58,108.55),'IRSTD-1k':(62.30,61.57,76.77,93.27,67.92),'NUDT-SIRST':(69.74,72.31,82.17,91.85,120.09)}
 for reference in read(OUT/'ours_protected.json'):
  ds=reference['dataset'];folder=ROOT/reference['evaluation_directory'];items=[];cp=ROOT/reference['checkpoint']
  digest=hashlib.sha256(cp.read_bytes()).hexdigest()
  for path in sorted(folder.glob('*.json')):
   saved=read(path);assert saved['checkpoint_sha256'][reference['variant']]==digest,path
   items.append(saved['rows'][reference['variant']])
  manifest=read(ROOT/f'outputs/comparison_50shot_5support/{ds}/protocol/manifest.json')
  assert len(items)==len(manifest['test_ids']) and {x['sample_id'] for x in items}==set(manifest['test_ids'])
  score=summarize(items);keys=('IoU_percent','nIoU_percent','F1_percent','Pd_percent','Fa_per_million')
  values=dict(zip(FIELDS,(score[k] for k in keys)))
  assert [f'{values[k]:.2f}' for k in FIELDS]==[f'{v:.2f}' for v in expected[ds]],(ds,values)
  target=OUT/'selected'/ds/'Ours';write(target/'summary.json',score)
  rows.append(dict(method='Ours',dataset=ds,**metadata('Ours'),values=values,status='verified',labels=50,epochs=120,
                   checkpoint=reference['checkpoint'],source=(target/'summary.json').relative_to(ROOT).as_posix(),
                   evaluation_directory=reference['evaluation_directory'],center_SAM=ds!='NUDT-SIRST',table_adaptation_star=False))
 # Never infer an unfinished MPCNet result or change its training files.
 evidence_path=ROOT/'experiments/evidence/comparison_three_datasets.json'
 old=read(evidence_path);mpc=[r for r in old.get('final_table_results',[]) if r['method']=='MPCNet']
 for ds in DATASETS:
  row=next((r for r in mpc if r['dataset']==ds),None)
  if row is None:row=dict(method='MPCNet',dataset=ds,**metadata('MPCNet'),labels=50,epochs=120,values=None,status='pending')
  rows.append(row)
 methods=[*METHODS,'MPCNet','Ours'];lookup={(r['method'],r['dataset']):r for r in rows}
 assert len(rows)==57 and len(lookup)==57
 def ranked(row,field):
  if not row['values']:return '--'
  raw=f"{row['values'][field]:.2f}"
  values=sorted({round(x['values'][field],2) for x in rows if x['dataset']==row['dataset'] and x['values']},reverse=field!='Fa')
  rank=values.index(round(row['values'][field],2))
  text=(r'\textbf{'+raw+'}' if rank==0 else r'\underline{'+raw+'}' if rank==1 else r'\textit{'+raw+'}' if rank==2 else raw)
  return text+(r'$^{*}$' if row.get('table_adaptation_star') else '')
 def venue(method):
  info=metadata(method)
  short={'IEEE TIP':'TIP','IEEE TGRS':'TGRS','IEEE TAES':'TAES','ACM Multimedia':'ACM MM'}.get(info['venue'],info['venue'])
  return short+"'"+str(info['year'])[-2:] if info['year'] else ''
 tex=[r'\begin{table*}[!t]',r'\centering',r'\caption{Comparison on SIRST, IRSTD-1k, and NUDT-SIRST. Bold, underlined, and italic values denote the best, second-best, and third-best results, respectively.}',r'\label{tab:comparison_three_datasets}',r'\begingroup',r'\renewcommand{\arraystretch}{1.15}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{ll|rrrrr|rrrrr|rrrrr}',r'\hline',r'\multirow{2}{*}{Method} & \multirow{2}{*}{Venue} & \multicolumn{5}{c|}{SIRST} & \multicolumn{5}{c|}{IRSTD-1k} & \multicolumn{5}{c}{NUDT-SIRST} \\',r'\cline{3-7}\cline{8-12}\cline{13-17}',r' & & '+' & '.join([f'{f} $'+(r'\downarrow' if f=='Fa' else r'\uparrow')+'$' for _ in DATASETS for f in FIELDS])+r' \\',r'\hline']
 category=None
 for method in methods:
  cat=metadata(method)['category']
  if cat!=category:
   if category is not None:tex.append(r'\hline')
   if cat!='Ours':tex.extend([r'\multicolumn{17}{l}{\textbf{'+('Few Shot Segmentation' if cat=='FSS' else 'IRSTD')+r'}} \\',r'\hline'])
   category=cat
  tex.append(' & '.join([method,venue(method),*[ranked(lookup[method,d],f) for d in DATASETS for f in FIELDS]])+r' \\')
 tex.extend([r'\hline',r'\end{tabular}}',r'\endgroup',r'\par\smallskip',r'\noindent\parbox{\linewidth}{\raggedright\footnotesize $^{*}$: Retained task-adapted or repeated settings; details are recorded in the retention index. --: No retained result. Historical budgets and schedules differ; settings were selected retrospectively.\par}',r'\end{table*}'])
 (ROOT/'experiments/tables/comparison_three_datasets.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
 header=['方法','类别','期刊／会议','年份','标注张数 S/I/N','轮数／更新 S/I/N',*[f'{d} {f}'+('↓' if f=='Fa' else '↑') for d in DATASETS for f in FIELDS]]
 def columns(method):
  records=[lookup[method,d] for d in DATASETS];info=metadata(method)
  def collapse(field):
   vals=[str(r.get(field) or '—') for r in records];return vals[0] if len(set(vals))==1 else '/'.join(vals)
  return [method,info['category'],info['venue'],str(info['year'] or '—'),collapse('labels'),collapse('epochs'),*[('—' if not r['values'] else f"{r['values'][f]:.2f}"+('*' if r.get('table_adaptation_star') else '')) for r in records for f in FIELDS]]
 md=['# 三数据集对比大表','',
 '按用户提供的最终表保留17个外部方法，另保留独立会话验证中的MPCNet，Ours单列。每个结果的配置、代码入口和唯一任务权重见[保留索引](comparison_retention.md)。旧适配定时任务已暂停，历史候选不会重新进入主表。','',
 '各域五指标来自同一设置。SIRST/IRSTD/NUDT官方测试分别为86/201/664张；IoU、nIoU、F1、Pd为%，Fa为误检像素/百万像素。原始字节与轮次已核对；13组旧指标目录在本次清理前已缺失，仅保留已有序列化数值，不声称重新推理核验。','',
 '| '+' | '.join(header)+' |','| '+' | '.join(['---']*3+['---:']*18)+' |']
 md.extend('| '+' | '.join(columns(m))+' |' for m in methods)
 md.extend(['','* 为用户选入的适配或原设置复跑组。ABCDFSS的两组及TVGTANet/NUDT仍使用原设置，不能称原方法完全失效；IFA/IRSTD从K50改K5，不构成单因素消融。NUDT的SCTransNet/ILNet保留125轮权重；MSHNet等按各自历史真实轮次保留，并非统一训练预算。','',
 'Ours采用SIRST、IRSTD有中心SAM的M2＋可学习径向，NUDT采用无中心版本；均为Top5固定权重结果。三域五指标已从保留计数独立汇总并核对checkpoint哈希。保留既有消融与Top-k实验，不在本次外部方法清理范围内。','',
 'HERA/IRSTD按用户表不汇报；MPCNet独立会话继续训练，仅其第120轮完整结果可以增量写入。重建本表：`python -m scripts.render_retained_comparison`；仅更新MPCNet：`python -m scripts.update_mpcnet_table`。'])
 (ROOT/'docs/comparison_three_datasets.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
 with (ROOT/'experiments/tables/comparison_three_datasets.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.writer(f);writer.writerow(header);writer.writerows(columns(m) for m in methods)
 write(evidence_path,dict(datasets=DATASETS,fields=FIELDS,results=rows,final_table_results=rows,
     new_method_results=[r for r in rows if r.get('origin')=='new_method_epoch120'],supplemental_method_results=mpc,
     method_order=methods,method_metadata=METADATA,external_method_count=18,external_category_counts=dict(FSS=7,IRSTD=11),
     retention_index=relative_index(),scope='user supplied table plus protected MPCNet',ranking='displayed ranks over retained mixed-protocol results; not uniform preregistered comparison'))
 print(json.dumps(dict(retained_external_methods=17,protected_MPCNet=True,table_cells=57,ours_count_and_hash_checks_passed=True)))
 if (ROOT/'outputs/comparison_sam_train50_20261109/protocol.json').exists():
  from scripts.update_sam_extension_metrics import main as update_sam
  update_sam()


def relative_index():return (OUT/'retention_plan.json').relative_to(ROOT).as_posix()


if __name__=='__main__':main()
