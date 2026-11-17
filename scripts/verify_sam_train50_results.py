"""Final/partial audit of train50 coverage, predictions, checkpoints and old rows."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.update_sam_extension_metrics import RUN,METHODS,DATASETS,audit,read,save,manifest

p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args()
results=[]
for method in METHODS:
    for ds in DATASETS:
        path=RUN/ds/method/'epoch_0120/metrics/summary.json'
        if not path.exists():
            assert a.partial,(method,ds,'missing final result')
            continue
        r=audit(path,manifest(ds))
        config=read(path.parents[2]/'config.json')
        archived=RUN/'runner_versions'/f'{config["runner_sha256"]}.py'
        assert hashlib.sha256(archived.read_bytes()).hexdigest()==config['runner_sha256']
        results.append(dict(method=method,dataset=ds,images=r['images'],checkpoint_sha256=r['checkpoint_sha256'],
                            annotation_ids_verified=True,all_120_epochs_verified=True,predictions_recount_verified=True))
baseline=read(RUN/'preserved_table_baseline.json')
md=(ROOT/'docs/comparison_three_datasets.md').read_text(encoding='utf-8').splitlines()
tex=(ROOT/'experiments/tables/comparison_three_datasets.tex').read_text(encoding='utf-8').splitlines()
with (ROOT/'experiments/tables/comparison_three_datasets.csv').open(encoding='utf-8-sig',newline='') as f:table=list(csv.reader(f))
evidence=read(ROOT/'experiments/evidence/comparison_three_datasets.json')
assert baseline['markdown_rows']==[r for r in md if r.startswith('| ') and r.split('|')[1].strip() not in METHODS]
assert baseline['tex_rows']==[r for r in tex if ' & ' in r and r.split(' & ')[0] not in METHODS]
assert baseline['csv_rows']==[r for r in table if r[0] not in METHODS]
assert baseline['evidence_rows']==[r for r in evidence['final_table_results'] if r['method'] not in METHODS]
record=dict(status='partial' if len(results)<12 else 'complete',verified_groups=len(results),expected_groups=12,
            test_counts={'SIRST':86,'IRSTD-1k':201,'NUDT-SIRST':664},previous_dataset_groups_preserved=57,
            previous_comparison_rows_preserved=True,groups=results)
save(RUN/'verification.json',record)
print(json.dumps({k:v for k,v in record.items() if k!='groups'}),flush=True)
