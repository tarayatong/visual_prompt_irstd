"""Apply explicitly recorded, audited task-adaptation display replacements."""
import hashlib
import json
from pathlib import Path
from scripts.collect_experiment_evidence import ROOT, read, metrics, check
from scripts.run_recent_a_comparison import manifest_for

OVERRIDES=ROOT/'experiments/evidence/comparison_adaptation_table_overrides.json'


def apply_replacements(rows):
    if not OVERRIDES.exists():return []
    plan=json.loads(OVERRIDES.read_text(encoding='utf-8'))
    index={(r['method'],r['dataset']):r for r in rows}
    for choice in plan['replacements']:
        row=index[choice['method'],choice['dataset']]
        assert row['values']==choice['previous_result']['values']
        assert row['source']==choice['previous_result']['source']
        detail_path=ROOT/choice['per_image']
        assert hashlib.sha256(detail_path.read_bytes()).hexdigest()==choice['per_image_sha256']
        audit=json.loads((ROOT/choice['audit']).read_text(encoding='utf-8'))
        report=json.loads((ROOT/choice['source']).read_text(encoding='utf-8'))
        assert audit['passed'] and report['status']=='complete'
        detail=read(detail_path);test_ids=set(manifest_for(choice['dataset'])['test_ids'])
        assert len(detail)==len(test_ids) and {r['sample_id'] for r in detail}==test_ids
        actual=metrics(detail);check(actual,choice['values'])
        expected=report.get('test_metrics') or report['final']['test_metrics']
        metric_keys=dict(IoU='IoU_percent',nIoU='nIoU_percent',F1='F1_percent',Pd='Pd_percent',Fa='Fa_per_million')
        check(actual,{k:expected[v] for k,v in metric_keys.items()})
        assert actual['IoU']>row['values']['IoU']
        row.update(values=actual,source=choice['source'],status='verified',labels=choice['annotation_budget'],
            epochs=choice['schedule_display'],origin='task_adaptation_replacement',
            previous_result=choice['previous_result'],adaptation_marker=choice['marker'],
            adaptation_tags=choice['adaptation_tags'],support_shots=choice['support_shots'],
            adapted_version=choice['version'],selection=choice['selection'],
            replacement_selection='user_requested_posthoc_higher_test_IoU_display; five metrics from one configuration')
    return plan['replacements']
