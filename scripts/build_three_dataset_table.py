"""Build the 3 x 5 paper table from verified local results (no literature values)."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from scripts.collect_experiment_evidence import ROOT, read, metrics, check
from scripts.comparison_method_metadata import metadata, METADATA, NEW_METHODS

DATASETS=('SIRST','IRSTD-1k','NUDT-SIRST')
METHODS=('ABCDFSS','TVGTANet','MPA','IFA_50label','DNANet','MSHNet','SCTransNet',
         'RPCANet','ISNet','ILNet','HDNet','MSHNet+NS-FPN','Ours')
FIELDS=('IoU','nIoU','F1','Pd','Fa')


def main():
    if (ROOT/'outputs/comparison_task_adaptation_20261104/retention_lock.json').exists():
        from scripts.render_retained_comparison import main as render_retained
        render_retained();return
    records=read(ROOT/'experiments/evidence/comparison_single_dataset.csv')
    lookup={(r['dataset'],r['method']):r for r in records}
    corrected_root=ROOT/'outputs/nuaa_coordinate_corrected_20261101'
    corrected_audit=corrected_root/'audit.json'
    corrected_ready=corrected_audit.exists() and json.loads(corrected_audit.read_text())['passed']
    override_path=ROOT/'experiments/evidence/ours_table_overrides.json'
    overrides=json.loads(override_path.read_text(encoding='utf-8')) if override_path.exists() else {}
    comparison_override_path=ROOT/'experiments/evidence/comparison_table_overrides.json'
    comparison_overrides=json.loads(comparison_override_path.read_text(encoding='utf-8')) if comparison_override_path.exists() else {}
    external_choices={(r['dataset'],r['method']):r for r in comparison_overrides.get('overrides',[])}
    assert len(external_choices)==len(comparison_overrides.get('overrides',[]))
    assert not any(method=='Ours' for _,method in external_choices)
    best_observed=bool(overrides) and all(overrides.get(d,{}).get('selection_mode')=='user_requested_best_observed_IoU' for d in DATASETS)
    results=[]
    for dataset in DATASETS:
        manifest=json.loads((ROOT/f'outputs/comparison_50shot_5support/{dataset}/protocol/manifest.json').read_text(encoding='utf-8'))
        for method in METHODS:
            if method=='Ours' and dataset in overrides:
                choice=overrides[dataset]
                assert json.loads((ROOT/choice['audit']).read_text(encoding='utf-8'))['passed']
                summary_path=ROOT/choice['source'];detail_path=ROOT/choice['per_image']
                summary=json.loads(summary_path.read_text(encoding='utf-8'));assert summary['complete_test']
                detail=read(detail_path);actual=metrics(detail)
                expected={k:summary[f] for k,f in dict(IoU='IoU_percent',nIoU='nIoU_percent',F1='F1_percent',Pd='Pd_percent',Fa='Fa_per_million').items()}
                check(actual,expected)
                result=dict(dataset=dataset,method=method,labels=50,status='verified',values=actual,
                    source=choice['source'],configuration=choice['configuration'],previous_source=choice['previous_source'],
                    previous_values=choice['previous_values'],selection_mode=choice.get('selection_mode'),
                    selection_scope=choice.get('selection_scope'),test_selection=choice.get('test_selection',False))
            elif dataset=='SIRST' and corrected_ready:
                summary_path=corrected_root/dataset/method/'metrics/summary.json'
                detail_path=summary_path.parent/'per_image.csv'
                summary=json.loads(summary_path.read_text(encoding='utf-8'))
                assert summary['complete_test'] and summary['legacy_binary_predictions_equal_images']==86
                detail=read(detail_path);actual=metrics(detail)
                expected={k:summary[f] for k,f in dict(IoU='IoU_percent',nIoU='nIoU_percent',F1='F1_percent',Pd='Pd_percent',Fa='Fa_per_million').items()}
                check(actual,expected)
                original=lookup.get((dataset,method),{})
                result=dict(dataset=dataset,method=method,labels=summary['actual_annotation_budget'],
                    status='replication_unverified' if '退化' in original.get('review','') else 'verified',
                    values=actual,source=summary_path.relative_to(ROOT).as_posix(),
                    spatial_protocol=summary['spatial_protocol'],review=original.get('review',''),
                    epochs=original.get('epochs',120),schedule_status=original.get('schedule_status','fixed120'),
                    previous_values=summary['historical'])
            elif method=='Ours':
                folder=ROOT/f'outputs/final6108_target_domains_20261101/{dataset}/metrics'
                summary_path=folder/'summary.json'
                detail_path=folder/'per_image.csv'
                result=dict(dataset=dataset,method=method,labels=50,status='pending',values=None,
                            source=summary_path.relative_to(ROOT).as_posix())
                if summary_path.exists():
                    summary=json.loads(summary_path.read_text(encoding='utf-8'))
                    assert summary.get('complete_test',summary.get('complete',False))
                    if dataset!='IRSTD-1k':summary=summary['metrics']['K5']
                    detail=read(detail_path)
                    actual=metrics(detail)
                    expected={k:summary[f] for k,f in dict(IoU='IoU_percent',nIoU='nIoU_percent',F1='F1_percent',Pd='Pd_percent',Fa='Fa_per_million').items()}
                    check(actual,expected)
                    result.update(status='verified',values=actual)
            else:
                original=lookup[dataset,method]
                summary_path=ROOT/original['source']
                detail_path=summary_path.parent/'per_image.csv'
                detail=read(detail_path);actual=metrics(detail);check(actual,original)
                result=dict(dataset=dataset,method=method,labels=int(original['annotation_budget']),
                            status='replication_unverified' if '退化' in original['review'] else 'verified',
                            values=actual,source=original['source'],review=original['review'],
                            epochs=original['epochs'],schedule_status=original['schedule_status'])
            if (dataset,method) in external_choices:
                choice=external_choices[dataset,method]
                assert result['status']=='verified'
                check(result['values'],choice['previous_result']['values'])
                assert result['source']==choice['previous_result']['source']
                assert json.loads((ROOT/choice['audit']).read_text(encoding='utf-8'))['passed']
                summary=json.loads((ROOT/choice['source']).read_text(encoding='utf-8'))
                assert summary['complete_test'] and summary['epochs']==choice['epochs']==125
                assert summary['method']==method
                detail_path=ROOT/choice['per_image']
                assert hashlib.sha256(detail_path.read_bytes()).hexdigest()==choice['per_image_sha256']
                detail=read(detail_path);actual=metrics(detail)
                check(actual,{k:summary[f] for k,f in dict(IoU='IoU_percent',nIoU='nIoU_percent',F1='F1_percent',Pd='Pd_percent',Fa='Fa_per_million').items()})
                previous_result=dict(result)
                result.update(values=actual,source=choice['source'],epochs=125,
                              schedule_status='user_selected_epoch125_display',
                              previous_result=previous_result,table_override=choice,
                              test_based_display_selection=True)
            if result['values'] is not None:
                assert len(detail)==len({r['sample_id'] for r in detail})==len(manifest['test_ids'])
                assert {r['sample_id'] for r in detail}==set(manifest['test_ids'])
            results.append(result)
            result.update(metadata(method))
    from scripts.render_final_comparison_table import render
    render(results, lookup, overrides, comparison_overrides, corrected_ready)


if __name__=="__main__":main()
