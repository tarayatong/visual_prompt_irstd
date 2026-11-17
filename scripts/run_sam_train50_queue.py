"""Persist/resume the user-authorized 12-job train50 epoch120 queue."""
import json
from pathlib import Path
import subprocess
import sys
import time
import os
from datetime import datetime,timezone
import msvcrt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/comparison_sam_train50_20261109'
METHODS=('IRPruneDet','IR-SAM2','IRSAM','SAMamba')
DATASETS=('SIRST','IRSTD-1k','NUDT-SIRST')

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(path)

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'queue.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    jobs=[dict(method=m,dataset=d,status='pending') for m in METHODS for d in DATASETS]
    protocol=dict(methods=METHODS,datasets=DATASETS,annotation_budget_per_domain=50,seed=111,epochs=120,
        user_confirmed='2026-11-09: train each dataset50, seed111, epoch120 then evaluate',
        test_counts={'SIRST':86,'IRSTD-1k':201,'NUDT-SIRST':664},training_runner='scripts/train_sam_extension_train50.py',
        main_results='docs/comparison_three_datasets.md',samamba_input_size=256,selection='epoch120 only',
        initializes_from_target_task_weights=False)
    save(RUN/'protocol.json',protocol)
    state=dict(status='running',pid=os.getpid(),jobs=jobs)
    def update():
        state['updated_utc']=datetime.now(timezone.utc).isoformat();save(RUN/'queue_status.json',state)
    for job in jobs:
        method=job['method'];ds=job['dataset'];folder=RUN/ds/method
        final=folder/'epoch_0120/metrics/summary.json'
        if final.exists():
            job['status']='complete';update();continue
        logs=RUN/'logs';logs.mkdir(exist_ok=True)
        logpath=logs/f'{method}_{ds}.log'
        with logpath.open('a',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/train_sam_extension_train50.py'),'--method',method,'--dataset',ds],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            job.update(status='running',pid=child.pid,log=str(logpath.relative_to(ROOT)));state['active']=job.copy();update()
            print(json.dumps(dict(event='start',method=method,dataset=ds,pid=child.pid)),flush=True)
            last=0
            while child.poll() is None:
                if time.monotonic()-last>=30:
                    status=folder/'status.json'
                    if status.exists():
                        try:
                            detail=json.loads(status.read_text(encoding='utf-8'));job['epoch']=detail.get('epoch',0)
                        except (OSError,json.JSONDecodeError):pass
                    state['active']=job.copy();update();last=time.monotonic()
                    print(json.dumps(dict(event='progress',method=method,dataset=ds,epoch=job.get('epoch',0))),flush=True)
                time.sleep(1)
        job.update(status='complete' if child.returncode==0 and final.exists() else 'failed',exit_code=child.returncode)
        state['active']=None;update()
        if job['status']=='complete':
            audit=subprocess.run([sys.executable,'-u',str(ROOT/'scripts/update_sam_extension_metrics.py')],cwd=ROOT,capture_output=True,text=True)
            job['audit_exit_code']=audit.returncode
            if audit.returncode:
                job['status']='verification_failed';job['audit_error']=audit.stderr[-3000:]
            else:print(audit.stdout.strip(),flush=True)
        update();print(json.dumps(dict(event='finish',**job)),flush=True)
    state['status']='complete' if all(j['status']=='complete' for j in jobs) else 'needs_attention';update()
    print(json.dumps(state,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
