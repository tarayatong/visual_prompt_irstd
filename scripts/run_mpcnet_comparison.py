"""MPCNet only: three fixed train50 splits, epoch120, isolated serial queue."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/comparison_mpcnet_20261105'
sys.path[:0] = [str(ROOT/'outputs/comparison_50shot_5support/runtime_packages'),
               str(ROOT/'outputs/comparison_recent_a_20261104/runtime_packages')]
import psutil
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from scripts import run_recent_a_comparison as core
from scripts.prepare_comparison import write_json


def setup(method, manifest, device, smoke=False):
    assert method == 'MPCNet'
    size = manifest.get('working_size', 256)
    source = RUN/'sources/MPCNet'
    sys.path.insert(0, str(source))
    from MPCNet import MPCNet
    from loss import SoftIoULoss
    from utils import weights_init_kaiming
    model = MPCNet(Train=True, size=size).to(device)
    model.apply(weights_init_kaiming)
    criterion = SoftIoULoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 120, eta_min=1e-4)
    stats = core.train_stats(manifest)

    class FixedTrain(Dataset):
        def __len__(self): return len(manifest['train_ids'])
        def __getitem__(self, index):
            sid = manifest['train_ids'][index]
            x, y = core.raw_pair(manifest, sid, size)
            if random.random() < .5: x, y = x[::-1], y[::-1]
            if random.random() < .5: x, y = x[:, ::-1], y[:, ::-1]
            if random.random() < .5: x, y = x.T, y.T
            x = (x-stats['mean'])/stats['std']
            return (torch.from_numpy(np.ascontiguousarray(x))[None],
                    torch.from_numpy(np.ascontiguousarray(y))[None], sid)

    loader = DataLoader(FixedTrain(), batch_size=4, shuffle=True, num_workers=0, drop_last=False)
    def objective(batch, epoch):
        model.Train = True
        x, y, _ = batch
        return criterion(model(x.to(device)), y.to(device))
    def predict(sid):
        model.Train = False
        # Inference reads images only; the common evaluator reads GT afterwards.
        with Image.open(manifest['records'][sid]['image']) as image:
            x = np.array(image.convert('L').resize((size,size), Image.Resampling.BILINEAR), dtype=np.float32)
        x = (x-stats['mean'])/stats['std']
        return model(torch.from_numpy(x)[None,None].to(device))[0,0]
    provenance = json.loads((source/'source_provenance.json').read_text(encoding='utf-8'))
    identity = dict(source_commit=provenance['commit'], source_repository=provenance['repository'],
        source_sha256={p.name:core.sha(p) for p in source.glob('*.py')},
        adapter_sha256=core.sha(Path(__file__)), variant='MPCNet_author_train50',
        category='IRSTD', venue='IEEE TGRS', year=2026, doi=provenance['doi'],
        loss='author SoftIoULoss averaged over three sigmoid heads',
        optimizer='author Adam', lr=.002, batch_size=4, input_size=size,
        initialization='author Kaiming / BatchNorm Normal(1,.02)', pretrained=None,
        normalization='train50-only grayscale mean/std', train_only_statistics=stats,
        inference_head='author output1 sigmoid, Train=False',
        deviations=f'fixed train50 seed111; resize{size} rather than native positive crop320; batch4 rather than8; author flips/transpose; cosine horizon120 rather than1000; no per-epoch test evaluation or test-selected checkpoint')
    return model, optimizer, scheduler, loader, objective, predict, identity


def worker(dataset, smoke):
    core.RUN = RUN
    core.setup_cnn = setup
    core.worker(SimpleNamespace(method='MPCNet', dataset=dataset, device='cuda', smoke=smoke))


def dispatch(wait_pids):
    RUN.mkdir(parents=True, exist_ok=True)
    lock = RUN/'queue.lock'
    if lock.exists():
        owner = int(lock.read_text())
        if psutil.pid_exists(owner): raise RuntimeError(f'Queue already owned by PID {owner}')
        lock.unlink()
    with os.fdopen(os.open(str(lock), os.O_WRONLY|os.O_CREAT|os.O_EXCL), 'w') as f: f.write(str(os.getpid()))
    plan = dict(method='MPCNet', maximum_epochs=120, retained_epochs=[120], annotation_budget=50,
                seed=111, gpu_parallelism=1, wait_pids=wait_pids,
                jobs=[dict(method='MPCNet',dataset=d,status='queued') for d in core.DATASETS])
    def save(state, **extra):
        write_json(RUN/'training_queue.json', plan)
        write_json(RUN/'execution_status.json', dict(status=state,pid=os.getpid(),**extra))
    try:
        watched={pid:psutil.Process(pid).create_time() for pid in wait_pids if psutil.pid_exists(pid)}
        while watched:
            active={}
            for pid,created in watched.items():
                try:
                    if psutil.Process(pid).create_time()==created: active[pid]=created
                except psutil.Error: pass
            watched=active
            if watched: save('waiting_for_existing_training',wait_pids=list(watched)); time.sleep(20)
        for job in plan['jobs']:
            d=job['dataset']
            summary=RUN/d/'MPCNet/epoch_0120/metrics/summary.json'
            if summary.exists():
                from scripts.report_recent_a_comparison import verified_summary
                verified_summary(summary,core.manifest_for(d),120)
                job.update(status='complete',reused_verified_result=True);save('running');continue
            for smoke in (True,False):
                stage='gpu_preflight' if smoke else 'training'
                command=[sys.executable,'-u','-m','scripts.run_mpcnet_comparison','--dataset',d]
                if smoke:command.append('--smoke')
                job.update(status=stage,command=command)
                with (RUN/(d+'.log')).open('a',encoding='utf-8') as log:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    save(stage,current_job=d,worker_pid=child.pid)
                    code=child.wait()
                if code: raise RuntimeError(f'{d} {stage} failed with code {code}')
            from scripts.report_recent_a_comparison import verified_summary
            verified_summary(summary,core.manifest_for(d),120)
            job.update(status='complete');save('running')
            subprocess.run([sys.executable,'-m','scripts.update_mpcnet_table'],cwd=ROOT,check=True)
        save('complete',completed=3,total=3)
    except Exception as e:
        save('failed',error=str(e));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=core.DATASETS)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--wait-pids',nargs='*',type=int,default=[])
    args=parser.parse_args()
    if args.dataset:worker(args.dataset,args.smoke)
    else:dispatch(args.wait_pids)
