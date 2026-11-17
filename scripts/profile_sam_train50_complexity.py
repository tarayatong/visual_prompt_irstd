"""Align the complexity table to the new retained IRSTD epoch120 weights."""
import argparse
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.profile_public_irstd_models import build,read,save,sha,gpu_state,manifest
from scripts.train_sam_extension_train50 import RUN,image_only
import torch
from torch.utils.flop_counter import FlopCounterMode


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(111)
    folder=RUN/'IRSTD-1k'/a.method;config=read(folder/'config.json');cp=folder/'checkpoints/epoch_0120.pt'
    model,forward,identity=build(a.method,task_weights=False)
    state=torch.load(cp,map_location='cpu',weights_only=False)
    assert state['epoch']==120 and state['config']==config
    model.load_state_dict(state['model'],strict=True);model.eval()
    total=sum(p.numel() for p in model.parameters());trainable=config['parameters_trainable']
    model.requires_grad_(False)
    m=manifest('IRSTD-1k');ids=m['test_ids'][:10]
    inputs=[image_only(m,sid,config['input_size'],a.method,config['train_only_statistics'])[None] for sid in ids]
    result=dict(identity,method=a.method,dataset='IRSTD-1k',parameters_total=total,parameters_trainable=trainable,
        input_geometry=f'RGB{config["input_size"]}; train50 seed111 epoch120; '+config['variant'],input_size=config['input_size'],
        profile_epoch=120,checkpoint=cp.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(cp),
        weights_status='project IRSTD-1k train50 seed111 epoch120 task checkpoint',
        component_parameters={k:sum(p.numel() for p in v.parameters()) for k,v in model.named_children()},
        flop_convention='FMA=2; torch FlopCounterMode supported convolution/matmul/attention; elementwise/interpolation/FFT/reference scan recurrence excluded',
        torch_version=torch.__version__,cuda_runtime=torch.version.cuda,profile_runner_sha256=sha(Path(__file__)),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark)
    with torch.inference_mode():
        with FlopCounterMode(display=False) as counter:y=forward(inputs[0])
        assert torch.isfinite(y).all()
        result.update(flops=counter.get_total_flops(),flops_device='cpu',flops_sample_ids=ids[:1],output_finite=True,output_shape=list(y.shape))
        before=gpu_state();result['gpu_before']=before
        if before['utilization']>10:
            result.update(timing_status='GPU busy; retry later',concurrent_gpu_load=True)
        else:
            model.cuda();inputs=[x.cuda() for x in inputs]
            for x in inputs[:3]:y=forward(x)
            assert torch.isfinite(y).all();torch.cuda.synchronize();elapsed=[]
            for _ in range(3):
                t=time.perf_counter()
                for x in inputs:y=forward(x)
                torch.cuda.synchronize();elapsed.append(time.perf_counter()-t)
            result.update(fps=10/statistics.median(elapsed),gpu=before['name'],gpu_after=gpu_state(),timing_repeat_seconds=elapsed,
                timing_query_ids=ids,batch_size_queries=1,timing_status='measured',concurrent_gpu_load=False,gpu_output_finite=bool(torch.isfinite(y).all()),
                timing_scope='GPU-resident input to probability, all branches, 3 warmups,10 unique queries x3; backend explicitly recorded')
    save(RUN/'trained_complexity'/f'{a.method}.json',result)
    if result.get('fps'):
        out=ROOT/'outputs/top8_analysis_20261105/complexity';old=out/f'{a.method}.json'
        archive=out/'public_release_initial_profiles'/old.name
        if old.exists() and not archive.exists():archive.parent.mkdir(exist_ok=True);shutil.copy2(old,archive)
        save(old,result)
    print(json.dumps({k:result.get(k) for k in ['method','input_size','profile_epoch','flops','fps','timing_status']}),flush=True)

if __name__=='__main__':main()
