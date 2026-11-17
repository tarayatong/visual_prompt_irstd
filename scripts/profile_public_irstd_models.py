"""Reproducible public-release complexity measurement, without training.

Run each method in an isolated process to avoid author module-name collisions.
Task checkpoints and backbone-only/initialized weights are explicitly separated.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT/'outputs/sam_extension_20261109'
OUT = ROOT/'outputs/top8_analysis_20261105/complexity'
sys.path.insert(0, str(EXP/'runtime_packages'))
from scripts.analyze_top8_comparison import read, save, sha, gpu_state, manifest


def package(name, path):
    """Load a namespace without executing an unrelated architecture registry."""
    mod=types.ModuleType(name);mod.__path__=[str(path)]
    sys.modules[name]=mod
    return mod


def build(method, task_weights=True, *, compact_scan=False):
    import torch
    source=EXP/'sources'/method
    sys.path.insert(0,str(source))
    identity=read(source/'source_provenance.json')
    identity.update(source_directory=str(source.relative_to(ROOT)), modifications=[])
    if method=='SAMamba':
        from scripts.samamba_reference_backend import install
        parity=install(EXP/'sources/mamba', compact=compact_scan)
        package('basicseg',source/'basicseg')
        package('basicseg.networks',source/'basicseg/networks')
        from basicseg.networks import SAMamba as samamba
        original=samamba.build_sam2
        samamba.build_sam2=lambda config,*args,**kwargs:original(config,device='cpu')
        model=samamba.SAMamba(checkpoint_path=None)
        ckpt=ROOT/'checkpoints/sam2.1_hiera_small.pt'
        weights=torch.load(ckpt,map_location='cpu',weights_only=True)['model']
        own=model.state_dict();loaded={}
        for k,v in own.items():
            if not k.startswith('encoder.'):continue
            key='image_encoder.trunk.'+k[len('encoder.'):]
            key=key.replace('.block.','.')
            if key in weights and v.shape==weights[key].shape:loaded[k]=weights[key]
        missing,unexpected=model.load_state_dict(loaded,strict=False)
        required={k for k in own if k.startswith('encoder.') and '.monaOp.' not in k}
        assert required <= loaded.keys(),required-loaded.keys()
        identity.update(checkpoint=str(ckpt.relative_to(ROOT)),checkpoint_sha256=sha(ckpt),
            weights_status='SAM2.1 Hiera-S pretrained trunk only; task adapters/heads seed111 initialized; task weights Google Drive not downloaded',
            loaded_pretrained_tensors=len(loaded),accuracy_reproduced=False,
            variant='author Hiera-S CSI128; PyTorch associative reference scan (not CUDA fused)',input_size=1024,
            backend='pytorch_compact_reference_scan' if compact_scan else 'pytorch_reference_scan',
            scan_numerical_parity=parity,
            mamba_source=read(EXP/'sources/mamba/source_provenance.json'),
            modifications=['Namespace-only imports avoid unrelated architecture registry; build SAM2 on CPU without absent author absolute checkpoint path',
                           'Official Mamba-1 class with unavailable CUDA selective scan replaced by numerically checked PyTorch prefix recurrence; no forward layers removed'])
        task=EXP/'weights/net_best_irstd.pth'
        if task_weights and task.exists():
            task_state=torch.load(task,map_location='cpu',weights_only=True)
            model.load_state_dict(task_state,strict=True)
            identity.update(checkpoint=str(task.relative_to(ROOT)),checkpoint_sha256=sha(task),
                weights_status='author IRSTD-1k trained task checkpoint; strict state_dict match (490 tensors)',
                task_checkpoint_strict_loaded=True,task_checkpoint_tensors=len(task_state),
                checkpoint_source=read(EXP/'weights/samamba_weight_provenance.json')['url'])
        return model,lambda x:model(x).sigmoid(),identity
    if method=='IR-SAM2':
        import yaml
        from model.base_model import make_adaptor
        cfg=yaml.safe_load((source/'model/configs/spl_t.yaml').read_text(encoding='utf-8'))
        cfg['sam_ckpt_path']=None
        model=make_adaptor(**cfg)
        ckpt=ROOT/'checkpoints/sam2.1_hiera_tiny.pt'
        weights=torch.load(ckpt,map_location='cpu',weights_only=True)['model']
        own=model.state_dict()
        loaded={k:v for k,v in weights.items() if k in own and own[k].shape==v.shape}
        required={k for k in own if k.startswith('image_encoder.trunk.') and 'promote_genertor' not in k}
        assert required <= loaded.keys(),required-loaded.keys()
        model.load_state_dict(loaded,strict=False);model._freeze_encoder()
        identity.update(checkpoint=str(ckpt.relative_to(ROOT)),checkpoint_sha256=sha(ckpt),
            weights_status='SAM2.1 tiny pretrained backbone only; task heads seed111 initialized; no task checkpoint released',
            loaded_pretrained_tensors=len(loaded),accuracy_reproduced=False,
            variant='author spl_t; warm_flag=True includes frequency enhancement',input_size=256)
        return model,lambda x:model(x,True)[0].sigmoid(),identity
    if method=='IRSAM':
        from segment_anything_training.build_IRSAM import build_sam_IRSAM
        model=build_sam_IRSAM()
        ckpt=ROOT/'checkpoints/mobile_sam.pt'
        state=torch.load(ckpt,map_location='cpu',weights_only=True)
        own=model.state_dict();loaded={k:v for k,v in state.items() if k in own and own[k].shape==v.shape}
        missing,unexpected=model.load_state_dict(loaded,strict=False)
        identity.update(checkpoint=str(ckpt.relative_to(ROOT)),checkpoint_sha256=sha(ckpt),
            weights_status='MobileSAM backbone/compatible tensors only; new task tensors seed111 initialized; task checkpoint unavailable',
            loaded_pretrained_tensors=len(loaded),missing_keys=missing,accuracy_reproduced=False,
            variant='current public release with WaveletGradientCompensation; not assumed identical to ECCV2024 architecture',input_size=512,
            encoder_internal_size=model.image_encoder.img_size)
        def forward(x):
            masks,edges=model([dict(image=x[0],original_size=tuple(x.shape[-2:]))])
            return masks.sigmoid()
        return model,forward,identity
    if method=='IRPruneDet':
        # These imports are used by training/pruning only. The released compact
        # network's forward depends solely on torch and model.fusion.
        path=source/'model/segmentation.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        removed=[ast.unparse(n) for n in tree.body if isinstance(n,ast.ImportFrom) and n.module.startswith('nni')]
        tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module.startswith('nni'))]
        mod=types.ModuleType('model.segmentation');mod.__file__=str(path)
        sys.modules[mod.__name__]=mod
        exec(compile(tree,str(path),'exec'),mod.__dict__)
        ckpt=source/'checkpoints/hard_finetuned/irprunedet_model.pkl'
        model=torch.load(ckpt,map_location='cpu',weights_only=False)
        identity.update(checkpoint=str(ckpt.relative_to(ROOT)),checkpoint_sha256=sha(ckpt),
            weights_status='author released hard-pruned finetuned NUAA-SIRST checkpoint; not project IRSTD train50',
            accuracy_reproduced=False,variant='released compact ASKCResUNet BiGlobal; docs/RESULTS.md distinguishes this from paper 0.1802M',input_size=512,
            modifications=['In-memory removal of unused training-only NNI imports; forward code unchanged: '+repr(removed)])
        return model,lambda x:model(x).sigmoid(),identity
    raise ValueError(method)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',required=True,choices=['IRSAM','IR-SAM2','IRPruneDet','SAMamba'])
    p.add_argument('--timing',action='store_true')
    a=p.parse_args()
    import torch
    import numpy as np
    from PIL import Image
    from torch.utils.flop_counter import FlopCounterMode
    torch.set_num_threads(2);torch.manual_seed(111)
    model,forward,identity=build(a.method)
    model.eval()
    total=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Preserve default training flags above; eliminate irrelevant gradient hooks
    # on raw token Parameters for FlopCounterMode under inference_mode.
    model.requires_grad_(False)
    m=manifest('IRSTD-1k');ids=m['test_ids'][:10];size=identity['input_size']
    inputs=[]
    stats=read(ROOT/'outputs/comparison_recent_a_20261104/IRSTD-1k/SAM-SPL/config.json')['train_only_statistics']
    for sid in ids:
        with Image.open(m['records'][sid]['image']) as im:
            data=np.asarray(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)
        if a.method=='IR-SAM2':data=(data-stats['mean'])/stats['std']
        elif a.method in ('IRPruneDet','SAMamba'):data=(data/255.-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
        inputs.append(torch.from_numpy(data.transpose(2,0,1).copy())[None])
    path=OUT/(a.method+'.json')
    result=read(path) if path.exists() else {}
    # Never carry timing/flops across a different checkpoint or code/backend.
    if any(result.get(k)!=identity.get(k) for k in ('checkpoint_sha256','commit','backend')):
        result={}
    result.update(method=a.method,dataset='IRSTD-1k',parameters_total=total,parameters_trainable=trainable,
        input_geometry=f'RGB{size}x{size}; '+identity['variant'],
        component_parameters={k:sum(p.numel() for p in v.parameters()) for k,v in model.named_children()},
        flop_convention='FMA=2; torch FlopCounterMode supported convolution/matmul/attention; elementwise/interpolation/FFT/CPU operations excluded',
        torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,**identity)
    save(path,result)
    print(a.method,'constructed',total,trainable,flush=True)
    with torch.inference_mode():
        if not a.timing:
            with FlopCounterMode(display=False) as counter:y=forward(inputs[0])
            result.update(flops=counter.get_total_flops(),flops_device='cpu',flops_sample_ids=ids[:1],
                flops_checkpoint_sha256=identity['checkpoint_sha256'],
                output_shape=list(y.shape),output_finite=bool(torch.isfinite(y).all()),
                output_min=float(y.min()),output_max=float(y.max()))
            assert result['output_finite']
        else:
            before=gpu_state();result['gpu_before']=before
            if before['utilization']>10:
                result.update(concurrent_gpu_load=True,timing_status='skipped: GPU busy before benchmark')
            else:
                model.cuda();inputs=[x.cuda() for x in inputs]
                for x in inputs[:3]:y=forward(x)
                assert torch.isfinite(y).all()
                torch.cuda.synchronize();elapsed=[]
                for repeat in range(3):
                    start=time.perf_counter()
                    for x in inputs:y=forward(x)
                    torch.cuda.synchronize();elapsed.append(time.perf_counter()-start)
                result.update(gpu=before['name'],gpu_after=gpu_state(),batch_size_queries=1,
                    timing_checkpoint_sha256=identity['checkpoint_sha256'],
                    fps=10/statistics.median(elapsed),timing_repeat_seconds=elapsed,timing_query_ids=ids,
                    timing_scope='GPU-resident input to probability output; all model branches; backend recorded separately; no disk read/model load; 3 warmups; 10 different images x3 rounds',
                    timing_status='measured',concurrent_gpu_load=False,
                    gpu_output_finite=bool(torch.isfinite(y).all()))
    save(path,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
