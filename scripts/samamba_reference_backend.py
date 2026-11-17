"""Windows measurement adapter for official Mamba-1, with an explicit reference backend.

This is not the CUDA fused kernel. Keep reference FPS labelled separately.
The scan uses the same real-valued recurrence, evaluated by an associative prefix
scan. Verify against the upstream sequential selective_scan_ref before use.
"""
import ast
from functools import partial
import importlib.util
import sys
import types
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat


def _paired_prefix_states(a, b):
    """Compute the same recurrent states with a linear-size reduction tree."""
    length = b.shape[-1]
    if length == 1:
        return b
    pair_count = length // 2
    a_even, b_even = a[..., ::2], b[..., ::2]
    a_odd, b_odd = a[..., 1::2], b[..., 1::2]
    odd_states = _paired_prefix_states(a_odd * a_even[..., :pair_count],
                                      b_odd + a_odd * b_even[..., :pair_count])
    preceding_odd = odd_states[..., :b_even.shape[-1] - 1]
    even_states = torch.cat((b_even[..., :1],
                             b_even[..., 1:] + a_even[..., 1:] * preceding_odd), dim=-1)
    paired = torch.stack((even_states[..., :pair_count], odd_states), dim=-1).flatten(-2)
    if length % 2:
        return torch.cat((paired, even_states[..., -1:]), dim=-1)
    return paired


def prefix_scan(u,delta,A,B,C,D=None,z=None,delta_bias=None,delta_softplus=False,return_last_state=False,*,compact=False):
    assert not A.is_complex() and B.ndim==3 and C.ndim==3
    dtype=u.dtype;u=u.float();delta=delta.float()
    if delta_bias is not None:delta=delta+delta_bias[...,None].float()
    if delta_softplus:delta=F.softplus(delta)
    a=torch.exp(delta[:,:,None,:]*A[None,:,:,None])
    b=delta[:,:,None,:]*B[:,None,:,:].float()*u[:,:,None,:]
    # Composition (a2,b2) o (a1,b1) = (a2*a1, b2+a2*b1).
    if compact:
        b = _paired_prefix_states(a, b)
    else:
        step=1
        while step<u.shape[-1]:
            b=torch.cat((b[...,:step], b[...,step:]+a[...,step:]*b[...,:-step]),dim=-1)
            a=torch.cat((a[...,:step], a[...,step:]*a[...,:-step]),dim=-1)
            step*=2
    state=b[...,-1]
    y=(b*C[:,None,:,:].float()).sum(dim=2)
    if D is not None:y=y+u*D[None,:,None]
    if z is not None:y=y*F.silu(z)
    y=y.to(dtype)
    return (y,state) if return_last_state else y


def compact_prefix_scan(*args, **kwargs):
    scan = partial(prefix_scan, compact=True)
    if torch.is_grad_enabled():
        return checkpoint(scan, *args, **kwargs, use_reentrant=False, preserve_rng_state=False)
    return scan(*args, **kwargs)


def install(source, *, compact=False):
    """Use the upstream Mamba class, only replacing the unavailable scan backend."""
    root=Path(source)
    def pkg(name,path):
        m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
        return m
    top=pkg('mamba_ssm',root/'mamba_ssm')
    pkg('mamba_ssm.ops',root/'mamba_ssm/ops')
    pkg('mamba_ssm.modules',root/'mamba_ssm/modules')
    scan_path=root/'mamba_ssm/ops/selective_scan_interface.py'
    tree=ast.parse(scan_path.read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='selective_scan_ref')
    env=dict(torch=torch,F=F,rearrange=rearrange,repeat=repeat)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(scan_path),'exec'),env)
    reference=env['selective_scan_ref']
    # Meaningful numerical check: dynamic B/C, gating, bias and last state.
    torch.manual_seed(111)
    u=torch.randn(2,8,129);delta=torch.randn_like(u)
    A=-torch.rand(8,16);B=torch.randn(2,16,129);C=torch.randn_like(B)
    D=torch.randn(8);z=torch.randn_like(u);bias=torch.randn(8)
    args=(u,delta,A,B,C,D,z,bias,True,True)
    with torch.no_grad():
        scan = compact_prefix_scan if compact else prefix_scan
        actual,actual_state=scan(*args);expected,expected_state=reference(*args)
    torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-4)
    torch.testing.assert_close(actual_state,expected_state,rtol=2e-4,atol=2e-4)
    parity=dict(max_absolute_error=float((actual-expected).abs().max()),
                last_state_error=float((actual_state-expected_state).abs().max()),
                atol=2e-4,rtol=2e-4,shape=list(u.shape),passed=True)
    interface=types.ModuleType('mamba_ssm.ops.selective_scan_interface')
    interface.selective_scan_fn=scan
    def unavailable(*args,**kwargs):raise RuntimeError('CUDA fused Mamba kernel unavailable on Windows')
    interface.mamba_inner_fn=unavailable
    sys.modules[interface.__name__]=interface
    name='mamba_ssm.modules.mamba_simple'
    spec=importlib.util.spec_from_file_location(name,root/'mamba_ssm/modules/mamba_simple.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    top.Mamba=mod.Mamba
    return parity
