"""Use the author's mathematical PyTorch reference for the missing CUDA kernel."""
import importlib
import sys
import types


def enable_author_reference_backend():
    # The author reference lives beside its compiled Function, whose module
    # eagerly imports the extension even when only the reference is needed.
    if "MultiScaleDeformableAttention" not in sys.modules:
        backend = types.ModuleType("MultiScaleDeformableAttention")
        def unavailable(*args, **kwargs):
            raise RuntimeError("Compiled MSDA unavailable; reference adapter must be used")
        backend.ms_deform_attn_forward = unavailable
        backend.ms_deform_attn_backward = unavailable
        sys.modules["MultiScaleDeformableAttention"] = backend
    functions = importlib.import_module("SFS_MSDeformAttn.ops.functions.ms_deform_attn_func")
    native_reference = functions.ms_deform_attn_core_pytorch
    class ReferenceFunction:
        @staticmethod
        def apply(value, shapes, starts, locations, weights, im2col_step):
            # starts/im2col_step only control kernel layout/tiling; the native
            # reference derives level partitions from shapes and uses autograd.
            return native_reference(value, shapes, locations, weights)
    module = importlib.import_module("SFS_MSDeformAttn.ops.modules.ms_deform_attn")
    module.MSDeformAttnFunction = ReferenceFunction
    return native_reference
