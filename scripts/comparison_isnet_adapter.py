"""ISNet native DCNv2 modules backed by torchvision's compiled equivalent op."""
import importlib
import sys
import types
import torch
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


def enable_torchvision_dcn(root):
    source = root / "sources/ISNet/model"
    sys.path[:0] = [str(source), str(source / "DCNv2")]
    sys.path.append(str((root / "runtime_packages").resolve()))
    if "_ext" not in sys.modules:
        sys.modules["_ext"] = types.ModuleType("_ext")
    native = importlib.import_module("dcn_v2")
    def operation(x, offset, mask, weight, bias, stride, padding, dilation, deformable_groups):
        assert offset.shape[1] == 2 * deformable_groups * weight.shape[-2] * weight.shape[-1]
        return deform_conv2d(x, offset, weight, bias, stride, padding, dilation, mask)
    native.dcn_v2_conv = operation
    return operation


def gradient_magnitude(x):
    """Exact author's no-padding-change image/label gradient, device-neutral."""
    channels = x.shape[1]
    vertical = x.new_tensor([[0,-1,0],[0,0,0],[0,1,0]])[None,None].repeat(channels,1,1,1)
    horizontal = vertical.transpose(2,3).contiguous()
    v = F.conv2d(x, vertical, padding=1, groups=channels)
    h = F.conv2d(x, horizontal, padding=1, groups=channels)
    return torch.sqrt(v.square() + h.square() + 1e-6)


class NativeISNetOutputs(torch.nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native

    def forward(self, x, warm_flag):
        prediction, edge = self.native(x, gradient_magnitude(x))
        self.edge_prediction = edge if self.training else None
        return [], prediction
