"""Query-only radial injections on top of the19-prototype memory model.

H0 arm: before the decoder's last upscaling activation.
Decoder arm: after that activation, before native mask-token dot products.
Both arms use the SAME zero-initialized64->32 projection and fixed descriptor.
"""
import types

from torch import nn

from visual_prompt.models.radial_sam_fusion import RadialSAMFusion


class QueryRadialInjection(nn.Module):
    def __init__(self, mode, mean, std):
        super().__init__()
        if mode not in ('h0', 'decoder'):
            raise ValueError(mode)
        self.mode = mode
        self.fusion = RadialSAMFusion(32, mean, std)
        self.inputs = None
        self.active = None
        self.cursor = 0
        self.applied = 0

    def begin(self, inputs):
        assert self.active is None
        self.inputs = inputs
        self.cursor = 0
        self.applied = 0

    def finish(self):
        assert self.active is None
        assert self.inputs is not None
        assert self.cursor == self.applied == len(self.inputs), (self.cursor, self.applied, len(self.inputs))
        self.inputs = None

    def inject(self, features):
        if self.active is None:
            return features
        assert features.shape[1:] == (32, 256, 256)
        self.applied += 1
        return self.fusion(features, *self.active)


def install_prototype_radial(model, mode, mean, std):
    branch = QueryRadialInjection(mode, mean, std).to(model.device)
    model.add_module('radial_branch', branch)
    native_decode = model.prototype_memory.decode

    def decode(interface, model, backbone, index, encoded):
        assert branch.active is None and branch.inputs is not None
        assert branch.cursor < len(branch.inputs)
        branch.active = branch.inputs[branch.cursor]
        branch.cursor += 1
        try:
            return native_decode(model, backbone, index, encoded)
        finally:
            branch.active = None

    model.prototype_memory.decode = types.MethodType(decode, model.prototype_memory)
    decoder = model.sam.sam_mask_decoder
    if mode == 'h0':
        def before_decoder(module, args, kwargs):
            if branch.active is not None:
                high = list(kwargs['high_res_features'])
                high[0] = branch.inject(high[0])
                kwargs = dict(kwargs, high_res_features=high)
            return args, kwargs
        decoder.register_forward_pre_hook(before_decoder, with_kwargs=True)
    else:
        # Native high-res decoder unpacks output_upscaling and invokes its
        # children directly; hook the actual final activation, not Sequential.
        def after_activation(module, args, output):
            return branch.inject(output)
        decoder.output_upscaling[-1].register_forward_hook(after_activation)
    return model
