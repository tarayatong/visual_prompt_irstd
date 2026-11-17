"""Combine the completed19-token memory and two-scale feature-radial modules."""
import types

import torch

from visual_prompt.models.tiny_decoder_feature_radial_gated import install_decoder_feature_radial_gated


def install_prototype_feature_radial(
    model, *, active_stages=(0, 1), neighbor_values=False, fixed_gate=False, fixed_scale=None,
):
    assert hasattr(model, 'prototype_memory')
    # Reuse the EXACT learned radial module and decoder hooks of the60.38 arm.
    # Only redirect its query activation to the prototype-memory decode entry.
    unused_native_decode = model._compute_decoder_out_w_mem
    install_decoder_feature_radial_gated(
        model, 'feature_decoder_gated', torch.zeros(64), torch.ones(64),
        active_stages=active_stages, neighbor_values=neighbor_values, fixed_gate=fixed_gate,
        fixed_scale=fixed_scale)
    model._compute_decoder_out_w_mem = unused_native_decode
    branch = model.radial_branch
    prototype_decode = model.prototype_memory.decode

    def decode(interface, model, backbone, index, encoded):
        assert branch.active is None and branch.inputs is not None
        assert branch.cursor < len(branch.inputs)
        branch.active = branch.inputs[branch.cursor]
        branch.cursor += 1
        try:
            return prototype_decode(model, backbone, index, encoded)
        finally:
            branch.active = None
            branch.encoded = None

    model.prototype_memory.decode = types.MethodType(decode, model.prototype_memory)
    return model
