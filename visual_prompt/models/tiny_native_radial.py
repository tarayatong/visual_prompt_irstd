"""Same radial arms on the native spatial-memory SAM2/SANSA baseline."""
import types

from visual_prompt.models.tiny_prototype_radial import QueryRadialInjection


def install_native_radial(model, mode, mean, std):
    branch = QueryRadialInjection(mode, mean, std).to(model.device)
    model.add_module('radial_branch', branch)
    native_decode = model._compute_decoder_out_w_mem

    def decode(self, backbone, index, memory_index, memory_bank):
        assert branch.active is None and branch.inputs is not None
        assert branch.cursor < len(branch.inputs)
        branch.active = branch.inputs[branch.cursor]
        branch.cursor += 1
        try:
            return native_decode(backbone, index, memory_index, memory_bank)
        finally:
            branch.active = None

    model._compute_decoder_out_w_mem = types.MethodType(decode, model)
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
        def after_activation(module, args, output):
            return branch.inject(output)
        decoder.output_upscaling[-1].register_forward_hook(after_activation)
    return model
