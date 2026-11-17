"""Count forward FLOPs for the four Tiny ablation architectures on CPU."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count

from scripts import run_recent_a_comparison as base
from scripts.run_tiny_ablation_three_seeds import read, write
from visual_prompt.models.tiny_shared_memory import support_regions, build_episode_keys
from visual_prompt.models.tiny_prototype_memory_replacement import install_prototype_memory
from visual_prompt.models.tiny_prototype_feature_radial import install_prototype_feature_radial
from visual_prompt.models.tiny_decoder_feature_radial_gated import install_decoder_feature_radial_gated


def cpu_sdpa_flops(query, key, value, *args, **kwargs):
    return sdpa_flop_count(query, key, value)


def count(call):
    # CPU fused SDPA is not registered in torch 2.7's default registry.
    mapping = {torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: cpu_sdpa_flops}
    with FlopCounterMode(display=False, custom_mapping=mapping) as counter:
        output = call()
    return output, counter.get_total_flops()


def profile(dataset, method, manifest, expected):
    base.seed_all()
    model, *_ = base.setup_sansa(manifest, torch.device('cpu'))
    from util.promptable_utils import build_prompt_dict, rescale_prompt

    use_memory = method in ('memory', 'combined')
    use_radial = method in ('radial', 'combined')
    if use_memory:
        install_prototype_memory(model)
    if method == 'combined':
        install_prototype_feature_radial(model)
    elif use_radial:
        install_decoder_feature_radial_gated(model, 'feature_decoder_gated', torch.zeros(64), torch.ones(64))
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert (params, trainable) == (expected['parameters_total'], expected['parameters_trainable'])
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]

    def normalize(image):
        return (torch.from_numpy(image.transpose(2, 0, 1)/255)-mean)/std

    ids = manifest['support_ids']
    pairs = [base.raw_pair(manifest, sid, 640, True) for sid in ids]
    supports = torch.stack([normalize(image) for image, _ in pairs])[None]
    masks = torch.stack([torch.from_numpy(mask) for _, mask in pairs])[None]
    regions = [support_regions(image.astype(np.uint8), mask > 0) for image, mask in pairs] if use_memory else None
    sid = manifest['test_ids'][0]
    with Image.open(manifest['records'][sid]['image']) as image:
        query = normalize(np.asarray(image.convert('RGB').resize((640, 640), Image.Resampling.BILINEAR),
                                     dtype=np.float32))[None, None]

    def encode(frames):
        processed, _, _, sizes = model._preprocess_visual_features(frames, 1024)
        return model._forward_backbone(processed, sizes), sizes

    def prepare_support():
        backbone, sizes = encode(supports)
        if use_memory:
            keys = build_episode_keys(model, backbone, range(5), regions)
            return model.prototype_memory.prepare(keys)
        prompt = build_prompt_dict(masks, 'mask', 5, False, torch.device('cpu'))
        bank = {}
        for index in range(5):
            mask = rescale_prompt(prompt[0][index]['prompt'], 'mask', sizes[index], 1024)
            decoded = model.sam._use_mask_as_output(backbone, mask, index)
            bank[index] = model._compute_memory_bank_dict(decoded, backbone, index)
        return bank

    def decode(backbone, cache):
        if use_radial:
            model.radial_branch.begin([(None, None)])
        if use_memory:
            result = model.prototype_memory.decode(model, backbone, 0, cache)
        else:
            result = model._compute_decoder_out_w_mem(backbone, 0, 5, cache)
        if use_radial:
            model.radial_branch.finish()
        return result.masks

    with torch.inference_mode():
        cache, support_flops = count(prepare_support)
        (backbone, _), encoder_flops = count(lambda: encode(query))
        output, decoder_flops = count(lambda: decode(backbone, cache))
        assert torch.isfinite(output).all()
    return dict(dataset=dataset, method=method, query_id=sid, support_ids=ids,
                valid_prototype_tokens=model.prototype_memory.last_count if use_memory else None,
                parameters_total=params, parameters_trainable=trainable,
                support_setup_flops=support_flops, query_encoder_flops=encoder_flops,
                query_decode_flops=decoder_flops, cached_query_flops=encoder_flops+decoder_flops)


def main():
    reference = read(ROOT/'outputs/tiny_ablation_three_seeds_20261110/summary.json')
    expected = {cell['method']: cell for cell in reference['cells'] if cell['seed'] == 113}
    result = dict(status='partial', torch_version=torch.__version__, device='CPU FP32',
                  inputs='640x640 input, SAM2 internal 1024x1024; one query, five cached supports',
                  weights='Architecture and valid token shapes profiled with initial weights; no checkpoint needed',
                  convention='FMA=2; supported convolution, matrix multiply and SDPA. Excludes grid_sample, '
                             'interpolation, normalization, elementwise and CPU image processing.',
                  rows=[])
    path = ROOT/'outputs/tiny_ablation_other_datasets_20261111/efficiency_flops.json'
    for dataset in ('SIRST', 'NUDT-SIRST', 'IRSTD-1k'):
        manifest = base.manifest_for(dataset)
        for method in ('native', 'memory', 'radial', 'combined'):
            row = profile(dataset, method, manifest, expected[method])
            result['rows'].append(row)
            write(path, result)
            print(row, flush=True)
    result['status'] = 'complete'
    write(path, result)


if __name__ == '__main__':
    main()
