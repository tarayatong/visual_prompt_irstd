"""Evaluate native or 19-slot memory and two-scale radial inference."""
import argparse
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from scripts import run_recent_a_comparison as base
from visual_prompt.models.tiny_decoder_feature_radial_gated import install_decoder_feature_radial_gated
from visual_prompt.models.tiny_dense_target_correction import correct_dense_target_memory
from visual_prompt.models.tiny_prototype_feature_radial import install_prototype_feature_radial
from visual_prompt.models.tiny_prototype_memory_replacement import install_prototype_memory
from visual_prompt.models.tiny_shared_memory import build_episode_keys, support_regions


def make_predictor(model, manifest, memory, radial, *, target_correction=0., foreground_gain=0., support_batch_size=None):
    size = manifest.get('working_size', 640)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]

    def normalize(image):
        return (torch.from_numpy(image.transpose(2, 0, 1) / 255) - mean) / std

    def encode(images):
        processed, _, _, sizes = model._preprocess_visual_features(
            images[None].to(model.device), model.sam.image_size)
        return model._forward_backbone(processed, sizes)

    ids = manifest['support_ids']
    pairs = [base.raw_pair(manifest, sid, size, True) for sid in ids]
    if memory == 'native':
        from util.promptable_utils import rescale_prompt

        bank = {}
        batch_size = support_batch_size or len(pairs)
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start+batch_size]
            support = encode(torch.stack([normalize(image) for image, _ in chunk]))
            for index, (_, mask) in enumerate(chunk):
                mask_prompt = rescale_prompt(torch.from_numpy(mask)[None, None].to(model.device),
                                             'mask', mask.shape, model.sam.image_size)
                decoded = model.sam._use_mask_as_output(support, mask_prompt, index)
                bank[start+index] = model._compute_memory_bank_dict(decoded, support, index)
            del support
        if target_correction or foreground_gain:
            correct_dense_target_memory(bank, pairs, target_correction, foreground_gain)

        def decode(query):
            return model._compute_decoder_out_w_mem(query, 0, len(ids), bank)
    else:
        support = encode(torch.stack([normalize(image) for image, _ in pairs]))
        geometry = [support_regions(image.astype(np.uint8), mask > 0) for image, mask in pairs]
        keys = build_episode_keys(model, support, range(len(ids)), geometry)
        encoded = model.prototype_memory.prepare(keys)

        def decode(query):
            return model.prototype_memory.decode(model, query, 0, encoded)

    def predict(sid):
        with Image.open(manifest['records'][sid]['image']) as image:
            array = np.array(image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR),
                             dtype=np.float32)
        query = encode(normalize(array)[None])
        if radial:
            model.radial_branch.begin([(None, None)])
        decoded = decode(query)
        if radial:
            model.radial_branch.finish()
        logits = F.interpolate(decoded.masks.float(), (size, size), mode='bilinear', align_corners=False)
        return logits[0, 0].sigmoid()

    return predict


@torch.inference_mode()
def evaluate(args, manifest, checkpoint, checkpoint_sha, memory, radial, output):
    base.seed_all()
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cpu':
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    model, _, _, _, _, _, identity = base.setup_sansa(manifest, device)
    fixed_scale = args.scale if radial == 'fixed' else None
    stages = {'both': (0, 1), '128': (0,), '256': (1,)}[args.fixed_radial_stages]
    if memory == 'native':
        if radial != 'off':
            install_decoder_feature_radial_gated(model, 'feature_decoder_gated',
                                                 torch.zeros(64), torch.ones(64),
                                                 fixed_scale=fixed_scale, active_stages=stages)
    else:
        install_prototype_memory(model, fixed=memory == 'fixed')
        if radial != 'off':
            install_prototype_feature_radial(model, fixed_scale=fixed_scale, active_stages=stages)
    if checkpoint is not None:
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        if radial == 'learned':
            names.update(name for name in model.state_dict() if name.startswith('radial_branch.'))
        weights = checkpoint['model']
        if missing := names - weights.keys():
            raise ValueError(f'Checkpoint is missing tensors for {memory}/{radial}: {sorted(missing)}')
        model.load_state_dict({name: weights[name] for name in names}, strict=False)
    model.eval()
    predict = make_predictor(model, manifest, memory, radial != 'off',
                             target_correction=args.target_correction, foreground_gain=args.foreground_gain)
    interface = getattr(model, 'prototype_memory', None)
    config = dict(method=output.name, dataset=args.dataset,
                  memory=memory, radial=radial, fixed_radial_scale=fixed_scale,
                  evaluation_split=args.split,
                  evaluation_device=device.type,
                  fixed_radial_stage_sizes=[(128, 256)[stage] for stage in stages] if radial == 'fixed' else None,
                  target_correction=args.target_correction, foreground_gain=args.foreground_gain,
                  support_ids=manifest['support_ids'], memory_slots=19 if interface is not None else None,
                  active_memory_tokens=interface.last_count if interface is not None else None,
                  memory_parameters=sum(p.numel() for p in interface.parameters()) if interface is not None else None,
                  radial_parameters=sum(p.numel() for p in model.radial_branch.parameters()) if radial != 'off' else 0,
                  manifest_sha256=manifest['manifest_sha256'],
                  training_horizon=checkpoint['epoch'] if checkpoint is not None else 0,
                  adapter_source='checkpoint' if checkpoint is not None else 'identity initialization',
                  initialization_sha256=identity['initialization_sha256'])
    base.evaluate(model, predict, manifest, output, config, config['training_horizon'], checkpoint_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=base.DATASETS, default='SIRST')
    parser.add_argument('--checkpoint', type=Path, help='Retained combined checkpoint with learned weights')
    parser.add_argument('--memory', choices=('native', 'fixed', 'learned', 'both'), default='fixed')
    parser.add_argument('--radial', choices=('off', 'fixed', 'learned', 'both'), default='fixed')
    parser.add_argument('--scale', type=float, default=.1)
    parser.add_argument('--fixed-radial-stages', choices=('both', '128', '256'), default='both')
    parser.add_argument('--split', choices=('test', 'support_holdout'), default='test',
                        help='Use the 45 non-support images from the 50-image allocation for selection')
    parser.add_argument('--device', choices=('cuda', 'cpu'))
    parser.add_argument('--target-correction', type=float, default=0.)
    parser.add_argument('--foreground-gain', type=float, default=0.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for value in (args.target_correction, args.foreground_gain):
        if not math.isfinite(value) or value < 0:
            parser.error('Dense memory coefficients must be finite and nonnegative')
    if (args.target_correction or args.foreground_gain) and args.memory != 'native':
        parser.error('Dense memory adjustments require --memory native')
    if args.fixed_radial_stages != 'both' and args.radial != 'fixed':
        parser.error('Selecting a fixed radial stage requires --radial fixed')
    if args.checkpoint is None and ({args.memory, args.radial} & {'learned', 'both'}):
        parser.error('Learned components require --checkpoint')
    manifest = base.manifest_for(args.dataset)
    if args.split == 'support_holdout':
        supports = set(manifest['support_ids'])
        manifest = dict(manifest, test_ids=[sid for sid in manifest['train_ids'] if sid not in supports])
        args.output /= 'support_holdout'
    checkpoint = checkpoint_sha = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if checkpoint['config']['manifest_sha256'] != manifest['manifest_sha256']:
            parser.error('Checkpoint and evaluation manifest differ')
        checkpoint_sha = base.sha(args.checkpoint)
    memory_modes = ('learned', 'fixed') if args.memory == 'both' else (args.memory,)
    radial_modes = ('learned', 'fixed') if args.radial == 'both' else (args.radial,)
    for memory in memory_modes:
        for radial in radial_modes:
            name = f'memory_{memory}_radial_{radial}'
            if radial == 'fixed' and args.fixed_radial_stages != 'both':
                name += f'_stage_{args.fixed_radial_stages}'
            if radial == 'fixed' and args.scale != .1:
                name += f'_scale_{args.scale:g}'
            if args.target_correction or args.foreground_gain:
                name += f'_correction_{args.target_correction:g}_gain_{args.foreground_gain:g}'
            evaluate(args, manifest, checkpoint, checkpoint_sha, memory, radial,
                     args.output / name)


if __name__ == '__main__':
    main()
