"""Combine19-token memory replacement with two-scale FEATURE-radial gated decoder.

Run --smoke first; --suite trains combined memory+feature-radial model, 120ep.
BF16 training, FP32 evaluation are identical for both arms. Last batch contains2.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from scripts import run_recent_a_comparison as base
from visual_prompt.models.tiny_shared_memory import (
    support_regions, build_episode_keys,
)
from visual_prompt.models.tiny_prototype_memory_replacement import install_prototype_memory
from visual_prompt.models.tiny_prototype_feature_radial import install_prototype_feature_radial
from visual_prompt.evaluation.comparison import image_metrics, summarize

OUT = ROOT / 'outputs/tiny_prototype_feature_radial_20261110/IRSTD-1k'
MODES = ('native', 'combined')


def autocast():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def state_for(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in names or k.startswith('radial_branch.')}


def restore(model, weights):
    assert set(weights) == set(state_for(model))
    model.load_state_dict(weights, strict=False)


def comparison():
    fields = ('IoU_percent', 'nIoU_percent', 'F1_percent', 'Pd_percent', 'Fa_per_million')
    references = {
        'native': 'outputs/tiny_shared_memory_batch4_20261110/IRSTD-1k/native/epoch_0120/metrics/summary.json',
        'memory_only': 'outputs/tiny_prototype_only_memory_20261110/IRSTD-1k/tokens/epoch_0120/metrics/summary.json',
        'feature_radial_only': 'outputs/tiny_decoder_feature_radial_gated_20261110/IRSTD-1k/feature_decoder_gated/epoch_0120/metrics/summary.json',
    }
    result = dict(batch_size=4, precision='BF16 training; FP32 evaluation', epochs=120,
                  seed=111, train_images=50, one_shared_encoder=True, results={})
    for name, relative in references.items():
        row = json.loads((ROOT/relative).read_text(encoding='utf-8'))
        assert row['batch_size']==4 and row['seed']==111 and row['epochs']==120
        result['results'][name] = {k:row[k] for k in fields}
    path = OUT/'combined/epoch_0120/metrics/summary.json'
    if path.exists():
        row = json.loads(path.read_text(encoding='utf-8'))
        values = {k:row[k] for k in fields}
        result['results']['combined'] = values
        result['delta_from'] = {name:{k:values[k]-ref[k] for k in fields}
                                for name,ref in result['results'].items() if name!='combined'}
    base.write_json(OUT/'comparison.json', result)
    return result


def worker(args):
    mode = args.variant
    folder = OUT / mode
    folder.mkdir(parents=True, exist_ok=True)
    base.seed_all()
    manifest = base.manifest_for('IRSTD-1k')
    model, unused_opt, unused_sched, unused_loader, unused_loss, native_predict, identity = base.setup_sansa(
        manifest, torch.device('cuda'))
    del unused_opt, unused_sched, unused_loader, unused_loss
    from util.promptable_utils import build_prompt_dict, rescale_prompt
    from util.losses import loss_masks

    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    pairs, geometry = {}, {}
    for sid in manifest['train_ids']:
        image, mask = base.raw_pair(manifest, sid, 640, True)
        pairs[sid] = ((torch.from_numpy(image.transpose(2, 0, 1)/255)-mean)/std,
                      torch.from_numpy(mask))
        if mode == 'combined':
            geometry[sid] = support_regions(image.astype(np.uint8), mask > 0)

    class Episodes(Dataset):
        def __len__(self):
            return 50

        def __getitem__(self, index):
            query = manifest['train_ids'][index]
            support = random.choice([sid for sid in manifest['train_ids'] if sid != query])
            selected = [pairs[support], pairs[query]]
            return (torch.stack([x for x, _ in selected]),
                    torch.stack([y for _, y in selected]), support, query)

    loader = DataLoader(Episodes(), batch_size=4, shuffle=True, num_workers=0, drop_last=False)
    assert len(loader) == 13
    sample = next(sid for sid in manifest['train_ids'] if sid not in manifest['support_ids'])
    with torch.random.fork_rng(devices=[]):
        install_prototype_memory(model)
    if args.smoke:
        model.eval()
        model.support_geometry = [[geometry[sid] for sid in manifest['support_ids']]]
        with torch.no_grad():
            memory_only_reference = native_predict(sample).float().detach().clone()
    with torch.random.fork_rng(devices=[]):
        install_prototype_feature_radial(model)

    def prepare_radial(ids):
        model.radial_branch.begin([(None,None) for _ in ids])

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all('adapter' in n or n.startswith('prototype_memory.') or n.startswith('radial_branch.')
               for n, p in model.named_parameters() if p.requires_grad)
    assert not hasattr(model, 'radial_fusion')
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 120*len(loader))

    def objective(batch):
        samples, masks, supports, queries = batch
        if mode == 'combined':
            model.support_geometry = [[geometry[sid]] for sid in supports]
        prepare_radial(list(queries))
        samples, masks = samples.cuda(), masks.cuda()
        with autocast():
            prompt = build_prompt_dict(masks, 'mask', 1, True, torch.device('cuda'))
            output = model(samples, prompt)
            model.radial_branch.finish()
            loss = sum(loss_masks(output['pred_masks'], masks[:, -1:], num_frames=1).values())
        return loss

    support_cache = {}

    @torch.no_grad()
    def predict(sid):
        with torch.autocast('cuda', enabled=False):
            if not support_cache:
                ids = manifest['support_ids']
                images = torch.stack([pairs[item][0] for item in ids])[None].cuda()
                processed, _, _, sizes = model._preprocess_visual_features(images, 1024)
                backbone = model._forward_backbone(processed, sizes)
                keys = build_episode_keys(model, backbone, range(5), [geometry[item] for item in ids])
                support_cache['encoded'] = model.prototype_memory.prepare(keys)
                support_cache['raw_keys'] = keys
            if sid in pairs:
                query = pairs[sid][0]
            else:
                with Image.open(manifest['records'][sid]['image']) as image:
                    array = np.array(image.convert('RGB').resize((640, 640), Image.Resampling.BILINEAR), dtype=np.float32)
                query = (torch.from_numpy(array.transpose(2, 0, 1)/255)-mean)/std
            processed, _, _, sizes = model._preprocess_visual_features(query[None, None].cuda(), 1024)
            backbone = model._forward_backbone(processed, sizes)
            prepare_radial([sid])
            decoded = model.prototype_memory.decode(model, backbone, 0, support_cache['encoded'])
            model.radial_branch.finish()
            logits = F.interpolate(decoded.masks.float(), (640, 640), mode='bilinear', align_corners=False)[0, 0]
            return logits.sigmoid()

    if args.smoke:
        model.eval()
        model.support_geometry = [[geometry[sid] for sid in manifest['support_ids']]]
        prepare_radial([sample])
        with torch.no_grad():
            reference_fp32 = native_predict(sample).float().detach().clone()
        model.radial_branch.finish()
        zero_error = float((reference_fp32-memory_only_reference).abs().max())
        assert torch.allclose(reference_fp32,memory_only_reference,atol=1e-6,rtol=1e-6),zero_error
        encoder_calls = []
        hook = model.sam.image_encoder.trunk.register_forward_pre_hook(
            lambda module, inputs: encoder_calls.append(int(inputs[0].shape[0])))
        with torch.no_grad():
            actual = predict(sample)
            predict(sample)
        hook.remove()
        assert encoder_calls == [5, 1, 1], encoder_calls
        error = float((reference_fp32-actual).abs().max())
        mean_error = float((reference_fp32-actual).abs().mean())
        assert error < .003 and mean_error < 1e-5, (error, mean_error)
        assert torch.isfinite(actual).all()
        model.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        batches = iter(loader)
        for step in range(3):
            batch = next(batches)
            assert batch[0].shape[0] == 4
            optimizer.zero_grad(set_to_none=True)
            loss = objective(batch)
            assert torch.isfinite(loss)
            loss.backward()
            for stage in model.radial_branch.fusion.stages:
                grad = stage.output.weight.grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)
            assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
            if mode == 'combined':
                grad = model.prototype_memory.projection.weight.grad
                assert grad is not None and grad.abs().sum() > 0
                if step == 2:
                    assert all(p.grad is not None and p.grad.abs().sum() > 0
                               for p in model.prototype_memory.parameters())
                    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.radial_branch.parameters())
            torch.nn.utils.clip_grad_norm_(trainable, .1)
            optimizer.step()
        peak = torch.cuda.max_memory_allocated()/1024**2
        result = dict(status='passed', actual_batch_size=4, accumulation_steps=1,
                      precision='BF16 training; FP32 evaluation', finite_forward_backward=True,
                      total_parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in trainable),
                      peak_allocated_MiB=peak, three_steps_seconds=time.monotonic()-begin,
                      native_dense_memory_encoder_removed=True, native_memory_attention_frozen=True,
                      radial_zero_initialization_max_probability_error=zero_error,
                      both_radial_heads_have_finite_nonzero_gradients=True,
                      valid_memory_tokens=int(support_cache['encoded'][0].shape[0]),
                      cached_support_zero_residual_max_probability_error=error,
                      cached_support_mean_probability_error=mean_error,
                      support_encoder_calls_for_two_queries=encoder_calls,
                      single_encoder=True, reference_encoder=False, radial=True)
        base.write_json(folder / 'preflight.json', result)
        print(json.dumps(dict(variant=mode, **result)), flush=True)
        return

    prior = json.loads((base.RUN / 'IRSTD-1k/SANSA/config.json').read_text(encoding='utf-8'))
    sources = [Path(__file__), ROOT / 'visual_prompt/models/tiny_shared_memory.py', ROOT / 'visual_prompt/models/tiny_prototype_memory_replacement.py', ROOT / 'visual_prompt/models/tiny_prototype_feature_radial.py', ROOT / 'visual_prompt/models/tiny_decoder_feature_radial_gated.py', ROOT / 'visual_prompt/models/tiny_decoder_radial_gated.py', ROOT / 'visual_prompt/models/tiny_radial_pre_h0.py', ROOT / 'visual_prompt/models/tiny_prototype_radial.py', ROOT / 'visual_prompt/models/radial_sam_fusion.py', Path(base.__file__),
               ROOT / 'visual_prompt/models/semantic_memory.py', ROOT / 'visual_prompt/core/semantic_direction.py']
    config = dict(prior, method='SAM2Tiny-PrototypeFeatureRadial-B4', variant=mode, batch_size=4,
                  precision='BF16 training; FP32 evaluation', seed=111, radial=True, one_shared_encoder=True,
                  radial_branch_parameters=sum(p.numel() for p in model.radial_branch.parameters()),
                  radial_source='each decoder stage feature; no raw image radial',
                  radial_stage_channels=[64,32], radial_stage_sizes=[128,256],
                  radial_radii_128grid=[.5,1,1.5,2,3,4,6,8], radial_radii_256grid=[1,2,3,4,6,8,12,16],
                  radial_compressed_channels=4, radial_descriptor_channels=256,
                  radial_directions=8, radial_residual_scale=.1, radial_output_initialization='zero',
                  warm_start='none; same pretrained weights and120ep protocol as both single-module arms',
                  memory_only_reference='outputs/tiny_prototype_only_memory_20261110/IRSTD-1k/tokens/epoch_0120/metrics/summary.json',
                  radial_only_reference='outputs/tiny_decoder_feature_radial_gated_20261110/IRSTD-1k/feature_decoder_gated/epoch_0120/metrics/summary.json',
                  reference_encoder_parameters=0, runner_sha256=base.sha(Path(__file__)),
                  source_hashes={p.relative_to(ROOT).as_posix(): base.sha(p) for p in sources},
                  parameters_total=sum(p.numel() for p in model.parameters()),
                  parameters_trainable=sum(p.numel() for p in trainable),
                  pipeline_parameters_total=sum(p.numel() for p in model.parameters()),
                  optimizer_steps_per_epoch=13, optimizer_steps_total=1560, last_batch_size=2,
                  lr=1e-4, train_support_shots=1, support_shots=5,
                  prototype_source='same current support backbone features; detached keys rebuilt per episode; no self-query support',
                  prototype_slots='2 global,8 corrected instances,9 negative; missing slots zero',
                  prototype_feature_grid=64, prototype_channels=352,
                  native_dense_memory=False, native_spatial_encoder_removed=True, native_object_pointers=False,
                  native_memory_attention_retained=True, native_memory_attention_trainable=False,
                  original_sansa_adapters_retained=True, extra_reader_residual=False,
                  fusion='19 prototype slots -> interface/type encoding -> frozen native attention -> native decoder with128/256 feature radial gated residuals',
                  semantic_key_rope=False, missing_prototype_slots='excluded from attention',
                  removed_spatial_encoder_parameters=model.removed_spatial_encoder_parameters,
                  matched_reference='outputs/tiny_shared_memory_batch4_20261110/IRSTD-1k/native/epoch_0120/metrics/summary.json',
                  query_feature_source='same current backbone H0/H1/F; shared with native SAM2',
                  evaluation_support_cache='prototype keys and interface encoding built once; no dense support memory',
                  selection='fixed epoch120 only; no test selection',
                  baseline_note='new matched batch4 BF16 native arm; historical batch1 FP32 is not a matched baseline')
    path = folder / 'config.json'
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == config
    else:
        base.write_json(path, config)
    for source in sources:
        target = folder / 'runner_sources' / (base.sha(source)+'.py')
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            target.write_bytes(source.read_bytes())
    latest = folder / 'checkpoints/latest.pt'
    latest.parent.mkdir(exist_ok=True)
    final = latest.parent / 'epoch_0120.pt'
    history, start = [], 0
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        assert state['config'] == config
        restore(model, state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history, start = state['history'], state['epoch']
        random.setstate(state['python_rng'])
        np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(start, 120):
        model.train()
        begin = time.monotonic()
        losses, ids, sizes = [], [], []
        for batch in loader:
            ids.extend(batch[-1])
            sizes.append(batch[0].shape[0])
            optimizer.zero_grad(set_to_none=True)
            loss = objective(batch)
            assert torch.isfinite(loss)
            loss.backward()
            for stage in model.radial_branch.fusion.stages:
                grad = stage.output.weight.grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)
            torch.nn.utils.clip_grad_norm_(trainable, .1)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
        assert len(ids) == 50 and set(ids) == set(manifest['train_ids'])
        assert sizes == [4]*12+[2]
        history.append(dict(epoch=epoch+1, loss=float(np.average(losses, weights=sizes)),
                            seconds=time.monotonic()-begin, unique_supervised_images=50,
                            optimizer_updates=(epoch+1)*13, batch_sizes=sizes))
        base.write_json(folder / 'training_history.json', history)
        base.write_json(folder / 'status.json', dict(status='training', epoch=epoch+1, maximum_epochs=120))
        if (epoch+1) % 5 == 0 or epoch == 119:
            state = dict(config=config, epoch=epoch+1, model=state_for(model), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), history=history,
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            temporary = latest.with_suffix('.tmp')
            torch.save(state, temporary)
            temporary.replace(latest)
            if epoch == 119:
                torch.save(state, final)
        print(f'{mode} batch4 epoch{epoch+1}/120 loss{history[-1]["loss"]:.6f} seconds{history[-1]["seconds"]:.2f}', flush=True)
    state = torch.load(final, map_location='cpu', weights_only=False)
    assert state['epoch'] == 120 and state['config'] == config
    assert [r['epoch'] for r in history] == list(range(1, 121))
    restore(model, state['model'])
    support_cache.clear()
    base.write_json(folder / 'status.json', dict(status='evaluating', epoch=120))
    base.evaluate(model, predict, manifest, folder, config, 120, base.sha(final))
    recount = []
    for sid in manifest['test_ids']:
        pred = np.array(Image.open(folder / f'epoch_0120/predictions/{sid}.png')) > 0
        height, width = manifest['records'][sid]['size_hw']
        gt = np.array(Image.open(manifest['records'][sid]['mask']).convert('L').resize(
            (width, height), Image.Resampling.NEAREST)) > 0
        assert pred.shape == gt.shape
        recount.append(image_metrics(pred, gt, sid))
    summary = folder / 'epoch_0120/metrics/summary.json'
    result = json.loads(summary.read_text(encoding='utf-8'))
    for key, value in summarize(recount).items():
        assert result[key] == value, key
    result.update(predictions_recount_verified=True, all_120_epochs_verified=True)
    base.write_json(summary, result)
    base.write_json(folder / 'status.json', dict(status='complete', epoch=120, predictions_recount_verified=True))
    comparison()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('combined',), default='combined')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--suite', action='store_true')
    args = parser.parse_args()
    if not args.suite:
        worker(args)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for smoke in (True, False):
        for mode in ('combined',):
            base.write_json(OUT / 'queue_status.json', dict(status='preflight' if smoke else 'running',
                                                           current_variant=mode, batch_size=4, radial=True))
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--variant', mode]
            if smoke:
                command.append('--smoke')
            code = subprocess.call(command, cwd=ROOT)
            if code:
                base.write_json(OUT / 'queue_status.json', dict(status='failed', variant=mode,
                                                               preflight=smoke, exit_code=code))
                raise SystemExit(code)
    base.write_json(OUT / 'queue_status.json', dict(status='complete', batch_size=4, radial=True))
    print(json.dumps(comparison()), flush=True)


if __name__ == '__main__':
    main()
