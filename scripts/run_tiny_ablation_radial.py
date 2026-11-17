"""Two-scale FEATURE-radial gated decoder bypass; matched native59.61 base.

Run --smoke first; --suite trains two-scale gated radial decoder bypass, 120ep.
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
from visual_prompt.models.tiny_decoder_feature_radial_gated import install_decoder_feature_radial_gated as install_native_radial
from visual_prompt.models.sam2_tiny_radial import DecoderGridRadialSampler
from visual_prompt.evaluation.comparison import image_metrics, summarize

OUT = ROOT/'outputs/tiny_ablation_three_seeds_20261110/seed111/IRSTD-1k'
RUN_SEED = 111
VARIANTS = ('feature_decoder_gated',)
MODES = ('native', *VARIANTS)


def autocast():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def state_for(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in names or k.startswith('radial_branch.')}


def restore(model, weights):
    assert set(weights) == set(state_for(model))
    model.load_state_dict(weights, strict=False)


def comparison():
    path = OUT/'feature_decoder_gated/epoch_0120/metrics/summary.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    base.write_json(OUT/'radial_completed.json', dict(seed=RUN_SEED,method='radial',
        summary=path.relative_to(ROOT).as_posix(),IoU_percent=result['IoU_percent']))
    return result


def worker(args):
    mode = args.variant
    dense_mlp = getattr(args, 'dense_mlp', False)
    active_stages = getattr(args, 'active_stages', (0, 1))
    folder = OUT / mode
    folder.mkdir(parents=True, exist_ok=True)
    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.manual_seed(RUN_SEED)
    torch.cuda.manual_seed_all(RUN_SEED)
    torch.set_num_threads(2)
    manifest = base.manifest_for(args.dataset)
    # No image-derived radial maps or fitted image statistics in this arm.
    radial_mean, radial_std = torch.zeros(64), torch.ones(64)
    model, unused_opt, unused_sched, unused_loader, unused_loss, native_predict, identity = base.setup_sansa(
        manifest, torch.device('cuda'))
    del unused_opt, unused_sched, unused_loader, unused_loss
    from util.promptable_utils import build_prompt_dict, rescale_prompt
    from util.losses import loss_masks

    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    pairs = {}
    for sid in manifest['train_ids']:
        image, mask = base.raw_pair(manifest, sid, 640, True)
        pairs[sid] = ((torch.from_numpy(image.transpose(2, 0, 1)/255)-mean)/std,
                      torch.from_numpy(mask))

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
    if dense_mlp:
        from scripts.run_tiny_ablation_native import install_dense_memory_mlp

        with torch.random.fork_rng(devices=[]):
            install_dense_memory_mlp(model)
        assert not any(p.requires_grad for p in model.sam.memory_encoder.parameters())
        assert not any(p.requires_grad for p in model.sam.memory_attention.parameters())
    if args.smoke:
        model.eval()
        with torch.no_grad():
            no_radial_reference = native_predict(sample).float().detach().clone()
    with torch.random.fork_rng(devices=[]):
        install_native_radial(model, 'feature_decoder_gated', radial_mean, radial_std,
                              active_stages=active_stages)
    radial_stages = [model.radial_branch.fusion.stages[i] for i in active_stages]

    def prepare_radial(ids):
        # Tokens only identify which query calls activate the feature-side hook.
        model.radial_branch.begin([(None, None) for _ in ids])

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_prefixes = ('radial_branch.', 'dense_memory_mlp.') if dense_mlp else ('radial_branch.',)
    assert all('adapter' in n or n.startswith(trainable_prefixes)
               for n, p in model.named_parameters() if p.requires_grad)
    assert not hasattr(model, 'radial_fusion')
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 120*len(loader))

    def objective(batch):
        samples, masks, supports, queries = batch
        prepare_radial(list(queries))
        samples, masks = samples.cuda(), masks.cuda()
        with autocast():
            prompt = build_prompt_dict(masks, 'mask', 1, True, torch.device('cuda'))
            output = model(samples, prompt)
            model.radial_branch.finish()
            loss = sum(loss_masks(output['pred_masks'], masks, num_frames=1).values())
        return loss

    support_cache = {}
    dense_spatial_tokens = 0

    @torch.no_grad()
    def predict(sid):
        nonlocal dense_spatial_tokens
        with torch.autocast('cuda', enabled=False):
            if not support_cache:
                ids = manifest['support_ids']
                images = torch.stack([pairs[item][0] for item in ids])[None].cuda()
                processed, _, _, sizes = model._preprocess_visual_features(images, 1024)
                backbone = model._forward_backbone(processed, sizes)
                masks = torch.stack([pairs[item][1] for item in ids])[None].cuda()
                prompt = build_prompt_dict(masks, 'mask', 5, False, torch.device('cuda'))
                bank = {}
                for index in range(5):
                    mask_prompt = rescale_prompt(prompt[0][index]['prompt'], 'mask', sizes[index], 1024)
                    decoded = model.sam._use_mask_as_output(backbone, mask_prompt, index)
                    bank[index] = model._compute_memory_bank_dict(decoded, backbone, index)
                if dense_mlp:
                    assert all(entry['maskmem_features'].shape == (1, 64, 64, 64)
                               for entry in bank.values())
                    dense_spatial_tokens = len(bank) * 64**2
                support_cache['bank'] = bank
            if sid in pairs:
                query = pairs[sid][0]
            else:
                with Image.open(manifest['records'][sid]['image']) as image:
                    array = np.array(image.convert('RGB').resize((640, 640), Image.Resampling.BILINEAR), dtype=np.float32)
                query = (torch.from_numpy(array.transpose(2, 0, 1)/255)-mean)/std
            processed, _, _, sizes = model._preprocess_visual_features(query[None, None].cuda(), 1024)
            backbone = model._forward_backbone(processed, sizes)
            prepare_radial([sid])
            decoded = model._compute_decoder_out_w_mem(backbone, 0, 5, support_cache['bank'])
            model.radial_branch.finish()
            logits = F.interpolate(decoded.masks.float(), (640, 640), mode='bilinear', align_corners=False)[0, 0]
            return logits.sigmoid()

    if args.smoke:
        model.eval()
        prepare_radial([sample])
        with torch.no_grad():
            reference_fp32 = native_predict(sample).float().detach().clone()
        model.radial_branch.finish()
        zero_error = float((reference_fp32-no_radial_reference).abs().max())
        assert torch.allclose(reference_fp32, no_radial_reference, atol=1e-6, rtol=1e-6), zero_error
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
        mean_error_limit = 1e-5 if args.dataset == 'IRSTD-1k' or not dense_mlp else 3e-5
        assert error < .003 and mean_error < mean_error_limit, (error, mean_error)
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
            for radial_stage in radial_stages:
                radial_grad = radial_stage.output.weight.grad
                assert radial_grad is not None and torch.isfinite(radial_grad).all() and radial_grad.abs().sum()>0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)
            assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
            if step == 2:
                assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.radial_branch.parameters())
            if dense_mlp:
                assert all(p.grad is not None and p.grad.abs().sum() > 0
                           for p in model.dense_memory_mlp.parameters())
            torch.nn.utils.clip_grad_norm_(trainable, .1)
            optimizer.step()
        peak = torch.cuda.max_memory_allocated()/1024**2
        result = dict(status='passed', actual_batch_size=4, accumulation_steps=1,
                      precision='BF16 training; FP32 evaluation', finite_forward_backward=True,
                      total_parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in trainable),
                      peak_allocated_MiB=peak, three_steps_seconds=time.monotonic()-begin,
                      native_dense_memory_encoder_removed=False, native_memory_attention_frozen=True,
                      radial_zero_initialization_max_probability_error=zero_error,
                      radial_projection_gradient_finite_nonzero=True, radial_mode=mode,
                      native_support_shots=5, prototype_memory=False,
                      cached_support_zero_residual_max_probability_error=error,
                      cached_support_mean_probability_error=mean_error,
                      support_encoder_calls_for_two_queries=encoder_calls,
                      single_encoder=True, reference_encoder=False, radial=True)
        if dense_mlp:
            result.update(dense_spatial_tokens=dense_spatial_tokens, dense_mlp_gradients_verified=True,
                          radial_stage_sizes=[(128, 256)[i] for i in active_stages])
        base.write_json(folder / 'preflight.json', result)
        print(json.dumps(dict(variant=mode, **result)), flush=True)
        return

    prior = json.loads((base.RUN / args.dataset / 'SANSA/config.json').read_text(encoding='utf-8'))
    sources = [Path(__file__), ROOT / 'visual_prompt/models/tiny_decoder_feature_radial_gated.py', ROOT / 'visual_prompt/models/tiny_decoder_radial_gated.py', ROOT / 'visual_prompt/models/tiny_radial_pre_h0.py', ROOT / 'visual_prompt/models/tiny_prototype_radial.py', ROOT / 'visual_prompt/models/radial_sam_fusion.py', ROOT / 'visual_prompt/models/sam2_tiny_radial.py', ROOT / 'visual_prompt/models/radial_profile.py', Path(base.__file__),
               ROOT / 'visual_prompt/models/semantic_memory.py', ROOT / 'visual_prompt/core/semantic_direction.py']
    if dense_mlp:
        sources.extend((ROOT / 'scripts/run_tiny_ablation_native.py', Path(args.experiment_source)))
    config = dict(prior, method='SAM2Tiny-DecoderFeatureRadialGated-B4', variant=mode, batch_size=4, training_random_seed=RUN_SEED, split_seed=111,
                  support_annotation_ids=manifest['support_ids'],
                  canonical_runner='run_tiny_decoder_feature_radial_gated.py', canonical_runner_sha256=base.sha(ROOT/'scripts/run_tiny_decoder_feature_radial_gated.py'),
                  precision='BF16 training; FP32 evaluation', seed=RUN_SEED, radial=True, one_shared_encoder=True,
                  radial_mode=mode, radial_branch_parameters=sum(p.numel() for p in model.radial_branch.parameters()),
                  radial_mean=radial_mean.tolist(), radial_std=radial_std.tolist(),
                  radial_statistics_source='dynamic decoder features; per-point L2 normalized contrast; no image radial maps or fitted image statistics',
                  radial_radii_256grid=[1,2,3,4,6,8,12,16], radial_radii_128grid=[.5,1,1.5,2,3,4,6,8],
                  radial_radius_units='pixels of each decoder feature grid; matched normalized coverage',
                  radial_directions=8, radial_compressed_channels=4, radial_descriptor_channels=256,
                  raw_image_radial=False, feature_radial_differentiable=True,
                  radial_fusion='per-stage feature radial descriptor; shared256to32 stem; same native-feature gates at128 and256 activations',
                  radial_residual_scale=.1, radial_output_initialization='zero',
                  radial_stage_channels=[64,32], radial_stage_sizes=[128,256],
                  radial_learned_stem='per-stage compression to4 -> 64 directional/radius contrasts -> shared256to32 + GroupNorm + GELU + depthwise3x3 + GELU',
                  image_radial_comparator='outputs/tiny_decoder_radial_gated_20261110/IRSTD-1k/decoder_gated/epoch_0120/metrics/summary.json',
                  radial_warm_start='none; same pretrained initialization and120ep recipe as59.61 native baseline',
                  reference_encoder_parameters=0, runner_sha256=base.sha(Path(__file__)),
                  source_hashes={p.relative_to(ROOT).as_posix(): base.sha(p) for p in sources},
                  parameters_total=sum(p.numel() for p in model.parameters()),
                  parameters_trainable=sum(p.numel() for p in trainable),
                  pipeline_parameters_total=sum(p.numel() for p in model.parameters()),
                  optimizer_steps_per_epoch=13, optimizer_steps_total=1560, last_batch_size=2,
                  lr=1e-4, train_support_shots=1, support_shots=5,
                  native_dense_memory=True, native_spatial_encoder_removed=False,
                  native_memory_attention_retained=True, native_memory_attention_trainable=False,
                  original_sansa_adapters_retained=True, prototype_memory=False,
                  fusion='native spatial support memory/attention/decoder; query-only radial residual',
                  matched_reference='outputs/tiny_shared_memory_batch4_20261110/IRSTD-1k/native/epoch_0120/metrics/summary.json',
                  query_feature_source='same current backbone H0/H1/F; shared with native SAM2',
                  evaluation_support_cache='fixed native support features and spatial memory built once',
                  selection='fixed epoch120 only; no test selection',
                  baseline_note='matched59.61 native spatial-memory baseline; from same pretrained initialization')
    if args.dataset != 'IRSTD-1k':
        assert prior['manifest_sha256'] == manifest['manifest_sha256']
        assert prior['train_annotation_ids'] == manifest['train_ids']
        for key in ('memory_only_reference', 'radial_only_reference', 'image_radial_comparator'):
            config.pop(key, None)
        config.update(matched_reference=(OUT/'native/epoch_0120/metrics/summary.json').relative_to(ROOT).as_posix(),
                      baseline_note='Compare only with the same dataset and training seed.')
    if dense_mlp:
        stages = [(128, 256)[i] for i in active_stages]
        config.update(
            method='SAM2Tiny-DenseMemoryMLP-RCR-B4',
            memory_control='dense_mlp',
            dense_memory_mapping='LayerNorm(64) -> Linear(64,176) -> GELU -> Linear(176,64)',
            dense_memory_mlp_parameters=sum(p.numel() for p in model.dense_memory_mlp.parameters()),
            dense_memory_grid=[64, 64], dense_spatial_tokens=5 * 64**2,
            native_object_pointers=True, native_memory_positions=True,
            radial_stage_channels=[(64, 32)[i] for i in active_stages], radial_stage_sizes=stages,
            radial_sampling_points_per_query=64 * sum(size**2 for size in stages),
            radial_control='only_256', radial_descriptor='center_minus_neighbor', radial_gate='adaptive',
            radial_fusion='adaptive feature radial residual at the final 256 decoder stage',
            fusion='native SAM2 mask-conditioned spatial memory -> per-location MLP -> native attention -> RCR',
            evaluation_support_cache='mapped native spatial memory with original positions and object pointers',
            matched_reference=args.full_reference,
            baseline_note='Native dense support encoding and MLP replace CSM with the same RCR and training recipe.',
        )
        config.pop('image_radial_comparator', None)
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
            for radial_stage in radial_stages:
                radial_grad = radial_stage.output.weight.grad
                assert radial_grad is not None and torch.isfinite(radial_grad).all() and radial_grad.abs().sum()>0
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
    if not dense_mlp:
        comparison()



def main():
    global OUT, RUN_SEED
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(111,112,113),required=True)
    parser.add_argument('--dataset', choices=base.DATASETS, default='IRSTD-1k')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    RUN_SEED=args.seed
    output_name = ('tiny_ablation_three_seeds_20261110' if args.dataset == 'IRSTD-1k'
                   else 'tiny_ablation_other_datasets_20261111')
    OUT=ROOT/'outputs'/output_name/f'seed{RUN_SEED}'/args.dataset
    worker(argparse.Namespace(variant='feature_decoder_gated',smoke=args.smoke,dataset=args.dataset))


if __name__=='__main__':
    main()
