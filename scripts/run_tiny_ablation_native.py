"""Matched native/c experiment: one encoder, actual batch4, no radial.

Run --smoke first; --suite trains native baseline then shared-memory c, 120ep.
BF16 training, FP32 evaluation are identical for both arms. Last batch contains2.
"""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from scripts import run_recent_a_comparison as base
from visual_prompt.models.tiny_shared_memory import (
    support_regions, descriptor, build_episode_keys, install_shared_memory,
)
from visual_prompt.models.tiny_memory_token_ablation import foreground_sampling_regions
from visual_prompt.evaluation.comparison import image_metrics, summarize

OUT = ROOT/'outputs/tiny_ablation_three_seeds_20261110/seed111/IRSTD-1k'
RUN_SEED = 111
MODES = ('native', 'c', 'dense_mlp', 'dense_e', 'dense_semantic_e', 'dense_mlp_semantic_e')


def semantic_memory_labels(image, mask):
    geometry = foreground_sampling_regions(image, mask)
    high_freq, high_value = geometry['negatives']
    labels = torch.zeros_like(high_freq, dtype=torch.long)
    labels[high_value] = 2
    labels[high_freq] = 1
    y, x = geometry['foreground_cells'].T
    labels[y, x] = 3
    return labels


def install_dense_semantic_embedding(model):
    model.dense_semantic_embedding = nn.Embedding(4, 64).to(model.device)
    nn.init.zeros_(model.dense_semantic_embedding.weight)
    encode = model._compute_memory_bank_dict

    def encode_with_embedding(self, decoder_out, backbone_out, index):
        entry = encode(decoder_out, backbone_out, index)
        labels = self.memory_semantic_labels[index]
        if labels is not None:
            positions = entry['maskmem_pos_enc']
            spatial = positions[-1]
            bias = self.dense_semantic_embedding(labels).permute(2, 0, 1)[None]
            assert bias.shape == spatial.shape
            entry['maskmem_pos_enc'] = [*positions[:-1], spatial + bias.to(spatial.dtype)]
        return entry

    model._compute_memory_bank_dict = types.MethodType(encode_with_embedding, model)


def install_dense_memory_mlp(model):
    model.dense_memory_mlp = nn.Sequential(
        nn.LayerNorm(64), nn.Linear(64, 176, bias=False),
        nn.GELU(), nn.Linear(176, 64, bias=False),
    ).to(model.device)
    encode = model._compute_memory_bank_dict

    def encode_with_mapping(self, decoder_out, backbone_out, index):
        entry = encode(decoder_out, backbone_out, index)
        spatial = entry['maskmem_features'].movedim(1, -1)
        entry['maskmem_features'] = self.dense_memory_mlp(spatial).movedim(-1, 1).contiguous()
        return entry

    model._compute_memory_bank_dict = types.MethodType(encode_with_mapping, model)


def install_dense_memory_embedding(model):
    model.dense_memory_embedding = nn.Parameter(torch.zeros(1, 64, 64, 64, device=model.device))
    encode = model._compute_memory_bank_dict

    def encode_with_embedding(self, decoder_out, backbone_out, index):
        entry = encode(decoder_out, backbone_out, index)
        positions = entry['maskmem_pos_enc']
        spatial = positions[-1]
        assert spatial.shape == self.dense_memory_embedding.shape
        entry['maskmem_pos_enc'] = [
            *positions[:-1], spatial + self.dense_memory_embedding.to(spatial.dtype)]
        return entry

    model._compute_memory_bank_dict = types.MethodType(encode_with_embedding, model)


def autocast():
    return torch.autocast('cuda', dtype=torch.bfloat16)


def state_for(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in names}


def restore(model, weights):
    assert set(weights) == set(state_for(model))
    model.load_state_dict(weights, strict=False)


def comparison():
    path = OUT/'native/epoch_0120/metrics/summary.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    base.write_json(OUT/'native_completed.json', dict(seed=RUN_SEED,method='native',
        summary=path.relative_to(ROOT).as_posix(),IoU_percent=result['IoU_percent']))
    return result


def worker(args):
    mode = args.variant
    use_mlp = mode in ('dense_mlp', 'dense_mlp_semantic_e')
    use_semantic_e = mode in ('dense_semantic_e', 'dense_mlp_semantic_e')
    folder = OUT / mode
    folder.mkdir(parents=True, exist_ok=True)
    random.seed(RUN_SEED)
    np.random.seed(RUN_SEED)
    torch.manual_seed(RUN_SEED)
    torch.cuda.manual_seed_all(RUN_SEED)
    torch.set_num_threads(2)
    manifest = base.manifest_for(args.dataset)
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
        if mode == 'c':
            geometry[sid] = support_regions(image.astype(np.uint8), mask > 0)
        if use_semantic_e:
            geometry[sid] = semantic_memory_labels(image.astype(np.uint8), mask > 0).cuda()

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
    if args.smoke:
        model.eval()
        with torch.no_grad(), autocast():
            reference = native_predict(sample).float().detach().clone()
        with torch.no_grad():
            reference_fp32 = native_predict(sample).float().detach().clone()
    if mode == 'c':
        with torch.random.fork_rng(devices=[]):
            install_shared_memory(model)
    if use_mlp:
        with torch.random.fork_rng(devices=[]):
            install_dense_memory_mlp(model)
    if mode == 'dense_e':
        install_dense_memory_embedding(model)
    if use_semantic_e:
        with torch.random.fork_rng(devices=[]):
            install_dense_semantic_embedding(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all('adapter' in n or n.startswith(('shared_memory.', 'dense_memory_mlp.'))
               or n in ('dense_memory_embedding', 'dense_semantic_embedding.weight')
               for n, p in model.named_parameters() if p.requires_grad)
    assert not hasattr(model, 'radial_fusion')
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 120*len(loader))

    def objective(batch):
        samples, masks, supports, _ = batch
        if mode == 'c':
            model.support_geometry = [[geometry[sid]] for sid in supports]
        if use_semantic_e:
            model.memory_semantic_labels = [label for sid in supports for label in (geometry[sid], None)]
        samples, masks = samples.cuda(), masks.cuda()
        with autocast():
            prompt = build_prompt_dict(masks, 'mask', 1, True, torch.device('cuda'))
            output = model(samples, prompt)
            loss = sum(loss_masks(output['pred_masks'], masks, num_frames=1).values())
        return loss

    support_cache = {}

    @torch.no_grad()
    def predict(sid):
        # Fixed support features/native memory/our prototypes are built once at
        # evaluation, then reused across all queries. Cache is reset after training.
        with torch.autocast('cuda', enabled=False):
            if not support_cache:
                ids = manifest['support_ids']
                if use_semantic_e:
                    model.memory_semantic_labels = [geometry[sid] for sid in ids]
                images = torch.stack([pairs[item][0] for item in ids])[None].cuda()
                masks = torch.stack([pairs[item][1] for item in ids])[None].cuda()
                processed, _, _, sizes = model._preprocess_visual_features(images, 1024)
                backbone = model._forward_backbone(processed, sizes)
                prompt = build_prompt_dict(masks, 'mask', 5, False, torch.device('cuda'))
                bank = {}
                for index in range(5):
                    mask_prompt = rescale_prompt(prompt[0][index]['prompt'], 'mask', sizes[index], 1024)
                    decoded = model.sam._use_mask_as_output(backbone, mask_prompt, index)
                    bank[index] = model._compute_memory_bank_dict(decoded, backbone, index)
                support_cache['bank'] = bank
                if mode == 'c':
                    support_cache['keys'] = build_episode_keys(
                        model, backbone, range(5), [geometry[item] for item in ids])
            if sid in pairs:
                query = pairs[sid][0]
            else:
                with Image.open(manifest['records'][sid]['image']) as image:
                    array = np.array(image.convert('RGB').resize((640, 640), Image.Resampling.BILINEAR), dtype=np.float32)
                query = (torch.from_numpy(array.transpose(2, 0, 1)/255)-mean)/std
            processed, _, _, sizes = model._preprocess_visual_features(query[None, None].cuda(), 1024)
            backbone = model._forward_backbone(processed, sizes)
            current = backbone.get_current_feats(0)
            position = backbone.get_current_pos_embeds(0)
            high = backbone.get_high_res_features(current)
            main = model.sam._prepare_memory_conditioned_features(
                frame_idx=5, current_vision_feats=current[-1:], current_vision_pos_embeds=position[-1:],
                feat_sizes=backbone.feat_sizes[-1:], num_frames=6, memory_bank=support_cache['bank'])
            if mode == 'c':
                main = model.shared_memory(main, descriptor(model, backbone, 0), support_cache['keys'])
            decoded = model.sam._forward_sam_heads(backbone_features=main, high_res_features=high, multimask_output=True)
            logits = F.interpolate(decoded.masks.float(), (640, 640), mode='bilinear', align_corners=False)[0, 0]
            return logits.sigmoid()

    if args.smoke:
        model.eval()
        if mode == 'c':
            model.support_geometry = [[geometry[sid] for sid in manifest['support_ids']]]
        if use_semantic_e:
            model.memory_semantic_labels = [geometry[sid] for sid in manifest['support_ids']] + [None]
        zero_error = None
        if use_mlp:
            with torch.no_grad():
                reference_fp32 = native_predict(sample).float().detach().clone()
        else:
            with torch.no_grad(), autocast():
                zero_output = native_predict(sample).float()
            zero_error = float((reference-zero_output).abs().max())
            assert torch.allclose(reference, zero_output, atol=1e-6, rtol=1e-6), zero_error
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
        # Separate FP32 encoder batches can select different numerical kernels.
        # Bound both local and full-image error; same-batch zero residual above
        # is separately required to agree to1e-6.
        assert error < .002 and mean_error < 1e-5, (error, mean_error)
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
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)
            assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
            if mode == 'c':
                grad = model.shared_memory.output.weight.grad
                assert grad is not None and grad.abs().sum() > 0
                if step == 2:
                    assert all(p.grad is not None and p.grad.abs().sum() > 0
                               for p in model.shared_memory.parameters())
            if use_mlp:
                assert all(p.grad is not None and p.grad.abs().sum() > 0
                           for p in model.dense_memory_mlp.parameters())
            if mode == 'dense_e':
                grad = model.dense_memory_embedding.grad
                assert grad is not None and grad.abs().sum() > 0
            if use_semantic_e:
                grad = model.dense_semantic_embedding.weight.grad
                assert grad is not None and grad.abs().sum() > 0
            torch.nn.utils.clip_grad_norm_(trainable, .1)
            optimizer.step()
        peak = torch.cuda.max_memory_allocated()/1024**2
        result = dict(status='passed', actual_batch_size=4, accumulation_steps=1,
                      precision='BF16 training; FP32 evaluation', finite_forward_backward=True,
                      total_parameters=sum(p.numel() for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in trainable),
                      peak_allocated_MiB=peak, three_steps_seconds=time.monotonic()-begin,
                      zero_residual_max_probability_error=zero_error,
                      cached_support_zero_residual_max_probability_error=error,
                      cached_support_mean_probability_error=mean_error,
                      support_encoder_calls_for_two_queries=encoder_calls,
                      single_encoder=True, reference_encoder=False, radial=False)
        base.write_json(folder / 'preflight.json', result)
        print(json.dumps(dict(variant=mode, **result)), flush=True)
        return

    prior = json.loads((base.RUN / args.dataset / 'SANSA/config.json').read_text(encoding='utf-8'))
    sources = [Path(__file__), ROOT / 'visual_prompt/models/tiny_shared_memory.py', Path(base.__file__),
               ROOT / 'visual_prompt/models/semantic_memory.py', ROOT / 'visual_prompt/core/semantic_direction.py']
    config = dict(prior, method=f'SAM2Tiny-Shared-B4-{mode}', variant=mode, batch_size=4, training_random_seed=RUN_SEED, split_seed=111,
                  support_annotation_ids=manifest['support_ids'],
                  canonical_runner='run_tiny_shared_memory_batch4.py', canonical_runner_sha256=base.sha(ROOT/'scripts/run_tiny_shared_memory_batch4.py'),
                  precision='BF16 training; FP32 evaluation', seed=RUN_SEED, radial=False, one_shared_encoder=True,
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
                  query_feature_source='same current backbone H0/H1/F; shared with native SAM2',
                  evaluation_support_cache='native memory and prototype keys built once per fixed trained model',
                  selection='fixed epoch120 only; no test selection',
                  baseline_note='new matched batch4 BF16 native arm; historical batch1 FP32 is not a matched baseline')
    if args.dataset != 'IRSTD-1k':
        assert prior['manifest_sha256'] == manifest['manifest_sha256']
        assert prior['train_annotation_ids'] == manifest['train_ids']
        for key in ('memory_only_reference', 'radial_only_reference', 'image_radial_comparator'):
            config.pop(key, None)
        config.update(matched_reference=(OUT/'native/epoch_0120/metrics/summary.json').relative_to(ROOT).as_posix(),
                      baseline_note='Compare only with the same dataset and training seed.')
    if use_mlp:
        config.update(prototype_source=None, prototype_slots=None, prototype_feature_grid=None,
                      prototype_channels=None, dense_memory_mapping='LN -> Linear(64,176) -> GELU -> Linear(176,64)',
                      dense_memory_mapping_residual=False, native_memory_attention_trainable=False,
                      evaluation_support_cache='mapped native spatial memory and original object pointers')
    if mode == 'dense_e':
        config.update(prototype_source=None, prototype_slots=None, prototype_feature_grid=None,
                      prototype_channels=None, dense_memory_embedding_shape=[64, 64, 64],
                      dense_memory_embedding_shared_across_supports=True,
                      dense_memory_embedding_path='add to spatial memory positions; attention keys only',
                      native_memory_attention_trainable=False,
                      evaluation_support_cache='native spatial memory with learned key positions and original object pointers')
    if use_semantic_e:
        config.update(prototype_source=None, prototype_slots=None, prototype_feature_grid=None,
                      prototype_channels=None, dense_memory_embedding_shape=[4, 64],
                      dense_memory_embedding_path='support semantic label lookup added to spatial key positions',
                      dense_memory_embedding_priority=['foreground', 'high_frequency', 'high_value', 'ordinary_background'],
                      native_memory_attention_trainable=False,
                      evaluation_support_cache=('mapped' if use_mlp else 'native')
                      + ' spatial memory with learned semantic key positions and original object pointers')
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
    if mode == 'native':
        comparison()



def main():
    global OUT, RUN_SEED
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=(111,112,113),required=True)
    parser.add_argument('--dataset', choices=base.DATASETS, default='IRSTD-1k')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--variant', choices=('native', *MODES[2:]), default='native')
    args=parser.parse_args()
    RUN_SEED=args.seed
    output_name = ('tiny_ablation_three_seeds_20261110' if args.dataset == 'IRSTD-1k'
                   else 'tiny_ablation_other_datasets_20261111')
    OUT=ROOT/'outputs'/output_name/f'seed{RUN_SEED}'/args.dataset
    if args.variant in MODES[2:]:
        OUT=ROOT/'outputs/tiny_dense_memory_mapping'/f'seed{RUN_SEED}'/args.dataset
    worker(argparse.Namespace(variant=args.variant,smoke=args.smoke,dataset=args.dataset))


if __name__=='__main__':
    main()
