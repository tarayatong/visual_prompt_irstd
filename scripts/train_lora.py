"""Full-mask, prompt-free MobileSAM training; test is evaluation-only."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from visual_prompt.core.metrics import PD_FAmeter
from visual_prompt.training.lora_model import MobileSAMKVLoRA
from visual_prompt.training.losses import SegmentationCriterion, LOSS_DEFAULTS


class IRSTDDataset(Dataset):
    def __init__(self, root, ids, size, augment=False):
        self.root, self.ids, self.size, self.augment = Path(root), ids, size, augment
        for sid in ids:
            for folder in ('IRSTD1k_Img', 'IRSTD1k_Label'):
                if not (self.root / folder / f'{sid}.png').is_file():
                    raise FileNotFoundError(f'{folder}/{sid}.png')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        with Image.open(self.root / 'IRSTD1k_Img' / f'{sid}.png') as im:
            image = np.array(im.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR))
        with Image.open(self.root / 'IRSTD1k_Label' / f'{sid}.png') as im:
            mask = np.array(im.convert('L').resize((self.size, self.size), Image.Resampling.NEAREST)) >= 127
        if self.augment:
            for axis in (0, 1):
                if torch.rand(()).item() < 0.5:
                    image, mask = np.flip(image, axis), np.flip(mask, axis)
            k = int(torch.randint(4, ()).item())
            image, mask = np.rot90(image, k), np.rot90(mask, k)
        return (torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255,
                torch.from_numpy(mask.copy()).unsqueeze(0).float(), sid)


def ids_from(path):
    ids = Path(path).read_text(encoding='utf-8').split()
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate IDs in {path}')
    return ids


def make_splits(root, seed, val_fraction):
    ids = ids_from(root / 'trainval.txt')
    test = ids_from(root / 'test.txt')
    if set(ids) & set(test):
        raise ValueError('trainval/test overlap')
    random.Random(seed).shuffle(ids)
    n = round(len(ids) * val_fraction)
    if not 0 < n < len(ids):
        raise ValueError('Validation split must leave nonempty training and validation sets')
    return {'train': ids[n:], 'val': ids[:n], 'test': test}


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0, criterion=None):
    criterion = criterion or SegmentationCriterion()
    component_sums = {}
    model.eval()
    pdfa = PD_FAmeter()
    inter = pred_area = gt_area = loss_sum = count = 0
    rows = []
    for step, (images, masks, ids) in enumerate(tqdm(loader, desc='evaluate')):
        if max_batches and step >= max_batches:
            break
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss, components = criterion(logits, masks)
        for key, value in components.items():
            component_sums[key] = component_sums.get(key, 0.0) + value.item() * len(ids)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite validation outputs')
        predictions = logits.sigmoid() >= 0.5
        loss_sum += loss.item() * len(ids)
        count += len(ids)
        pdfa.update(predictions[:, 0].cpu(), masks[:, 0].cpu(), masks.shape[-2:])
        for sid, pred, gt in zip(ids, predictions, masks.bool()):
            i, p, g = int((pred & gt).sum()), int(pred.sum()), int(gt.sum())
            inter += i; pred_area += p; gt_area += g
            rows.append({'sample_id': sid, 'iou': i / max(p + g - i, 1),
                         'intersection': i, 'pred_pixels': p, 'gt_pixels': g})
    pd = pdfa.PD / pdfa.target if pdfa.target else 0.0
    fa = pdfa.FA / pdfa.all_pixel if pdfa.all_pixel else 0.0
    return {'loss': loss_sum / max(count, 1), 'samples': count,
            **{'val_' + k: v / max(count, 1) for k, v in component_sums.items()},
            'iou': inter / max(pred_area + gt_area - inter, 1),
            'f1': 2 * inter / max(pred_area + gt_area, 1),
            'mean_image_iou': sum(r['iou'] for r in rows) / max(count, 1),
            'pd': pd, 'fa_per_million': float(fa) * 1e6}, rows


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config')
    known, _ = pre.parse_known_args()
    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument('--data-root', default='dataset/IRSTD-1k')
    p.add_argument('--checkpoint', default='checkpoints/mobile_sam.pt')
    p.add_argument('--output-dir', default='outputs/training/mobilesam_kv_r8_mask_output')
    p.add_argument('--device', default='auto')
    p.add_argument('--rank', type=int, default=8)
    p.add_argument('--alpha', type=float, default=8.0)
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accum-steps', type=int, default=4)
    p.add_argument('--lora-lr', type=float, default=1e-4)
    p.add_argument('--merging-lr', type=float, default=1e-5)
    p.add_argument('--mask-output-lr', type=float, default=1e-4)
    p.add_argument('--train-mask-output', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-fraction', type=float, default=0.2)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    for key, default in LOSS_DEFAULTS.items():
        p.add_argument('--' + key.replace('_', '-'), type=type(default), default=default)
    p.add_argument('--hard-start-epoch', type=int, default=4,
                   help='1-based epoch to enable FP/FN hard mining')
    p.add_argument('--resume')
    p.add_argument('--evaluate', help='Evaluate a saved training checkpoint; never trains')
    p.add_argument('--eval-split', choices=['val', 'test'], default='test')
    p.add_argument('--max-train-batches', type=int, default=0, help='Smoke testing only')
    p.add_argument('--max-val-batches', type=int, default=0, help='Smoke testing only')
    p.add_argument('--dry-run', action='store_true')
    if known.config:
        config = yaml.safe_load(Path(known.config).read_text(encoding='utf-8'))
        unknown = set(config) - {a.dest for a in p._actions}
        if unknown:
            p.error(f'Unknown config keys: {unknown}')
        p.set_defaults(**config)
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.accum_steps, args.image_size, args.threads) < 1:
        p.error('Epochs, batch size, accumulation, image size and threads must be positive')
    if min(args.max_train_batches, args.max_val_batches, args.workers) < 0:
        p.error('Batch limits and worker count must be nonnegative')
    if args.hard_start_epoch < 1:
        p.error('hard-start-epoch must be >= 1')
    if args.resume and args.evaluate:
        p.error('--resume and --evaluate are mutually exclusive')
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu or a CUDA-enabled machine')
    saved = torch.load(args.resume or args.evaluate, map_location='cpu', weights_only=True) if args.resume or args.evaluate else None
    if saved:
        if args.resume and args.hard_start_epoch != saved['config'].get('hard_start_epoch', 1):
            raise ValueError('Hard-loss schedule differs from checkpoint; use the saved schedule or a new run.')
        if 'hard_start_epoch' in saved['config']:
            args.hard_start_epoch = saved['config']['hard_start_epoch']
        saved_loss = {k: saved['config'].get(k, v) for k, v in LOSS_DEFAULTS.items()}
        if args.resume and any(getattr(args, k) != v for k, v in saved_loss.items()):
            raise ValueError('Loss settings differ from checkpoint. Use a new run for loss comparisons.')
        for key, value in saved_loss.items():
            setattr(args, key, value)
        saved_output = saved['config'].get('train_mask_output', False)
        if args.resume and args.train_mask_output != saved_output:
            raise ValueError('Resume requires the same train-mask-output setting. Start a new output directory '
                             'for the unfrozen output head; use --no-train-mask-output to resume the old run.')
        args.train_mask_output = saved_output
        args.mask_output_lr = saved['config'].get('mask_output_lr', args.mask_output_lr)
        # Do not silently change architecture/preprocessing on resume or evaluation.
        for key in ('rank', 'alpha', 'image_size', 'seed', 'val_fraction', 'gradient_checkpointing'):
            setattr(args, key, saved['config'][key])
        if args.resume:
            for key in ('batch_size', 'accum_steps', 'lora_lr', 'merging_lr', 'weight_decay', 'amp'):
                setattr(args, key, saved['config'][key])
    base_criterion = SegmentationCriterion(loss_mode='bce_dice',
                                           dice_weight=args.dice_weight)
    hard_criterion = SegmentationCriterion(**{k: getattr(args, k) for k in LOSS_DEFAULTS})
    def criterion_for_epoch(epoch_index):
        return (base_criterion, 'bce_dice') if epoch_index + 1 < args.hard_start_epoch else (hard_criterion, 'weighted_bce_dice_hard')
    root, out = Path(args.data_root), Path(args.output_dir)
    splits = saved['splits'] if saved else make_splits(root, args.seed, args.val_fraction)
    expected = make_splits(root, args.seed, args.val_fraction)
    if splits != expected:
        raise ValueError('Dataset split differs from saved checkpoint')
    model = MobileSAMKVLoRA(args.checkpoint, args.rank, args.alpha, args.gradient_checkpointing,
                           train_mask_output=args.train_mask_output).to(device)
    if saved:
        model.load_state_dict(saved['model'], strict=True)
    print(json.dumps({'device': str(device), 'loss_schedule': {'bce_dice_epochs': f'1-{args.hard_start_epoch-1}', 'hard_from_epoch': args.hard_start_epoch}, 'parameters': model.parameter_counts(),
                      'splits': {k: len(v) for k, v in splits.items()}}, indent=2), flush=True)
    if args.dry_run:
        return
    if not saved and (out / 'last.pt').exists():
        raise FileExistsError('Existing run: use --resume or a new --output-dir')
    out.mkdir(parents=True, exist_ok=True)
    def loader(which, train=False, generator=None):
        return DataLoader(IRSTDDataset(root, splits[which], args.image_size, train),
                          batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
                          pin_memory=device.type == 'cuda', generator=generator)
    if args.evaluate:
        checkpoint_epoch = saved.get('epoch', 0)
        eval_criterion, _ = criterion_for_epoch(checkpoint_epoch)
        metrics, rows = evaluate(model, loader(args.eval_split), device, criterion=eval_criterion)
        (out / f'{args.eval_split}_metrics.json').write_text(json.dumps(metrics, indent=2))
        with (out / f'{args.eval_split}_detail.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        print(json.dumps(metrics, indent=2))
        return
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))
    (out / 'splits.json').write_text(json.dumps(splits, indent=2))
    (out / 'parameters.json').write_text(json.dumps(model.parameter_counts(), indent=2))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    def is_output(name):
        return any(name.startswith('sam.mask_decoder.' + part + '.') for part in
                   ('output_upscaling', 'output_hypernetworks_mlps', 'mask_tokens'))
    groups = [
        {'params': [p for n, p in trainable if '.downsample.' in n], 'lr': args.merging_lr},
        {'params': [p for n, p in trainable if '.downsample.' not in n and not is_output(n)], 'lr': args.lora_lr},
    ]
    if args.train_mask_output:
        groups.append({'params': [p for n, p in trainable if is_output(n)], 'lr': args.mask_output_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')
    generator = torch.Generator().manual_seed(args.seed)
    start, best = 0, -1.0
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        start, best = saved['epoch'] + 1, saved['best_iou']
        torch.set_rng_state(saved['rng'])
        generator.set_state(saved['loader_rng'])
        if device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    train_loader, val_loader = loader('train', True, generator), loader('val')
    for epoch in range(start, args.epochs):
        criterion, loss_stage = criterion_for_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        steps = min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader)
        total_loss = 0.0
        component_sums = {}
        for step, (images, masks, _) in enumerate(tqdm(train_loader, desc=f'epoch {epoch + 1}')):
            if step >= steps:
                break
            images, masks = images.to(device), masks.to(device)
            # Correct divisor for an incomplete final accumulation group.
            group_size = min(args.accum_steps, steps - (step // args.accum_steps) * args.accum_steps)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss, components = criterion(model(images), masks)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            total_loss += loss.item()
            for key, value in components.items():
                component_sums[key] = component_sums.get(key, 0.0) + value.item()
            scaler.scale(loss / group_size).backward()
            if (step + 1) % args.accum_steps == 0 or step + 1 == steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0, error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
        metrics, _ = evaluate(model, val_loader, device, args.max_val_batches, criterion=criterion)
        improved = metrics['iou'] > best
        best = max(best, metrics['iou'])
        result = {'epoch': epoch + 1, 'loss_stage': loss_stage, 'train_loss': total_loss / steps,
                  **{'train_' + k: v / steps for k, v in component_sums.items()}, **metrics}
        with (out / 'history.jsonl').open('a') as handle:
            handle.write(json.dumps(result) + '\n')
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scaler': scaler.state_dict(), 'epoch': epoch, 'best_iou': best,
                 'config': vars(args), 'splits': splits, 'rng': torch.get_rng_state(),
                 'loader_rng': generator.get_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
        torch.save(state, out / 'last.tmp')
        (out / 'last.tmp').replace(out / 'last.pt')
        if improved:
            torch.save(state, out / 'best.tmp')
            (out / 'best.tmp').replace(out / 'best.pt')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
