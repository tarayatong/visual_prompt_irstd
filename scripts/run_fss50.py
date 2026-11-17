"""Native IFA/MPA target adaptation using all fixed train50 labels, isolated v2."""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from scripts.prepare_comparison import DEFAULT_OUTPUT, write_json
from scripts.run_comparison_ours import save_csv
from visual_prompt.evaluation.comparison import image_metrics, summarize


def native_model(root, method):
    sys.path.append(str((root / 'runtime_packages').resolve()))
    if method == 'MPA':
        from scripts.comparison_mpa_adapter import load_native
        model, dataset, lossfn = load_native(root)
    else:
        from scripts.run_comparison_ifa import build_native
        model, _ = build_native(root)
        from data.lung_IFA import DatasetLungIFA
        dataset, lossfn = DatasetLungIFA, None
    for layer in (model.layer0, model.layer1):
        for p in layer.parameters():
            p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for p in module.parameters():
                p.requires_grad_(False)
    return model, dataset, lossfn


def episode_dataset(native, manifest, transform, folder, method):
    """Reuse native augmentation/pseudoquery creation; replace only file sampling."""
    import albumentations as A
    ids = manifest['train_ids']
    assert len(ids) == len(set(ids)) == 50
    masks = folder / 'canonical_train_masks'
    masks.mkdir(parents=True, exist_ok=True)
    mask_to_image = {}
    for sid in ids:
        record = manifest['records'][sid]
        with Image.open(record['image']) as im:
            size = im.size
        with Image.open(record['mask']) as im:
            a = np.array(im.convert('L').resize(size, Image.Resampling.NEAREST)) > 0
        path = masks / f'{sid}.png'
        Image.fromarray(a.astype(np.uint8) * 255).save(path)
        mask_to_image[str(path)] = record['image']

    class BudgetDataset(native):
        def sample_episode(self, index):
            names = [str(masks / f'{sid}.png') for sid in self.active_ids]
            return names[0], names, 0

        def load_frame(self, query_name, support_names):
            images = [Image.open(mask_to_image[p]).convert('RGB') for p in support_names]
            return images[0], self.read_mask(support_names[0]), images, support_names

    d = BudgetDataset.__new__(BudgetDataset)
    d.shot, d.num, d.transform, d.active_ids = 5, 10, transform, ids[:5]
    color = A.Compose([A.RandomBrightnessContrast(p=.5),
                       A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                                            val_shift_limit=20, always_apply=False, p=1)])
    if method == 'MPA':
        d.q_colorAug = color
    else:
        d.q_transform = A.Compose([A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5), A.RandomRotate90()])
        d.q_transform2 = color
    return d


def ifa_loss(out, qmask, smask, ce):
    # Exact refine=True coefficients from official ifa.py.
    return (ce(out[0], qmask) + ce(out[1], qmask) + ce(out[2], qmask)
            + .2 * ce(out[3], smask) + .4 * ce(out[4], smask)
            + .1 * ce(out[5], qmask) + .1 * ce(out[6], smask)
            + .1 * ce(out[7], qmask) + .1 * ce(out[8], smask))


def groups_for_epoch(ids, epoch):
    order = list(ids)
    random.Random(111 + epoch).shuffle(order)
    return [order[i:i+5] for i in range(0, 50, 5)]


def full_test(model, manifest, transform, folder, method, config):
    model.eval()
    model.shot = 50
    sx, sy = [], []
    for sid in manifest['train_ids']:
        rec = manifest['records'][sid]
        sx.append(transform(Image.open(rec['image']).convert('RGB'))[None].cuda())
        with Image.open(rec['mask']) as im:
            a = np.array(im.convert('L').resize((400, 400), Image.Resampling.NEAREST)) > 0
        sy.append(torch.from_numpy(a.astype(np.float32))[None].cuda())
    predictions = folder / 'predictions'
    predictions.mkdir(exist_ok=True)
    rows = []
    for i, sid in enumerate(manifest['test_ids']):
        rec = manifest['records'][sid]
        h, w = rec['size_hw']
        destination = predictions / f'{sid}.npy'
        if not destination.exists():
            with torch.no_grad():
                query = transform(Image.open(rec['image']).convert('RGB'))[None].cuda()
                out = model(sx, sy, [query], [None]) if method == 'MPA' else model(sx, sy, query, None)
                prediction = out[0].argmax(1)[0].float().cpu().numpy()
                del out, query
            np.save(destination, cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST))
        a = np.load(destination)
        assert a.shape == (h, w) and np.isfinite(a).all()
        gt = np.array(Image.open(rec['mask']).convert('L').resize((w, h), Image.Resampling.NEAREST)) > 0
        rows.append(image_metrics(a, gt, sid))
        print(f'{method} {manifest["dataset"]} test {i+1}/{len(manifest["test_ids"])}', flush=True)
    assert {p.stem for p in predictions.glob('*.npy')} == set(manifest['test_ids'])
    result = summarize(rows)
    result.update(config, dataset=manifest['dataset'], images=len(rows), complete_test=True,
                  epochs=config['maximum_epochs'], schedule_status=config.get('schedule_status', 'fixed_cap500_train_curve_review'),
                  prediction_coverage_verified=True)
    save_csv(folder / 'metrics/per_image.csv', rows)
    write_json(folder / 'metrics/summary.json', result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['IFA', 'MPA'], required=True)
    p.add_argument('--dataset', choices=['IRSTD-1k', 'NUDT-SIRST', 'SIRST'], required=True)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--gpu-smoke', action='store_true')
    p.add_argument('--run-root', type=Path, help='Isolated result root; does not resume the historical fss50_v2 run')
    p.add_argument('--retain-epochs', type=int, nargs='+', default=[])
    args = p.parse_args()
    assert 1 <= args.epochs <= 500
    if any(e < 1 or e > args.epochs for e in args.retain_epochs):
        raise ValueError('Retained epochs must fall within the fixed training horizon')
    torch.set_num_threads(2)
    random.seed(111); np.random.seed(111); torch.manual_seed(111)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(111)
    root = DEFAULT_OUTPUT
    folder = (args.run_root if args.run_root else root / 'fss50_v2') / args.dataset / args.method
    manifest = json.loads((root / args.dataset / 'protocol/manifest.json').read_text(encoding='utf-8'))
    model, native, lossfn = native_model(root, args.method)
    transform = transforms.Compose([transforms.Resize((400, 400)), transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    dataset = episode_dataset(native, manifest, transform, folder, args.method)
    if args.preflight:
        seen = []
        for group in groups_for_epoch(manifest['train_ids'], 0):
            dataset.active_ids = group
            sx, sy, qx, qy, *_ = dataset[0]
            assert sx.shape == (5, 3, 400, 400) and sy.shape == (5, 400, 400)
            assert torch.isfinite(qx).all() and set(torch.unique(sy).tolist()) <= {0., 1.}
            seen.extend(group)
        assert set(seen) == set(manifest['train_ids']) and len(seen) == 50
        model.eval(); model.shot = 50
        imgs = [torch.rand(1, 3, 64, 64) for _ in range(50)]
        masks = [torch.zeros(1, 64, 64) for _ in imgs]
        for a in masks: a[:, 25:35, 25:35] = 1
        with torch.no_grad():
            out = model(imgs, masks, [imgs[0]], [None]) if args.method == 'MPA' else model(imgs, masks, imgs[0], None)
        assert out[0].shape == (1, 2, 64, 64) and torch.isfinite(out[0]).all()
        print('PASS native augmentations; epoch coverage50; native50-support CPU forward', flush=True)
        return
    model.cuda()
    lr = .0005 if args.method == 'IFA' else .001
    commit = subprocess.check_output(['git', '-C', str(root/'sources'/args.method), 'rev-parse', 'HEAD'], text=True).strip()
    weight = root/'weights'/('ifa_fss_resnet50_5shot.pth' if args.method == 'IFA' else 'resnet50_deepstem_mpa.pth')
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    expected = ('3119cc60fc84964c3222035cf1cce6e84fcb2d2ca27da601bb975aa2dbf136ce' if args.method == 'IFA'
                else '1235e2d602f14e4634811a03c172e2e04a04fd86c82b37a2a9ed75f109f45524')
    assert digest == expected, 'Pretrained weight hash mismatch'
    config = dict(method=args.method, protocol='fss50_v2_native_adaptation', actual_annotation_budget=50,
                  allocated_annotation_budget=50, train_annotation_ids=manifest['train_ids'],
                  training_episode_support_shots=5, inference_support_shots=50, support_shots=50,
                  episodes_per_epoch=10, maximum_epochs=args.epochs, maximum_optimizer_updates=args.epochs*10,
                  epoch_definition='one balanced pass over all50 labels;10 five-support native pseudoquery episodes',
                  seed=111, input_size=400, batch_size=1, optimizer='SGD', lr=lr, momentum=.9,
                  weight_decay=.0005, source_commit=commit, initialization_sha256=digest,
                  manifest_sha256=manifest['manifest_sha256'], target_gradient_updates=True,
                  query_labels_for_inference=False, test_selection='final checkpoint only; no test-selected epochs',
                  stopping='fixed declared cap; augmentation curves are not validation; no unverified convergence claim',
                  learning_rate_schedule='native step decay at one-third and two-thirds total optimizer updates',
                  baseline_comparability='old5 results retained; IFA now adds target adaptation, not a pure sample-only ablation')
    if args.retain_epochs:
        config.update(retained_epochs=sorted(set(args.retain_epochs)),
                      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      test_selection='predeclared epoch checkpoints; evaluate after training; never select using test')
    folder.mkdir(parents=True, exist_ok=True)
    cp = folder/'checkpoints/latest.pt'
    if (folder/'config.json').exists():
        assert json.loads((folder/'config.json').read_text(encoding='utf-8')) == config
    if not args.gpu_smoke: write_json(folder/'config.json', config)
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad], lr=lr, momentum=.9, weight_decay=.0005)
    ce = torch.nn.CrossEntropyLoss(ignore_index=255)
    start, phase, counter, best, history = 0, 1, 0, 0., []
    if cp.exists() and not args.gpu_smoke:
        state = torch.load(cp, map_location='cpu', weights_only=False)
        assert state['config'] == config
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start, phase, counter, best, history = [state[k] for k in ('epoch','phase','counter','best','history')]
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(start, args.epochs):
        t0 = time.monotonic(); model.train(); model.shot = 5
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d): module.eval()
        used, losses, scores = [], [], []
        groups = groups_for_epoch(manifest['train_ids'], epoch)
        if args.gpu_smoke: groups = groups[:1]
        for group_index, group in enumerate(groups):
            dataset.active_ids = group
            sx, sy, qx, qy, *_ = dataset[group_index]
            sx = [a[None].cuda() for a in sx]; sy = [a[None].cuda() for a in sy]
            smask = torch.cat(sy).long()
            optimizer.zero_grad(set_to_none=True)
            if args.method == 'MPA':
                nphase = 6 if args.gpu_smoke else phase
                qx = [a[None].cuda() for a in qx[:nphase]]; qy = [a[None].long().cuda() for a in qy[:nphase]]
                out = model(sx, sy, qx, qy)
                ls = lossfn(*out, qy, smask, True, ce)
                loss = ls[0] + .4*ls[1] + .1*ls[2] + ls[3] + .2*ls[4]
                pred, target = out[0][0].detach().argmax(1), qy[0]
            else:
                qx, qy = qx[None].cuda(), qy[None].long().cuda()
                out = model(sx, sy, qx, qy)
                loss = ifa_loss(out, qy, smask, ce)
                pred, target = out[0].detach().argmax(1), qy
            assert torch.isfinite(loss), f'nonfinite loss epoch{epoch+1}'
            loss.backward()
            assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
            optimizer.step()
            losses.append(float(loss.detach()))
            scores.append(100*((pred==1)&(target==1)).sum().item()/max(((pred==1)|(target==1)).sum().item(),1))
            used.extend(group)
            update = epoch*10+group_index+1
            if update in (args.epochs*10//3, args.epochs*20//3):
                for pg in optimizer.param_groups: pg['lr'] /= 10
            del out, loss, sx, sy, smask, qx, qy, pred, target
            if args.method == 'MPA': del ls
        if args.gpu_smoke:
            model.eval(); model.shot=50
            imgs = [transform(Image.open(manifest['records'][sid]['image']).convert('RGB'))[None].cuda() for sid in manifest['train_ids']]
            masks=[]
            for sid in manifest['train_ids']:
                a=np.array(Image.open(folder/'canonical_train_masks'/f'{sid}.png').resize((400,400),Image.Resampling.NEAREST))>0
                masks.append(torch.from_numpy(a.astype(np.float32))[None].cuda())
            with torch.no_grad():
                out=model(imgs,masks,[imgs[0]],[None]) if args.method=='MPA' else model(imgs,masks,imgs[0],None)
            assert torch.isfinite(out[0]).all()
            print(f'PASS GPU400 {args.method} native training/backward,50-support inference; peakMiB={torch.cuda.max_memory_allocated()/2**20:.0f}',flush=True)
            return
        assert len(used)==50 and set(used)==set(manifest['train_ids'])
        history.append(dict(epoch=epoch+1, loss=float(np.mean(losses)), train_aug_IoU_percent=float(np.mean(scores)),
                            phase=phase, optimizer_updates=(epoch+1)*10, unique_supervised_images=50,
                            supervised_ids=used, seconds=time.monotonic()-t0))
        write_json(folder/'training_history.json', history)
        if (epoch+1)%25==0 or epoch+1==args.epochs or epoch+1 in args.retain_epochs:
            if args.method=='MPA' and (epoch+1)%25==0:
                score=float(np.mean([v['train_aug_IoU_percent'] for v in history[-25:]]))
                best=max(best,score)
                counter=0 if score>=best*1.01 else counter+1
                if counter>=3 and phase<6: phase,counter=phase+1,0
            state=dict(config=config,epoch=epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),
                       phase=phase,counter=counter,best=best,history=history,python_rng=random.getstate(),
                       numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            cp.parent.mkdir(parents=True,exist_ok=True)
            tmp=cp.with_suffix('.tmp'); torch.save(state,tmp); tmp.replace(cp)
            torch.save(state,cp.parent/f'epoch_{epoch+1:04d}.pt')
            write_json(folder/'schedule_decision.json',dict(evaluated_epoch=epoch+1,maximum_epochs=args.epochs,
                       convergence_epoch=None,status='fixed_cap_needs_curve_review',test_labels_used=False))
        print(f'{args.method} {args.dataset}: epoch{epoch+1}/{args.epochs} loss{history[-1]["loss"]:.6f} phase{phase} unique50 updates{(epoch+1)*10} seconds{history[-1]["seconds"]:.2f}',flush=True)
    if args.retain_epochs:
        for retained_epoch in sorted(set(args.retain_epochs)):
            state = torch.load(folder/'checkpoints'/f'epoch_{retained_epoch:04d}.pt', map_location='cpu', weights_only=False)
            if state['config'] != config or state['epoch'] != retained_epoch:
                raise ValueError('Retained checkpoint identity mismatch')
            model.load_state_dict(state['model'])
            evaluation = folder/f'epoch_{retained_epoch:04d}'
            evaluation.mkdir(parents=True, exist_ok=True)
            full_test(model,manifest,transform,evaluation,args.method,
                      dict(config, maximum_epochs=retained_epoch, training_horizon=args.epochs,
                           checkpoint_sha256=hashlib.sha256((folder/'checkpoints'/f'epoch_{retained_epoch:04d}.pt').read_bytes()).hexdigest(),
                           schedule_status='predeclared_checkpoint_from_fixed500_trajectory'))
    else:
        full_test(model,manifest,transform,folder,args.method,config)


if __name__=='__main__': main()
