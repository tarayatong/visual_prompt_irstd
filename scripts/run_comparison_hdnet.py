"""Train official HDNet from scratch using only each frozen train50 manifest."""
from __future__ import annotations

import argparse
import json
import importlib
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.prepare_comparison import DEFAULT_OUTPUT, write_json
from scripts.run_comparison_ours import save_csv
from visual_prompt.evaluation.comparison import image_metrics, summarize


def train_metric_stability(history):
    recent = history[-5:]
    foreground = len(recent) == 5 and all(
        row["tp"] > 0 and row["predictions"] > 0
        and row.get("pixels", 0) - row["tp"] - row.get("fp", 0) - row.get("fn", 0) > 0
        for row in recent)
    values = [row["IoU_percent"] for row in recent]
    stable = (foreground and np.ptp(values) <= 1.0
              and np.ptp(values) / max(float(np.mean(values)), 1e-8) <= .05)
    return bool(stable), bool(foreground)


class ManifestDataset(Dataset):
    def __init__(self, manifest, training, official_data_class):
        self.manifest, self.training = manifest, training
        self.ids = manifest["train_ids"] if training else manifest["test_ids"]
        # Reuse the author's scale/crop/mirror/blur and normalization exactly.
        self.transformer = official_data_class.__new__(official_data_class)
        self.transformer.crop_size = self.transformer.base_size = 256

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        record = self.manifest["records"][sid]
        image = Image.open(record["image"]).convert("RGB")
        mask = Image.open(record["mask"]).convert("L").resize(image.size, Image.Resampling.NEAREST)
        if self.training:
            image, mask = self.transformer._sync_transform(image, mask)
        else:
            image, mask = self.transformer._testval_sync_transform(image, mask)
        x = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255)
        x = (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        # Some author augmentations expand replicated binary masks to RGB.
        mask_array = np.array(mask.convert("L") if isinstance(mask, Image.Image) else mask)
        if mask_array.ndim == 3:
            mask_array = np.asarray(Image.fromarray(mask_array).convert("L"))
        y = torch.from_numpy((mask_array > 0).astype(np.float32))[None]
        return x, y, sid


class NativeDeepOutputs(torch.nn.Module):
    """Expose native same-resolution deep outputs through the shared loop."""
    def __init__(self, native):
        super().__init__()
        self.native = native

    def forward(self, x, warm_flag):
        outputs = self.native(x)
        if isinstance(outputs, (list, tuple)):
            return list(outputs[:-1]), outputs[-1]
        return [], outputs


class GrayCropManifestDataset(Dataset):
    def __init__(self, manifest, training, functions):
        self.manifest, self.training = manifest, training
        self.ids = manifest["train_ids"] if training else manifest["test_ids"]
        self.random_crop, self.augment = functions["random_crop"], functions["augment"]()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        record = self.manifest["records"][sid]
        image = Image.open(record["image"]).convert("L")
        mask = Image.open(record["mask"]).convert("L").resize(image.size, Image.Resampling.NEAREST)
        image = np.array(image, dtype=np.float32)
        mask = (np.array(mask) > 0).astype(np.float32)
        stats = self.manifest["train50_gray_stats"]
        image = (image - stats["mean"]) / stats["std"]
        if self.training:
            image, mask = self.random_crop(image, mask, 256, pos_prob=.5)
            image, mask = self.augment(image, mask)
        else:
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(np.ascontiguousarray(image[None])), torch.from_numpy(np.ascontiguousarray(mask[None])), sid


class RPCAManifestDataset(Dataset):
    """Native grayscale /255 resize; native augmentation only for SIRST."""
    def __init__(self, manifest, training, augmentation):
        self.manifest, self.training = manifest, training
        self.ids = manifest["train_ids"] if training else manifest["test_ids"]
        self.augment = augmentation()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        record = self.manifest["records"][sid]
        image = cv2.imread(record["image"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(record["mask"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.training and self.manifest.get("dataset") == "SIRST":
            image, mask = self.augment(image, mask)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        # Guard empty masks, avoiding the author's division by zero on tiny targets.
        mask = mask.astype(np.float32) / max(float(mask.max()), 1)
        return torch.from_numpy(image[None]), torch.from_numpy(mask[None]), sid


class RPCANativeOutputs(torch.nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native

    def forward(self, x, warm_flag):
        reconstructed, target = self.native(x)
        self.reconstruction_loss = F.mse_loss(reconstructed, x) if self.training else None
        return [], target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", default=["IRSTD-1k", "NUDT-SIRST", "SIRST"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--method", choices=["HDNet", "MSHNet", "DNANet", "SCTransNet", "RPCANet", "ILNet", "MSHNet+NS-FPN", "ISNet"], default="HDNet")
    parser.add_argument("--output-group", default="")
    parser.add_argument("--adaptive-schedule", action="store_true")
    args = parser.parse_args()
    if args.output_group and (Path(args.output_group).is_absolute() or ".." in Path(args.output_group).parts):
        raise ValueError("output-group must be a relative folder within the result root")
    source = args.root / "sources" / ("NS-FPN" if args.method == "MSHNet+NS-FPN" else args.method)
    sys.path.insert(0, str(source.resolve()))
    sys.path.append(str((args.root / "runtime_packages").resolve()))
    dataset_class = ManifestDataset
    prediction_is_probability = args.method == "SCTransNet"
    multiscale_targets = args.method != "DNANet"
    if args.method == "ISNet":
        from scripts.comparison_isnet_adapter import enable_torchvision_dcn, NativeISNetOutputs, gradient_magnitude
        enable_torchvision_dcn(args.root)
        from ISNet import ISNet
        from loss import SoftLoULoss1
        from utils1.dataset_IRSTD1K import SirstDataset as IRSTD_Dataset
        multiscale_targets = False
        def HDNet(channels):
            native = ISNet([4,4,4], [8,16,32,64])
            def initialize(module):
                if isinstance(module, torch.nn.Conv2d):
                    torch.nn.init.normal_(module.weight, 0, .02)
                elif isinstance(module, torch.nn.BatchNorm2d):
                    torch.nn.init.normal_(module.weight, 1, .02)
                    torch.nn.init.normal_(module.bias, 0)
            native.apply(initialize)
            return NativeISNetOutputs(native)
        def SLSIoULoss():
            return lambda prediction, target, warm, epoch: SoftLoULoss1()(prediction, target)
    elif args.method == "MSHNet+NS-FPN":
        from scripts.comparison_nsfpn_adapter import enable_author_reference_backend
        enable_author_reference_backend()
        from model.MSHNet_NSFPN import MSHNet_NSFPN as HDNet
        from model.loss import SLSIoULoss
        from utils.data import IRSTD_Dataset
    elif args.method == "ILNet":
        from model.ilnet import ILNet_L
        from utils.data import SirstDataset
        from utils.Adan import Adan
        from utils.lr_scheduler import create_lr_scheduler
        from scripts.comparison_ilnet_adapter import ILNetManifestDataset, ILNetNativeOutputs, train50_stats
        dataset_class, IRSTD_Dataset = ILNetManifestDataset, SirstDataset
        multiscale_targets = False
        def HDNet(channels):
            return ILNetNativeOutputs(ILNet_L())
        def SLSIoULoss():
            return lambda prediction, target, warm, epoch: F.binary_cross_entropy_with_logits(prediction, target)
    elif args.method == "RPCANet":
        from models import get_model
        from utils.data import augmentation
        from utils.loss import SoftLoULoss
        from utils.lr_scheduler import LR_Scheduler
        IRSTD_Dataset = augmentation
        dataset_class = RPCAManifestDataset
        multiscale_targets = False
        def HDNet(channels):
            return RPCANativeOutputs(get_model("rpcanet"))
        def SLSIoULoss():
            return lambda prediction, target, warm, epoch: SoftLoULoss()(prediction, target)
    elif args.method == "SCTransNet":
        from model.SCTransNet import SCTransNet, get_CTranS_config
        from utils import random_crop, weights_init_kaiming
        from dataset import augumentation
        IRSTD_Dataset = dict(random_crop=random_crop, augment=augumentation)
        dataset_class = GrayCropManifestDataset
        multiscale_targets = False
        def HDNet(channels):
            native = SCTransNet(get_CTranS_config(), mode="train", deepsuper=True)
            native.apply(weights_init_kaiming)
            return NativeDeepOutputs(native)
        def SLSIoULoss():
            return lambda prediction, target, warm, epoch: F.binary_cross_entropy(prediction, target)
    elif args.method == "DNANet":
        from model.model_DNANet import DNANet, Res_CBAM_block
        from model.loss import SoftIoULoss
        from model.utils import TrainSetLoader, TestSetLoader, weights_init_xavier
        class IRSTD_Dataset(TrainSetLoader):
            _testval_sync_transform = TestSetLoader._testval_sync_transform
        def HDNet(channels):
            native = DNANet(1, channels, Res_CBAM_block, [2, 2, 2, 2], [16, 32, 64, 128, 256], deep_supervision=True)
            native.apply(weights_init_xavier)
            return NativeDeepOutputs(native)
        def SLSIoULoss():
            return lambda prediction, target, warm, epoch: SoftIoULoss(prediction, target)
    else:
        HDNet = getattr(importlib.import_module(f"model.{args.method}"), args.method)
        from model.loss import SLSIoULoss
        from utils.data import IRSTD_Dataset
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if args.smoke:
        model = HDNet(3).cuda()
        x = torch.zeros(2, 1 if args.method in ("SCTransNet", "RPCANet") else 3, 256, 256, device="cuda")
        y = torch.zeros(2, 1, 256, 256, device="cuda")
        y[:, :, 120:124, 120:124] = 1
        masks, prediction = model(x, True)
        loss = SLSIoULoss()(prediction, y, 5, 6)
        if args.method == "RPCANet":
            loss = loss + .01 * model.reconstruction_loss
        loss.backward()
        print(dict(shape=list(prediction.shape), loss=float(loss.detach()), commit=commit))
        return
    for dataset_name in args.datasets:
        root = args.root / dataset_name
        manifest = json.loads((root / "protocol/manifest.json").read_text(encoding="utf-8"))
        manifest["dataset"] = dataset_name
        if args.method == "ILNet":
            manifest["train50_rgb_stats"] = train50_stats(manifest)
        if args.method == "SCTransNet":
            total, total_squared, count = 0.0, 0.0, 0
            for sid in manifest["train_ids"]:
                pixels = np.array(Image.open(manifest["records"][sid]["image"]).convert("L"), dtype=np.float64)
                total += float(pixels.sum())
                total_squared += float(np.square(pixels).sum())
                count += pixels.size
            mean = total / count
            manifest["train50_gray_stats"] = dict(mean=mean, std=max((total_squared / count - mean**2)**.5, 1e-6), source="fixed train50 only")
        output = args.root / args.output_group / dataset_name / args.method
        checkpoint_path = output / "checkpoints/latest.pt"
        random.seed(manifest["seed"])
        np.random.seed(manifest["seed"])
        torch.manual_seed(manifest["seed"])
        torch.cuda.manual_seed_all(manifest["seed"])
        generator = torch.Generator().manual_seed(manifest["seed"])
        batch_size = 2 if args.method == "ILNet" else 4
        loader = DataLoader(dataset_class(manifest, True, IRSTD_Dataset), batch_size=batch_size,
                            shuffle=True, num_workers=0, drop_last=True, generator=generator)
        model = HDNet(3).cuda()
        optimizer = (Adan(model.parameters(), lr=.001, weight_decay=1e-4) if args.method == "ILNet"
                     else torch.optim.Adam(model.parameters(), lr=5e-4) if args.method == "RPCANet"
                     else torch.optim.Adam(model.parameters(), lr=.001) if args.method == "SCTransNet"
                     else torch.optim.Adagrad(model.parameters(), lr=.05, weight_decay=1e-4 if args.method == "ISNet" else 0))
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25, eta_min=.045)
                     if args.method == "HDNet" else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
                     if args.method in ("DNANet", "SCTransNet") else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0))
        criterion = SLSIoULoss()
        if args.method == "ISNet":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30)
        if args.method == "ILNet":
            scheduler = create_lr_scheduler(optimizer, len(loader), args.epochs, warmup=True, warmup_epochs=10)
        scaler = torch.amp.GradScaler("cuda", enabled=args.method == "ILNet")
        native_scheduler = LR_Scheduler("poly", 5e-4, args.epochs, len(loader), quiet=True) if args.method == "RPCANet" else None
        start, history, metric_history, convergence_epoch = 0, [], [], None
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if ckpt["manifest_sha256"] != manifest["manifest_sha256"] or ckpt["commit"] != commit:
                raise ValueError("HDNet checkpoint source or supervision changed")
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if ckpt.get("scaler"):
                scaler.load_state_dict(ckpt["scaler"])
            start, history = ckpt["epoch"], ckpt["history"]
            metric_history = ckpt.get("metric_history", [])
            convergence_epoch = ckpt.get("convergence_epoch")
            if (args.adaptive_schedule and convergence_epoch is not None
                    and not train_metric_stability(metric_history)[1]):
                print("Discarding a degenerate all-background/all-foreground plateau; continue from saved training state", flush=True)
                convergence_epoch = None
            random.setstate(ckpt["random_state"])
            np.random.set_state(ckpt["numpy_state"])
            torch.set_rng_state(ckpt["torch_state"])
            torch.cuda.set_rng_state_all(ckpt["cuda_state"])
            generator.set_state(ckpt["loader_state"])
        write_json(output / "config.json", dict(commit=commit, epochs=args.epochs, train_images=50,
                   initialization="random", optimizer="Adam" if args.method == "SCTransNet" else "Adagrad", lr=.001 if args.method == "SCTransNet" else .05, batch_size=4, input_size=256,
                   scheduler="author CosineAnnealingLR(T_max=25, eta_min=.045)" if args.method == "HDNet" else f"CosineAnnealingLR(T_max={args.epochs}, eta_min=1e-5); horizon is declared safety cap" if args.method in ("DNANet", "SCTransNet") else "author constant learning rate", warm_epoch=5,
                   method=args.method, adaptive_schedule=args.adaptive_schedule,
                   normalization=manifest.get("train50_gray_stats", "ImageNet RGB"),
                   test_resize="256x256 before original-resolution evaluation",
                   selection="final fixed epoch; never evaluate test during training", manifest_sha256=manifest["manifest_sha256"]))
        if args.method == "MSHNet+NS-FPN":
            path = output / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config.update(deformable_attention_backend="author ms_deform_attn_core_pytorch reference, autograd; CUDA extension unavailable",
                          backend_validation="CPU finite-difference gradcheck and native64px forward/backward pass; CUDA-kernel numerical comparison unavailable",
                          deviations="reference backend changes runtime performance; architecture and learned parameterization preserved;500epoch cap")
            write_json(path, config)
        if args.method == "ISNet":
            path = output / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config.update(weight_decay=1e-4, scheduler="native CosineAnnealingLR(T_max=30,eta_min=0),step(epoch)",
                          model="native [4,4,4] blocks,[8,16,32,64] channels and native normal initialization",
                          loss="native SoftIoU1(main)+10*BCE(edge)+SoftIoU1(edge)",
                          edge_target="native gradient magnitude of existing segmentation mask, clamped max1 for BCE validity; no extra labels",
                          deformable_backend="torchvision compiled modulated deform_conv2d; native DCN modules/offset/mask layout retained",
                          backend_validation="CPU zero-offset/mask1 conv equivalence, all5 operand gradients, native64px forward/backward passed; author _ext comparison unavailable",
                          deviations="256 crop/resize rather than480/512; batch4 rather than12; clamp edge target max1; alternative equivalent compiled DCN backend;500epoch cap")
            write_json(path, config)
        if args.method == "ILNet":
            path = output / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config.update(optimizer="native Adan", lr=.001, weight_decay=1e-4, batch_size=2, input_size=512,
                          scheduler=f"native per-batch warmup10+cosine; horizon={args.epochs}", amp=True,
                          normalization=manifest["train50_rgb_stats"], loss="native sum BCEWithLogits(main-first and auxiliary heads)",
                          test_resize="512x512 before original-resolution evaluation",
                          deviations="batch2 rather than author8 for10GB GPU; train50-only normalization instead of author dataset constants;500epoch hard cap")
            write_json(path, config)
        if args.method == "RPCANet":
            path = output / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config.update(optimizer="Adam", lr=5e-4, scheduler=f"native per-batch poly power0.9; horizon={args.epochs} safety cap",
                          normalization="grayscale /255", loss="native SoftIoU(target)+0.01*MSE(reconstruction,input)",
                          augmentation="native SIRST augmentation; native IRSTD/NUDT resize only",
                          deviations="batch4 rather than author8; empty-mask divide-by-zero guarded; adaptive schedule instead of author50epochs")
            write_json(path, config)
        final_epoch = start
        for epoch in tqdm(range(start, args.epochs), desc=f"{args.method} {dataset_name}"):
            if args.adaptive_schedule and convergence_epoch is not None and epoch >= convergence_epoch + 100:
                break
            model.train()
            losses = []
            for batch_index, (x, y, _) in enumerate(loader):
                if native_scheduler is not None:
                    native_scheduler(optimizer, batch_index, epoch, 0)
                x, y = x.cuda(), y.cuda()
                with torch.autocast(device_type="cuda", enabled=args.method == "ILNet"):
                    masks, pred = model(x, epoch > 5)
                    loss = criterion(pred, y, 5, epoch)
                if args.method == "ISNet":
                    edge_target = gradient_magnitude(y).clamp(max=1)
                    edge = model.edge_prediction
                    loss = loss + 10 * F.binary_cross_entropy(edge, edge_target) + criterion(edge, edge_target, 5, epoch)
                if args.method == "RPCANet":
                    loss = loss + .01 * model.reconstruction_loss
                current = y
                for j, mask in enumerate(masks):
                    if j and multiscale_targets:
                        current = F.max_pool2d(current, 2, 2)
                    with torch.autocast(device_type="cuda", enabled=args.method == "ILNet"):
                        loss = loss + criterion(mask, current, 5, epoch)
                if args.method not in ("SCTransNet", "ILNet"):
                    loss = loss / (len(masks) + 1)
                if not torch.isfinite(loss):
                    raise ValueError(f"HDNet nonfinite loss at epoch {epoch}")
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if args.method == "ILNet":
                    scheduler.step()
                losses.append(float(loss.detach()))
            if args.method == "ISNet":
                scheduler.step(epoch)
            elif args.method != "ILNet":
                scheduler.step()
            history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses))))
            final_epoch = epoch + 1
            if args.adaptive_schedule and final_epoch % 25 == 0:
                train_view = dict(manifest, test_ids=manifest["train_ids"])
                metric_loader = DataLoader(dataset_class(train_view, False, IRSTD_Dataset), batch_size=1, num_workers=0)
                model.eval()
                train_rows = []
                with torch.no_grad():
                    for mx, _, mids in metric_loader:
                        sid = mids[0]
                        _, logits = model(mx.cuda(), True)
                        record = manifest["records"][sid]
                        h, w = record["size_hw"]
                        probability = cv2.resize((logits if prediction_is_probability else torch.sigmoid(logits))[0, 0].cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
                        gt = np.array(Image.open(record["mask"]).convert("L").resize((w, h), Image.Resampling.NEAREST)) > 0
                        train_rows.append(image_metrics(probability, gt, sid))
                scores = summarize(train_rows)
                scores.update(epoch=final_epoch, split="train50_no_augmentation", used_for_schedule=True)
                metric_history.append(scores)
                write_json(output / "training_metric_history.json", metric_history)
                loss_values = [row["loss"] for row in history]
                stable_loss = final_epoch >= 200
                for end in (final_epoch - 50, final_epoch - 25, final_epoch):
                    old = float(np.median(loss_values[max(0, end-100):max(0, end-50)])) if end >= 100 else 0
                    new = float(np.median(loss_values[max(0, end-50):end]))
                    stable_loss = stable_loss and abs(new-old)/max(abs(old), 1e-8) < .05
                stable_metrics, foreground_learned = train_metric_stability(metric_history)
                if convergence_epoch is None and stable_loss and stable_metrics:
                    convergence_epoch = final_epoch
                    print(f"Convergence confirmed at {final_epoch}; final epoch={final_epoch+100}", flush=True)
                write_json(output / "schedule_decision.json", dict(
                    convergence_epoch=convergence_epoch, final_epoch=convergence_epoch + 100 if convergence_epoch else None,
                    evaluated_epoch=final_epoch, stable_loss=bool(stable_loss), stable_train_iou=bool(stable_metrics),
                    nondegenerate_foreground=foreground_learned,
                    criterion="adjacent50 loss median change<5% at three25 checks; nonzero TP/predictions/TN at five train50 evaluations; IoU range<=1pp AND relative range<=5% across100epochs; min200; then+100; hardcap500",
                    test_labels_used=False, maximum_epochs=args.epochs))
            if (epoch + 1) % 25 == 0 or epoch + 1 == args.epochs:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                epoch=epoch + 1, history=history, manifest_sha256=manifest["manifest_sha256"], commit=commit,
                                metric_history=metric_history, convergence_epoch=convergence_epoch,
                                random_state=random.getstate(), numpy_state=np.random.get_state(),
                                torch_state=torch.get_rng_state(), cuda_state=torch.cuda.get_rng_state_all(),
                                loader_state=generator.get_state(), scaler=scaler.state_dict())
                torch.save(checkpoint_state, checkpoint_path)
                if epoch + 1 <= 500:
                    torch.save(checkpoint_state, checkpoint_path.parent / f"epoch_{epoch+1:04d}.pt")
                write_json(output / "training_history.json", history)
        model.eval()
        config_path = output / "config.json"
        resolved_config = json.loads(config_path.read_text(encoding="utf-8"))
        resolved_config.update(actual_final_epoch=final_epoch, convergence_epoch=convergence_epoch,
                               epoch_argument_is="safety cap" if args.adaptive_schedule else "fixed final epoch")
        write_json(config_path, resolved_config)
        pred_dir = output / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        test_loader = DataLoader(dataset_class(manifest, False, IRSTD_Dataset), batch_size=1, num_workers=0)
        with torch.no_grad():
            for x, _, ids in tqdm(test_loader, desc=f"HDNet test {dataset_name}"):
                sid = ids[0]
                _, logits = model(x.cuda(), True)
                record = manifest["records"][sid]
                h, w = record["size_hw"]
                probability = cv2.resize((logits if prediction_is_probability else torch.sigmoid(logits))[0, 0].cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
                np.save(pred_dir / f"{sid}.npy", probability)
                gt = np.array(Image.open(record["mask"]).convert("L").resize((w, h), Image.Resampling.NEAREST)) > 0
                rows.append(image_metrics(probability, gt, sid))
        result = summarize(rows)
        result.update(method=args.method, dataset=dataset_name, complete_test=True, actual_annotation_budget=50,
                      support_shots=0, threshold=.5, epochs=final_epoch, source_commit=commit,
                      convergence_epoch=convergence_epoch, adaptive_schedule=args.adaptive_schedule,
                      schedule_status=("cap_before_post_convergence100" if convergence_epoch is not None and final_epoch < convergence_epoch + 100
                                       else "convergence_plus100" if convergence_epoch is not None
                                       else "cap_without_confirmed_convergence" if args.adaptive_schedule else "fixed_pilot"),
                      manifest_sha256=manifest["manifest_sha256"])
        save_csv(output / "metrics/per_image.csv", rows)
        write_json(output / "metrics/summary.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
