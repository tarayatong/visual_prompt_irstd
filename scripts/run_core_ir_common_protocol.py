"""Retrain the complexity-table methods at a common external resolution."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.flop_counter import FlopCounterMode

from scripts import run_recent_a_comparison as base
from scripts.eval_tiny_fixed_memory_radial import make_predictor
from scripts.run_tiny_ablation_combined import restore, state_for
from visual_prompt.models.tiny_prototype_feature_radial import install_prototype_feature_radial
from visual_prompt.models.tiny_prototype_memory_replacement import install_prototype_memory
from visual_prompt.models.tiny_shared_memory import support_regions

OUT = base.ROOT / "outputs/core_ir_followup_20261117"
CNN = ("ISNet", "DNANet", "MSHNet", "SCTransNet", "RPCANet", "ILNet", "HDNet", "MSHNet+NS-FPN")
SAM = ("IRPruneDet", "IRSAM", "IR-SAM2", "SAMamba")
MEMORY = ("Ours", "SANSA")
METHODS = (*MEMORY, *CNN, "SAM-SPL", "MLP-Net", "MPCNet", *SAM)
SUPPORT_DRAW_SEEDS = (20261116, 20261117, 20261118)
RETAINED_CORE = {
    ("SIRST", 111): "tiny_radial_detail_other_datasets_seed111/SIRST/only_256/combined",
    ("IRSTD-1k", 113): "tiny_radial_detail_irstd_seed113/only_256/combined",
    ("NUDT-SIRST", 111): "tiny_radial_detail_other_datasets_seed111/NUDT-SIRST/only_256/combined",
}
DEVICE = torch.device("cuda")
EPOCHS = 120
BATCH = 4
NATIVE_FULL_METHODS = ("Ours", "SANSA", "ILNet")
NATIVE_FULL_SIZES = {"Ours": 640, "SANSA": 640, "ILNet": 512}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def manifest(dataset, size, full):
    result = base.manifest_for(dataset)
    ids = result["source_train_ids"] if full else result["train_ids"]
    if len(set(ids)) != len(ids) or set(ids) & set(result["test_ids"]):
        raise ValueError(f"Invalid train/test split for {dataset}")
    identity = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    return dict(result, working_size=size, train_ids=ids, train_ids_sha256=identity)


def support_draw(m, seed, candidates=None):
    ids = m["train_ids"] if candidates is None else candidates
    return random.Random(f"{m['dataset']}:{seed}").sample(sorted(ids), 5)


def read_image(m, sid, rgb):
    size = m["working_size"]
    with Image.open(m["records"][sid]["image"]) as image:
        return np.array(image.convert("RGB" if rgb else "L").resize(
            (size, size), Image.Resampling.BILINEAR), dtype=np.float32)


def normalized(x, mean, std):
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1) if x.ndim == 3 else x[None]
    return (x - mean) / std


RGB_MEAN = torch.tensor([.485, .456, .406])[:, None, None] * 255
RGB_STD = torch.tensor([.229, .224, .225])[:, None, None] * 255


def sam_objective(model, batch, ids, m, ours, rng):
    from util.losses import loss_masks
    from util.promptable_utils import build_prompt_dict

    images, masks, geometry = [], [], []
    size = m["working_size"]
    pool = list(ids)
    for query in batch:
        index = rng.randrange(len(pool) - 1)
        # Skip the query's slot while preserving a uniform draw from its peers.
        support = pool[-1] if pool[index] == query else pool[index]
        pairs = [base.raw_pair(m, sid, size, True) for sid in (support, query)]
        images.append(torch.stack([normalized(x, RGB_MEAN, RGB_STD) for x, _ in pairs]))
        masks.append(torch.stack([torch.from_numpy(y) for _, y in pairs]))
        if ours:
            x, y = pairs[0]
            geometry.append([support_regions(x.astype(np.uint8), y > 0)])
    x, y = torch.stack(images).cuda(), torch.stack(masks).cuda()
    if ours:
        model.support_geometry = geometry
        model.radial_branch.begin([(None, None)] * len(batch))
    try:
        prompt = build_prompt_dict(y, "mask", 1, True, DEVICE)
        output = model(x, prompt)["pred_masks"]
        target = y[:, -1:] if ours else y
        return sum(loss_masks(output, target, num_frames=1).values())
    finally:
        if ours:
            model.radial_branch.finish()


def memory_model(method, m):
    model, _, _, _, _, _, identity = base.setup_sansa(m, DEVICE)
    ours = method == "Ours"
    if ours:
        install_prototype_memory(model)
        install_prototype_feature_radial(model, active_stages=(1,))
    return model, dict(identity, variant="CoRe-IR final decoder" if ours else "SANSA Tiny",
                       backbone_input_size=model.sam.image_size)


def memory_predictor(method, model, m):
    return make_predictor(model, m, "learned" if method == "Ours" else "native",
                          method == "Ours", support_batch_size=1)


def cnn_adapter(method, m):
    from scripts.comparison_ilnet_adapter import train50_stats
    from scripts.run_nuaa_coordinate_corrected import build_infrared
    from scripts.comparison_isnet_adapter import gradient_magnitude

    ns = build_infrared(method)
    size = m["working_size"]
    if method == "SCTransNet":
        native = ns["SCTransNet"](ns["get_CTranS_config"](), img_size=size, mode="train", deepsuper=True)
        native.apply(ns["weights_init_kaiming"])
        model = ns["NativeDeepOutputs"](native)
    else:
        model = ns["HDNet"](3)
    model.cuda()
    criterion = ns["SLSIoULoss"]()
    rgb = method not in ("SCTransNet", "RPCANet")
    mean, std = RGB_MEAN, RGB_STD
    if method == "ILNet":
        stats = train50_stats(m)
        mean, std = (torch.tensor(stats[key])[:, None, None] * 255 for key in ("mean", "std"))
    elif method == "SCTransNet":
        stats = base.train_stats(m)
        mean, std = stats["mean"], stats["std"]
    elif not rgb:
        mean, std = 0, 255

    def examples(ids):
        pairs = [base.raw_pair(m, sid, size, rgb) for sid in ids]
        return (torch.stack([normalized(x, mean, std) for x, _ in pairs]).cuda(),
                torch.stack([torch.from_numpy(y)[None] for _, y in pairs]).cuda())

    def objective(ids, epoch):
        x, y = examples(ids)
        aux, pred = model(x, epoch > 5)
        loss = criterion(pred, y, 5, epoch)
        if method == "ISNet":
            edges = model.edge_prediction
            target = gradient_magnitude(y).clamp(max=1)
            loss = loss + 10 * F.binary_cross_entropy(edges, target) + criterion(edges, target, 5, epoch)
        if method == "RPCANet":
            loss = loss + .01 * model.reconstruction_loss
        for head in aux:
            target = F.adaptive_max_pool2d(y, head.shape[-2:])
            loss = loss + criterion(head, target, 5, epoch)
        return loss if method in ("ILNet", "SCTransNet") else loss / (len(aux) + 1)

    def predict(sid):
        x = normalized(read_image(m, sid, rgb), mean, std)[None].cuda()
        pred = model(x, True)[1][0, 0]
        return pred if ns["prediction_is_probability"] else pred.sigmoid()

    source = base.DEFAULT_OUTPUT / "sources" / ("NS-FPN" if method == "MSHNet+NS-FPN" else method)
    return model, objective, predict, dict(source_commit=base.commit(source))


def ilnet_full_adapter(m):
    from scripts.comparison_ilnet_adapter import train50_stats
    from scripts.run_nuaa_coordinate_corrected import build_infrared

    ns = build_infrared("ILNet")
    from model.loss import criterion
    from utils.data import IRSTD1K_Dataset, SirstDataset

    model = ns["ILNet_L"]().cuda()
    dataset = m["dataset"]
    data_class = SirstDataset if dataset == "SIRST" else IRSTD1K_Dataset
    transform = data_class.__new__(data_class)
    transform.img_size = m["working_size"]
    if dataset == "SIRST":
        mean, std = [.35619214, .35575104, .35673013], [.2614548, .26135704, .26168558]
    elif dataset == "IRSTD-1k":
        mean, std = [.28450727, .28450724, .28450724], [.22880708, .22880709, .22880709]
    else:
        stats = train50_stats(m)
        mean, std = stats["mean"], stats["std"]
    normalization = "train-only NUDT statistics" if dataset == "NUDT-SIRST" else "author dataset constants"
    mean, std = (torch.tensor(value)[:, None, None] * 255 for value in (mean, std))

    def objective(ids, epoch):
        images, masks = [], []
        for sid in ids:
            rec = m["records"][sid]
            with Image.open(rec["image"]) as image, Image.open(rec["mask"]) as mask:
                rgb = image.convert("RGB")
                raw = np.asarray(mask.convert("L").resize(rgb.size, Image.Resampling.NEAREST))
            label = Image.fromarray((raw > 0).astype(np.uint8) * 255)
            rgb, label = transform._sync_transform(rgb, label)
            images.append(normalized(np.array(rgb, dtype=np.float32), mean, std))
            masks.append(torch.from_numpy((np.array(label) > 0).astype(np.float32))[None])
        return criterion(model(torch.stack(images).cuda()), torch.stack(masks).cuda())

    def predict(sid):
        image = normalized(read_image(m, sid, True), mean, std)
        result = model(image[None].cuda())
        top = result[0] if isinstance(result, (list, tuple)) else result
        return top[0, 0].sigmoid()

    source = base.DEFAULT_OUTPUT / "sources/ILNet"
    return model, objective, lambda view: predict, dict(source_commit=base.commit(source),
        source_schedule=ns["create_lr_scheduler"], optimizer_class=ns["Adan"],
        normalization=normalization, augmentation="author random flip, scale, pad/crop and Gaussian blur")


def make_adapter(method, m, smoke):
    if method in MEMORY:
        model, identity = memory_model(method, m)

        def objective(ids, epoch):
            return sam_objective(model, ids, m["train_ids"], m, method == "Ours", random)

        return model, objective, lambda view: memory_predictor(method, model, view), identity
    if method in CNN:
        model, objective, predict, identity = cnn_adapter(method, m)
    elif method in SAM:
        from scripts import train_sam_extension_train50 as ext

        model, _, _, _, loss, forward, identity, epoch_ref, train = ext.setup(method, m)
        indices = {sid: index for index, sid in enumerate(m["train_ids"])}

        def objective(ids, epoch):
            epoch_ref[0] = epoch + 1
            pairs = [train[indices[sid]] for sid in ids]
            return loss(*(torch.stack(tensors).cuda() for tensors in zip(*(pair[:2] for pair in pairs))))

        def predict(sid):
            x = ext.image_only(m, sid, m["working_size"], method, train.stats)
            return forward(x[None].cuda())[0, 0]
    else:
        if method == "MPCNet":
            from scripts.run_mpcnet_comparison import setup
        else:
            from scripts.comparison_added_irstd_adapter import setup
        model, _, _, loader, loss, predict, identity = setup(method, m, DEVICE, smoke)
        indices = {sid: index for index, sid in enumerate(m["train_ids"])}

        def objective(ids, epoch):
            return loss(loader.collate_fn([loader.dataset[indices[sid]] for sid in ids]), epoch)

    return model, objective, lambda view: predict, identity


def count_call(function):
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        result = function()
    return result, counter.get_total_flops()


def check_forward(adapter_predict, view, sid, memory):
    predictor, support_ops = count_call(lambda: adapter_predict(view))
    probability, query_ops = count_call(lambda: predictor(sid))
    if probability.ndim != 2 or not torch.isfinite(probability).all() or query_ops <= 0:
        raise RuntimeError(f"Invalid inference output or FLOP count for {sid}")
    return dict(GFLOPs=query_ops / 1e9, support_GFLOPs=support_ops / 1e9 if memory else None)


def worker(method, dataset, size, full, smoke, *, native_full=False, native_microbatch=None):
    base.seed_all()
    if native_full:
        size = NATIVE_FULL_SIZES[method]
    m = manifest(dataset, size, full)
    phase = "full_native" if native_full else "full" if full else "complexity"
    folder = OUT / phase / dataset / str(size) / method
    report = folder / "result.json"
    if not smoke and report.exists():
        return
    device = torch.cuda.get_device_name()
    free_at_start, _ = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    native_ilnet = native_full and method == "ILNet"
    native_sansa = native_full and method == "SANSA"
    model, objective, predict_for, identity = (
        ilnet_full_adapter(m) if native_ilnet else make_adapter(method, m, smoke))
    memory = method in MEMORY
    params = [p for p in model.parameters() if p.requires_grad]
    batch_size = 8 if native_ilnet else 1 if native_sansa else BATCH
    epochs = 600 if native_ilnet else EPOCHS
    steps = (len(m["train_ids"]) + batch_size - 1) // batch_size
    learning_rate, weight_decay = (.001, 1e-4) if native_ilnet else (1e-4, 0.)
    if native_ilnet:
        optimizer = identity["optimizer_class"](params, lr=learning_rate, weight_decay=weight_decay)
        scheduler = identity["source_schedule"](optimizer, steps, epochs, warmup=True, warmup_epochs=10)
        microbatch = native_microbatch if smoke else read(folder / "preflight.json")["microbatch"]
        amp_dtype = torch.float16
    else:
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * steps)
        microbatch = batch_size if method == "Ours" else 1
        amp_dtype = torch.bfloat16 if (memory or method == "ILNet") and not native_sansa else None
    scaler = torch.amp.GradScaler("cuda", enabled=native_ilnet)
    settings = dict(method=method, dataset=dataset, input_size=size, batch_size=batch_size,
                    microbatch=microbatch, seed=111, actual_annotation_budget=len(m["train_ids"]),
                    train_annotation_ids=m["train_ids"], train_ids_sha256=m["train_ids_sha256"],
                    manifest_sha256=m["manifest_sha256"], training_horizon=epochs, threshold=.5,
                    optimizer=optimizer.__class__.__name__, learning_rate=learning_rate,
                    weight_decay=weight_decay, updates=epochs * steps,
                    backbone_input_size=identity.get("backbone_input_size", identity.get("encoder_internal_size", size)),
                    parameters_total=sum(p.numel() for p in model.parameters()),
                    parameters_trainable=sum(p.numel() for p in params), device=device,
                    source_commit=identity.get("source_commit"), runner_sha256=base.sha(Path(__file__)),
                    precision={torch.float16: "FP16 AMP", torch.bfloat16: "BF16 autocast", None: "FP32"}[amp_dtype],
                    flop_convention="CUDA FP32 torch FlopCounterMode, FMA=2; supported operators only")
    if native_full:
        settings.update(training_protocol="method_specific_full", normalization=identity.get("normalization", "ImageNet RGB"),
                        augmentation=identity.get("augmentation", "training query and randomly drawn distinct support"),
                        schedule="author warm-up 10 epochs then per-update cosine" if native_ilnet else "per-update cosine",
                        lr_warmup_epochs=10 if native_ilnet else 0)
    cp = folder / "latest.pt"
    history, start = [], 0
    if not smoke and cp.exists():
        state = torch.load(cp, map_location="cpu", weights_only=False)
        previous_settings = dict(state["settings"])
        previous_runner = previous_settings.pop("runner_sha256")
        if previous_settings != {key: value for key, value in settings.items() if key != "runner_sha256"}:
            raise ValueError(f"Checkpoint protocol changed for {folder}")
        if previous_runner != settings["runner_sha256"]:
            print(json.dumps(dict(event="resuming_checkpoint", saved_runner_sha256=previous_runner,
                                  current_runner_sha256=settings["runner_sha256"])), flush=True)
        if method == "Ours":
            restore(model, state["weights"])
        elif method == "SANSA":
            base.load_saved(model, state["weights"])
        else:
            model.load_state_dict(state["weights"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if native_ilnet:
            scaler.load_state_dict(state["scaler"])
        random.setstate(state["random"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        history, start = state["history"], state["epoch"]
        del state
    draws = [m["support_ids"]] if native_full and memory else [
        support_draw(m, seed) if memory else [] for seed in
        (SUPPORT_DRAW_SEEDS if memory else SUPPORT_DRAW_SEEDS[:1])]
    for epoch in range(start, 1 if smoke else epochs):
        begin = time.monotonic()
        ids = list(m["train_ids"])
        random.shuffle(ids)
        if smoke:
            ids = ids[:batch_size]
        model.train()
        losses = []
        for index in range(0, len(ids), batch_size):
            batch = ids[index:index + batch_size]
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for offset in range(0, len(batch), microbatch):
                part = batch[offset:offset + microbatch]
                precision = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
                with precision:
                    loss = objective(part, epoch) * len(part) / len(batch)
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite loss for {method} at epoch {epoch + 1}")
                scaler.scale(loss).backward()
                total += loss.item()
            if not native_ilnet:
                torch.nn.utils.clip_grad_norm_(params, .1 if memory else 1., error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(total)
        row = dict(epoch=epoch + 1, loss=float(np.mean(losses)), seconds=round(time.monotonic() - begin, 2))
        history.append(row)
        print(json.dumps(dict(method=method, dataset=dataset, **row)), flush=True)
        if not smoke and ((epoch + 1) % 5 == 0 or epoch + 1 == epochs):
            weights = (state_for(model) if method == "Ours" else base.saved_model(model)
                       if method == "SANSA" else model.state_dict())
            save_checkpoint(cp, dict(settings=settings, epoch=epoch + 1, history=history,
                weights=weights, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                random=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    # FlopCounterMode's module hooks must not attach to decoder token gradients
    # during an inference-only measurement. Trainable parameters were counted above.
    model.requires_grad_(False)
    view = dict(m, support_ids=draws[0])
    probe = m["train_ids"][0] if smoke else m["test_ids"][0]
    costs = check_forward(predict_for, view, probe, memory)
    if smoke:
        peak = torch.cuda.max_memory_allocated()
        if native_ilnet and peak > free_at_start:
            raise MemoryError(f"ILNet batch {microbatch} allocated {peak} bytes with {free_at_start} free")
        base.write_json(folder / "preflight.json", {**settings, **costs, "peak_allocated_MiB": peak / 2**20})
        return
    results = []
    for index, ids in enumerate(draws):
        view = dict(m, support_ids=ids)
        with torch.no_grad():
            predict = predict_for(view)
            output = folder / f"support_{index}"
            base.evaluate(model, predict, view, output, dict(settings, support_ids=ids), epochs, base.sha(cp))
            del predict
        results.append(read(output / f"epoch_{epochs:04d}/metrics/summary.json"))
    ious = [row["IoU_percent"] for row in results]
    base.write_json(report, dict(settings, **costs, checkpoint_sha256=base.sha(cp),
                    test_images=len(m["test_ids"]), support_draws=draws,
                    IoU_percent=float(np.mean(ious)), IoU_by_draw=ious))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dataset", default="IRSTD-1k", choices=base.DATASETS)
    parser.add_argument("--size", type=int, default=640, choices=(256, 640))
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--native-full", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.native_full and args.method not in NATIVE_FULL_METHODS:
        parser.error("native-full is configured for CoRe-IR, SANSA and ILNet")
    if args.native_full and args.smoke and args.method == "ILNet":
        for microbatch in (8, 4, 2, 1):
            try:
                worker(args.method, args.dataset, args.size, True, True,
                       native_full=True, native_microbatch=microbatch)
            except (torch.cuda.OutOfMemoryError, MemoryError) as error:
                print(f"ILNet microbatch {microbatch}: {error}", flush=True)
            else:
                return
            torch.cuda.empty_cache()
        raise RuntimeError("ILNet could not fit even one image")
    worker(args.method, args.dataset, args.size, args.full or args.native_full, args.smoke,
           native_full=args.native_full)


if __name__ == "__main__":
    main()
