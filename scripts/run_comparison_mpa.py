"""MPA support-only budget adapter;500 passes, no target-test model selection."""
import argparse
import json
import random
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from scripts.prepare_comparison import DEFAULT_OUTPUT, write_json
from scripts.comparison_mpa_adapter import load_native, fixed_support_dataset
from scripts.run_comparison_ours import save_csv
from visual_prompt.evaluation.comparison import image_metrics, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["IRSTD-1k", "NUDT-SIRST", "SIRST"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 500:
        raise ValueError("MPA support passes must stay within user hard cap500")
    root = DEFAULT_OUTPUT
    model, native_dataset, lossfn = load_native(root)
    if args.preflight:
        print("Native MPA backbone and augmentation/loss imports passed; GPU400px verification pending")
        return
    model.cuda()
    transform = transforms.Compose([transforms.Resize((400, 400)), transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    commit = subprocess.check_output(["git", "-C", str(root / "sources/MPA"), "rev-parse", "HEAD"], text=True).strip()
    # A separate process is dispatched for each dataset, preventing trained model reuse.
    if len(args.datasets) != 1:
        raise ValueError("Dispatch one dataset per process to isolate initialization")
    name = args.datasets[0]
    manifest = json.loads((root / name / "protocol/manifest.json").read_text(encoding="utf-8"))
    output = root / "full16" / name / "MPA"
    checkpoint = output / "checkpoints/latest.pt"
    config = dict(method="MPA", source_commit=commit, allocated_annotation_budget=50, actual_annotation_budget=5,
                  support_shots=5, input_size=400, epochs=args.epochs, optimizer="native SGD", lr=.001,
                  momentum=.9, weight_decay=5e-4, batch_size=1, refinement=True,
                  initialization_sha256="1235e2d602f14e4634811a03c172e2e04a04fd86c82b37a2a9ed75f109f45524",
                  native_components="6 support-derived augmentations; parallel/chain/self weighted losses;6phases; frozen layer0/1 and BN",
                  budget_adaptation="one pass over fixed5 supports per optimizer step; hard500 passes replaces author48000 episodes; batch1 replaces4; snapshot25 replaces1200",
                  phase_selection="native counter logic on rolling training-augmentation foreground IoU, replacing forbidden test-mIoU",
                  test_selection="final fixed support pass; full test once; no test-based best checkpoint",
                  manifest_sha256=manifest["manifest_sha256"])
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise ValueError("MPA configuration changed")
    write_json(path, config)
    random.seed(111); np.random.seed(111); torch.manual_seed(111); torch.cuda.manual_seed_all(111)
    dataset = fixed_support_dataset(native_dataset, manifest, transform, args.epochs)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001, momentum=.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    start, phase, counter, best, history = 0, 1, 0, 0., []
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["config"] != config: raise ValueError("MPA checkpoint protocol mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start, phase, counter, best, history = [state[k] for k in ("epoch", "phase", "counter", "best", "history")]
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
    for epoch in range(start, args.epochs):
        model.train()
        for layer in model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d): layer.eval()
        sx, sy, qx, qy, *_ = dataset[epoch]
        supports = [x[None].cuda() for x in sx]
        support_masks = [y[None].cuda() for y in sy]
        queries = [x[None].cuda() for x in qx[:phase]]
        query_masks = [y[None].long().cuda() for y in qy[:phase]]
        optimizer.zero_grad(set_to_none=True)
        outputs = model(supports, support_masks, queries, query_masks)
        losses = lossfn(*outputs, query_masks, torch.cat(support_masks).long(), True, criterion)
        loss = losses[0] + .4 * losses[1] + .1 * losses[2] + losses[3] + .2 * losses[4]
        if not torch.isfinite(loss): raise ValueError(f"MPA nonfinite loss at pass{epoch+1}")
        pred = outputs[0][0].detach().argmax(1)
        target = query_masks[0]
        intersection = ((pred == 1) & (target == 1)).sum().item()
        union = ((pred == 1) | (target == 1)).sum().item()
        score = 100 * intersection / max(union, 1)
        loss.backward(); optimizer.step()
        history.append(dict(epoch=epoch+1, loss=float(loss.detach()), phase=phase, train_aug_IoU_percent=score))
        if epoch+1 in (args.epochs//3, args.epochs*2//3):
            optimizer.param_groups[0]["lr"] /= 10
        del outputs, losses, loss, supports, support_masks, queries, query_masks
        if (epoch+1) % 25 == 0 or epoch+1 == args.epochs:
            score = float(np.mean([r["train_aug_IoU_percent"] for r in history[-25:]]))
            best = max(best, score)
            counter = 0 if score >= best * 1.01 else counter + 1
            if counter >= 3 and phase < 6: phase, counter = phase+1, 0
            state = dict(config=config, epoch=epoch+1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         phase=phase, counter=counter, best=best, history=history, python_rng=random.getstate(),
                         numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, checkpoint); torch.save(state, checkpoint.parent / f"epoch_{epoch+1:04d}.pt")
            write_json(output / "training_history.json", history)
            print(f"MPA {name}: {epoch+1}/{args.epochs} support passes, phase{phase}, train_aug_IoU={score:.4f}", flush=True)
    model.eval()
    predictions = output / "predictions"; predictions.mkdir(exist_ok=True)
    supports, support_masks = [], []
    for sid in manifest["support_ids"]:
        record = manifest["records"][sid]
        supports.append(transform(Image.open(record["image"]).convert("RGB"))[None].cuda())
        mask = Image.open(record["mask"]).convert("L").resize((400,400), Image.Resampling.NEAREST)
        support_masks.append(torch.from_numpy((np.array(mask)>0).astype(np.float32))[None].cuda())
    rows = []
    with torch.no_grad():
        for index, sid in enumerate(manifest["test_ids"]):
            record = manifest["records"][sid]; h,w = record["size_hw"]; destination = predictions / f"{sid}.npy"
            if not destination.exists():
                query = transform(Image.open(record["image"]).convert("RGB"))[None].cuda()
                pred = model(supports, support_masks, [query], [None])[0].argmax(1)[0].float().cpu().numpy()
                np.save(destination, cv2.resize(pred,(w,h),interpolation=cv2.INTER_NEAREST))
                print(f"MPA {name} test: {index+1}/{len(manifest['test_ids'])}",flush=True)
            gt = np.array(Image.open(record["mask"]).convert("L").resize((w,h),Image.Resampling.NEAREST))>0
            rows.append(image_metrics(np.load(destination),gt,sid))
    result = summarize(rows); result.update(config,dataset=name,complete_test=True,schedule_status="fixed500_support_pass_budget_adapter")
    save_csv(output / "metrics/per_image.csv",rows); write_json(output / "metrics/summary.json",result)


if __name__ == "__main__": main()
