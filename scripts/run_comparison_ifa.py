"""Official IFA FSS1000 checkpoint, directly transferred with fixed IR support5."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from scripts.prepare_comparison import DEFAULT_OUTPUT, write_json
from scripts.run_comparison_ours import save_csv
from visual_prompt.evaluation.comparison import image_metrics, summarize


def build_native(root):
    source = root / "sources/IFA"
    sys.path.insert(0, str(source.resolve()))
    import model.resnet as resnet
    from model.IFA_matching import IFA_MatchingNet
    original = resnet.resnet50
    # Constructor insists on a cwd-relative ImageNet checkpoint. Full strict
    # task-state loading below replaces every tensor, so bypass only that I/O.
    resnet.resnet50 = lambda pretrained=True: original(pretrained=False)
    try:
        model = IFA_MatchingNet("resnet50", refine=True, shot=5)
    finally:
        resnet.resnet50 = original
    weight = root / "weights/ifa_fss_resnet50_5shot.pth"
    state = torch.load(weight, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval(), source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["IRSTD-1k", "NUDT-SIRST", "SIRST"])
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    root = DEFAULT_OUTPUT
    model, source = build_native(root)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if args.preflight:
        torch.set_num_threads(2)
        images = [torch.rand(1, 3, 64, 64) for _ in range(5)]
        masks = [torch.zeros(1, 64, 64) for _ in range(5)]
        for mask in masks:
            mask[:, 25:35, 25:35] = 1
        with torch.no_grad():
            outputs = model(images, masks, images[0], None)
        assert outputs[0].shape == (1, 2, 64, 64) and torch.isfinite(outputs[0]).all()
        print(json.dumps(dict(strict_full_checkpoint="passed", cpu_native_inference="passed", source_commit=commit)))
        return
    model.cuda()
    transform = transforms.Compose([transforms.Resize((400, 400)), transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    for name in args.datasets:
        manifest = json.loads((root / name / "protocol/manifest.json").read_text(encoding="utf-8"))
        output = root / "full16" / name / "IFA"
        predictions = output / "predictions"
        predictions.mkdir(parents=True, exist_ok=True)
        config = dict(method="IFA", source_commit=commit, allocated_annotation_budget=50, actual_annotation_budget=5,
                      support_shots=5, input_size=400, target_gradient_updates=False, refinement=True,
                      external_checkpoint="official FSS1000 ResNet50 5shot; selected before infrared testing",
                      checkpoint_sha256="3119cc60fc84964c3222035cf1cce6e84fcb2d2ca27da601bb975aa2dbf136ce",
                      protocol="native pretrained-model evaluation transferred to infrared; no additional infrared IFA fine-tuning",
                      query_labels_for_inference=False, manifest_sha256=manifest["manifest_sha256"])
        path = output / "config.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
            raise ValueError("IFA configuration changed; refusing to reuse predictions")
        write_json(path, config)
        supports, masks = [], []
        for sid in manifest["support_ids"]:
            record = manifest["records"][sid]
            supports.append(transform(Image.open(record["image"]).convert("RGB"))[None].cuda())
            mask = Image.open(record["mask"]).convert("L").resize((400, 400), Image.Resampling.NEAREST)
            masks.append(torch.from_numpy((np.array(mask) > 0).astype(np.float32))[None].cuda())
        rows = []
        with torch.no_grad():
            for index, sid in enumerate(manifest["test_ids"]):
                record = manifest["records"][sid]
                h, w = record["size_hw"]
                destination = predictions / f"{sid}.npy"
                if not destination.exists():
                    query = transform(Image.open(record["image"]).convert("RGB"))[None].cuda()
                    logits = model(supports, masks, query, None)[0]
                    prediction = logits.argmax(1)[0].float().cpu().numpy()
                    prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)
                    np.save(destination, prediction)
                    print(f"IFA {name}: {index+1}/{len(manifest['test_ids'])} {sid}", flush=True)
                gt = np.array(Image.open(record["mask"]).convert("L").resize((w, h), Image.Resampling.NEAREST)) > 0
                rows.append(image_metrics(np.load(destination), gt, sid))
        scores = summarize(rows)
        scores.update(config, dataset=name, complete_test=True)
        save_csv(output / "metrics/per_image.csv", rows)
        write_json(output / "metrics/summary.json", scores)
        print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()
