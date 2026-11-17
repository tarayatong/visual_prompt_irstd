"""Native ABCDFSS every-episode adaptation with fixed infrared supports."""
import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from scripts.prepare_comparison import DEFAULT_OUTPUT, write_json
from scripts.run_comparison_ours import save_csv
from visual_prompt.evaluation.comparison import image_metrics, summarize
from visual_prompt.evaluation.spatial_alignment import restore_to_original, SPATIAL_PROTOCOL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ABCDFSS", "TVGTANet"], default="ABCDFSS")
    parser.add_argument("--datasets", nargs="+", default=["IRSTD-1k", "NUDT-SIRST", "SIRST"])
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--coordinate-output", type=Path)
    args = parser.parse_args()
    root = DEFAULT_OUTPUT
    source = root / "sources" / args.method
    sys.path.insert(0, str(source.resolve()))
    sys.path.append(str((root / "runtime_packages").resolve()))
    if args.method == "TVGTANet":
        from scripts.comparison_tvgtanet_adapter import load_native, NativeInfraredCAM
        runner, native_cam, source = load_native(root)
    else:
        from core import runner
    from core.backbone import Backbone
    weight = root / "weights/resnet50-11ad3fa6.pth"
    if not weight.exists():
        raise FileNotFoundError(f"Official torchvision ResNet50 DEFAULT weight missing: {weight}")
    hub = root / "runtime_cache/torch_hub"
    torch.hub.set_dir(str(hub.resolve()))
    cached = hub / "checkpoints" / weight.name
    cached.parent.mkdir(parents=True, exist_ok=True)
    if not cached.exists():
        shutil.copyfile(weight, cached)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if args.preflight:
        if args.method == "TVGTANet" and not (root / "weights/ViT-B-16.pt").exists():
            raise FileNotFoundError("Official CLIP ViT-B16 weight missing")
        print(json.dumps(dict(imports="passed", source_commit=commit, weights_bytes=weight.stat().st_size)))
        return
    device = torch.device("cuda")
    backbone = Backbone("resnet50").to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    cam_generator = NativeInfraredCAM(native_cam, root / "weights/ViT-B-16.pt") if args.method == "TVGTANet" else None
    transform = transforms.Compose([transforms.Resize((400, 400)), transforms.ToTensor(),
                                    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    for name in args.datasets:
        manifest = json.loads((root / name / "protocol/manifest.json").read_text(encoding="utf-8"))
        old_output = root / "full16" / name / args.method
        output = args.coordinate_output / name / args.method if args.coordinate_output else old_output
        prediction_dir = output / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        runner.set_args(SimpleNamespace(benchmark="infrared" if cam_generator else "lung", datapath="", nshot=5, adapt_to="every-episode",
                                        postprocessing="off", logpath=str(output / "logs"), verbosity=0))
        config = runner.makeConfig()
        config.featext.fit_every_episode = True
        def get_cam(sid):
            path = output / "clip_cams" / f"{sid}.npy"
            previous = old_output / "clip_cams" / f"{sid}.npy"
            if args.coordinate_output and previous.exists() and not path.exists():
                path.parent.mkdir(exist_ok=True)
                shutil.copyfile(previous, path)
            if not path.exists():
                result = cam_generator(manifest["records"][sid]["image"])
                path.parent.mkdir(exist_ok=True)
                np.save(path, result.numpy())
                print(f"TVGTANet CAM {name}: {sid}", flush=True)
            return torch.from_numpy(np.load(path)).float()
        maker = runner.makeFeatureMaker(SimpleNamespace(class_ids=[0]), config, device=device,
                                       feat_extr_method=backbone.extract_feats)
        supports, masks = [], []
        for sid in manifest["support_ids"]:
            record = manifest["records"][sid]
            supports.append(transform(Image.open(record["image"]).convert("RGB")))
            mask = Image.open(record["mask"]).convert("L").resize((400, 400), Image.Resampling.NEAREST)
            masks.append(torch.from_numpy((np.array(mask) > 0).astype(np.float32)))
        supports, masks = torch.stack(supports)[None].cuda(), torch.stack(masks)[None].cuda()
        settings = dict(method=args.method, source_commit=commit, input_size=400, support_shots=5,
                        actual_annotation_budget=5, allocated_annotation_budget=50,
                        adaptation="native every-episode,25epochs; fixed support and unlabeled individual query",
                        query_labels_for_adaptation=False, postprocessing="native default off",
                        rng_policy="native augmentation seeded2; independent query head RNG index+2",
                        manifest_sha256=manifest["manifest_sha256"])
        if cam_generator:
            settings.update(adaptation="native every-episode,20epochs (generic branch); fixed support and unlabeled individual query",
                            clip_weight_sha256="5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
                            clip_foreground="infrared small target", clip_background="infrared background",
                            clip_template="a clean origami {}.", cam="native1008px attention-refined GradCAM; uint8 threshold128",
                            packaging_adapter="comparison_tvgtanet_adapter.py; missing helpers from official PI-CLIP utils.py")
        if args.coordinate_output:
            settings.update(spatial_protocol=SPATIAL_PROTOCOL, coordinate_correction_only=True,
                            original_run=str(old_output), inference_label_restore='INTER_NEAREST_EXACT')
        config_path = output / "config.json"
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("ABCDFSS settings changed; refusing to reuse predictions")
        write_json(config_path, settings)
        support_cams = torch.stack([get_cam(sid) for sid in manifest["support_ids"]])[None].cuda() if cam_generator else None
        rows = []
        for index, sid in enumerate(manifest["test_ids"]):
            record = manifest["records"][sid]
            h, w = record["size_hw"]
            destination = prediction_dir / f"{sid}.npy"
            if not destination.exists():
                query_cam = get_cam(sid)[None].cuda() if cam_generator else None
                random.seed(index + 2)
                runner.utils.fix_randseed(index + 2)
                batch = dict(query_img=transform(Image.open(record["image"]).convert("RGB"))[None].cuda(),
                             support_imgs=supports, support_masks=masks, class_id=torch.tensor([0], device=device))
                if cam_generator:
                    batch.update(query_pascal_label=query_cam, support_pascal_labels=support_cams)
                task = runner.SingleSampleEval(batch, maker)
                _, prediction = task.forward()
                prediction = prediction[0].detach().float().cpu().numpy()
                if args.coordinate_output:
                    native_dir=output/'working_predictions';native_dir.mkdir(exist_ok=True)
                    np.save(native_dir/f'{sid}.npy',prediction)
                    prediction=restore_to_original(prediction,(h,w),discrete=True)
                else:
                    prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_NEAREST)
                np.save(destination, prediction)
                del task, batch
                print(f"{args.method} {name}: {index+1}/{len(manifest['test_ids'])} {sid}", flush=True)
            gt = np.array(Image.open(record["mask"]).convert("L").resize((w, h), Image.Resampling.NEAREST)) > 0
            rows.append(image_metrics(np.load(destination), gt, sid))
        scores = summarize(rows)
        scores.update(settings, dataset=name, complete_test=True)
        save_csv(output / "metrics/per_image.csv", rows)
        write_json(output / "metrics/summary.json", scores)
        print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
