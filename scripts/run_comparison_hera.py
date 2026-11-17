"""Apply official HERA's support-only adaptation to fixed infrared support sets.

No infrared-specific architecture/loss tuning and no training on the other 45 images.
With fixed support and no query-dependent adaptation, adapt once then reuse the
identical adapted model for all independent queries (recorded implementation choice).
"""
from __future__ import annotations

import argparse
import json
import logging
import gc
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import torch

from scripts.prepare_comparison import DEFAULT_OUTPUT, write_json
from scripts.run_comparison_ours import save_csv
from visual_prompt.evaluation.comparison import image_metrics, summarize


def main():
    # Native logs contain Unicode layer-selection markers on Windows consoles.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", default=["IRSTD-1k", "NUDT-SIRST", "SIRST"])
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="write isolated results under OUTPUT_ROOT/<dataset>/HERA; source/data stay under --root")
    parser.add_argument("--query-ids-json", type=Path, default=None,
                        help="optional JSON list of query IDs (e.g. the fixed val10 split)")
    parser.add_argument("--refine-mode", choices=("auto", "off", "always"), default="auto",
                        help="native HERA refinement policy; fixed across this run")
    args = parser.parse_args()
    source = (args.root / "sources/HERA").resolve()
    dino = (args.root / "sources/dinov3").resolve()
    # The latest upstream hubconf imports unrelated segmentation heads and their
    # optional training dependencies. Load the identical official backbone through
    # a minimal local hub entry point; no architecture or weights are changed.
    hub = (args.root / "sources/dinov3_backbone_hub").resolve()
    hub.mkdir(parents=True, exist_ok=True)
    (hub / "hubconf.py").write_text("from dinov3.hub.backbones import dinov3_vitl16\ndependencies = ['torch', 'numpy']\n", encoding="utf-8")
    sys.path.insert(0, str(dino))
    weights = (args.root / "weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth").resolve()
    torch.hub.set_dir(str((args.root / "runtime_cache/torch_hub").resolve()))
    sys.path.append(str((args.root / "runtime_packages").resolve()))
    sys.path.insert(0, str(source))
    import main_hera as native
    from torchvision import transforms
    from common.logger import Logger
    from common.evaluation import Evaluator
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if not weights.is_file():
        raise FileNotFoundError(f"HERA DINOv3 weights not downloaded: {weights}")
    if args.preflight:
        model = native.SSP_MatchingNet("DINOv3", dinov3_repo=str(hub), dinov3_ckpt_path=str(weights))
        model.eval()
        with torch.no_grad():
            feature = model.extract_feat_dino_single(torch.zeros(1, 3, 400, 400), model.backbone, [23], attn_strategy="dual_attn_gauss")
        print(dict(source_commit=commit, weights_bytes=weights.stat().st_size, imports="passed",
                   feature_shape=list(feature.shape),
                   policy="native support-only TTA; not gradient-free"))
        return
    transform = transforms.Compose([transforms.Resize((400, 400)), transforms.ToTensor(),
                                     transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    for name in args.datasets:
        root = args.root / name
        manifest = json.loads((root / "protocol/manifest.json").read_text(encoding="utf-8"))
        output = ((args.output_root / name / "HERA") if args.output_root is not None else root / "HERA")
        pred_dir = output / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output / "metrics/summary.json"
        if summary_path.exists() and not args.max_test:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            if (previous.get("complete_test") and previous.get("source_commit") == commit
                    and previous.get("manifest_sha256") == manifest["manifest_sha256"]
                    and all((pred_dir / f"{sid}.npy").exists() for sid in manifest["test_ids"])):
                print(f"HERA {name}: complete matching run already exists; skipping", flush=True)
                continue
        if args.query_ids_json is not None:
            ids = json.loads(args.query_ids_json.read_text(encoding="utf-8"))
            if len(ids) != len(set(ids)) or any(sid not in manifest["records"] for sid in ids):
                raise ValueError("query ID file contains duplicate or unknown sample IDs")
        else:
            ids = manifest["test_ids"]
        ids = ids[:args.max_test or None]
        native.utils.fix_randseed(manifest["seed"])
        model = native.SSP_MatchingNet("DINOv3", dinov3_repo=str(hub), dinov3_ckpt_path=str(weights)).cuda()
        # Candidate scoring requires no parameter graph; native adaptation re-enables
        # only the selected FC or fusion parameters in its fine-tuning branch.
        for p in model.parameters():
            p.requires_grad_(False)
        fusion = native.FusionMLP().cuda()
        refine = native.RefineModule().cuda()
        support_rgb, support_masks = [], []
        for sid in manifest["support_ids"]:
            rec = manifest["records"][sid]
            support_rgb.append(transform(Image.open(rec["image"]).convert("RGB")))
            m = Image.open(rec["mask"]).convert("L").resize((400, 400), Image.Resampling.NEAREST)
            support_masks.append(torch.from_numpy((np.array(m) > 0).astype(np.float32)))
        support_rgb, support_masks = torch.stack(support_rgb), torch.stack(support_masks)
        first_id = ids[0]
        rec = manifest["records"][first_id]
        first_query = transform(Image.open(rec["image"]).convert("RGB"))[None]
        batch = dict(support_imgs=support_rgb[None], support_masks=support_masks[None],
                     query_img=first_query, query_mask=torch.zeros(1, 400, 400),
                     class_id=torch.tensor([0]))
        loader = SimpleNamespace(dataset=SimpleNamespace(benchmark="lung", class_ids=[0]))

        class SingleEpisode:
            dataset = loader.dataset

            def __iter__(self):
                yield batch

            def __len__(self):
                return 1

        chosen = {}
        resumed_adaptation = False
        adaptation_path = output / "checkpoints/support_adaptation.pt"
        config_path = output / "config.json"
        if adaptation_path.exists() and config_path.exists() and (pred_dir / f"{first_id}.npy").exists():
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            saved = torch.load(adaptation_path, map_location="cpu", weights_only=False)
            if (saved.get("source_commit") == commit
                    and saved.get("manifest_sha256") == manifest["manifest_sha256"]
                    and saved_config.get("source_commit") == commit
                    and saved_config.get("manifest_sha256") == manifest["manifest_sha256"]
                    and len(saved["selected_feat_ids"]) == 1):
                parameters = dict(model.named_parameters())
                with torch.no_grad():
                    for key, value in saved["adapted_parameters"].items():
                        parameters[key].copy_(value)
                        parameters[key].requires_grad_(True)
                model.feat_ids = saved["selected_feat_ids"]
                selected_layer = model.feat_ids[0]
                def restored_extractor(img, bb, ids, **kwargs):
                    return model.extract_feat_dino_single(img, model.backbone, [selected_layer],
                                                          attn_strategy="dual_attn_gauss", **kwargs)
                model.extract_feats = restored_extractor
                fusion.load_state_dict(saved["fusion"])
                refine.load_state_dict(saved["refine"])
                chosen.update(apply_refine=bool(saved["apply_refine"]), vote_stats=saved_config.get("vote_stats"))
                resumed_adaptation = True
                print(f"HERA {name}: restored exact single-layer support adaptation", flush=True)
        vote_function = native.should_apply_refine_by_vote

        def record_vote(**kwargs):
            apply, stats = vote_function(**kwargs)
            chosen.update(apply_refine=bool(apply), vote_stats=stats)
            return apply, stats

        native.should_apply_refine_by_vote = record_vote
        original_classify = Evaluator.classify_prediction

        def capture_first(prediction, ignored_batch):
            h, w = rec["size_hw"]
            p = cv2.resize(prediction[0].float().cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
            np.save(pred_dir / f"{first_id}.npy", p)
            # These upstream episode metrics are diagnostics only; final metrics
            # below use original-resolution GT, after all predictions are produced.
            return original_classify(prediction, ignored_batch)

        Evaluator.classify_prediction = capture_first
        # Upstream basicConfig does not replace handlers on later datasets.
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
            handler.close()
        Logger.initialize(str(output / "logs"), "resume_inference.log" if resumed_adaptation else "native_support_adaptation.log")
        Evaluator.initialize()
        settings = SimpleNamespace(fusion="on", attn_strategy="dual_attn_gauss", refine=args.refine_mode, benchmark="lung")
        try:
            if not resumed_adaptation:
                native.test(model, SingleEpisode(), 5, torch.device("cuda"), fusion, refine, settings)
        finally:
            Evaluator.classify_prediction = original_classify
            native.should_apply_refine_by_vote = vote_function
        if args.refine_mode != "auto":
            chosen["apply_refine"] = args.refine_mode == "always"
        # The benchmark alias only supplies a one-class upstream logger; refine
        # voting's current dataset_name argument does not change its computations.
        write_json(output / "config.json", dict(source_commit=commit, actual_annotation_budget=5,
                   allocated_annotation_budget=50, support_shots=5, backbone="DINOv3 ViT-L/16",
                   input_size=400, target_gradient_updates=True, query_labels_for_adaptation=False,
                   adaptation="once on fixed support; native HLS/PGR/PAC; no changes between queries",
                   upstream_logger_benchmark_alias="lung", upstream_query_metrics="discarded; dummy GT",
                   adaptation_rng_seed=0,
                   restored_saved_adaptation=resumed_adaptation,
                   refine=settings.refine, selected_feat_ids=list(model.feat_ids),
                   query_ids=ids, query_split_file=str(args.query_ids_json) if args.query_ids_json else None,
                   manifest_sha256=manifest["manifest_sha256"], **chosen))
        checkpoint_dir = output / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dict(source_commit=commit, manifest_sha256=manifest["manifest_sha256"],
                        selected_feat_ids=list(model.feat_ids),
                        adapted_parameters={key: p.detach().cpu() for key, p in model.named_parameters() if p.requires_grad},
                        fusion=fusion.state_dict(), refine=refine.state_dict(),
                        apply_refine=chosen.get("apply_refine", False)),
                   checkpoint_dir / "support_adaptation.pt")
        model.eval()
        for parameter in model.parameters():
            parameter.grad = None
        for parameter in fusion.parameters():
            parameter.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        write_json(output / "inference_memory.json", dict(
            allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
            policy="release unused adaptation allocator cache and gradients before fixed-model inference"))
        rgb_list = [x[None].cuda() for x in support_rgb]
        mask_list = [x[None].cuda() for x in support_masks]
        with torch.no_grad():
            if resumed_adaptation:
                resume_query = first_query.cuda()
                resume_logits = model(rgb_list, mask_list, resume_query, None)[0]
                if chosen.get("apply_refine"):
                    resume_feature, resume_attn = model.get_feat_and_attn(resume_query, return_attn=True)
                    resume_logits = refine(resume_feature, resume_logits, resume_query, resume_attn)
                height, width = manifest["records"][first_id]["size_hw"]
                resume_prediction = cv2.resize(resume_logits.argmax(1)[0].float().cpu().numpy(),
                                               (width, height), interpolation=cv2.INTER_NEAREST)
                stored_prediction = np.load(pred_dir / f"{first_id}.npy")
                mismatch = int(np.count_nonzero(resume_prediction != stored_prediction))
                write_json(output / "resume_prediction_check.json", dict(sample_id=first_id, mismatched_pixels=mismatch, passed=mismatch == 0))
                if mismatch:
                    raise ValueError("Restored HERA adaptation changed the existing first-query prediction")
                del resume_query, resume_logits, resume_prediction, stored_prediction
            fixed_support_features = model(rgb_list, mask_list, None, None, tta_flag=True)
            cache_verified = False
            for sid in ids[1:]:
                dest = pred_dir / f"{sid}.npy"
                if dest.exists():
                    continue
                rec = manifest["records"][sid]
                query = transform(Image.open(rec["image"]).convert("RGB"))[None].cuda()
                query_feature = model(None, None, query, None, q_flag=True)
                logits = model.ssp_func(rgb_list, mask_list, query, None, fixed_support_features, query_feature)[0]
                if not cache_verified:
                    reference = model(rgb_list, mask_list, query, None)[0]
                    torch.testing.assert_close(logits, reference, rtol=1e-5, atol=1e-5)
                    write_json(output / "support_cache_check.json", dict(
                        sample_id=sid, max_abs_difference=float((logits - reference).abs().max()),
                        rtol=1e-5, atol=1e-5, passed=True))
                    del reference
                    cache_verified = True
                if chosen.get("apply_refine"):
                    feat_q, attn = model.get_feat_and_attn(query, return_attn=True)
                    logits = refine(feat_q, logits, query, attn)
                h, w = rec["size_hw"]
                # Match upstream argmax before nearest original-size mapping.
                pred = cv2.resize(logits.argmax(1)[0].float().cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
                np.save(dest, pred)
                print(f"HERA {name}: {sid}", flush=True)
        rows = []
        for sid in ids:
            rec = manifest["records"][sid]
            h, w = rec["size_hw"]
            gt = np.array(Image.open(rec["mask"]).convert("L").resize((w, h), Image.Resampling.NEAREST)) > 0
            rows.append(image_metrics(np.load(pred_dir / f"{sid}.npy"), gt, sid))
        result = summarize(rows)
        result.update(method="HERA", dataset=name, complete_test=len(ids) == len(manifest["test_ids"]),
                      actual_annotation_budget=5, allocated_annotation_budget=50, support_shots=5,
                      target_gradient_updates=True, source_commit=commit, manifest_sha256=manifest["manifest_sha256"])
        save_csv(output / "metrics/per_image.csv", rows)
        write_json(output / "metrics/summary.json", result)
        print(json.dumps(result, indent=2))
        del model, fusion, refine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
