"""Compare frozen transfer, five target supports, and five-shot updates."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import random
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch

from scripts import background_replacement_data as bg
from scripts import run_background_replacement_baselines as infrared
from scripts.run_core_ir_common_protocol import (OUT, RETAINED_CORE, base, read, memory_model, memory_predictor,
    sam_objective, save_checkpoint)
from scripts.run_nudt_irstd_shot_curve import train_ir
from visual_prompt.models.tiny_shared_memory import support_regions
from scripts.run_tiny_ablation_combined import restore, state_for

SOURCE = "NUDT-SIRST"
METHODS = ("Ours", "SANSA", "ILNet")
DOMAINS = ("NUDT-SIRST-Sea", "DenseSIRST", "Open-SIRST-v2")
STEPS = 100


def split(text):
    return [PurePosixPath(line.strip()).stem for line in text.splitlines() if line.strip()]


def image_record(image, mask):
    with Image.open(image) as frame:
        size = list(reversed(frame.size))
    if not mask.is_file():
        raise FileNotFoundError(mask)
    return dict(image=str(image), mask=str(mask), size_hw=size)


def public_pairs(domain):
    datasets = Path(base.manifest_for(SOURCE)["root"]).parent
    if domain == "DenseSIRST":
        root = datasets / "DenseSIRST/SIRSTdevkit"
        images, masks = root / "PNGImages", root / "SIRST/BinaryMask"
        train, test = root / "Splits/trainval_v2.txt", root / "Splits/test_v2.txt"
    else:
        root = datasets / "open-sirst-v2-master"
        images, masks = root / "mixed", root / "annotations/masks"
        train, test = root / "splits/trainval_full.txt", root / "splits/test_full.txt"
    ids, heldout = (split(path.read_text(encoding="utf-8-sig")) for path in (train, test))
    return ids, heldout, {sid: (images / f"{sid}.png", masks / f"{sid}_pixels0.png")
                           for sid in set(ids + heldout)}


def sea_pairs():
    root = OUT / "external_data/NUDT-SIRST-Sea"
    prefix = "NUDT-SIRST(sea)/"
    with ZipFile(base.ROOT / "NUDT-SIRST(sea).zip") as archive:
        ids, test = (split(archive.read(prefix + f"idx_4961+847/{name}.txt").decode("utf-8-sig"))
                     for name in ("train", "test"))
        # Keep training and test scenes apart using the first two filename fields.
        groups = {tuple(sid.split("-")[:2]) for sid in test}
        available = set(archive.namelist())
        ids = [sid for sid in ids if tuple(sid.split("-")[:2]) not in groups]
        if not ids:
            raise ValueError("NUDT-SIRST-Sea has no scene-disjoint adaptation pool in the official split")
        pairs = {}
        for sid in set(ids + test):
            paths = [Path(kind, f"{sid}.png") for kind in ("images", "masks")]
            for part in paths:
                entry = prefix + part.as_posix()
                if entry not in available:
                    break
                path = root / part
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    content = archive.read(entry)
                    with Image.open(BytesIO(content)) as image:
                        image.verify()
                    path.write_bytes(content)
            else:
                pairs[sid] = tuple(root / part for part in paths)
    return ids, test, pairs


def signatures(path):
    with Image.open(path) as image:
        full = image.convert("L")
        digest = hashlib.sha256(str(full.size).encode() + full.tobytes()).hexdigest()
        array = np.asarray(full.resize((17, 16), Image.Resampling.BILINEAR))
    return digest, np.packbits((array[:, 1:] > array[:, :-1]).ravel())


def positive(record):
    with Image.open(record["mask"]) as mask:
        return bool(np.asarray(mask.convert("L")).any())


def preflight():
    sources = [base.manifest_for(dataset) for dataset in base.DATASETS]
    seen_ids = {sid for m in sources for sid in m["records"]}
    known = [signatures(row["image"]) for m in sources for row in m["records"].values()]
    checksums = {key for key, _ in known}
    fingerprints = np.stack([value for _, value in known])
    bitcount = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)

    def overlaps(digest, fingerprint, reference):
        return digest in checksums or (bitcount[np.bitwise_xor(reference, fingerprint)].sum(axis=1) <= 6).any()

    for domain in DOMAINS:
        output = OUT / "transfer" / domain / "protocol.json"
        try:
            ids, test, pairs = sea_pairs() if domain == "NUDT-SIRST-Sea" else public_pairs(domain)
            usable, rejected, records = {}, {}, {}
            for sid in dict.fromkeys(test + ids):
                paths = pairs.get(sid)
                if sid in seen_ids or not paths or not all(path.is_file() for path in paths):
                    rejected[sid] = "same ID as main benchmark or missing image/mask"
                    continue
                digest, fingerprint = signatures(paths[0])
                if overlaps(digest, fingerprint, fingerprints):
                    rejected[sid] = "exact image match or 256-bit dHash distance at most 6 to a main-benchmark image"
                    continue
                usable[sid] = (digest, fingerprint)
                records[sid] = image_record(*paths)
            test_ids = [sid for sid in test if sid in usable]
            if not test_ids:
                raise ValueError("No eligible test images")
            positives_in_test = sum(positive(records[sid]) for sid in test_ids)
            if not positives_in_test:
                raise ValueError(f"All {len(test_ids)} eligible test images are background-only")
            test_hashes = {usable[sid][0] for sid in test_ids}
            test_fingerprints = np.stack([usable[sid][1] for sid in test_ids])
            eligible = []
            for sid in ids:
                if sid not in usable:
                    continue
                digest, fingerprint = usable[sid]
                if digest in test_hashes or overlaps(digest, fingerprint, test_fingerprints):
                    rejected[sid] = "duplicate or near-duplicate of target test image"
                    continue
                if positive(records[sid]):
                    eligible.append(sid)
            if len(eligible) < 5:
                raise ValueError(f"Only {len(eligible)} eligible positive adaptation images; need 5")
            selected = random.Random(111).sample(sorted(eligible), 5)
            assert not set(selected) & set(test_ids)
            protocol = dict(dataset=domain, source=SOURCE, working_size=640, test_ids=test_ids,
                            train_ids=selected, support_ids=selected, target_updates=STEPS,
                            official_test_size=len(test), candidate_training_images=len(ids),
                            rejected=rejected, positive_eligible=len(eligible), positive_test=positives_in_test,
                            records={sid: records[sid] for sid in selected + test_ids})
            protocol["manifest_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
            if any(output.parent.glob("*/*/epoch_0120/metrics/summary.json")) and read(output) != protocol:
                raise ValueError("New target manifest differs from the evaluated protocol")
            base.write_json(output, protocol)
            output.with_name("blocked.json").unlink(missing_ok=True)
            print(domain, "test", len(test_ids), "eligible train", len(eligible), "five", selected, flush=True)
        except (FileNotFoundError, ValueError) as error:
            base.write_json(output.with_name("blocked.json"), dict(dataset=domain, error=str(error)))
            print(domain, "blocked", error, flush=True)


def load_source(method):
    source = base.manifest_for(SOURCE)
    if method in ("Ours", "SANSA"):
        model, identity = memory_model(method, source)
        if method == "Ours":
            path = base.ROOT / "outputs" / RETAINED_CORE[(SOURCE, 111)] / "checkpoints/epoch_0120.pt"
        else:
            path = base.RUN / SOURCE / method / "checkpoints/epoch_0120.pt"
        weights = torch.load(path, map_location="cpu", weights_only=False)
        assert weights["config"]["manifest_sha256"] == source["manifest_sha256"]
        (restore if method == "Ours" else base.load_saved)(model, weights["model"])
        return model, path, dict(source, working_size=640), identity
    path, config, identity = bg.checkpoint(method, SOURCE)
    model = infrared.build(method).cuda()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"], strict=True)
    return model, path, source, read(config)


def transfer(method, domain, smoke):
    from unittest.mock import patch

    base.seed_all()
    root = OUT / "transfer" / domain
    m = read(root / "protocol.json")
    memory = method in ("Ours", "SANSA")
    for condition in ("direct", "support5", "update5") if memory else ("direct", "update5"):
        folder = root / method / condition
        report = folder / "epoch_0120/metrics/summary.json"
        if not smoke and report.exists():
            continue
        model, cp, source, identity = load_source(method)
        steps = (1 if smoke else STEPS) if condition == "update5" else 0
        ids = m["train_ids"] if condition != "direct" else []
        if memory:
            if steps:
                model.train()
                params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.)
                rng = random.Random(111)
                query_rng = random.Random(112)
                for _ in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    for query in query_rng.sample(ids, 4):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss = sam_objective(model, [query], ids, m, method == "Ours", rng) / 4
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite target loss")
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, .1, error_if_nonfinite=True)
                    optimizer.step()
                del optimizer
            model.eval()
            view = (dict(m, records={**m["records"], **{f"source:{sid}": source["records"][sid]
                        for sid in source["support_ids"]}},
                        support_ids=[f"source:{sid}" for sid in source["support_ids"]])
                    if condition == "direct" else m)
            with torch.no_grad():
                predict = memory_predictor(method, model, view)
        else:
            def read_target(_, sid):
                rec = m["records"][sid]
                with Image.open(rec["image"]) as image, Image.open(rec["mask"]) as mask:
                    im = np.array(image.convert("RGB"))
                    gt = np.array(mask.convert("L").resize(image.size, Image.Resampling.NEAREST)) > 0
                return im, gt.astype(np.uint8), None

            if steps:
                with patch.object(bg, "manifest", return_value=m), patch.object(bg, "read", side_effect=read_target):
                    train_ir(model, method, identity, ids, steps)
            model.eval()

            def predict(sid):
                rec = m["records"][sid]
                with Image.open(rec["image"]) as image:
                    im = np.array(image.convert("RGB"))
                x = infrared.transform(im, method, identity, rec["image"])
                return model(x[None].cuda())[0, 0]

        if smoke:
            with torch.no_grad():
                probability = predict(m["train_ids"][0])
                if probability.ndim != 2 or not torch.isfinite(probability).all():
                    raise ValueError("Transfer preflight returned an invalid mask")
            del predict, model
            continue
        initial_sha = base.sha(cp)
        settings = dict(method=method, dataset=domain, source=SOURCE, condition=condition, target_updates=steps,
                        target_annotation_ids=ids, source_checkpoint=str(cp), source_sha256=initial_sha,
                        support_ids=source["support_ids"] if memory and condition == "direct" else ids if memory else [],
                        training_horizon=120, manifest_sha256=m["manifest_sha256"])
        if steps:
            cp = folder / "adapted.pt"
            save_checkpoint(cp, dict(config=settings, model=(state_for if method == "Ours" else base.saved_model)(model)))
        base.evaluate(model, predict, m, folder, settings, 120, base.sha(cp) if steps else initial_sha)
        del predict, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--dataset", choices=DOMAINS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.method and args.dataset:
        transfer(args.method, args.dataset, args.smoke)
    else:
        parser.error("Specify --preflight or both --method and --dataset")
