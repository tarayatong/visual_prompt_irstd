"""Audit downloaded datasets and freeze the 50-image / 5-support protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/comparison_50shot_5support"
DATASETS = {
    "IRSTD-1k": ("D:/workspace/code/dataset/IRSTD-1k", "trainval.txt", "test.txt"),
    "NUDT-SIRST": ("D:/workspace/code/dataset/NUDT-SIRST", "50_50/train.txt", "50_50/test.txt"),
    "SIRST": ("D:/workspace/code/dataset/sirst-master", "idx_427/trainval.txt", "idx_427/test.txt"),
}
DATASET_OVERRIDES = {
    "IRSTD-1k": ("--irstd-root", "CORE_IR_IRSTD_ROOT"),
    "NUDT-SIRST": ("--nudt-root", "CORE_IR_NUDT_ROOT"),
    "SIRST": ("--sirst-root", "CORE_IR_SIRST_ROOT"),
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ids_from(path):
    ids = [Path(line.strip()).stem for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate IDs in {path}")
    return ids


def resolve_pair(root, sid):
    image_dirs = [root / x for x in ("IRSTD1k_Img", "images", "images/images")]
    mask_dirs = [root / x for x in ("IRSTD1k_Label", "masks", "masks/masks")]
    images = [p / f"{sid}.png" for p in image_dirs if (p / f"{sid}.png").is_file()]
    masks = [p / f"{sid}{suffix}.png" for p in mask_dirs for suffix in ("", "_pixels0")
             if (p / f"{sid}{suffix}.png").is_file()]
    if len(images) != 1 or len(masks) != 1:
        raise ValueError(f"Expected one exact image/mask pair for {sid}: {images}, {masks}")
    return images[0], masks[0]


def prepare(output=DEFAULT_OUTPUT, seed=111, dataset_roots=None):
    output = Path(output)
    audits = []
    for name, (directory, trainfile, testfile) in DATASETS.items():
        _, environment = DATASET_OVERRIDES[name]
        root = Path((dataset_roots or {}).get(name) or os.environ.get(environment) or directory).expanduser()
        train, test = ids_from(root / trainfile), ids_from(root / testfile)
        if set(train) & set(test):
            raise ValueError(f"Train/test overlap in {name}")
        records = {}
        for sid in train + test:
            image, mask = resolve_pair(root, sid)
            with Image.open(image) as im, Image.open(mask) as ma:
                size = list(reversed(im.size))
            records[sid] = dict(image=str(image.resolve()), mask=str(mask.resolve()), size_hw=size)
            with Image.open(mask) as ma:
                if list(reversed(ma.size)) != size:
                    records[sid]["mask_native_size_hw"] = list(reversed(ma.size))
                    records[sid]["mask_alignment"] = "nearest-neighbor resize to original image size; source files unchanged"
        selected = random.Random(seed).sample(train, 50)
        # Only inspect masks inside the budget to choose valid support images.
        positive = [sid for sid in selected if (np.array(Image.open(records[sid]["mask"]).convert("L")) > 0).any()]
        support = random.Random(seed + 1).sample(positive, 5)
        payload = dict(protocol="random50_support5_v1", seed=seed, dataset=name,
                       root=str(root), train_split=trainfile, test_split=testfile,
                       source_train_ids=train, train_ids=selected, support_ids=support, test_ids=test,
                       annotation_budget=50, support_shots=5,
                       validation_policy="fixed hyperparameters; no target-label model/threshold selection",
                       grouping="official image split; scene/sequence metadata unavailable; no new group split inferred",
                       records=records,
                       source_split_sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest()
                                            for f in (trainfile, testfile)})
        encoded = json.dumps(payload, sort_keys=True).encode()
        payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
        dest = output / name / "protocol"
        manifest = dest / "manifest.json"
        if manifest.exists():
            old = json.loads(manifest.read_text(encoding="utf-8"))
            if old != payload:
                raise ValueError(f"Refusing to replace a different frozen manifest: {manifest}")
        write_json(manifest, payload)
        for filename, ids in (("train50.txt", selected), ("support5.txt", support), ("test_full.txt", test)):
            (dest / filename).write_text("\n".join(ids) + "\n", encoding="utf-8")
        audits.append(dict(dataset=name, source_train=len(train), train=50, support=5,
                           test=len(test), overlap=0, manifest_sha256=payload["manifest_sha256"]))
    write_json(output / "dataset_audit.json", audits)
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=111)
    for flag, environment in DATASET_OVERRIDES.values():
        parser.add_argument(flag, type=Path, help=f"Dataset root (overrides {environment}).")
    args = parser.parse_args()
    roots = {name: getattr(args, flag[2:].replace("-", "_")) for name, (flag, _) in DATASET_OVERRIDES.items()}
    prepare(args.output, args.seed, roots)
