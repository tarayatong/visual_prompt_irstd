import os
import random

import numpy as np
import torch
from PIL import Image


def read_split_ids(data_root, split_name):
    with open(os.path.join(data_root, split_name), "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def raw_irstd_paths(data_root, sample_id):
    return (
        os.path.join(data_root, "IRSTD1k_Img", f"{sample_id}.png"),
        os.path.join(data_root, "IRSTD1k_Label", f"{sample_id}.png"),
    )


def parse_id_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]


class RawIRSTDDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split_name, image_size):
        self.data_root = data_root
        self.image_size = image_size
        self.sample_ids = read_split_ids(data_root, split_name)
        paths = [raw_irstd_paths(data_root, sample_id) for sample_id in self.sample_ids]
        self.images = [item[0] for item in paths]
        self.masks = [item[1] for item in paths]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        mask = Image.open(self.masks[index]).convert("L").resize(
            (self.image_size, self.image_size), Image.NEAREST
        )
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        mask_np = (np.asarray(mask, dtype=np.uint8) >= 127).astype(np.float32)
        return (
            torch.from_numpy(image_np.transpose(2, 0, 1)),
            torch.from_numpy(mask_np[None]),
            os.path.basename(self.images[index]),
        )


def positive_trainval_ids(data_root):
    result = []
    for sample_id in read_split_ids(data_root, "trainval.txt"):
        _, mask_path = raw_irstd_paths(data_root, sample_id)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if int((mask >= 127).sum()) > 0:
            result.append(sample_id)
    return result


def build_raw_support_paths(
    data_root,
    k_shot,
    support_ids=None,
    fixed_first_ids=None,
    fixed_last_ids=None,
    random_seed=None,
):
    positive_ids = positive_trainval_ids(data_root)
    if not positive_ids:
        raise RuntimeError("No positive support masks found in the IRSTD-1k trainval split.")

    explicit_ids = parse_id_list(support_ids)
    if explicit_ids:
        selected_ids = explicit_ids
    else:
        first_ids = parse_id_list(fixed_first_ids)
        last_ids = parse_id_list(fixed_last_ids)
        fixed_ids = first_ids + last_ids
        if len(fixed_ids) > k_shot:
            raise ValueError(f"Fixed support ids exceed k_shot={k_shot}: {fixed_ids}")
        pool = [sample_id for sample_id in positive_ids if sample_id not in set(fixed_ids)]
        needed = k_shot - len(fixed_ids)
        if needed > len(pool):
            raise ValueError(f"Need {needed} support ids, but only {len(pool)} are available.")
        selected_ids = first_ids + random.Random(random_seed).sample(pool, needed) + last_ids

    missing = [sample_id for sample_id in selected_ids if sample_id not in set(positive_ids)]
    if missing:
        raise ValueError(f"Support id(s) are not positive trainval samples: {', '.join(missing)}")
    if len(selected_ids) != k_shot:
        raise ValueError(f"Expected k_shot={k_shot}, got {len(selected_ids)}: {selected_ids}")
    return [raw_irstd_paths(data_root, sample_id) for sample_id in selected_ids]


def distance_to_gt(mask, y, x):
    gt_y, gt_x = np.nonzero(mask > 0)
    if len(gt_y) == 0:
        return float("nan")
    return float(np.sqrt((gt_y - float(y)) ** 2 + (gt_x - float(x)) ** 2).min())


def dilated_hit(mask, y, x, radius):
    height, width = mask.shape
    y0, y1 = max(0, int(y) - radius), min(height, int(y) + radius + 1)
    x0, x1 = max(0, int(x) - radius), min(width, int(x) + radius + 1)
    return bool((mask[y0:y1, x0:x1] > 0).any())
