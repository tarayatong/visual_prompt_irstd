"""Feature extraction and similarity helpers for the numerical pipeline."""

import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _feature_to_bchw(feature):
    if feature.ndim == 3:
        batch, tokens, channels = feature.shape
        side = int(round(tokens**0.5))
        if side * side != tokens:
            raise ValueError(f"Cannot reshape feature {tuple(feature.shape)} to a square map.")
        return feature.reshape(batch, side, side, channels).permute(0, 3, 1, 2).contiguous()
    if feature.ndim != 4:
        raise ValueError(f"Expected a 3D token sequence or 4D feature map, got {tuple(feature.shape)}.")
    if feature.shape[1] == feature.shape[2] and feature.shape[-1] != feature.shape[1]:
        return feature.permute(0, 3, 1, 2).contiguous()
    return feature


def _resolve_module(root, layer_name):
    module = root
    for part in layer_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def resolve_sim_feature_layers(image_encoder, layer_names):
    names = [name.strip() for name in layer_names.split(",") if name.strip()]
    if len(names) == 1 and names[0].lower() == "all":
        return ["patch_embed"] + [f"layers.{index}" for index in range(len(image_encoder.layers))]
    return names


def register_sim_feature_hooks(predictor, layer_names):
    names = resolve_sim_feature_layers(predictor.model.image_encoder, layer_names)
    if not names:
        raise ValueError("At least one similarity feature layer is required.")
    predictor.sim_feature_maps = {}
    predictor.sim_feature_layers = names
    predictor.sim_feature_hooks = []
    for layer_name in names:
        module = _resolve_module(predictor.model.image_encoder, layer_name)

        def capture(_module, _inputs, output, name=layer_name):
            predictor.sim_feature_maps[name] = _feature_to_bchw(output.detach())

        predictor.sim_feature_hooks.append(module.register_forward_hook(capture))


def get_named_sim_features(predictor, align=True):
    maps = getattr(predictor, "sim_feature_maps", None)
    layers = getattr(predictor, "sim_feature_layers", None)
    if not maps or not layers:
        return [("image_embedding", predictor.features.detach())]
    features = []
    target_size = None
    for layer_name in layers:
        if layer_name not in maps:
            raise RuntimeError(f"Similarity feature layer was not captured: {layer_name}")
        feature = maps[layer_name].detach()
        if target_size is None:
            target_size = feature.shape[-2:]
        elif align and feature.shape[-2:] != target_size:
            feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
        features.append((layer_name, feature))
    return features


def get_sim_features(predictor):
    return torch.cat([feature for _, feature in get_named_sim_features(predictor)], dim=1)


def load_rgb_and_mask(image_path: str, mask_path: str, size_hw: Tuple[int, int]):
    height, width = size_hw
    image = Image.open(image_path).convert("RGB").resize((width, height), Image.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST)
    image_np = np.asarray(image, dtype=np.uint8)
    mask_np = (np.asarray(mask, dtype=np.uint8) >= 127).astype(np.uint8)
    mask_3c = np.repeat(mask_np[:, :, None] * 255, 3, axis=2).astype(np.uint8)
    return image_np, mask_np, mask_3c


@torch.no_grad()
def compute_sim_from_features(predictor, target_feat, cur_feat_bchw, sim_size=256):
    target_feat = target_feat / (target_feat.norm(dim=-1, keepdim=True) + 1e-8)
    feature = cur_feat_bchw.squeeze(0)
    channels, height, width = feature.shape
    feature = feature / (feature.norm(dim=0, keepdim=True) + 1e-8)
    similarity = target_feat @ feature.reshape(channels, height * width)
    sim_raw = similarity.reshape(1, 1, height, width)
    if sim_size > 0 and sim_raw.shape[-2:] != (sim_size, sim_size):
        sim_raw = F.interpolate(sim_raw, size=(sim_size, sim_size), mode="bilinear", align_corners=False)
    sim_post = predictor.model.postprocess_masks(sim_raw, predictor.input_size, predictor.original_size)
    return sim_raw.squeeze(0), sim_post.squeeze()
