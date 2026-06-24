from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SemanticDirectionConfig:
    use_negative_subspace: bool = False
    use_high_freq_negative: bool = True
    edge_method: str = "canny"
    edge_quantile: float | None = None
    edge_exclude_dilate: int = 5
    use_bright_negative: bool = True
    use_hard_negative: bool = False
    hard_negative_min_pixels: int = 1
    hard_negative_exclude_known: bool = False
    bright_threshold_mode: str = "target_mean"
    bright_quantile: float = 0.95
    bright_flat_edge_quantile: float = 0.50
    negative_subspace_ridge: float = 0.0
    target_pool: str = "mean_max"
    unit_features: bool = True
    target_rf_radius_scale: float = 1.0
    target_rf_radius_min: int = 0
    target_rf_radius_max: int | None = None


def parse_optional_float(value):
    if isinstance(value, str) and value.lower() in {"none", "null", "all"}:
        return None
    return float(value)


def parse_optional_int(value):
    if isinstance(value, str) and value.lower() in {"none", "null", "all"}:
        return None
    return int(value)


def edge_map_from_image(image, method="sobel"):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == "canny":
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    elif method == "log":
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = np.abs(cv2.Laplacian(blur, cv2.CV_32F, ksize=3))
        edges = edges / (edges.max() + 1e-8)
    else:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(gx * gx + gy * gy)
        edges = edges / (edges.max() + 1e-8)
    return edges.astype(np.float32)


def edge_mask_from_image(image, method="sobel", edge_quantile=None):
    edge_map = edge_map_from_image(image, method=method)
    if edge_quantile is None:
        return edge_map > 0
    edge_threshold = np.quantile(edge_map, edge_quantile)
    if edge_threshold <= 0 and edge_map.max() > 0:
        return edge_map > 0
    return edge_map >= edge_threshold


def bright_flat_mask_from_image(image, edge_method="sobel", bright_quantile=0.95, flat_edge_quantile=0.50):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bright_threshold = np.quantile(gray, bright_quantile)
    return gray >= bright_threshold


def bright_flat_mask_from_target_mean(image, target_mask, edge_method="sobel", flat_edge_quantile=0.50):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    target_values = gray[target_mask > 0]
    if target_values.size == 0:
        return np.zeros_like(gray, dtype=bool)
    bright_threshold = float(target_values.mean())
    return gray > bright_threshold


def receptive_radius_pixels(mask_hw, feature_hw, radius_scale=0.0, min_radius=0, max_radius=None):
    if radius_scale <= 0:
        return 0
    stride_y = float(mask_hw[0]) / float(feature_hw[0])
    stride_x = float(mask_hw[1]) / float(feature_hw[1])
    radius = int(np.ceil(max(stride_y, stride_x) * radius_scale))
    radius = max(int(min_radius), radius)
    if max_radius is not None:
        radius = min(radius, int(max_radius))
    return max(0, radius)


def mask_to_feature_bool(
    mask_bchw,
    feature_hw,
    threshold=0.0,
    rf_radius_scale=0.0,
    rf_radius_min=0,
    rf_radius_max=None,
):
    mask = mask_bchw[:, :1].float()
    radius = receptive_radius_pixels(
        mask.shape[-2:],
        feature_hw,
        radius_scale=rf_radius_scale,
        min_radius=rf_radius_min,
        max_radius=rf_radius_max,
    )
    if radius > 0:
        kernel_size = 2 * radius + 1
        mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=radius)
    coverage = F.interpolate(mask, size=feature_hw, mode="area")
    return coverage.squeeze(0).squeeze(0) > threshold


def pool_feature_token(feat_hw_c, mask_hw, pool_mode="mean_max", unit_features=True):
    feat = feat_hw_c
    if unit_features:
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
    selected = feat[mask_hw]
    if selected.numel() == 0:
        return None
    if pool_mode == "mean":
        token = selected.mean(0, keepdim=True)
    elif pool_mode == "max":
        token = selected.max(dim=0, keepdim=True)[0]
    elif pool_mode == "mean_max":
        token = (selected.mean(0, keepdim=True) + selected.max(dim=0, keepdim=True)[0]) / 2
    else:
        raise ValueError(f"Unknown prototype pooling mode: {pool_mode}")
    return token / (token.norm(dim=-1, keepdim=True) + 1e-8)


def hard_negative_mask_from_target_token(feat_hw_c, target_mask, target_token, min_pixels=1):
    if target_token is None or target_mask.sum().item() == 0:
        return torch.zeros_like(target_mask, dtype=torch.bool)
    feat_unit = feat_hw_c / (feat_hw_c.norm(dim=-1, keepdim=True) + 1e-8)
    target_unit = target_token.squeeze(0)
    target_unit = target_unit / (target_unit.norm(dim=-1, keepdim=True) + 1e-8)
    sim = torch.sum(feat_unit * target_unit.view(1, 1, -1), dim=-1)
    target_values = sim[target_mask]
    if target_values.numel() == 0:
        return torch.zeros_like(target_mask, dtype=torch.bool)
    target_mean = target_values.mean()
    hard_mask = torch.logical_and(sim > target_mean, ~target_mask)
    if hard_mask.sum().item() < min_pixels:
        return torch.zeros_like(target_mask, dtype=torch.bool)
    return hard_mask


def negative_subspace_project_token(pos_token, neg_tokens, ridge=0.0, rank_tol=1e-5):
    pos_token = pos_token / (pos_token.norm(dim=-1, keepdim=True) + 1e-8)
    valid_negs = []
    for token in neg_tokens:
        if token is None or token.numel() == 0:
            continue
        token = token / (token.norm(dim=-1, keepdim=True) + 1e-8)
        valid_negs.append(token)
    if not valid_negs:
        return pos_token, torch.zeros(0, device=pos_token.device, dtype=pos_token.dtype), {
            "negative_rank": 0,
            "projection_norm": 0.0,
            "residual_norm": 1.0,
            "target_score_cos": 1.0,
        }

    neg_basis = torch.cat(valid_negs, dim=0)
    _, singular_values, vh = torch.linalg.svd(neg_basis.float(), full_matrices=False)
    rank = int((singular_values > singular_values.max() * rank_tol).sum().item()) if singular_values.numel() else 0
    if rank == 0:
        return pos_token, torch.zeros(len(valid_negs), device=pos_token.device, dtype=pos_token.dtype), {
            "negative_rank": 0,
            "projection_norm": 0.0,
            "residual_norm": 1.0,
            "target_score_cos": 1.0,
        }

    ortho_basis = vh[:rank].to(device=pos_token.device, dtype=pos_token.dtype)
    projected = (pos_token @ ortho_basis.t()) @ ortho_basis
    score_token = pos_token - projected
    residual_norm = float(score_token.norm(dim=-1).item())
    if residual_norm >= 1e-8:
        score_token = score_token / (score_token.norm(dim=-1, keepdim=True) + 1e-8)
    else:
        score_token = torch.zeros_like(pos_token)

    gram = neg_basis @ neg_basis.t()
    rhs = neg_basis @ pos_token.t()
    if ridge > 0:
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        coeffs = torch.linalg.solve(gram + ridge * eye, rhs).squeeze(-1)
    else:
        coeffs = (torch.linalg.pinv(gram.float()).to(gram.dtype) @ rhs).squeeze(-1)
    return score_token, coeffs, {
        "negative_rank": rank,
        "projection_norm": float(projected.norm(dim=-1).item()),
        "residual_norm": residual_norm,
        "target_score_cos": float(torch.sum(pos_token * score_token).item()),
    }


def extract_support_semantic_tokens(image, mask, mask_pad, sim_features, config: SemanticDirectionConfig):
    feat_hw_c = sim_features.squeeze(0).permute(1, 2, 0)
    target_mask = mask_to_feature_bool(
        mask_pad,
        feat_hw_c.shape[:2],
        rf_radius_scale=config.target_rf_radius_scale,
        rf_radius_min=config.target_rf_radius_min,
        rf_radius_max=config.target_rf_radius_max,
    )
    target_token = pool_feature_token(
        feat_hw_c,
        target_mask,
        pool_mode=config.target_pool,
        unit_features=config.unit_features,
    )

    negative_tokens = {}
    if not config.use_negative_subspace:
        return target_token, negative_tokens

    exclude_kernel = np.ones((config.edge_exclude_dilate, config.edge_exclude_dilate), dtype=np.uint8)
    exclude_mask = cv2.dilate((mask > 0).astype(np.uint8), exclude_kernel, iterations=1) > 0

    edge_feat_mask = torch.zeros_like(target_mask, dtype=torch.bool)
    if config.use_high_freq_negative:
        edge_mask = edge_mask_from_image(image, method=config.edge_method, edge_quantile=config.edge_quantile)
        edge_mask = np.logical_and(edge_mask, ~exclude_mask)
        edge_t = torch.from_numpy(edge_mask.astype(np.float32))[None, None].to(feat_hw_c.device)
        edge_feat_mask = mask_to_feature_bool(edge_t, feat_hw_c.shape[:2])
        edge_feat_mask = torch.logical_and(edge_feat_mask, ~target_mask)
        edge_token = pool_feature_token(
            feat_hw_c,
            edge_feat_mask,
            pool_mode="mean",
            unit_features=config.unit_features,
        )
        if edge_token is not None:
            negative_tokens["high_freq"] = edge_token

    if config.use_bright_negative:
        if config.bright_threshold_mode == "target_mean":
            bright_mask = bright_flat_mask_from_target_mean(
                image,
                mask,
                edge_method=config.edge_method,
                flat_edge_quantile=config.bright_flat_edge_quantile,
            )
        elif config.bright_threshold_mode == "global_quantile":
            bright_mask = bright_flat_mask_from_image(
                image,
                edge_method=config.edge_method,
                bright_quantile=config.bright_quantile,
                flat_edge_quantile=config.bright_flat_edge_quantile,
            )
        else:
            raise ValueError(f"Unknown bright threshold mode: {config.bright_threshold_mode}")
        bright_mask = np.logical_and(bright_mask, ~exclude_mask)
        bright_t = torch.from_numpy(bright_mask.astype(np.float32))[None, None].to(feat_hw_c.device)
        bright_feat_mask = mask_to_feature_bool(bright_t, feat_hw_c.shape[:2])
        bright_feat_mask = torch.logical_and(bright_feat_mask, ~target_mask)
        bright_token = pool_feature_token(
            feat_hw_c,
            bright_feat_mask,
            pool_mode="mean",
            unit_features=config.unit_features,
        )
        if bright_token is not None:
            negative_tokens["bright_flat"] = bright_token

    if config.use_hard_negative:
        hard_feat_mask = hard_negative_mask_from_target_token(
            feat_hw_c,
            target_mask,
            target_token,
            min_pixels=config.hard_negative_min_pixels,
        )
        if config.hard_negative_exclude_known:
            known_negative_mask = edge_feat_mask
            if config.use_bright_negative:
                known_negative_mask = torch.logical_or(known_negative_mask, bright_feat_mask)
            hard_feat_mask = torch.logical_and(hard_feat_mask, ~known_negative_mask)
        hard_token = pool_feature_token(
            feat_hw_c,
            hard_feat_mask,
            pool_mode="mean",
            unit_features=config.unit_features,
        )
        if hard_token is not None:
            negative_tokens["hard_negative"] = hard_token

    return target_token, negative_tokens


def build_score_token(target_tokens, negative_tokens_by_name, config: SemanticDirectionConfig):
    target_tokens = [token for token in target_tokens if token is not None]
    if not target_tokens:
        raise RuntimeError("No valid target tokens were extracted from the support set.")
    target_token = torch.stack([token.squeeze(0) for token in target_tokens], dim=0).mean(0, keepdim=True)
    target_token = target_token / (target_token.norm(dim=-1, keepdim=True) + 1e-8)
    score_token = target_token
    info = {
        **asdict(config),
        "used_negative_subspace": False,
        "negative_names": [],
        "subspace_coeffs": [],
        "negative_rank": 0,
        "projection_norm": 0.0,
        "residual_norm": 1.0,
        "target_score_cos": 1.0,
        "score_negative_cosines": [],
    }

    if not config.use_negative_subspace:
        return score_token, info

    negative_names = []
    negative_tokens = []
    for name, tokens in negative_tokens_by_name.items():
        if not tokens:
            continue
        neg = torch.stack([token.squeeze(0) for token in tokens], dim=0).mean(0, keepdim=True)
        neg = neg / (neg.norm(dim=-1, keepdim=True) + 1e-8)
        negative_names.append(name)
        negative_tokens.append(neg)

    score_token, coeffs, projection_info = negative_subspace_project_token(
        target_token,
        negative_tokens,
        ridge=config.negative_subspace_ridge,
    )
    info.update(projection_info)
    info["used_negative_subspace"] = len(negative_tokens) > 0
    info["negative_names"] = negative_names
    info["subspace_coeffs"] = [float(v) for v in coeffs.detach().cpu().tolist()]
    info["score_negative_cosines"] = [float(torch.sum(score_token * neg).item()) for neg in negative_tokens]
    return score_token, info
