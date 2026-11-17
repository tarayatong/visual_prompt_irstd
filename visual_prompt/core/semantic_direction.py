from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SemanticDirectionConfig:
    use_negative_subspace: bool = True
    use_high_freq_negative: bool = True
    edge_method: str = "sobel"
    edge_quantile: float | None = 0.90
    edge_exclude_dilate: int = 9
    use_bright_negative: bool = True
    bright_threshold_mode: str = "target_mean"
    bright_quantile: float = 0.95
    bright_flat_edge_quantile: float = 0.50
    negative_subspace_ridge: float = 0.0
    target_pool: str = "mean_max"
    unit_features: bool = True
    target_rf_radius_scale: float = 1.0
    target_rf_radius_min: int = 0
    target_rf_radius_max: int | None = None


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


def bright_flat_mask_from_image(image, bright_quantile=0.95):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return gray >= np.quantile(gray, bright_quantile)


def bright_flat_mask_from_target_mean(image, target_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    target_values = gray[target_mask > 0]
    if target_values.size == 0:
        return np.zeros_like(gray, dtype=bool)
    return gray > float(target_values.mean())


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


def mask_to_feature_bool(mask_bchw, feature_hw, threshold=0.0, rf_radius_scale=0.0, rf_radius_min=0, rf_radius_max=None):
    mask = mask_bchw[:, :1].float()
    radius = receptive_radius_pixels(mask.shape[-2:], feature_hw, rf_radius_scale, rf_radius_min, rf_radius_max)
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
        raise ValueError(f"Unknown target_pool: {pool_mode}")
    return token / (token.norm(dim=-1, keepdim=True) + 1e-8)


def mean_token(tokens):
    valid = [token for token in tokens if token is not None and token.numel() > 0]
    if not valid:
        return None
    token = torch.stack([item.squeeze(0) for item in valid], dim=0).mean(0, keepdim=True)
    return token / (token.norm(dim=-1, keepdim=True) + 1e-8)


def negative_subspace_project_token(pos_token, neg_tokens, ridge=0.0, rank_tol=1e-5):
    pos_token = pos_token / (pos_token.norm(dim=-1, keepdim=True) + 1e-8)
    valid_negs = []
    for token in neg_tokens:
        if token is not None and token.numel() > 0:
            valid_negs.append(token / (token.norm(dim=-1, keepdim=True) + 1e-8))
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
    feature = sim_features.detach()
    if feature.ndim != 4:
        raise ValueError(f"Expected BCHW similarity features, got {tuple(feature.shape)}")
    feature_hw = feature.shape[-2:]
    feat_hw_c = feature[0].permute(1, 2, 0).contiguous()
    target_mask = mask_to_feature_bool(
        mask_pad,
        feature_hw,
        threshold=0.0,
        rf_radius_scale=config.target_rf_radius_scale,
        rf_radius_min=config.target_rf_radius_min,
        rf_radius_max=config.target_rf_radius_max,
    )
    target_token = pool_feature_token(feat_hw_c, target_mask, config.target_pool, config.unit_features)
    negative_tokens = {}
    known_negative_mask = torch.zeros_like(target_mask, dtype=torch.bool)
    if config.use_high_freq_negative:
        edge_mask = edge_mask_from_image(image, method=config.edge_method, edge_quantile=config.edge_quantile).astype(np.uint8)
        if config.edge_exclude_dilate > 0:
            kernel = np.ones((config.edge_exclude_dilate, config.edge_exclude_dilate), dtype=np.uint8)
            exclude = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
            edge_mask = np.logical_and(edge_mask > 0, ~exclude).astype(np.uint8)
        edge_pad = torch.as_tensor(edge_mask, device=feature.device, dtype=torch.float32)[None, None]
        edge_feat_mask = mask_to_feature_bool(edge_pad, feature_hw, threshold=0.0)
        known_negative_mask |= edge_feat_mask
        negative_tokens["high_freq"] = pool_feature_token(feat_hw_c, edge_feat_mask, "mean", config.unit_features)
    if config.use_bright_negative:
        if config.bright_threshold_mode == "target_mean":
            bright_mask = bright_flat_mask_from_target_mean(image, mask)
        else:
            bright_mask = bright_flat_mask_from_image(image, bright_quantile=config.bright_quantile)
        bright_mask = np.logical_and(bright_mask, mask <= 0).astype(np.uint8)
        bright_pad = torch.as_tensor(bright_mask, device=feature.device, dtype=torch.float32)[None, None]
        bright_feat_mask = mask_to_feature_bool(bright_pad, feature_hw, threshold=0.0)
        known_negative_mask |= bright_feat_mask
        negative_tokens["bright_flat"] = pool_feature_token(feat_hw_c, bright_feat_mask, "mean", config.unit_features)
    return target_token, negative_tokens


def build_score_token(target_tokens, negative_tokens_by_name, config: SemanticDirectionConfig):
    target_token = mean_token(target_tokens)
    if target_token is None:
        raise RuntimeError("No valid target tokens were provided.")
    negative_names = []
    negative_tokens = []
    for name, tokens in negative_tokens_by_name.items():
        token = mean_token(tokens)
        if token is None:
            continue
        negative_names.append(name)
        negative_tokens.append(token)
    if config.use_negative_subspace:
        score_token, coeffs, projection_info = negative_subspace_project_token(target_token, negative_tokens, config.negative_subspace_ridge)
    else:
        score_token = target_token
        coeffs = torch.zeros(len(negative_tokens), device=target_token.device, dtype=target_token.dtype)
        projection_info = {"negative_rank": 0, "projection_norm": 0.0, "residual_norm": 1.0, "target_score_cos": 1.0}
    info = asdict(config)
    info.update(projection_info)
    info["negative_names"] = negative_names
    info["subspace_coeffs"] = [float(v) for v in coeffs.detach().cpu().tolist()]
    info["score_negative_cosines"] = [float(torch.sum(score_token * neg).item()) for neg in negative_tokens]
    info["negative_token_count"] = len(negative_tokens)
    return score_token, info
