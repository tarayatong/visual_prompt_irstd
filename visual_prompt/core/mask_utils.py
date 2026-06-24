import cv2
import numpy as np

from visual_prompt.pipelines.semantic_prompt_pipeline import load_rgb_and_mask


def top_area_mask(score, area):
    flat = score.reshape(-1)
    area = int(max(1, min(area, flat.size)))
    indices = np.argpartition(flat, -area)[-area:]
    result = np.zeros(flat.size, dtype=np.uint8)
    result[indices] = 1
    return result.reshape(score.shape)


def binary_metrics(pred, gt):
    pred_b, gt_b = pred > 0, gt > 0
    inter = int(np.logical_and(pred_b, gt_b).sum())
    union = int(np.logical_or(pred_b, gt_b).sum())
    pred_pixels, gt_pixels = int(pred_b.sum()), int(gt_b.sum())
    iou = inter / union if union else 0.0
    f1 = (2 * inter) / (pred_pixels + gt_pixels) if pred_pixels + gt_pixels else 0.0
    return iou, f1, inter, pred_pixels, gt_pixels


def component_at_seed(score, area, grow_factor, seed_y, seed_x, seed_radius):
    candidate = top_area_mask(score, max(1, int(round(area * grow_factor))))
    _, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    seed_y, seed_x = int(seed_y), int(seed_x)
    comp_id = labels[seed_y, seed_x]
    if comp_id <= 0:
        h, w = score.shape
        y0, y1 = max(0, seed_y - seed_radius), min(h, seed_y + seed_radius + 1)
        x0, x1 = max(0, seed_x - seed_radius), min(w, seed_x + seed_radius + 1)
        local = np.where(labels[y0:y1, x0:x1] > 0, score[y0:y1, x0:x1], -np.inf)
        if np.isfinite(local).any():
            py, px = np.unravel_index(int(np.argmax(local)), local.shape)
            seed_y, seed_x = y0 + int(py), x0 + int(px)
            comp_id = labels[seed_y, seed_x]
        if comp_id <= 0:
            return np.zeros_like(score, dtype=np.uint8), seed_y, seed_x, "no_seed_component"
    component = labels == comp_id
    if int(component.sum()) <= area:
        return component.astype(np.uint8), seed_y, seed_x, "component"
    return top_area_mask(np.where(component, score, -np.inf), area), seed_y, seed_x, "component_clipped"


def support_area_prior(support_paths, image_size, mode):
    areas = []
    for image_path, mask_path in support_paths:
        _, mask, _ = load_rgb_and_mask(image_path, mask_path, (image_size, image_size))
        areas.append(int(mask.sum()))
    if not areas:
        raise RuntimeError("No support mask areas were available.")
    if mode == "median":
        return int(max(1, round(float(np.median(areas)))))
    if mode == "mean":
        return int(max(1, round(float(np.mean(areas)))))
    raise ValueError(f"Unknown area prior mode: {mode}")
