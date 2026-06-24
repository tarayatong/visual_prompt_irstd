import numpy as np


def normalize_map(values):
    values = np.asarray(values, dtype=np.float32)
    low, high = np.percentile(values, [1, 99])
    return np.clip((values - low) / (high - low + 1e-8), 0, 1)


def overlay_mask(image, mask, color=(0, 255, 80), alpha=0.45):
    result = image.astype(np.float32).copy()
    selected = mask > 0
    result[selected] = result[selected] * (1 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)
