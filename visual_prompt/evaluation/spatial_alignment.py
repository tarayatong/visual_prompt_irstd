"""Pixel-center restoration shared by the coordinate-corrected comparisons."""
import cv2
import numpy as np

SPATIAL_PROTOCOL = 'pixel_center_v1_bilinear_probability_nearest_exact_labels'


def restore_to_original(value, size_hw, *, discrete):
    value = np.asarray(value)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError('Expected a finite 2-D prediction')
    h, w = map(int, size_hw)
    return cv2.resize(value, (w, h), interpolation=cv2.INTER_NEAREST_EXACT if discrete else cv2.INTER_LINEAR)
