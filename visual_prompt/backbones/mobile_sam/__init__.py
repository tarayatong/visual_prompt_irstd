"""Local MobileSAM backend used by the visual-prompt pipeline."""

from .builder import DEFAULT_CHECKPOINT, build_mobile_sam, sam_model_registry
from .predictor import SamPredictor

__all__ = [
    "DEFAULT_CHECKPOINT",
    "SamPredictor",
    "build_mobile_sam",
    "sam_model_registry",
]

