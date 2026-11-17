"""Frozen original MobileSAM with concatenated intermediate encoder features."""
import numpy as np
import torch
import torch.nn.functional as F

from visual_prompt.backbones.mobile_sam import build_mobile_sam, SamPredictor
from visual_prompt.pipelines.semantic_prompt_pipeline import _feature_to_bchw


class MobileSimilarityPredictor:
    supports_external_sim_features = True

    def __init__(self, checkpoint="checkpoints/mobile_sam.pt", device="cuda:0", image_size=512):
        torch.set_num_threads(4)
        self.model = build_mobile_sam(checkpoint,image_size=image_size).eval().requires_grad_(False).to(device)
        self.predictor = SamPredictor(self.model)
        self.device = torch.device(device)
        self.feature_source = "mobile_encoder_concat"
        self.sim_feature_layers = ["patch_embed"] + [f"layers.{i}" for i in range(len(self.model.image_encoder.layers))]
        self._features = {}
        self.features = None
        self.input_size = (image_size, image_size)
        self.original_size = None
        self.hooks = []
        for name in self.sim_feature_layers:
            module = self.model.image_encoder
            for part in name.split('.'):
                module = module[int(part)] if part.isdigit() else getattr(module, part)
            def capture(_module, _inputs, output, key=name):
                self._features[key] = _feature_to_bchw(output).detach()
            self.hooks.append(module.register_forward_hook(capture))

    @torch.no_grad()
    def set_image_batch(self, images):
        self.original_size = images[0].shape[:2]
        if any(im.shape[:2] != self.original_size for im in images):
            raise ValueError("A batch must contain equal image sizes")
        transformed = np.stack([self.predictor.transform.apply_image(im) for im in images])
        x = torch.as_tensor(transformed, device=self.device).permute(0, 3, 1, 2).contiguous()
        self.input_size = tuple(x.shape[-2:])
        self._features.clear()
        self.features = self.model.image_encoder(self.model.preprocess(x))

    def set_image(self, image, mask=None, image_format="RGB"):
        if image_format != "RGB":
            image = image[..., ::-1]
        self.set_image_batch([image])
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask[..., 0]
        return torch.as_tensor(mask.copy(), device=self.device, dtype=torch.float32)[None, None] / 255

    def get_named_sim_features(self, align=True):
        named = [(name, self._features[name]) for name in self.sim_feature_layers]
        if not align:
            return named
        size = named[0][1].shape[-2:]
        return [(name, feat if feat.shape[-2:] == size else F.interpolate(
            feat, size=size, mode="bilinear", align_corners=False)) for name, feat in named]

    def postprocess_masks(self, masks, input_size, original_size):
        return self.model.postprocess_masks(masks, input_size, original_size)
