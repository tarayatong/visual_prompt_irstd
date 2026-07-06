import torch
import torch.nn.functional as F


DEFAULT_SAM2_BPLUS_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEFAULT_SAM2_BPLUS_CHECKPOINT = "checkpoints/sam2.1_hiera_base_plus.pt"


class Sam2SimilarityPredictor:
    def __init__(self, model):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.model = model
        self.predictor = SAM2ImagePredictor(model)
        self.supports_external_sim_features = True
        self.sim_feature_layers = ["high_res_0", "high_res_1", "image_embed"]
        self.features = None
        self.original_size = None
        self.input_size = (model.image_size, model.image_size)

    @property
    def device(self):
        return self.predictor.device

    def set_image(self, image, mask=None, image_format="RGB"):
        if image_format != "RGB":
            image = image[..., ::-1]
        self.predictor.set_image(image)
        self.original_size = image.shape[:2]
        self.input_size = (self.model.image_size, self.model.image_size)
        self.features = self.predictor._features["image_embed"]
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask[..., 0]
        return torch.as_tensor(mask, device=self.device, dtype=torch.float32)[None, None] / 255.0

    def postprocess_masks(self, masks, input_size, original_size):
        masks = F.interpolate(
            masks,
            (self.model.image_size, self.model.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return F.interpolate(masks, original_size, mode="bilinear", align_corners=False)

    def get_named_sim_features(self, align=True):
        feats = self.predictor._features
        named = [
            ("high_res_0", feats["high_res_feats"][0].detach()),
            ("high_res_1", feats["high_res_feats"][1].detach()),
            ("image_embed", feats["image_embed"].detach()),
        ]
        if not align:
            return named
        target_size = named[0][1].shape[-2:]
        result = []
        for name, feat in named:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            result.append((name, feat))
        return result


def build_sam2_bplus(checkpoint=DEFAULT_SAM2_BPLUS_CHECKPOINT, config=DEFAULT_SAM2_BPLUS_CONFIG, device="cpu"):
    from sam2.build_sam import build_sam2

    model = build_sam2(config, checkpoint, device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
