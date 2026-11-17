import torch
import torch.nn.functional as F


DEFAULT_SAM2_BPLUS_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
DEFAULT_SAM2_BPLUS_CHECKPOINT = "checkpoints/sam2.1_hiera_base_plus.pt"


class Sam2SimilarityPredictor:
    def __init__(self, model, feature_source="decoder"):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.model = model
        self.predictor = SAM2ImagePredictor(model)
        self.predictor._bb_feat_sizes = [(model.image_size // stride, model.image_size // stride)
                                         for stride in (4, 8, 16)]
        self.feature_source = feature_source
        self.supports_external_sim_features = True
        self.sim_feature_layers = self._default_sim_feature_layers()
        self.features = None
        self._features = None
        self.original_size = None
        self.input_size = (model.image_size, model.image_size)

    @property
    def device(self):
        return self.predictor.device

    def _default_sim_feature_layers(self):
        if self.feature_source == "encoder_fpn":
            return ["encoder_fpn_s4", "encoder_fpn_s8", "encoder_fpn_s16"]
        if self.feature_source == "decoder":
            return ["high_res_0", "high_res_1", "image_embed"]
        raise ValueError(f"Unknown SAM2 feature_source: {self.feature_source}")

    def _set_decoder_features(self, image_or_images, is_batch):
        if is_batch:
            self.predictor.set_image_batch(image_or_images)
            self.original_size = image_or_images[0].shape[:2]
        else:
            self.predictor.set_image(image_or_images)
            self.original_size = image_or_images.shape[:2]
        self.input_size = (self.model.image_size, self.model.image_size)
        self._features = self.predictor._features
        self.features = self._features["image_embed"]

    @torch.no_grad()
    def _set_encoder_fpn_features(self, image_or_images, is_batch):
        if is_batch:
            images = image_or_images
            self.predictor._orig_hw = [image.shape[:2] for image in images]
            input_image = self.predictor._transforms.forward_batch(images).to(self.device)
            batch_size = len(images)
            self.original_size = images[0].shape[:2]
        else:
            image = image_or_images
            self.predictor._orig_hw = [image.shape[:2]]
            input_image = self.predictor._transforms(image)[None, ...].to(self.device)
            batch_size = 1
            self.original_size = image.shape[:2]

        backbone_out = self.model.image_encoder(input_image)
        _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(backbone_out)
        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats, feat_sizes)
        ]
        names = ["encoder_fpn_s4", "encoder_fpn_s8", "encoder_fpn_s16"]
        self._features = {name: feat.detach() for name, feat in zip(names, feats)}
        self.features = self._features["encoder_fpn_s16"]
        self.input_size = (self.model.image_size, self.model.image_size)

    def set_image(self, image, mask=None, image_format="RGB"):
        if image_format != "RGB":
            image = image[..., ::-1]
        if self.feature_source == "encoder_fpn":
            self._set_encoder_fpn_features(image, is_batch=False)
        else:
            self._set_decoder_features(image, is_batch=False)
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask[..., 0]
        return torch.as_tensor(mask, device=self.device, dtype=torch.float32)[None, None] / 255.0

    def set_image_batch(self, images):
        if self.feature_source == "encoder_fpn":
            self._set_encoder_fpn_features(images, is_batch=True)
        else:
            self._set_decoder_features(images, is_batch=True)

    def postprocess_masks(self, masks, input_size, original_size):
        masks = F.interpolate(
            masks,
            (self.model.image_size, self.model.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return F.interpolate(masks, original_size, mode="bilinear", align_corners=False)

    def get_named_sim_features(self, align=True):
        if self.feature_source == "encoder_fpn":
            feats = self._features
            named = [(name, feats[name].detach()) for name in self._default_sim_feature_layers()]
        else:
            feats = self._features or self.predictor._features
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


def build_sam2_bplus(checkpoint=DEFAULT_SAM2_BPLUS_CHECKPOINT, config=DEFAULT_SAM2_BPLUS_CONFIG, device="cpu", image_size=512):
    from sam2.build_sam import build_sam2

    if image_size not in (512, 1024):
        raise ValueError('SAM2 input size must be 512 or 1024')
    model = build_sam2(config, checkpoint, device=device,
                       hydra_overrides_extra=[f'model.image_size={image_size}'])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
