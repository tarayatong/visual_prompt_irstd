"""Build the single MobileSAM variant used by this project."""

from pathlib import Path

import torch

from .modeling import MaskDecoder, PromptEncoder, Sam, TinyViT, TwoWayTransformer


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[3] / "checkpoints" / "mobile_sam.pt"


def build_mobile_sam(checkpoint: str | Path | None = DEFAULT_CHECKPOINT) -> Sam:
    prompt_embed_dim = 256
    image_size = 1024
    image_embedding_size = image_size // 16
    model = Sam(
        image_encoder=TinyViT(
            img_size=image_size,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    model.eval()

    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"MobileSAM checkpoint not found: {checkpoint_path}. "
                "Place it at checkpoints/mobile_sam.pt or pass --sam_ckpt."
            )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    return model


sam_model_registry = {"vit_t": build_mobile_sam}
