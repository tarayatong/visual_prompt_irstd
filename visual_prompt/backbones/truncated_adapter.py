"""Frozen pretrained encoder prefixes; later stages/FPN/decoders are absent.

This adapter exposes native intermediate features, without a random projection.
Only loading uses the full official model on CPU; the runtime module and export
state contain solely the selected prefix.
"""
import gc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from visual_prompt.backbones.sam2_adapter import build_sam2_bplus
from visual_prompt.backbones.mobile_sam import build_mobile_sam
from visual_prompt.pipelines.semantic_prompt_pipeline import _feature_to_bchw


class MobilePrefix(nn.Module):
    def __init__(self, encoder, groups):
        super().__init__()
        self.patch_embed=encoder.patch_embed
        self.layers=nn.ModuleList(list(encoder.layers)[:groups])

    def forward(self,x):
        x=self.patch_embed(x);outputs=[x]
        for layer in self.layers:
            x=layer(x);outputs.append(_feature_to_bchw(x))
        return outputs


class PrefixModel(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.image_encoder=encoder


class TruncatedSimilarityPredictor:
    supports_external_sim_features=True

    def __init__(self,checkpoint,config=None,stages=2,mobile=False,device='cuda:0',image_size=512):
        self.device=torch.device(device);self.image_size=image_size;self.mobile=mobile;self.stages=stages
        if mobile:
            if stages not in (1,2,3):raise ValueError('Mobile prefix supports1..3 groups')
            full=build_mobile_sam(checkpoint,image_size=image_size)
            encoder=MobilePrefix(full.image_encoder,stages)
            self.register_mean=full.pixel_mean.detach().clone().to(self.device)
            self.register_std=full.pixel_std.detach().clone().to(self.device)
            from visual_prompt.backbones.mobile_sam.utils.transforms import ResizeLongestSide
            self.transform=ResizeLongestSide(image_size)
            self.sim_feature_layers=['patch_embed']+[f'layers.{i}' for i in range(stages)]
        else:
            if stages not in (1,2,3):raise ValueError('Hiera prefix supports1..3 stages')
            full=build_sam2_bplus(checkpoint,config,'cpu',image_size)
            encoder=full.image_encoder.trunk
            end=encoder.stage_ends[stages-1]
            encoder.blocks=nn.ModuleList(list(encoder.blocks)[:end+1])
            encoder.stage_ends=encoder.stage_ends[:stages]
            encoder.return_interm_layers=True
            encoder.channel_list=[encoder.blocks[i].dim_out for i in reversed(encoder.stage_ends)]
            from sam2.utils.transforms import SAM2Transforms
            self.transform=SAM2Transforms(image_size,0.)
            self.sim_feature_layers=[f'stage{i+1}' for i in range(stages)]
        self.model=PrefixModel(encoder).eval().requires_grad_(False).to(self.device)
        del full;gc.collect()
        self._features={};self.features=None;self.original_size=None;self.input_size=(image_size,image_size)
        self.feature_source='pretrained_prefix_native_concat'

    @torch.no_grad()
    def set_image_batch(self,images):
        self.original_size=images[0].shape[:2]
        if any(im.shape[:2]!=self.original_size for im in images):raise ValueError('Equal-sized images required')
        if self.mobile:
            a=np.stack([self.transform.apply_image(im) for im in images])
            x=torch.as_tensor(a,device=self.device).permute(0,3,1,2).contiguous()
            self.input_size=tuple(x.shape[-2:]);x=(x-self.register_mean)/self.register_std
            x=F.pad(x,(0,self.image_size-x.shape[-1],0,self.image_size-x.shape[-2]))
        else:
            x=self.transform.forward_batch(images).to(self.device)
        outputs=self.model.image_encoder(x)
        self._features=dict(zip(self.sim_feature_layers,outputs,strict=True));self.features=outputs[-1]

    def set_image(self,image,mask=None,image_format='RGB'):
        if image_format!='RGB':image=image[...,::-1]
        self.set_image_batch([image])
        if mask is None:return None
        if mask.ndim==3:mask=mask[...,0]
        return torch.as_tensor(mask.copy(),device=self.device,dtype=torch.float32)[None,None]/255.

    def get_named_sim_features(self,align=True):
        named=[(name,self._features[name]) for name in self.sim_feature_layers]
        if not align:return named
        shape=named[0][1].shape[-2:]
        return [(n,f if f.shape[-2:]==shape else F.interpolate(f,size=shape,mode='bilinear',align_corners=False)) for n,f in named]

    def postprocess_masks(self,masks,input_size,original_size):
        masks=F.interpolate(masks,(self.image_size,self.image_size),mode='bilinear',align_corners=False)
        if self.mobile:masks=masks[...,:input_size[0],:input_size[1]]
        return F.interpolate(masks,original_size,mode='bilinear',align_corners=False)
