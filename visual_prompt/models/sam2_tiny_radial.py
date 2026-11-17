"""Image-only radial sampling and native SAM2 query-decoder feature fusion."""
import types
import cv2
import numpy as np
from PIL import Image
import torch
from visual_prompt.models.radial_profile import DirectionalRadialOperator,RadialProfileConfig


class DecoderGridRadialSampler:
    """Sample8x8 contrast profiles at native high-resolution decoder grid sites.

    Radii are original-image pixels. Grid centers follow align_corners=False
    resize coordinates. No masks, candidate centers or query labels are used.
    """
    def __init__(self,device,grid_size=256):
        self.device=torch.device(device);self.grid_size=grid_size
        self.operator=DirectionalRadialOperator(1,RadialProfileConfig(
            components=('contrast',),fusion='concat',
            radii=(1.,2.,3.,4.,6.,8.,12.,16.),chunk_size=8192)).to(device).eval()

    @torch.no_grad()
    def __call__(self,image_path):
        rgb=np.array(Image.open(image_path).convert('RGB'))
        h,w=rgb.shape[:2];s=self.grid_size
        gray=torch.from_numpy(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255).to(self.device)[None,None]
        yy=(torch.arange(s,device=self.device,dtype=torch.float32)+.5)*(h/s)-.5
        xx=(torch.arange(s,device=self.device,dtype=torch.float32)+.5)*(w/s)-.5
        y,x=torch.meshgrid(yy,xx,indexing='ij')
        profiles=self.operator.sample_profiles(gray,torch.stack((x,y),-1).reshape(1,-1,2))
        raw=profiles.values.reshape(1,s*s,64).transpose(1,2).reshape(1,64,s,s).half()
        valid=profiles.valid.reshape(1,s*s,64).transpose(1,2).reshape(1,64,s,s)
        return raw.cpu(),valid.cpu()


def install_query_radial_fusion(model,fusion):
    """Adapt a SANSA SAM2-Tiny instance without editing upstream source files.

    Its existing SAM2 encoder/memory/decoder and adapter initialization remain.
    Only query high_res_features[0] gets the radial residual, immediately before
    the native decoder. Support GT memory encoding and object pointers remain
    on the original path. Callers must set _radial_inputs for each query.
    """
    if hasattr(model,'radial_fusion'):raise ValueError('Radial fusion already installed')
    model.add_module('radial_fusion',fusion)
    model._radial_inputs=None

    def decode(self,backbone_out,idx,memory_idx,memory_bank):
        if self._radial_inputs is None:raise RuntimeError('Prepare image-only radial query inputs before forward')
        current=backbone_out.get_current_feats(idx)
        pos=backbone_out.get_current_pos_embeds(idx)
        high_res=backbone_out.get_high_res_features(current)
        raw,valid=self._radial_inputs
        high_res[0]=self.radial_fusion(high_res[0],raw,valid)
        with_memory=self.sam._prepare_memory_conditioned_features(
            frame_idx=memory_idx,current_vision_feats=current[-1:],current_vision_pos_embeds=pos[-1:],
            feat_sizes=backbone_out.feat_sizes[-1:],num_frames=memory_idx+1,memory_bank=memory_bank)
        return self.sam._forward_sam_heads(backbone_features=with_memory,high_res_features=high_res,
                                          multimask_output=memory_idx>0)
    model._compute_decoder_out_w_mem=types.MethodType(decode,model)
    return model
