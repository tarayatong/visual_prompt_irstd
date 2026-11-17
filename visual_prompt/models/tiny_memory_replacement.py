"""Replace native SAM2 memory with the existing external Reader/radial branch."""
import types
import torch
from torch import nn
from torch.nn import functional as F


class RemovedNativeMemory(nn.Module):
    def forward(self,*args,**kwargs):
        raise RuntimeError('Native SAM2 memory must not execute in replacement arm')


def install_memory_replacement(model,fusion):
    """Keep native image encoder/heads; encode query only and use no-memory input.

    External instance/negative keys and Reader reach the decoder through fusion.
    No native support masks, mask-memory encoder, memory attention or recurrent
    memory bank are used. Multimask selection remains identical to the prior arm.
    """
    if hasattr(model,'radial_fusion'):raise ValueError('Install replacement on a fresh native model')
    removed={}
    for name in ('memory_encoder','memory_attention'):
        module=getattr(model.sam,name)
        removed[name]=sum(p.numel() for p in module.parameters())
        setattr(model.sam,name,RemovedNativeMemory())
    model.add_module('radial_fusion',fusion)
    model._radial_inputs=None;model._fusion_disabled=False
    model.native_memory_removed=removed
    model.native_memory_bank_writes=0

    def forbidden(self,*args,**kwargs):
        raise RuntimeError('Native SAM2 memory encoding/reading is forbidden')
    model.sam._encode_new_memory=types.MethodType(forbidden,model.sam)
    model.sam._prepare_memory_conditioned_features=types.MethodType(forbidden,model.sam)

    def forward(self,samples,prompt_dict=None):
        if samples.ndim!=5 or samples.shape[1]!=1:
            raise ValueError('Replacement forward accepts exactly one query image per example')
        samples,b,t,orig=self._preprocess_visual_features(samples,self.sam.image_size)
        assert t==1
        bb=self._forward_backbone(samples,orig)
        outputs=[]
        for idx in range(b):
            current=bb.get_current_feats(idx)
            high_res=bb.get_high_res_features(current)
            if not self._fusion_disabled:
                if self._radial_inputs is None:raise RuntimeError('Prepare query radial inputs')
                high_res[0]=self.radial_fusion(high_res[0],*self._radial_inputs)
            h,w=bb.feat_sizes[-1]
            feature=current[-1]+self.sam.no_mem_embed
            feature=feature.permute(1,2,0).reshape(1,256,h,w)
            decoded=self.sam._forward_sam_heads(backbone_features=feature,
                high_res_features=high_res,multimask_output=True)
            outputs.append(decoded.masks[0])
        masks=torch.cat(outputs)
        masks=F.interpolate(masks[None],size=orig[0],mode='bilinear',align_corners=False)[0]
        return {'pred_masks':masks}
    model.forward=types.MethodType(forward,model)
    return model
