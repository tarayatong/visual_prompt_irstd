"""Unchanged49windows, encoded in microbatches and stitched on the GPU."""
import numpy as np
import torch
from torch.nn import functional as F
from visual_prompt.evaluation import eval_sliding_window_semantic_mask as ev
from visual_prompt.pipelines.semantic_prompt_pipeline import get_sim_features


@torch.no_grad()
def batched_memory_scores(model,image,batch_size=8):
    args=model.args;h,w=image.shape[:2];step=round(args.window_size*(1-args.overlap));predictor=model.pred()
    positions=[(y,x) for y in ev.sliding_starts(h,args.window_size,step) for x in ev.sliding_starts(w,args.window_size,step)]
    keys=model.memory['keys'].to(model.device);total=torch.zeros((len(keys),h,w),device=model.device,dtype=torch.float32)
    canonical=torch.zeros((h,w),device=model.device,dtype=torch.float32);weight=torch.zeros_like(canonical)
    dummy=np.zeros((h,w),np.uint8)
    for start in range(0,len(positions),batch_size):
        points=positions[start:start+batch_size]
        crops=[ev.crop_with_padding(image,dummy,y,x,args.window_size)[0] for y,x in points]
        ev.encode_crop_batch(predictor,crops);feature=get_sim_features(predictor).float()
        single=ev.compute_sim_batch_from_features(predictor,model.e0,feature,args.window_sim_size)
        raw=torch.einsum('kc,bchw->bkhw',keys,F.normalize(feature,dim=1))
        if raw.shape[-2:]!=(args.window_sim_size,args.window_sim_size):raw=F.interpolate(raw,size=(args.window_sim_size,args.window_sim_size),mode='bilinear',align_corners=False)
        patches=predictor.postprocess_masks(raw,predictor.input_size,predictor.original_size)
        for j,(y,x) in enumerate(points):
            canonical[y:y+args.window_size,x:x+args.window_size]+=single[j]
            total[:,y:y+args.window_size,x:x+args.window_size]+=patches[j]
            weight[y:y+args.window_size,x:x+args.window_size]+=1
    canonical/=weight.clamp_min(1e-6);total/=weight.clamp_min(1e-6)[None];total[0]=canonical
    return model.reader.score_map(total).cpu().numpy(),canonical.cpu().numpy()
