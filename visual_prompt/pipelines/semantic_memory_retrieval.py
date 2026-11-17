"""Opt-in support memory retrieval over existing sliding-window encoder features."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from visual_prompt.evaluation import eval_sliding_window_semantic_mask as ev
from visual_prompt.pipelines.semantic_prompt_pipeline import get_sim_features


@torch.no_grad()
def stitched_memory_maps(predictor, image, keys, args):
    h, w = image.shape[:2]
    total = np.zeros((len(keys), h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    stride = round(args.window_size * (1-args.overlap))
    if stride <= 0:
        raise ValueError('Memory retrieval requires a positive window stride')
    dummy = np.zeros((h, w), np.uint8)
    windows = [(y,x) for y in ev.sliding_starts(h,args.window_size,stride)
               for x in ev.sliding_starts(w,args.window_size,stride)]
    batch_size = max(1, int(args.window_batch_size))
    for start in range(0,len(windows),batch_size):
        batch = windows[start:start+batch_size]
        crops = [ev.crop_with_padding(image,dummy,y,x,args.window_size)[0] for y,x in batch]
        ev.encode_crop_batch(predictor,crops)
        feature = F.normalize(get_sim_features(predictor).float(),dim=1)
        raw = torch.einsum('kc,bchw->bkhw',keys,feature)
        if raw.shape[-2:] != (args.window_sim_size,args.window_sim_size):
            raw = F.interpolate(raw,size=(args.window_sim_size,args.window_sim_size),mode='bilinear',align_corners=False)
        patches = predictor.postprocess_masks(raw,predictor.input_size,predictor.original_size).cpu().numpy()
        for patch_set,(y,x) in zip(patches,batch):
            for i, patch in enumerate(patch_set):
                if patch.shape != (args.window_size, args.window_size):
                    patch = cv2.resize(patch, (args.window_size, args.window_size), interpolation=cv2.INTER_LINEAR)
                total[i, y:y+args.window_size, x:x+args.window_size] += patch
            weight[y:y+args.window_size, x:x+args.window_size] += 1
    return total / np.maximum(weight, 1e-6)[None]


@torch.no_grad()
def score_stitched_memory(predictor, image, memory, args, reader=None, mode='corrected', return_original=False):
    """Return a score map accepted by existing greedy_score_peaks/Obj/Mask routing.

    Explicitly opt in for the domain being adapted. No global hooks or changes to
    other domain defaults. Old e0 remains the input to downstream Mask decoding.
    """
    keys = memory['keys'].to(args.device)
    maps = stitched_memory_maps(predictor, image, keys, args)
    if mode == 'corrected':
        score = maps[1]
    elif mode == 'multi':
        positive = maps[2:2+memory['positive_count']]
        weights = np.asarray(memory['positive_weights'], np.float32)[:, None, None]
        peak = positive.max(0)
        score = peak + np.log((np.exp((positive-peak)*20)*weights).sum(0).clip(1e-10))/20
    elif mode == 'learned' and reader is not None:
        reader.eval()
        device = next(reader.parameters()).device
        score = reader.score_map(torch.from_numpy(maps).to(device)).cpu().numpy()
    else:
        raise ValueError('Choose corrected/multi or provide a reader for learned mode')
    # Obj can retain its original semantic descriptor without another encoder pass.
    return (score, maps[0]) if return_original else score
