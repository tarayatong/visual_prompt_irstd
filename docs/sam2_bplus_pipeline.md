# SAM2.1 Hiera-Base+ Backbone

This project can run the training-free similarity pipeline with SAM2.1 Hiera-Base+ features.

## Setup

Install the official SAM2 package in the same environment used by this project:

```powershell
C:\Users\80923\miniconda3\Scripts\conda.exe run -n visual-prompt python -m pip install "git+https://github.com/facebookresearch/sam2.git"
```

Download the checkpoint:

```powershell
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt" -OutFile "checkpoints\sam2.1_hiera_base_plus.pt"
```

The checkpoint is large and should not be committed to git.

## Quick Run

```powershell
C:\Users\80923\miniconda3\Scripts\conda.exe run -n visual-prompt python -m visual_prompt.evaluation.eval_sliding_window_semantic_mask `
  --backbone sam2_bplus `
  --data_root D:/workspace/code/visual_prompt/visual_prompt_irstd/dataset/IRSTD-1k `
  --device cpu `
  --image_size 256 `
  --score_layer fused_all_layers `
  --support_negative_scope global `
  --support_ids XDU514,XDU646,XDU904,XDU660,XDU347 `
  --edge_method sobel `
  --edge_quantile 0.90 `
  --edge_exclude_dilate 9 `
  --bright_threshold_mode target_mean `
  --local_contrast_sigma 7.0 `
  --target_rf_radius_scale 1.0 `
  --target_rf_radius_min 0 `
  --mask_score_mode sim_times_contrast `
  --use_vector_correction `
  --use_high_freq_negative `
  --use_bright_negative `
  --no-use_sliding_window `
  --no-use_hard_negative `
  --sample_ids XDU189,XDU946,XDU734 `
  --use_iterative_refine `
  --iter_refine_rounds 2 `
  --iter_topk_points 8 `
  --iter_debug_visualizations 3 `
  --layer_debug_visualizations 3 `
  --output_dir outputs/debug_iter/sam2_bplus_three_cases
```

## Notes

- `--backbone sam2_bplus` uses SAM2 features only; the pipeline is still training-free.
- `--final_multi_source auto` resolves to `peaks` for SAM2 and `branches` for MobileSAM. SAM2's score peaks were cleaner than all branch candidates in the initial three-case check.
- Use `--final_multi_source hybrid` to include both refined score peaks and branch candidates when recall is more important than false positives.
- Use `--layer_debug_visualizations N` to save per-layer native/postprocessed score maps such as `high_res_0`, `high_res_1`, and `image_embed`.
- On CPU, SAM2 B+ is much slower than MobileSAM. Use CUDA on the test machine if available.
