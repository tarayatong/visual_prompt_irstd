# SAM2 Main Baseline

The retained baseline is `experiments/configs/sam2/main.yaml`.

Key choices:

```text
backbone: sam2_bplus
encoder input: 512 (updated 2026-10-11)
score layer: concat_all_layers only
support_negative_scope: global
sliding window: 128, overlap 0.5
window_batch_size: 4
score-token correction: on
high-frequency negative: on
bright negative: on
hard negative: off
mask score: sim_times_contrast equivalent
local_contrast_sigma: 7.0
size constraint: final_area_scales 0.25..6.0
```

Each 128-pixel window is resized to 512 before encoding. MobileSAM similarity
inference also defaults to 512. Checkpoint tensors are loaded unchanged. Earlier
results using 1024 encoder input remain historical results, not measurements of
the current 512 setting.

Rationale: SAM2 `high_res_0` independent score maps were noisy under uniform per-layer min-max fusion. The clean branch scores the concatenated feature tensor directly, so high-resolution noise is mixed in channel space instead of being independently amplified.
