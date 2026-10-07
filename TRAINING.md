# MobileSAM PatchMerging + K/V LoRA training

This is a separate, full-mask-supervised training path. It does not use the
five-shot semantic tokens, negative-direction correction, or iterative mask
selection from the training-free experiments. No ground-truth points or boxes
are supplied at inference: the original prompt encoder receives empty prompts,
and the original mask decoder returns its single-mask output.

## Trainable modules

- All three encoder PatchMerging modules, including BN affine parameters:
  **233,376 parameters**. Their BN running statistics update during training.
- Independent rank-8 K and V LoRA in all 10 encoder attention modules:
  **59,392 parameters**.
- Independent rank-8 K and V LoRA in all 7 decoder attention modules:
  **47,104 parameters**.
- Total: **974,080 trainable parameters**. Alpha=8, scale=alpha/rank=1.
- Decoder mask output is fully unfrozen: output_upscaling (including LayerNorm),
  all four output_hypernetworks_mlps, and mask_tokens: **634,208 parameters**.
  Single-mask supervision directly trains hypernetwork 0; hypernetworks 1-3
  are enabled but receive no direct loss gradient in this mode.
- Original attention weights/biases, Q, attention output projections, other
  encoder convolutions, Transformer MLPs, prompt encoder, IoU token and IoU
  prediction head remain frozen. Frozen BN running statistics also remain unchanged.

Encoder QKV is packed per head; the implementation changes only the K/V slices
within each head. LoRA B matrices start at zero. Frozen layers still participate
in autograd to propagate gradients to earlier trainable modules.

## Data and defaults

Run all commands from `visual_prompt_irstd/` with the project environment active.
Use the original `checkpoints/mobile_sam.pt` and raw IRSTD-1k folder documented
in `docs/reproduction_setup.md`.

Seed 42 partitions the existing 800-image trainval list into 640 train and 160
validation images. The 201-image test list is disjoint and never participates in
training or best-checkpoint selection. Exact IDs are saved in `splits.json` and
each checkpoint. Selection uses pooled foreground validation IoU.

Images and masks are resized to 256x256 for supervision (bilinear/nearest).
The encoder input is subsequently upsampled to 1024x1024 to preserve the existing
MobileSAM architecture. Augmentation is synchronized flips and 90-degree
rotations. Loss is unweighted BCE-with-logits plus per-image soft Dice.

Defaults: 50 epochs, batch size 1, accumulation 4, AdamW, LoRA LR 1e-4,
PatchMerging LR 1e-5, mask-output LR 1e-4, weight decay 1e-4, gradient clipping 1.0. CUDA AMP is enabled
only on CUDA; frozen-BN blocks use non-reentrant gradient checkpointing, while
trainable PatchMerging BN runs once per forward. This reduces memory demand but
does not guarantee a particular GPU memory footprint. CPU is supported.

## Commands

Validate parameter counts and data split without training:

```powershell
python -m scripts.train_lora --config experiments/configs/train_lora_r8.yaml --dry-run
```

Start full training (auto selects CUDA if available, otherwise CPU):

```powershell
python -m scripts.train_lora --config experiments/configs/train_lora_r8.yaml
```

Force CUDA on a machine with CUDA-enabled PyTorch:

```powershell
python -m scripts.train_lora --config experiments/configs/train_lora_r8.yaml --device cuda:0
```

Resume after a completed epoch, preserving optimizer/scaler and torch RNG state:

```powershell
python -m scripts.train_lora --config experiments/configs/train_lora_r8.yaml --resume outputs/training/mobilesam_kv_r8_mask_output/last.pt
```

Evaluate the selected checkpoint on the complete test split:

```powershell
python -m scripts.train_lora --config experiments/configs/train_lora_r8.yaml --evaluate outputs/training/mobilesam_kv_r8_mask_output/best.pt --eval-split test
```

The evaluation mode inherits architecture/preprocessing and split IDs from the
checkpoint, and writes `test_metrics.json` and `test_detail.csv`. It uses threshold
0.5, centroid distance <3 at the evaluation resolution, and corrected unmatched-
component Fa. Native-resolution evaluation is not implemented in this version.

One training batch and one validation batch, for code verification only:

```powershell
python -m scripts.train_lora --config experiments/configs/train_lora_r8.yaml --epochs 1 --max-train-batches 1 --max-val-batches 1 --output-dir outputs/training_smoke/new_run
```

Do not treat smoke results as model performance. Use a separate output directory
for every fresh experiment. Checkpoints save the full model (including frozen
weights and BN buffers), optimizer, scaler, configuration, splits and RNG states.
The original MobileSAM checkpoint must still be available for model construction.
`last.pt` supports epoch-boundary resume; `best.pt` is selected by validation IoU.

## Verification performed

CPU checks passed for zero-init equivalence of the LoRA QKV wrapper, head-wise Q
preservation, exact parameter count, frozen BN policy, a real optimizer step,
checkpoint saving/reloading, epoch resume, batch-size-two forward/backward,
nonzero finite gradients in all trainable module groups, and unchanged frozen
weights after the smoke optimizer steps. Full convergence and CUDA memory/AMP
behavior have not been tested on this Intel Iris Xe host.

The output-head setting is saved in checkpoints. Old checkpoints without this
field retain the old frozen-head behavior during evaluation. Resuming requires
the same setting: use --no-train-mask-output for an old run; start a fresh run
in the new output directory for the unfrozen-head configuration. Editing code
does not change an already-running training process.
