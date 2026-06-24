# Semantic Visual Prompting for Infrared Small Targets

This repository contains a self-contained training-free visual-prompt pipeline
for infrared small-target localization and mask extraction.

The active method combines:

```text
5-shot support targets
-> receptive-field-aware target tokens
-> support-derived high-frequency and bright-flat negative directions
-> negative-subspace-adjusted semantic score token
-> multi-layer MobileSAM feature similarity
-> overlapping local-window fusion
-> similarity-guided local-contrast mask extraction
```

The method code is implemented under `visual_prompt/`. It does not import or
modify a separate PR-MaGIC, Personalize-SAM, SAM-SPL, or MobileSAM source
checkout. The bundled MobileSAM backend retains TinyViT, prompt encoding, mask
decoding, and predictor inference so decoder comparisons remain reproducible.
License notices remain beside the code.

## External Assets

Weights and datasets are intentionally not stored in Git. Download and directory
instructions are in `docs/reproduction_setup.md`.

## Environment

```powershell
pip install -r requirements.txt
```

The default backbone checkpoint is:

```text
checkpoints/mobile_sam.pt
```

Expected SHA-256:

```text
6DBB90523A35330FEDD7F1D3DFC66F995213D81B29A5CA8108DBCDD4E37D6C2F
```

## Run Experiments

Run commands from the repository root:

```powershell
cd D:\workspace\code\visual_prompt
conda run -n sam python -m scripts.run_experiment `
  --config experiments/configs/ablation/overall/03_full.yaml
```

Inspect a resolved command without running inference:

```powershell
conda run -n sam python -m scripts.run_experiment `
  --config experiments/configs/ablation/overall/03_full.yaml `
  --dry-run
```

## Minimal Repository Layout

```text
visual_prompt/
  backbones/       local MobileSAM runtime
  core/            semantic directions, embedding utilities, and metrics
  data/            dataset adapters
  pipelines/       reusable semantic visual-prompt pipeline
  evaluation/      unified numerical evaluation and ablations

experiments/configs/  canonical experiment settings
scripts/              experiment and output-management entry points
docs/                 public download and reproduction instructions
```

The following paths exist only after local setup and are ignored by Git:

```text
dataset/              IRSTD-1k images, labels, and split files
checkpoints/          downloaded `.pt`/`.pth` model weights
outputs/              generated metrics, metadata, and figures
visual_prompt/visualization/  local paper/test plotting tools
```

The public minimal-reproduction guide is `docs/reproduction_setup.md`. Local
ablation notes, generated results, and paper/test plotting code are excluded by
`.gitignore`.
