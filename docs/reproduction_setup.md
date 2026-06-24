# Minimal Reproduction Setup

The Git repository contains only the MobileSAM runtime, semantic-direction code,
unified numerical evaluator, ablation YAML files, and experiment runner. Model
weights, datasets, generated results, and paper/test plotting scripts are local-only.

## 1. MobileSAM Checkpoint

Download `mobile_sam.pt` from the official MobileSAM project:

```text
project: https://github.com/ChaoningZhang/MobileSAM
direct file: https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

Place it as:

```text
visual_prompt/
  checkpoints/
    mobile_sam.pt
```

Expected file size is approximately 40.7 MB. The checkpoint used for the retained
experiments has SHA-256:

```text
6DBB90523A35330FEDD7F1D3DFC66F995213D81B29A5CA8108DBCDD4E37D6C2F
```

The full checkpoint is loaded strictly, including TinyViT, prompt encoder, and
mask decoder weights. This preserves both feature-similarity evaluation and the
optional decoder comparison path.

An alternative location can be passed with `--sam_ckpt <path>`.

## 2. IRSTD-1k Dataset

Obtain IRSTD-1k from its author-provided distribution (the ISNet project is a
common source: `https://github.com/RuiZhang97/ISNet`) and organize the raw files as:

```text
visual_prompt/
  dataset/
    IRSTD-1k/
      IRSTD1k_Img/
        XDU0.png
        ...
      IRSTD1k_Label/
        XDU0.png
        ...
      trainval.txt
      test.txt
```

Each split file contains one sample ID per line without `.png`. An external dataset
location can be used with `--data_root D:/path/to/IRSTD-1k`.

Every retained ablation YAML has an explicit `args.data_root` entry. Update that
single field in the YAML files when the dataset is stored elsewhere; it overrides
the evaluator's fallback path and is recorded in the resolved run command.

## 3. Environment and Validation

```powershell
conda create -n sam python=3.12 -y
conda run -n sam pip install -r requirements.txt
conda run -n sam python -m scripts.run_experiment `
  --config experiments/configs/ablation/overall/03_full.yaml `
  --dry-run
```

Run one numerical setting:

```powershell
conda run -n sam python -m scripts.run_experiment `
  --config experiments/configs/ablation/overall/03_full.yaml
```

Results are created locally under `outputs/` and remain ignored by Git.
