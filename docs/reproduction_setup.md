# CoRe-IR 复现设置

使用 Python 3.12。训练和在线评估需要 CUDA。按 GPU 环境安装匹配的 PyTorch 与 torchvision，然后在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## 数据准备

准备三个数据集的官方划分。IRSTD-1k 根目录使用 `IRSTD1k_Img/`、`IRSTD1k_Label/`、`trainval.txt` 和 `test.txt`。SIRST 使用 `idx_427/trainval.txt` 和 `idx_427/test.txt`。NUDT-SIRST 使用 `50_50/train.txt` 和 `50_50/test.txt`。

```powershell
python -m scripts.prepare_comparison --irstd-root D:/data/IRSTD-1k --nudt-root D:/data/NUDT-SIRST --sirst-root D:/data/sirst-master
```

同样可通过 `CORE_IR_IRSTD_ROOT`、`CORE_IR_NUDT_ROOT` 和 `CORE_IR_SIRST_ROOT` 指定数据路径。默认种子 111 生成每域 50 张训练图和五张固定支持。已有清单受一致性检查保护。

## SAM2 Tiny 与 SANSA 初始化

将 SANSA 源码检出至 `outputs/comparison_recent_a_20261104/sources/SANSA`，使用论文实验版本 `f562b5642940e08a57e3da2608b804b99af3763b`。[SANSA 源码](https://github.com/ClaudiaCuttano/SANSA)

将官方 `sam2_hiera_tiny.pt` 放在 `outputs/comparison_recent_a_20261104/weights/`。[SAM2 Tiny 初始化权重](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt)

## 训练和评估

```powershell
python -m scripts.run_core_ir_final_three_seeds --plan-only
python -m scripts.run_core_ir_final_three_seeds --dataset IRSTD-1k --seed 113
```

省略数据集与种子会执行三域九组。存在归档权重时复用对应检查点，否则按相同设置训练。输出目录为 `outputs/core_ir_final_three_seeds_20261116/fixed_support/`。

图 3 需要冻结评估结果、基线预测及逐图计数。可用 `--reference` 指定包含 `bad_samples/` 和 `qualitative_full/predictions/` 的本地目录。

```powershell
python -m scripts.render_core_ir_final_qualitative
```

全监督协议和补充实验见[运行协议](core_ir_followup_20261117.txt)。其他模型的源码版本、权重与来源记录见[实验总档](core_ir_experiments.md)。

```powershell
python -m unittest discover -s tests
```
