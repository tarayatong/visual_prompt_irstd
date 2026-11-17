# CoRe-IR 论文实验总档

## 最终论文设置与实验顺序

最终 CoRe-IR 用 **SAM2 Hiera Tiny + 19 槽 CSM + 一个 RCR**。RCR 放在最终上采样和 skip 融合之后、mask prediction 之前，论文中称为 *final decoder refinement before mask prediction*。在当前 1024 输入的实现中，这里的特征是 32 通道、空间大小 256 × 256。层的位置由解码流程定义。

| 顺序 | 实验 | 协议与报告方式 |
| --- | --- | --- |
| 1 | **最终设置三种子与图 3** | SIRST、IRSTD-1k、NUDT-SIRST 各用训练 seed 111/112/113。每域固定 train50、120 epochs 和原清单中的 5 张推理支持。每个种子在该域完整测试一次，从三个种子计算均值及样本标准差。每域复用一份已验证的最终架构权重，补训两个种子。图 3 沿用七个样本，使用相同固定支持下的最终位置 RCR 掩码。 [训练与复评入口](../scripts/run_core_ir_final_three_seeds.py) · [支持图 ID 和种子](../outputs/core_ir_final_three_seeds_20261116/fixed_support/protocol.json) · [图 3 入口](../scripts/render_core_ir_final_qualitative.py) |
| 独立实验 | **训练和推理支持集随机化** | 三种子实验沿用现有训练采样器，每个 episode 从预定的 train50 池抽一张与查询不同的支持，推理固定为清单上的五张。用户另行设计和执行训练与推理支持集共同随机化的对照。 |
| 2 | **输入分辨率主表和独立对照** | 主表保留可复现的各方法输入配置，统一披露样本划分、标注量、输出回原图的规则、预算与阈值选择，并列出外部工作尺寸及实际编码器尺寸。对照先在原图统一限分辨率，再接各模型的正常预处理来测同信息量。以 IRSTD-1k 的 256 与 512 为首个对照，先做同权重分辨率敏感性，再对差别明显的代表方法按目标分辨率重新训练。首批取 CoRe-IR、SANSA Tiny、ILNet 与 HDNet。等信息测试和等算力测试分别报告。 |
| 3 | **更多数据集泛化** | 先定义与训练集隔离的目标域，再预先指定源模型。报告源支持的 0-shot 和仅换五张目标支持的冻结复评，如需要再统一 5-shot 更新次数。DenseSIRST 用于密集目标压力测试，本地 DenseSIRST 与 Open SIRST v2 的 256 个测试 ID 都有 107 个出现在现有 SIRST 清单，其中 11 个进入 train50。剩余 149 个 ID 还需检查背景和近重复。额外独立目标域候选是有像素掩码的 BIT-SIRST，拿到完整数据后先核对数据来源和场景重合。 [DenseSIRST 官方资料](https://github.com/GrokCV/DenseSIRST) · [Open SIRST v2 官方资料](https://github.com/YimianDai/open-sirst-v2) · [BIT-SIRST 官方资料](https://github.com/EdwardBao1006/bit_sirst#dataset-description) |
| 4 | **骨干计算预算** | 当前已经用官方 SAM2 Hiera 的最小型号 Tiny。完整 CoRe-IR 约 38.58M 总参数、1.154M 可训练参数。报告参数、单图 FLOPs、峰值显存、缓存后延迟与五支持构建时间。实际内部编码分辨率的对照优先于换骨干。Small 仅作为将来检验模块跨骨干规模的可选对照。 [SAM2 官方型号表](https://github.com/facebookresearch/sam2#model-description) |

Adapter-only 已由逐步构造中「Native memory + adapters」一格覆盖。它与原生冻结五支持、完整 CSM + RCR 一起解释逐步收益，沿用 [已有逐步构造结果](../outputs/irstd_stepwise_csm_20261115/results.csv)。

文件核对也确认了重合。Open SIRST v2 的 107 个同 ID 测试图与本地 SIRST 原图逐字节一致。DenseSIRST 的对应 107 个文件中有 3 个完全一致，其余同 ID 的样本仍需按同源场景处理。

分辨率容易混淆的三个数值是原始数据分辨率、送进模型预处理的工作尺寸，以及骨干内部网格。目前 IRSTD-1k 测试原图为 512 × 512，NUDT-SIRST 为 256 × 256，SIRST 有多种尺寸。CoRe-IR 正式训练使用 640 × 640 工作尺寸，SAM2 内部为 1024 × 1024。将 CoRe-IR 和 SANSA 都接 256 外部输入时，SAM2 仍使用 1024 内部网格。做计算预算对照时应显式改内部尺寸，核对支持和查询、CSM 池化网格、RoPE、mask decoder 和评价恢复全链路。

### IRSTD-1k 的分辨率披露

| 表 1 的实际工作输入 | 对应方法 |
| --- | --- |
| 640，SAM2 内部 1024 | CoRe-IR、SANSA |
| 512 | ILNet、IRSAM（IRSAM 编码器内部 1024） |
| 400 | ABCDFSS、IFA、TVGTANet、MPA |
| 257 | PAHNet |
| 256 | ISNet、DNANet、SCTransNet、MSHNet、RPCANet、HDNet、SAM-SPL、MLP-Net、SAMamba、IR-SAM2、MSHNet+NS-FPN |

这些是表 1 所用运行的配置。部分是已披露的复现适配，例如 SAMamba 受当时显存及 scan 后端限制，实跑工作输入为 256。主表应称各方法的实际评估配置。等信息量对照采用抗混叠降采样，支持图及查询图按同一规则处理，再经过各自预处理和原图指标计算。若必须比较 512 网格，给 ILNet、HDNet 等允许该尺寸的方法重新训练与评估同一配置，并同时记录成本。 [表 1 运行配置索引](../experiments/evidence/core_ir_paper_index.json)

当前表 2 用各方法的实际评测尺寸统计单图 GFLOPs。ILNet 使用第 500 轮权重，工作和编码器尺寸均为 512 × 512，实测 169.50 GFLOPs，IoU 引用主表的 61.78%。IRSAM 的工作尺寸为 512 × 512，编码器输入补齐至 1024 × 1024，实测 94.14 GFLOPs。SANSA 和 CoRe-IR 的工作尺寸为 640 × 640，编码器为 1024 × 1024，缓存支持后的单图成本为 767.08 和 419.76 GFLOPs。CoRe-IR 在表 2 报告参与量测的 seed 113 权重的 64.47% IoU。 [表 2 逐行来源](../outputs/top8_analysis_20261105/complexity_eval_input/table_rows.csv) · [复算脚本](../scripts/profile_irstd_common_input.py)

### 最终三种子结果

三个数据集各完成三次全测试集评估。每域使用同一批训练图和五张推理支持，第 120 轮评估。SD 为三个训练种子的样本标准差。九组权重哈希一致，2,853 张预测掩码的重算结果与归档指标一致。

| 数据集 | 训练种子 | IoU (%) | nIoU (%) | F1 (%) | Pd (%) | FA／百万像素 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SIRST | 111 | 77.61 | 77.39 | 87.40 | 97.25 | 59.22 |
| SIRST | 112 | 57.15 | 74.41 | 72.73 | 95.41 | 284.82 |
| SIRST | 113 | 72.17 | 74.54 | 83.84 | 96.33 | 101.88 |
| **SIRST** | **均值 ± SD** | 68.98 ± 10.60 | 75.45 ± 1.68 | 81.32 ± 7.65 | 96.33 ± 0.92 | 148.64 ± 119.85 |
| IRSTD-1k | 111 | 62.23 | 61.77 | 76.72 | 92.26 | 56.04 |
| IRSTD-1k | 112 | 62.95 | 62.48 | 77.27 | 91.58 | 55.93 |
| IRSTD-1k | 113 | 64.47 | 63.05 | 78.40 | 92.26 | 54.60 |
| **IRSTD-1k** | **均值 ± SD** | 63.22 ± 1.14 | 62.43 ± 0.64 | 77.46 ± 0.85 | 92.03 ± 0.39 | 55.53 ± 0.80 |
| NUDT-SIRST | 111 | 74.18 | 77.02 | 85.18 | 98.41 | 133.28 |
| NUDT-SIRST | 112 | 70.36 | 75.78 | 82.60 | 98.62 | 177.48 |
| NUDT-SIRST | 113 | 70.45 | 76.10 | 82.66 | 98.73 | 189.82 |
| **NUDT-SIRST** | **均值 ± SD** | 71.66 ± 2.18 | 76.30 ± 0.64 | 83.48 ± 1.47 | 98.59 ± 0.16 | 166.86 ± 29.72 |

[固定支持清单](../outputs/core_ir_final_three_seeds_20261116/fixed_support/protocol.json) · [逐种子原始结果](../outputs/core_ir_final_three_seeds_20261116/fixed_support/runs.csv) · [全精度汇总](../outputs/core_ir_final_three_seeds_20261116/fixed_support/summary.json) · [图 3](../outputs/core_ir_final_three_seeds_20261116/fixed_support/qualitative_best/qualitative_comparison.png)

## 正文与附录的结果来源

| 稿件位置 | 实际配置与证据 |
| --- | --- |
| 表 1 三域对比 | CoRe-IR 使用 19 槽 CSM、单级 256 RCR。每域用 50 张训练图、固定 5 张推理支持和种子 111/112/113 的第 120 轮权重。对比方法及档案路径见[论文证据索引](../experiments/evidence/core_ir_paper_index.json)。 |
| 表 2 复杂度 | 按方法实际评测尺寸统计单图 GFLOPs，表内 CoRe-IR 使用 IRSTD-1k seed 113。ILNet 使用第 500 轮权重。 [逐行来源](../outputs/top8_analysis_20261105/complexity_eval_input/table_rows.csv) |
| 表 3 模块四格 | 仅 RCR 为双级，完整行为单级 256。SIRST 的原生和仅 RCR 为 seed 113，CSM 与完整行为 seed 111。索引同时保留了原表和双级完整行的精确来源。 [模块消融](tiny_memory_radial_ablation.md) · [单级完整行来源](../outputs/csm_dense_memory_controls/results.csv) |
| 表 4 / 表 A1 CSM | 固定单级 256 RCR。Dense Mem. 是原生 dense + pointwise MLP，五支持共 20,480 个空间 token。Raw Target 只替换修正实例槽，两个全局槽、背景及类型标记保持。SIRST/IRSTD-1k/NUDT-SIRST 分别为 seed 111/113/111。 [三域五指标](../outputs/csm_dense_memory_controls/results.csv) · [配对 Bootstrap](../outputs/csm_dense_memory_controls/paired_image_bootstrap.csv) |
| 表 5 / 表 A2 RCR | 三域各五个设置。表中的 Pixel value 与 Fixed (0.5) 均只接 256 级。[IRSTD-1k 正式对照](../outputs/tiny_radial_256_irstd_seed113/comparison.json) · [SIRST、NUDT-SIRST 正式对照](../outputs/tiny_radial_256_other_datasets_seed111/comparison.json) |
| 表 6 逐步构造 | IRSTD-1k 的冻结 query-only、冻结原生五支持、原生记忆 + adapters、单级 CoRe-IR。IoU 依次为 0.09、32.29、60.99、64.47。前两行同用 SAM2 Tiny 初始化权重。 [五指标与额外对照](../outputs/irstd_stepwise_csm_20261115/results.csv) · [Bootstrap](../outputs/irstd_stepwise_csm_20261115/paired_image_bootstrap.csv) |
| 图 1(a)、图 1(c) / 图 4 | NUDT→IRSTD，10 个方法 × 5 个预算。CoRe-IR 使用双级 RCR 的 NUDT seed 111 源权重。非零预算均重置源权重并更新 100 次。目标测试用对应预算的 IRSTD 支持，源域回测用原 NUDT 五支持。1-shot 训练以源图作支持。 [图 1(a) 数据](../outputs/intro_figure1/fig1a_cross_domain.csv) · [50 组源/目标结果](../outputs/intro_figure1/fig1c_adaptation_retention.csv) · [训练与支持协议](../outputs/nudt_irstd_shot_curve_20261112/protocol.json) |
| 图 1(b) / 图 A1 | IRSTD-1k 双级 CoRe-IR、seed 111。五张训练支持构造全局原型，201 张测试图提供诊断区域。比较发生在记忆投影前。目标 / 高频 / 高值区域数为 201 / 201 / 171。 [来源与区域规则](../outputs/intro_figure1/fig1b_provenance.json) · [逐区域余弦](../outputs/intro_figure1/fig1b_heldout_cosines.csv) · [配对间隔](../outputs/intro_figure1/fig1b_paired_gaps.csv) |
| 图 2 / 图 3 | 图 2 为方法示意。图 3 沿用七个样本，CoRe-IR 使用固定支持与最终解码器的单级 RCR 权重。 [复现与逐图计数](../outputs/core_ir_final_three_seeds_20261116/fixed_support/qualitative_best/provenance.json) |
| 附录推理支持敏感性 | 三个训练种子比较固定五支持与三组随机五支持。 [逐抽样结果](../outputs/core_ir_followup_20261117/support/per_draw.csv) · [三域汇总](../outputs/core_ir_followup_20261117/support/by_dataset.csv) |

### 模块消融能用上的原始组合

下表统一列出双级 RCR 组合。IoU 为百分比，括号为训练种子。

| 数据集 | 原生 | 仅 CSM | 仅 RCR（双级） | CSM + RCR（双级） | 单级 256 完整版 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIRST | 61.88 (111) / 61.48 (113) | 76.45 (111) | 75.18 (113) | 76.77 (111) / 76.25 (113) | 77.61 (111) |
| IRSTD-1k | 60.99 (113) | 62.67 (113) | 61.50 (113) | 63.86 (113) | 64.47 (113) |
| NUDT-SIRST | 65.98 (111) | 67.53 (111) | 69.55 (111) | 72.71 (111) | 74.18 (111) |

目前 IRSTD-1k 和 NUDT-SIRST 拥有同种子、同级数的完整四格。SIRST 的 seed 111 已有原生、仅 CSM 和双级完整三格，seed 113 已有原生、仅 RCR 和双级完整三格。 [SIRST 原生 seed 111](../outputs/tiny_ablation_other_datasets_20261111/seed111/SIRST/native/epoch_0120/metrics/summary.json) · [原始四格及主结果运行](../experiments/evidence/tiny_memory_radial_retained.csv)

## 全量实验

全监督训练与其余补充实验沿用正在执行的队列。[实验协议](core_ir_followup_20261117.txt) · [实时状态](../outputs/core_ir_followup_20261117/queue.json)
