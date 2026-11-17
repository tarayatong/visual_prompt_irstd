# 红外稀疏小目标少样本分割

CoRe-IR 使用 SAM2 Tiny、紧凑语义记忆（CSM）和径向对比细化（RCR）。当前稿件的实验协议、原始指标、配置和权重由[论文实验总档](docs/core_ir_experiments.md)索引。

- [三种子固定支持复评](scripts/run_core_ir_final_three_seeds.py)
- [图 3 定性对比复现](scripts/render_core_ir_final_qualitative.py)
- [补充实验与全监督训练协议](docs/core_ir_followup_20261117.txt)
- [安装与数据准备](docs/reproduction_setup.md)

数据集、模型权重、预测掩码和运行日志保存在本地。三个基准数据集的划分与固定支持清单由准备脚本写入 `outputs/comparison_50shot_5support/<数据集>/protocol/`。[论文证据索引](experiments/evidence/core_ir_paper_index.json)记录图表对应的本地产物。
