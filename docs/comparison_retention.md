# 比较实验保留索引

按用户最终表保留17个外部方法及其50组非空结果；另完整保留独立训练中的MPCNet和Ours。每个需要固定任务权重的外部方法／数据集只留1份checkpoint，共44份。已删除17709个文件，回收42.42 GiB；44份保留权重的SHA256在删除前后完全一致。

配置、源代码commit、执行器哈希、权重路径及SHA256集中在[selected_assets.json](../experiments/retention_comparison_20261105/selected_assets.json)。删除清单和逐文件日志位于同目录；清理前大表及完整适配统计保存在before/，不包含额外权重副本。

ABCDFSS与TVGTANet是逐查询适配，不存在一个静态的目标域最终checkpoint；它们保留唯一共享预训练底座、锁定参数和逐图结果。HERA需要保留支持适配权重及DINOv3底座；SANSA等也需要各自冻结底座。这些共享依赖不属于重复任务设置。

13组历史指标目录在此次清理前已经缺失（11组SIRST坐标修正结果、2组NUDT第125轮结果）。保留表中序列化数值、真实设置和对应轮次权重，未重新推理验证这些缺失来源；具体列表见retention_plan.json的legacy_missing_metric_sources。其他保留结果按现存计数／预测核对。

旧适配自动化gpu已暂停，历史队列被retention_lock.json锁定。Ours的既有消融／Top-k实验属于另一个保留范围，仍完整保留；本表Ours对应SIRST、IRSTD有中心SAM，NUDT无中心SAM的M2可学习径向、Top5设置。

删除的方法为Matcher、PI-CLIP、DSV-LFS、PConv+SD、SFDTNet、IOVarNet；未入选的DGNet／ISA来源也已清理。MPCNet目录、训练代码及运行进程未纳入删除范围。其表格增量更新器已适配缩减后的方法数。

| 方法 | 数据集 | IoU% | 真实轮次／更新 | 配置 | 唯一任务权重 |
| --- | --- | ---: | --- | --- | --- |
| ABCDFSS | SIRST | 14.65 | 25q | [设置](../outputs/comparison_task_adaptation_20261104/ABCDFSS/SIRST/selection.json) | 逐查询适配；共享底座 |
| ABCDFSS | IRSTD-1k | 7.07 |  | [设置](../outputs/comparison_50shot_5support/full16/IRSTD-1k/ABCDFSS/config.json) | 逐查询适配；共享底座 |
| ABCDFSS | NUDT-SIRST | 9.36 | 25q | [设置](../outputs/comparison_task_adaptation_20261104/ABCDFSS/NUDT-SIRST/selection.json) | 逐查询适配；共享底座 |
| IFA | SIRST | 37.68 | 500 | [设置](../outputs/comparison_50shot_5support/fss50_v2/SIRST/IFA/config.json) | [权重](../outputs/comparison_50shot_5support/fss50_v2/SIRST/IFA/checkpoints/epoch_0500.pt) |
| IFA | IRSTD-1k | 19.55 | 800u | [设置](../outputs/comparison_task_adaptation_20261104/IFA/IRSTD-1k/full50/training_config.json) | [权重](../outputs/comparison_task_adaptation_20261104/IFA/IRSTD-1k/full50/model.pt) |
| IFA | NUDT-SIRST | 29.20 | 500 | [设置](../outputs/comparison_50shot_5support/fss50_v2/NUDT-SIRST/IFA/config.json) | [权重](../outputs/comparison_50shot_5support/fss50_v2/NUDT-SIRST/IFA/checkpoints/epoch_0500.pt) |
| TVGTANet | SIRST | 21.87 |  | [设置](../outputs/comparison_50shot_5support/full16/SIRST/TVGTANet/config.json) | 逐查询适配；共享底座 |
| TVGTANet | IRSTD-1k | 11.62 |  | [设置](../outputs/comparison_50shot_5support/full16/IRSTD-1k/TVGTANet/config.json) | 逐查询适配；共享底座 |
| TVGTANet | NUDT-SIRST | 20.73 | 20q | [设置](../outputs/comparison_task_adaptation_20261104/TVGTANet/NUDT-SIRST/selection.json) | 逐查询适配；共享底座 |
| SANSA | SIRST | 62.08 | 120 | [设置](../outputs/comparison_recent_a_20261104/SIRST/SANSA/config.json) | [权重](../outputs/comparison_recent_a_20261104/SIRST/SANSA/checkpoints/epoch_0120.pt) |
| SANSA | IRSTD-1k | 62.48 | 120 | [设置](../outputs/comparison_recent_a_20261104/IRSTD-1k/SANSA/config.json) | [权重](../outputs/comparison_recent_a_20261104/IRSTD-1k/SANSA/checkpoints/epoch_0120.pt) |
| SANSA | NUDT-SIRST | 65.96 | 120 | [设置](../outputs/comparison_recent_a_20261104/NUDT-SIRST/SANSA/config.json) | [权重](../outputs/comparison_recent_a_20261104/NUDT-SIRST/SANSA/checkpoints/epoch_0120.pt) |
| PAHNet | SIRST | 10.31 | 120 | [设置](../outputs/comparison_recent_a_20261104/SIRST/PAHNet/config.json) | [权重](../outputs/comparison_recent_a_20261104/SIRST/PAHNet/checkpoints/epoch_0120.pt) |
| PAHNet | IRSTD-1k | 6.24 | 120 | [设置](../outputs/comparison_recent_a_20261104/IRSTD-1k/PAHNet/config.json) | [权重](../outputs/comparison_recent_a_20261104/IRSTD-1k/PAHNet/checkpoints/epoch_0120.pt) |
| PAHNet | NUDT-SIRST | 17.72 | 120 | [设置](../outputs/comparison_recent_a_20261104/NUDT-SIRST/PAHNet/config.json) | [权重](../outputs/comparison_recent_a_20261104/NUDT-SIRST/PAHNet/checkpoints/epoch_0120.pt) |
| MPA | SIRST | 26.67 | 500 | [设置](../outputs/comparison_50shot_5support/full16/SIRST/MPA/config.json) | [权重](../outputs/comparison_50shot_5support/full16/SIRST/MPA/checkpoints/epoch_0500.pt) |
| MPA | IRSTD-1k | 1.52 | 500 | [设置](../outputs/comparison_50shot_5support/full16/IRSTD-1k/MPA/config.json) | [权重](../outputs/comparison_50shot_5support/full16/IRSTD-1k/MPA/checkpoints/epoch_0500.pt) |
| MPA | NUDT-SIRST | 24.79 | 500 | [设置](../outputs/comparison_50shot_5support/full16/NUDT-SIRST/MPA/config.json) | [权重](../outputs/comparison_50shot_5support/full16/NUDT-SIRST/MPA/checkpoints/epoch_0500.pt) |
| HERA | SIRST | 3.64 |  | [设置](../outputs/comparison_50shot_5support/SIRST/HERA/config.json) | [权重](../outputs/comparison_50shot_5support/SIRST/HERA/checkpoints/support_adaptation.pt) |
| HERA | NUDT-SIRST | 1.66 |  | [设置](../outputs/comparison_50shot_5support/NUDT-SIRST/HERA/config.json) | [权重](../outputs/comparison_50shot_5support/NUDT-SIRST/HERA/checkpoints/support_adaptation.pt) |
| ISNet | SIRST | 56.66 | 500 | [设置](../outputs/comparison_50shot_5support/full16/SIRST/ISNet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/SIRST/ISNet/checkpoints/epoch_0500.pt) |
| ISNet | IRSTD-1k | 5.06 | 1000u | [设置](../outputs/comparison_task_adaptation_20261104/followups/ISNet/IRSTD-1k/full50/config.json) | [权重](../outputs/comparison_task_adaptation_20261104/followups/ISNet/IRSTD-1k/full50/model.pt) |
| ISNet | NUDT-SIRST | 36.52 | 500 | [设置](../outputs/comparison_50shot_5support/full16/NUDT-SIRST/ISNet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/NUDT-SIRST/ISNet/checkpoints/epoch_0500.pt) |
| DNANet | SIRST | 57.35 | 500 | [设置](../outputs/comparison_50shot_5support/full16/SIRST/DNANet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/SIRST/DNANet/checkpoints/epoch_0500.pt) |
| DNANet | IRSTD-1k | 58.87 | 500 | [设置](../outputs/comparison_50shot_5support/full16/IRSTD-1k/DNANet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/IRSTD-1k/DNANet/checkpoints/epoch_0500.pt) |
| DNANet | NUDT-SIRST | 62.20 | 500 | [设置](../outputs/comparison_50shot_5support/full16/NUDT-SIRST/DNANet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/NUDT-SIRST/DNANet/checkpoints/epoch_0500.pt) |
| MSHNet | SIRST | 60.08 | 1175 | [设置](../outputs/comparison_50shot_5support/full16/SIRST/MSHNet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/SIRST/MSHNet/checkpoints/latest.pt) |
| MSHNet | IRSTD-1k | 55.52 | 1200 | [设置](../outputs/comparison_50shot_5support/full16/IRSTD-1k/MSHNet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/IRSTD-1k/MSHNet/checkpoints/latest.pt) |
| MSHNet | NUDT-SIRST | 57.61 | 1200 | [设置](../outputs/comparison_50shot_5support/full16/NUDT-SIRST/MSHNet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/NUDT-SIRST/MSHNet/checkpoints/latest.pt) |
| SCTransNet | SIRST | 59.03 | 500 | [设置](../outputs/comparison_50shot_5support/full16/SIRST/SCTransNet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/SIRST/SCTransNet/checkpoints/epoch_0500.pt) |
| SCTransNet | IRSTD-1k | 56.35 | 500 | [设置](../outputs/comparison_50shot_5support/full16/IRSTD-1k/SCTransNet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/IRSTD-1k/SCTransNet/checkpoints/epoch_0500.pt) |
| SCTransNet | NUDT-SIRST | 63.72 | 125 | [设置](../outputs/comparison_50shot_5support/full16/NUDT-SIRST/SCTransNet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/NUDT-SIRST/SCTransNet/checkpoints/epoch_0125.pt) |
| RPCANet | SIRST | 46.31 | 500 | [设置](../outputs/comparison_50shot_5support/full16/SIRST/RPCANet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/SIRST/RPCANet/checkpoints/epoch_0500.pt) |
| RPCANet | IRSTD-1k | 43.41 | 500 | [设置](../outputs/comparison_50shot_5support/full16/IRSTD-1k/RPCANet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/IRSTD-1k/RPCANet/checkpoints/epoch_0500.pt) |
| RPCANet | NUDT-SIRST | 17.37 | 1000u | [设置](../outputs/comparison_task_adaptation_20261104/RPCANet/NUDT-SIRST/full50/config.json) | [权重](../outputs/comparison_task_adaptation_20261104/RPCANet/NUDT-SIRST/full50/model.pt) |
| ILNet | SIRST | 63.21 | 500 | [设置](../outputs/comparison_50shot_5support/full16/SIRST/ILNet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/SIRST/ILNet/checkpoints/epoch_0500.pt) |
| ILNet | IRSTD-1k | 61.78 | 500 | [设置](../outputs/comparison_50shot_5support/full16/IRSTD-1k/ILNet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/IRSTD-1k/ILNet/checkpoints/epoch_0500.pt) |
| ILNet | NUDT-SIRST | 56.57 | 125 | [设置](../outputs/comparison_50shot_5support/full16/NUDT-SIRST/ILNet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/NUDT-SIRST/ILNet/checkpoints/epoch_0125.pt) |
| HDNet | SIRST | 62.28 | 500 | [设置](../outputs/comparison_50shot_5support/full16/SIRST/HDNet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/SIRST/HDNet/checkpoints/epoch_0500.pt) |
| HDNet | IRSTD-1k | 59.29 | 1200 | [设置](../outputs/comparison_50shot_5support/full16/IRSTD-1k/HDNet/config.json) | [权重](../outputs/comparison_50shot_5support/full16/IRSTD-1k/HDNet/checkpoints/latest.pt) |
| HDNet | NUDT-SIRST | 61.61 | 1000u | [设置](../outputs/comparison_task_adaptation_20261104/HDNet/NUDT-SIRST/full50/config.json) | [权重](../outputs/comparison_task_adaptation_20261104/HDNet/NUDT-SIRST/full50/model.pt) |
| SAM-SPL | SIRST | 54.25 | 120 | [设置](../outputs/comparison_recent_a_20261104/SIRST/SAM-SPL/config.json) | [权重](../outputs/comparison_recent_a_20261104/SIRST/SAM-SPL/checkpoints/epoch_0120.pt) |
| SAM-SPL | IRSTD-1k | 30.82 | 120 | [设置](../outputs/comparison_recent_a_20261104/IRSTD-1k/SAM-SPL/config.json) | [权重](../outputs/comparison_recent_a_20261104/IRSTD-1k/SAM-SPL/checkpoints/epoch_0120.pt) |
| SAM-SPL | NUDT-SIRST | 46.80 | 120 | [设置](../outputs/comparison_recent_a_20261104/NUDT-SIRST/SAM-SPL/config.json) | [权重](../outputs/comparison_recent_a_20261104/NUDT-SIRST/SAM-SPL/checkpoints/epoch_0120.pt) |
| MLP-Net | SIRST | 21.98 | 120 | [设置](../outputs/comparison_recent_a_20261104/SIRST/MLP-Net/config.json) | [权重](../outputs/comparison_recent_a_20261104/SIRST/MLP-Net/checkpoints/epoch_0120.pt) |
| MLP-Net | IRSTD-1k | 6.83 | 120 | [设置](../outputs/comparison_recent_a_20261104/IRSTD-1k/MLP-Net/config.json) | [权重](../outputs/comparison_recent_a_20261104/IRSTD-1k/MLP-Net/checkpoints/epoch_0120.pt) |
| MLP-Net | NUDT-SIRST | 16.86 | 120 | [设置](../outputs/comparison_recent_a_20261104/NUDT-SIRST/MLP-Net/config.json) | [权重](../outputs/comparison_recent_a_20261104/NUDT-SIRST/MLP-Net/checkpoints/epoch_0120.pt) |
| MSHNet+NS-FPN | SIRST | 58.35 | 500 | [设置](../outputs/comparison_50shot_5support/full16/SIRST/MSHNet+NS-FPN/config.json) | [权重](../outputs/comparison_50shot_5support/full16/SIRST/MSHNet+NS-FPN/checkpoints/epoch_0500.pt) |
| MSHNet+NS-FPN | IRSTD-1k | 53.17 | 500 | [设置](../outputs/comparison_50shot_5support/full16/IRSTD-1k/MSHNet+NS-FPN/config.json) | [权重](../outputs/comparison_50shot_5support/full16/IRSTD-1k/MSHNet+NS-FPN/checkpoints/epoch_0500.pt) |
| MSHNet+NS-FPN | NUDT-SIRST | 53.00 | 1000u | [设置](../outputs/comparison_task_adaptation_20261104/MSHNet+NS-FPN/NUDT-SIRST/full50/config.json) | [权重](../outputs/comparison_task_adaptation_20261104/MSHNet+NS-FPN/NUDT-SIRST/full50/model.pt) |

Ours的三份表中权重及计数来源见[保护记录](../experiments/retention_comparison_20261105/ours_protected.json)，其余保留消融见[已有消融索引](retained_table_experiments.md)。

只重建当前大表：`python -m scripts.render_retained_comparison`。原执行器用于保留模型的构建和历史流程，加载权重时应使用本索引的具体路径；不要重新运行已停用的历史全方法队列。

<!-- SAM_TRAIN50_BEGIN -->
## 新增红外方法 train50（2026-11-09）

每域50张、seed111、第120轮；配置和权重逐组唯一。初始化及实现差异见[执行协议](sam_train50_protocol_20261109.md)。

| 方法 | 数据集 | 状态 | IoU | 配置 | 第120轮权重 |
|---|---|---|---:|---|---|
| IRPruneDet | SIRST | verified | 25.55 | [config](../outputs/comparison_sam_train50_20261109/SIRST/IRPruneDet/config.json) | [checkpoint](../outputs\comparison_sam_train50_20261109\SIRST\IRPruneDet\checkpoints\epoch_0120.pt) |
| IRPruneDet | IRSTD-1k | verified | 21.73 | [config](../outputs/comparison_sam_train50_20261109/IRSTD-1k/IRPruneDet/config.json) | [checkpoint](../outputs\comparison_sam_train50_20261109\IRSTD-1k\IRPruneDet\checkpoints\epoch_0120.pt) |
| IRPruneDet | NUDT-SIRST | verified | 12.72 | [config](../outputs/comparison_sam_train50_20261109/NUDT-SIRST/IRPruneDet/config.json) | [checkpoint](../outputs\comparison_sam_train50_20261109\NUDT-SIRST\IRPruneDet\checkpoints\epoch_0120.pt) |
| IR-SAM2 | SIRST | verified | 47.51 | [config](../outputs/comparison_sam_train50_20261109/SIRST/IR-SAM2/config.json) | [checkpoint](../outputs\comparison_sam_train50_20261109\SIRST\IR-SAM2\checkpoints\epoch_0120.pt) |
| IR-SAM2 | IRSTD-1k | verified | 40.25 | [config](../outputs/comparison_sam_train50_20261109/IRSTD-1k/IR-SAM2/config.json) | [checkpoint](../outputs\comparison_sam_train50_20261109\IRSTD-1k\IR-SAM2\checkpoints\epoch_0120.pt) |
| IR-SAM2 | NUDT-SIRST | verified | 46.59 | [config](../outputs/comparison_sam_train50_20261109/NUDT-SIRST/IR-SAM2/config.json) | [checkpoint](../outputs\comparison_sam_train50_20261109\NUDT-SIRST\IR-SAM2\checkpoints\epoch_0120.pt) |
| IRSAM | SIRST | verified | 0.00 | [config](../outputs/comparison_sam_train50_20261109/SIRST/IRSAM/config.json) | [checkpoint](../outputs\comparison_sam_train50_20261109\SIRST\IRSAM\checkpoints\epoch_0120.pt) |
| IRSAM | IRSTD-1k | verified | 45.85 | [config](../outputs/comparison_sam_train50_20261109/IRSTD-1k/IRSAM/config.json) | [checkpoint](../outputs\comparison_sam_train50_20261109\IRSTD-1k\IRSAM\checkpoints\epoch_0120.pt) |
| IRSAM | NUDT-SIRST | verified | 56.28 | [config](../outputs/comparison_sam_train50_20261109/NUDT-SIRST/IRSAM/config.json) | [checkpoint](../outputs\comparison_sam_train50_20261109\NUDT-SIRST\IRSAM\checkpoints\epoch_0120.pt) |
| SAMamba | SIRST | verified | 63.89 | [config](../outputs/comparison_sam_train50_20261109/SIRST/SAMamba/config.json) | [checkpoint](../outputs\comparison_sam_train50_20261109\SIRST\SAMamba\checkpoints\epoch_0120.pt) |
| SAMamba | IRSTD-1k | verified | 47.25 | [config](../outputs/comparison_sam_train50_20261109/IRSTD-1k/SAMamba/config.json) | [checkpoint](../outputs\comparison_sam_train50_20261109\IRSTD-1k\SAMamba\checkpoints\epoch_0120.pt) |
| SAMamba | NUDT-SIRST | verified | 61.12 | [config](../outputs/comparison_sam_train50_20261109/NUDT-SIRST/SAMamba/config.json) | [checkpoint](../outputs\comparison_sam_train50_20261109\NUDT-SIRST\SAMamba\checkpoints\epoch_0120.pt) |
<!-- SAM_TRAIN50_END -->
