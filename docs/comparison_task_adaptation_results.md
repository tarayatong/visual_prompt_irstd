# 表中保留的任务适配结果

按用户最终对比表，仅保留以下8组适配／复跑结果及对应设置、代码、唯一任务权重。旧18项适配队列已停用，自动化gpu已暂停；未选中的权重和方法已清理。MPCNet在独立会话保留。完整范围见[保留索引](comparison_retention.md)。

| 方法 | 数据集 | 标记 | IoU% | nIoU% | F1% | Pd% | Fa/百万 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ABCDFSS | NUDT-SIRST | 原设置复跑 | 9.36 | 21.05 | 17.13 | 78.52 | 4700.07 |
| ABCDFSS | SIRST | 原设置复跑 | 14.65 | 19.14 | 25.56 | 69.72 | 3031.34 |
| HDNet | NUDT-SIRST | T,L | 61.61 | 62.97 | 76.25 | 89.21 | 181.91 |
| IFA | IRSTD-1k | T,P,S | 19.55 | 23.32 | 32.70 | 87.54 | 501.64 |
| ISNet [Adam-edge] | IRSTD-1k | T,L | 5.06 | 6.66 | 9.63 | 76.77 | 4678.91 |
| MSHNet+NS-FPN | NUDT-SIRST | T,L | 53.00 | 54.65 | 69.28 | 86.03 | 212.57 |
| RPCANet | NUDT-SIRST | T,L | 17.37 | 51.05 | 29.59 | 90.37 | 2268.23 |
| TVGTANet | NUDT-SIRST | 原设置复跑 | 20.73 | 23.74 | 34.34 | 78.31 | 1463.28 |

T为训练日程，L为损失，P为支持表示，S为输入尺度。ABCDFSS两组和TVGTANet/NUDT保留原设置，复跑波动不算模块收益。IFA/IRSTD原部署K50、本版K5；本轮验证选型计入B50。设置由40/10验证选择，最终使用全50训练或固定5支持逐查询适配；最终表的版本取舍属于观察测试后的展示选择。

以上保存预测均已重新计分核验。其余历史数值保存在清理前快照中，不再保留未选中的任务权重。

- ABCDFSS / NUDT-SIRST：[来源](../outputs\comparison_task_adaptation_20261104\ABCDFSS\NUDT-SIRST\report.json)。
- ABCDFSS / SIRST：[来源](../outputs\comparison_task_adaptation_20261104\ABCDFSS\SIRST\report.json)。
- HDNet / NUDT-SIRST：[来源](../outputs\comparison_task_adaptation_20261104\HDNet\NUDT-SIRST\report.json)。
- IFA / IRSTD-1k：[来源](../outputs\comparison_task_adaptation_20261104\IFA\IRSTD-1k\report.json)。
- ISNet [Adam-edge] / IRSTD-1k：[来源](../outputs\comparison_task_adaptation_20261104\followups\ISNet\IRSTD-1k\report.json)。
- MSHNet+NS-FPN / NUDT-SIRST：[来源](../outputs\comparison_task_adaptation_20261104\MSHNet+NS-FPN\NUDT-SIRST\report.json)。
- RPCANet / NUDT-SIRST：[来源](../outputs\comparison_task_adaptation_20261104\RPCANet\NUDT-SIRST\report.json)。
- TVGTANet / NUDT-SIRST：[来源](../outputs\comparison_task_adaptation_20261104\TVGTANet\NUDT-SIRST\report.json)。
