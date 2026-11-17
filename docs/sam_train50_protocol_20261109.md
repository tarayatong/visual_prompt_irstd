# 新增红外方法三域 train50 / epoch120 协议

用户于2026-11-09确认：IRPruneDet、IRSAM、IR-SAM2、SAMamba 分别在 SIRST、IRSTD-1k、NUDT-SIRST 上使用每域固定50张、seed111，训练120轮后汇报第120轮。共12组，不跑500轮，不用测试集选模型、阈值、输入尺寸或训练轮次。

固定样本清单沿用 `outputs/comparison_50shot_5support/DATASET/protocol/manifest.json`，训练ID写入每组 config；每轮完整经过50张，包含最后不足整批的样本。三域测试分别86/201/664张。训练仅加载50张图及GT，测试标签只在最终评测使用。

| 方法 | 输入 / batch | 初始化 | 优化器 / 损失 |
|---|---|---|---|
| IRPruneDet | RGB512 / 4 | 保留作者发布紧凑拓扑；全部卷积、转置卷积与BN参数/统计重置；不保留任务权重数值 | Adagrad lr0.01 wd1e-4；作者 SoftLoULoss |
| IRSAM 当前公开版 | RGB512，内部编码器1024 / 1 | 加载 MobileSAM 可兼容预训练张量，其余任务模块 seed111 初始化；全模型可训练 | AdamW lr1e-4 wd0；作者 Dice+SoftIoU+10×edgeBCE；训练GT Canny100/200边缘 |
| IR-SAM2 | RGB256 / 2 | SAM2.1 Tiny trunk；任务模块 seed111 初始化，按作者冻结 trunk | 作者 Adan lr0.01 wd1e-4；所有输出头 CSALoss，含 SLS、LLoss、0.2×TDA；尺寸/对比度统计只取train50 |
| SAMamba | RGB256 / 2 | SAM2.1 Hiera-S trunk；任务模块 seed111 初始化；不加载已下载的作者目标域权重 | AdamW lr5e-4 wd1e-4；作者 SoftIoU+WeightedDice+Focal 等权 |

共同使用120轮余弦日程（SAMamba eta_min1e-5，其余1e-6），FP32，不使用AMP或compile；输入缩放、独立水平/垂直翻转、转置。IRPruneDet、SAMamba 用 ImageNet 归一化；IR-SAM2 用各域train50灰度均值/标准差；IRSAM保留原始0–255，由模型内部归一化。

这是一组公开实现的固定小样本适配实验，不能称作作者原生全监督成绩。特别是：IRPruneDet不重新进行剪枝搜索，只对发布拓扑从头训练；IRSAM当前源码有 WaveletGradientCompensation，参数与ECCV论文不同；SAMamba在10GB显存Windows环境使用可微PyTorch参考scan，输入由原生1024改为256，全部差异随结果披露。前述1024推理复杂度测量与此256训练设置不可直接混为同一配置。

评测用概率阈值0.5，双线性恢复到原图尺寸；IoU/nIoU/F1/Pd为%，Fa为全图误检像素/百万像素。Pd采用质心距离严格小于3像素的一对一匹配。保存每张二值预测与TP/FP/FN等计数；独立重计全部预测并匹配五指标、ID覆盖和权重哈希后才填入主表。退化结果仍按第120轮原值保留，不据测试结果选更好轮次。

- [队列状态](../outputs/comparison_sam_train50_20261109/queue_status.json)
- [结果与逐组来源](../outputs/comparison_sam_train50_20261109/results.json)
- [训练入口](../scripts/train_sam_extension_train50.py)、[串行可恢复队列](../scripts/run_sam_train50_queue.py)、[核验与增量填表](../scripts/update_sam_extension_metrics.py)

新结果增量写入主Markdown、CSV、TeX及证据JSON；其他方法行逐项保留。每5轮保存可恢复状态，最终仅保留第120轮作为报告权重。新增四种方法的复杂度在GPU空闲时按IRSTD-1k第120轮权重及实际输入尺寸重新测量，原始公开权重测量归档于 `outputs/top8_analysis_20261105/complexity/public_release_initial_profiles/`；其他方法复杂度与原有低IoU分析保留。

SAMamba参考后端已与官方串行 `selective_scan_ref` 对照完整Mamba层前向及反向：前向最大绝对误差1.49e-8、输入梯度2.24e-8、参数梯度2.38e-7。核验记录见[梯度对照](../outputs/comparison_sam_train50_20261109/preflight/SAMamba_gradient_reference_audit.json)。参考实现的FPS仅代表本机后端，不代表原生CUDA实现速度。
