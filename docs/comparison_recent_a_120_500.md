# 2025–2026 年 A 类会议对比补充

<!-- current-seven-batch -->
**当前执行范围（覆盖下方早期120/500计划）：**最终23个外部方法（FSS10、IRSTD13）另列Ours，已有成绩保留。仅补新增 PConv+SD、SANSA、PAHNet、SAM-SPL、SFDTNet、IOVarNet、MLP-Net；7×3域＝21组120轮结果，不跑500轮。北京时间2026-11-04晚22:00定时启动。已保存的120轮检查点直接复用，保留并披露原学习率日程；其余训练120轮。下方“选定5个”和“120/500”为原始调研历史，当前执行以协议JSON和队列为准；更新[三数据集主表](comparison_three_datasets.md)。

按2026-11-04检索快照，范围为CCF A类主会、单帧红外小目标与2D少样本/跨域分割；相关但监督或输入不同的方法另外标注。A类依据：[CCF人工智能目录](https://www.ccf.org.cn/Academic_Evaluation/AI/)及[图形多媒体目录](https://www.ccf.org.cn/Academic_Evaluation/CGAndMT/)。ECCV、ICLR、WACV和Workshop/Findings不自动归入本次CCF A类主会。NeurIPS2026截至当前日期尚未正式发表，不将录用/预印本充作已发表论文。

共登记 **62** 条相关记录。以下是本次可追溯检索得到的全部记录，不声称已经证明全球文献零遗漏：CVPR/ICCV和ICML/IJCAI按官方目录标题扫描，AAAI、ACM MM、NeurIPS另由官方页面/出版元数据补查；部分目录访问失败保留在原始清单，未将失败解释成没有论文。原始证据见experiments/evidence/recent_a_2025_2026/。

## 选定5个方法

| 方法 | 会议 | 选择原因 | 作者代码 |
|---|---|---|---|
| [PConv+SD](https://ojs.aaai.org/index.php/AAAI/article/view/32996) | AAAI2025 | 红外物理形状先验和尺度动态损失；补充近期红外专用方法 | [代码](https://github.com/JN-Yang/PConv-SDloss-Data) |
| [SANSA](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ac6ab84809cef04dbe11e9b80d63c281-Abstract-Conference.html) | NeurIPS2025 | 冻结SAM2、轻量适配、支持提示；与本方法最相关 | [代码](https://github.com/ClaudiaCuttano/SANSA) |
| [PAHNet](https://openaccess.thecvf.com/content/ICCV2025/html/Zou_Balancing_Conservatism_and_Aggressiveness_Prototype-Affinity_Hybrid_Network_for_Few-Shot_Segmentation_ICCV_2025_paper.html) | ICCV2025 | 原型与密集亲和互补；补充2025年FSS强基线 | [代码](https://github.com/tianyu-zou/PAHNet) |
| [MPA](https://openaccess.thecvf.com/content/CVPR2026/html/Nie_Cross-Domain_Few-Shot_Segmentation_via_Multi-view_Progressive_Adaptation_CVPR_2026_paper.html) | CVPR2026 | 近期跨域适配基线；重跑独立120/500组 | [代码](https://github.com/niejiahao1998/MPA) |
| [MSHNet+NS-FPN](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.html) | CVPR2026 | 近期噪声抑制方法；以固定轮数复核已有历史日程结果 | [代码](https://github.com/mengduann/NS-FPN) |

新增PConv+SD、SANSA、PAHNet；MPA、NS-FPN虽已有历史比较，仍按本次120/500固定设置独立运行。PAHNet所需作者源域5-shot完整权重单独下载核验，不能用随机原型预测器代替。DGNet/ADGNet作者声明ACM MM2026，但当前尚未独立核实正式论文集；DGNet给定DOI在Crossref返回404，暂列待核验，不算正式入选。Text-IRSTD作者仓库仅README；InvDet代码仍待发布；ISA作者仓库为空，均不得假装已能训练。

## 固定120/500协议

复用三个原manifest、各50张训练标签、seed111、完整官方测试201/664/86张。每个方法每数据集训练一条500轮轨迹；保存完整第120/500轮模型、优化器、日程及RNG。120组是500轮日程中的120轮权重，不是重新设置学习率终点为120的独立实验。两组都在训练结束后评测，测试指标不用于选epoch、支持图、阈值或超参数。共15次训练、30条结果。

普通监督网络每轮覆盖全部50张，保留最后不足batch的样本；MPA每轮10个5支持增强任务，共50张；SANSA每轮50个查询任务，支持也只从train50中抽取。部署支持量分别记录，不能将训练标注预算B=50写成部署K=50。指标为全图IoU/nIoU/F1，质心Hungarian距离<3像素的Pd，Fa=全图FP/总像素×10^6；GT>0。

所有新实验保存在outputs/comparison_recent_a_20261104/。逐方法/数据集保存epoch_0120/和epoch_0500/的predictions、metrics/per_image.csv、metrics/summary.json；检查点在checkpoints/，保留training_history.json。summary必须通过测试ID/数量核验才记完成，未完成保留空值，不填0。历史输出及暂停的旧MPA50任务不动。

PConv为作者插件与SDM损失接入现有MSHNet，明确标注本地低层放置选择；SANSA为作者支持的tiny骨干和目标域适配版，不能冒充论文large模型原生结果；NS-FPN使用作者PyTorch参考算子。所有模型规模、预训练、损失、适配和原生差异写入每次config.json。

## 论文清单（包含未入选及不适用项）

| ID | 会议 | 论文 | 适用范围 | 代码核查/入选 |
|---|---|---|---|---|
| R001 | AAAI2025 | [FAMNet: Frequency-aware Matching Network for Cross-domain Few-shot Medical Image Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/32184) | 扩展领域：医学分割 | not_audited |
| R002 | AAAI2025 | [Foreground-Covering Prototype Generation and Matching for SAM-Aided Few-Shot Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/32688) | 直接候选：FSS/跨域FSS | not_audited |
| R003 | AAAI2025 | [Motion Prior Knowledge Learning with Homogeneous Language Descriptions for Moving Infrared Small Target Detection](https://ojs.aaai.org/index.php/AAAI/article/view/32217) | 相关红外：多帧，不能直接复用单帧协议 | not_audited |
| R004 | AAAI2025 | [Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection](https://ojs.aaai.org/index.php/AAAI/article/view/32996) | 直接候选：单帧红外 | **入选 PConv+SD** |
| R005 | AAAI2025 | [SAM-Aware Graph Prompt Reasoning Network for Cross-Domain Few-Shot Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/32695) | 直接候选：FSS/跨域FSS | not_audited |
| R006 | AAAI2025 | [Semi-supervised Infrared Small Target Detection with Thermodynamic-Inspired Uneven Perturbation and Confidence Adaptation](https://ojs.aaai.org/index.php/AAAI/article/view/33086) | 相关红外：半监督，需额外无标签预算 | not_audited |
| R007 | AAAI2026 | [Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-domain Few-shot Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/37879) | 直接候选：FSS/跨域FSS | not_audited |
| R008 | AAAI2026 | [Cross-domain Joint Learning with Prototype-guided Mixture-of-Experts for Infrared Moving Small Target Detection](https://ojs.aaai.org/index.php/AAAI/article/view/37373) | 相关红外：多帧，不能直接复用单帧协议 | not_audited |
| R009 | AAAI2026 | [DEFANet: Dual-Path Edge-Target Collaboration with Frequency-Aware Enhancement for Infrared Small Target Detection](https://ojs.aaai.org/index.php/AAAI/article/download/37368/41330) | 直接候选：单帧红外 | not_audited |
| R010 | AAAI2026 | [Divide-and-Conquer Decoupled Network for Cross-Domain Few-Shot Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/37338) | 直接候选：FSS/跨域FSS | not_audited |
| R011 | AAAI2026 | [Domain-Auxiliary Infrared Moving Small Target Detection by Learning to Overlook Domain Discrepancy](https://ojs.aaai.org/index.php/AAAI/article/download/37293/41255) | 相关红外：多帧，不能直接复用单帧协议 | not_audited |
| R012 | AAAI2026 | [SeViL: Semi-supervised Vision-Language Learning with Text Prompt Guiding for Moving Infrared Small Target Detection](https://ojs.aaai.org/index.php/AAAI/article/download/37372/41334) | 相关红外：多帧，不能直接复用单帧协议 | not_audited |
| R013 | AAAI2026 | [Spatio-Temporal Context Learning with Temporal Difference Convolution for Moving Infrared Small Target Detection](https://ojs.aaai.org/index.php/AAAI/article/download/37385/41347) | 相关红外：多帧，不能直接复用单帧协议 | not_audited |
| R014 | AAAI2026 | [Training-free Boosting for Few-shot Segmentation via Generalizing Semantic Mining](https://ojs.aaai.org/index.php/AAAI/article/view/38076) | 直接候选：FSS/跨域FSS | not_audited |
| R015 | CVPR2025 | [DSV-LFS: Unifying LLM-Driven Semantic Cues with Visual Features for Robust Few-Shot Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Karimi_DSV-LFS_Unifying_LLM-Driven_Semantic_Cues_with_Visual_Features_for_Robust_CVPR_2025_paper.html) | 直接候选：FSS/跨域FSS | existing_local_integration_missing_verified_task_weights |
| R016 | CVPR2025 | [Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Dual-Agent_Optimization_framework_for_Cross-Domain_Few-Shot_Segmentation_CVPR_2025_paper.html) | 直接候选：FSS/跨域FSS | not_audited |
| R017 | CVPR2025 | [Generalized Few-shot 3D Point Cloud Segmentation with Vision-Language Model](https://openaccess.thecvf.com/content/CVPR2025/html/An_Generalized_Few-shot_3D_Point_Cloud_Segmentation_with_Vision-Language_Model_CVPR_2025_paper.html) | 扩展领域：3D/视频/动作 | not_audited |
| R018 | CVPR2025 | [Hyperbolic Uncertainty-Aware Few-Shot Incremental Point Cloud Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Sur_Hyperbolic_Uncertainty-Aware_Few-Shot_Incremental_Point_Cloud_Segmentation_CVPR_2025_paper.html) | 扩展领域：3D/视频/动作 | not_audited |
| R019 | CVPR2025 | [SAIST: Segment Any Infrared Small Target Model Guided by Contrastive Language-Image Pretraining](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_SAIST_Segment_Any_Infrared_Small_Target_Model_Guided_by_Contrastive_CVPR_2025_paper.html) | 直接候选：单帧红外 | not_audited |
| R020 | CVPR2025 | [Text Augmented Correlation Transformer For Few-shot Classification & Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Nandam_Text_Augmented_Correlation_Transformer_For_Few-shot_Classification__Segmentation_CVPR_2025_paper.html) | 直接候选：FSS/跨域FSS | not_audited |
| R021 | CVPR2025 | [The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_The_Devil_is_in_Low-Level_Features_for_Cross-Domain_Few-Shot_Segmentation_CVPR_2025_paper.html) | 直接候选：FSS/跨域FSS | not_audited |
| R022 | CVPR2026 | [Bayesian Decomposition and Semantic Completion for Few-shot Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_Bayesian_Decomposition_and_Semantic_Completion_for_Few-shot_Semantic_Segmentation_CVPR_2026_paper.html) | 直接候选：FSS/跨域FSS | not_audited |
| R023 | CVPR2026 | [CHAL: Causal-guided Hierarchical Anomaly-aware Learning for Moving Infrared Small Target Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Duan_CHAL_Causal-guided_Hierarchical_Anomaly-aware_Learning_for_Moving_Infrared_Small_Target_CVPR_2026_paper.html) | 相关红外：多帧，不能直接复用单帧协议 | not_audited |
| R024 | CVPR2026 | [Cross-Domain Few-Shot Segmentation via Multi-view Progressive Adaptation](https://openaccess.thecvf.com/content/CVPR2026/html/Nie_Cross-Domain_Few-Shot_Segmentation_via_Multi-view_Progressive_Adaptation_CVPR_2026_paper.html) | 直接候选：FSS/跨域FSS | **入选 MPA** |
| R025 | CVPR2026 | [Focus on Background: Exploring SAM's Potential in Few-shot Medical Image Segmentation with Background-centric Prompting](https://openaccess.thecvf.com/content/CVPR2026/html/Bo_Focus_on_Background_Exploring_SAMs_Potential_in_Few-shot_Medical_Image_CVPR_2026_paper.html) | 扩展领域：医学分割 | not_audited |
| R026 | CVPR2026 | [SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Li_SD-FSMIS_Adapting_Stable_Diffusion_for_Few-Shot_Medical_Image_Segmentation_CVPR_2026_paper.html) | 扩展领域：医学分割 | not_audited |
| R027 | CVPR2026 | [Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.html) | 直接候选：单帧红外 | **入选 MSHNet+NS-FPN** |
| R028 | CVPR2026 | [Selective, Regularized, and Calibrated: Harnessing Vision Foundation Models for Cross-Domain Few-Shot Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_Selective_Regularized_and_Calibrated_Harnessing_Vision_Foundation_Models_for_Cross-Domain_CVPR_2026_paper.html) | 直接候选：FSS/跨域FSS | not_audited |
| R029 | CVPR2026 | [Target-Aware Invertible Encoder with Reconstruction Guidance for Infrared Small Target Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_CVPR_2026_paper.html) | 直接候选：单帧红外 | author_code_pending_institutional_review_20261104 |
| R030 | ICCV2025 | [Adapting In-Domain Few-Shot Segmentation to New Domains without Source Domain Retraining](https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Adapting_In-Domain_Few-Shot_Segmentation_to_New_Domains_without_Source_Domain_ICCV_2025_paper.html) | 直接候选：FSS/跨域FSS | author_repository_empty_20261104 |
| R031 | ICCV2025 | [Balancing Conservatism and Aggressiveness: Prototype-Affinity Hybrid Network for Few-Shot Segmentation](https://openaccess.thecvf.com/content/ICCV2025/html/Zou_Balancing_Conservatism_and_Aggressiveness_Prototype-Affinity_Hybrid_Network_for_Few-Shot_Segmentation_ICCV_2025_paper.html) | 直接候选：FSS/跨域FSS | **入选 PAHNet** |
| R032 | ICCV2025 | [DISTA-Net: Dynamic Closely-Spaced Infrared Small Target Unmixing](https://openaccess.thecvf.com/content/ICCV2025/html/Han_DISTA-Net_Dynamic_Closely-Spaced_Infrared_Small_Target_Unmixing_ICCV_2025_paper.html) | 相关红外：亚像素解混，输出不同 | not_audited |
| R033 | ICCV2025 | [DeFSS: Image-to-Mask Denoising Learning for Few-shot Segmentation](https://openaccess.thecvf.com/content/ICCV2025/html/Qin_DeFSS_Image-to-Mask_Denoising_Learning_for_Few-shot_Segmentation_ICCV_2025_paper.html) | 直接候选：FSS/跨域FSS | not_audited |
| R034 | ICCV2025 | [DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup](https://openaccess.thecvf.com/content/ICCV2025/html/Qu_DictAS_A_Framework_for_Class-Generalizable_Few-Shot_Anomaly_Segmentation_via_Dictionary_ICCV_2025_paper.html) | 扩展领域：异常分割 | not_audited |
| R035 | ICCV2025 | [From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision](https://openaccess.thecvf.com/content/ICCV2025/html/Yu_From_Easy_to_Hard_Progressive_Active_Learning_Framework_for_Infrared_ICCV_2025_paper.html) | 相关红外：点监督，监督形式不同 | not_audited |
| R036 | ICCV2025 | [Generalized Few-Shot Point Cloud Segmentation via LLM-Assisted Hyper-Relation Matching](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Generalized_Few-Shot_Point_Cloud_Segmentation_via_LLM-Assisted_Hyper-Relation_Matching_ICCV_2025_paper.html) | 扩展领域：3D/视频/动作 | not_audited |
| R037 | ICCV2025 | [Incremental Few-Shot Semantic Segmentation via Multi-Level Switchable Visual Prompts](https://openaccess.thecvf.com/content/ICCV2025/html/Wan_Incremental_Few-Shot_Semantic_Segmentation_via_Multi-Level_Switchable_Visual_Prompts_ICCV_2025_paper.html) | 扩展设置：增量/广义FSS | not_audited |
| R038 | ICCV2025 | [MOVE: Motion-Guided Few-Shot Video Object Segmentation](https://openaccess.thecvf.com/content/ICCV2025/html/Ying_MOVE_Motion-Guided_Few-Shot_Video_Object_Segmentation_ICCV_2025_paper.html) | 扩展领域：3D/视频/动作 | not_audited |
| R039 | ICCV2025 | [Multi-Modal Few-Shot Temporal Action Segmentation](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_Multi-Modal_Few-Shot_Temporal_Action_Segmentation_ICCV_2025_paper.html) | 扩展领域：3D/视频/动作 | not_audited |
| R040 | ICCV2025 | [Object-level Correlation for Few-Shot Segmentation](https://openaccess.thecvf.com/content/ICCV2025/html/Wen_Object-level_Correlation_for_Few-Shot_Segmentation_ICCV_2025_paper.html) | 直接候选：FSS/跨域FSS | not_audited |
| R041 | ICCV2025 | [Probabilistic Prototype Calibration of Vision-language Models for Generalized Few-shot Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Probabilistic_Prototype_Calibration_of_Vision-language_Models_for_Generalized_Few-shot_Semantic_ICCV_2025_paper.html) | 扩展设置：增量/广义FSS | not_audited |
| R042 | ICCV2025 | [Text-IRSTD: Leveraging Semantic Text to Promote Infrared Small Target Detection in Complex Scenes](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Text-IRSTD_Leveraging_Semantic_Text_to_Promote_Infrared_Small_Target_Detection_ICCV_2025_paper.html) | 直接候选：单帧红外 | author_repository_readme_only_20261104 |
| R043 | ICML2025 | [Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation](https://proceedings.mlr.press/v267/tong25b.html) | 直接候选：FSS/跨域FSS | not_audited |
| R044 | ICML2025 | [DyPolySeg: Taylor Series-Inspired Dynamic Polynomial Fitting Network for Few-shot Point Cloud Semantic Segmentation](https://proceedings.mlr.press/v267/wang25aa.html) | 扩展领域：3D/视频/动作 | not_audited |
| R045 | ICML2025 | [Self-Disentanglement and Re-Composition for Cross-Domain Few-Shot Segmentation](https://proceedings.mlr.press/v267/tong25d.html) | 直接候选：FSS/跨域FSS | not_audited |
| R046 | ICML2025 | [Unlocking the Power of SAM 2 for Few-Shot Segmentation](https://proceedings.mlr.press/v267/xu25an.html) | 直接候选：FSS/跨域FSS | not_audited |
| R047 | ICML2026 | [Diffuse to Detect: Bi-Level Sample Rebalancing with Pseudo-Label Diffusion for Point-Supervised Infrared Small-Target Detection](https://proceedings.mlr.press/v306/liu26fj.html) | 相关红外：点监督，监督形式不同 | not_audited |
| R048 | ICML2026 | [From Coarse to Fine: Deep Prototype Refinement Network for Few-Shot Point Cloud Semantic Segmentation](https://proceedings.mlr.press/v306/wang26dm.html) | 扩展领域：3D/视频/动作 | not_audited |
| R049 | ICML2026 | [PromptPilot: Game-Theoretic Multi-Agent Prompt Optimization for Segment Anything](https://proceedings.mlr.press/v306/shi26p.html) | 直接候选：FSS/跨域FSS | not_audited |
| R050 | ICML2026 | [Temporal-Emerged Prompting for Segment Anything in Multiframe Infrared Small Target Detection](https://proceedings.mlr.press/v306/xing26b.html) | 相关红外：多帧，不能直接复用单帧协议 | not_audited |
| R051 | IJCAI2025 | [Concentrate on Weakness: Mining Hard Prototypes for Few-Shot Medical Image Segmentation](https://www.ijcai.org/proceedings/2025/139) | 扩展领域：医学分割 | not_audited |
| R052 | IJCAI2026 | [Bridging the Objective Gap: A Unified Pre-Training Framework for Few-Shot Medical Image Segmentation](https://www.ijcai.org/proceedings/2026/112) | 扩展领域：医学分割 | not_audited |
| R053 | IJCAI2026 | [Learning with Semantic Priors: Stabilizing Point-Supervised Infrared Small Target Detection via Hierarchical Knowledge Distillation](https://www.ijcai.org/proceedings/2026/224) | 相关红外：点监督，监督形式不同 | not_audited |
| R054 | IJCAI2026 | [Rethinking Multimodal Few-Shot 3D Point Cloud Segmentation: From Fused Refinement to Decoupled Arbitration](https://www.ijcai.org/proceedings/2026/103) | 扩展领域：3D/视频/动作 | not_audited |
| R055 | MM2025 | [MM-Prompt: Multi-modality and Multi-granularity Prompts for Few-Shot Segmentation](https://doi.org/10.1145/3746027.3754967) | 直接候选：FSS/跨域FSS | not_audited |
| R056 | MM2025 | [Spatial-Frequency Mamba Collaborative Learning Network for Infrared Small Target Detection](https://doi.org/10.1145/3746027.3754572) | 直接候选：单帧红外 | not_audited |
| R057 | MM2025 | [Textual and Visual Guided Task Adaptation for Source-Free Cross-Domain Few-Shot Segmentation](https://doi.org/10.1145/3746027.3755772) | 直接候选：FSS/跨域FSS | not_audited |
| R058 | MM2025 | [The Overlooked Matters: Revisiting Background, Prototype, and Activation in Few-Shot Medical Image Segmentation](https://doi.org/10.1145/3746027.3754987) | 扩展领域：医学分割 | not_audited |
| R059 | NeurIPS2025 | [Reasoning Beyond Points: A Visual Introspective Approach for Few-Shot 3D Segmentation](https://doi.org/10.52202/085713-3914) | 扩展领域：3D/视频/动作 | not_audited |
| R060 | NeurIPS2025 | [Rethinking Evaluation of Infrared Small Target Detection](https://papers.nips.cc/paper_files/paper/2025/file/a81051ae2c8b1e46bd51480917b8ab84-Paper-Datasets_and_Benchmarks_Track.pdf) | 评价/基准，不是训练型对比方法 | not_audited |
| R061 | NeurIPS2025 | [SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ac6ab84809cef04dbe11e9b80d63c281-Abstract-Conference.html) | 直接候选：FSS/跨域FSS | **入选 SANSA** |
| R062 | NeurIPS2025 | [SegGraph: Leveraging Graphs of SAM Segments for Few-Shot 3D Part Segmentation](https://doi.org/10.52202/085713-0435) | 扩展领域：3D/视频/动作 | not_audited |

## 执行入口

```powershell
python -m scripts.run_recent_a_queue --wait-pids 23468 18036 29476
python -m scripts.report_recent_a_comparison
```

队列只在已有训练退出后串行派发，每个任务先执行GPU原生前向/反向检查，再正式训练；失败写日志并继续其余任务。重新执行从最新完整检查点恢复。实际状态以execution_status.json、training_queue.json和各日志为准，本页不把排队写成训练完成。
