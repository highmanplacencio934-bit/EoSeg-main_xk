# 创新点1 ASEE：本轮训练记录

用户最新授权审核/修复后启动训练；本轮选用 `configs/synapse/vit_query_mul_scale_fusion_asee_plan.yaml`。真实GPU冒烟及checkpoint核验已经通过；本轮正式220轮训练已启动。

## 固定对照与研究变量

保留220轮、batch16、workers8、384输入/patch16、9类、16query、seed0、FP16、默认compile、LR2e-4、AdamW/WD0.05、LLRD0.8、原warmup/poly、损失/采样/增强和固定18+12划分。继续原空掩码Dice：pred非空/GT空=1，双方空=0。HD95仍为体素网格单位；物理HD95暂停。

创新仅加入ASEE：两份Plan外层LN公式，文本映射可训练，patch/query及文本/共现双分支启用。在最后3个block前注入一次，保留CLS/register、预测头和原融合。ASEE共7,360,768个可训练参数，归入原非主干优化组；冻结医学文本编码器不随分割训练加载。创新从原timm离线预训练初始化，不从baseline最终checkpoint微调。

## 正式先验已生成

- 本地官方编码器：`pretrained/Bio_ClinicalBERT/d5892b39a4adaed74b92212a44081509db72f87b`。
- 官方仓库：https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/tree/d5892b39a4adaed74b92212a44081509db72f87b 。通过本机官方HTTPS下载到SSHFS；服务器提取保持离线，没有安装/升级依赖。
- 权重435,778,770字节，SHA256=`a18c4c260fb5c0978b86658615106d5617050b5f14dac6ceb5e0d8beb2f9f719`，匹配官方LFS；其他文件匹配官方Git blob，完整来源在编码器目录source_manifest.json。
- `priors/synapse_asee_v1.pt`，SHA256=`eceb52e294151028be69433d628182a317ef707dd8e5da927b69149eb89232ff`。
- 仅使用固定18训练病例/2211原始训练切片；原始文本CLS为8×768，共现矩阵8×8；未用测试图像/标签构建先验。
- label1..8出现切片数依次1215、228、543、547、911、430、532、594；编号顺序主动脉/胆囊/左肾/右肾/肝脏/胰腺/脾脏/胃。训练和后续汇总继续按核验名称表关联标签，不含背景。
- AutoModel只加载BERT编码器，官方checkpoint中的8个预训练任务head键列为UNEXPECTED；没有缺失编码器键，未将任务head加入ASEE。

## 启动前环境与GPU冒烟

服务器5090，RTX5090D、驱动580.178.04、PyTorch2.13.0+cu132、CUDA13.2，明确使用DeepLearning_xiakun环境Python；可用磁盘约1.18TB。

首轮冒烟 `outputs/innovation1/asee_plan_smoke/20261022T043352Z` 因独立RuntimeAudit在首步发现非有限梯度而退出1，未进入正式训练。该检查发生在GradScaler正常缩放处理之前，需区分初始FP16缩放溢出与模型数值问题；没有降低初始scale、改精度或改模型。

复核冒烟 `outputs/innovation1/asee_plan_smoke/20261022T043730Z`：8步、batch16、workers8、FP16、默认compile，完整12测试病例。仅加入独立诊断回调记录原GradScaler的skip/scale和实际Adam更新；正式训练不带该回调。首轮与诊断产物分别保留，不覆盖。

复核期间另有其他用户GPU作业约5.7GiB；未对其执行任何控制。启动前空闲25.6GiB，原baseline同路径峰值reserved17.78GiB；按实际余量放行，启动器要求至少24GiB空闲。若后续不足，不自动改变baseline参数。

## 证据目录

`outputs/innovation1/asee_launch_preparation_20261022T043035Z`：官方API/下载来源、训练先验命令与日志、服务器核验、受保护文件哈希、独立启动器、AMP诊断、真实checkpoint恢复核验脚本。最终冒烟/正式启动状态将附在下方，以实际status.json为准。

## 最终GPU冒烟结论

复核冒烟正常退出0，完成8个训练batch及完整12病例测试；8个batch的loss均有限。第1个batch只有backbone.blocks.22.mlp.fc2.weight出现缩放后的溢出，原GradScaler将65536降到32768并跳过该次Adam更新，未产生optimizer state；随后7个batch的全部401组梯度张量均有限，完成7次真实Adam更新，ASEE全部28组参数收到梯度。没有调整初始scale、精度、公式、学习率或其他训练参数。

首轮失败来自隔离检查回调在GradScaler处理前强制要求所有梯度有限；修正的是本轮输出目录内的独立诊断检查，使其验证原GradScaler确实缩放/跳过，并要求后续有限梯度和真实更新。原公共审计脚本、baseline源码与正式训练回调未改。该结论针对本次启动短测，不声称未来全程不会再次发生缩放调整。

峰值GPU allocated=17.302570GiB，reserved=18.261719GiB；没有OOM。真实best checkpoint在CPU严格恢复：417个模型张量逐值相等，missing/unexpected均为空，原始先验buffer和额外配置/来源元数据一致。动态编译记录与全部步骤/逐病例CSV均保留；短测成绩不作正式创新分数。

## 已启动的正式运行

- 启动UTC：2026-10-22 04:45:51（America/Denver为2026-10-21 22:45:51）。
- 输出：`outputs/innovation1/asee_plan/20261022T044551Z`。
- 训练PID=3981715，独立supervisor PID=3981714；SSH断开不终止训练，后续以status.json和真实进程为准。
- 权重/CSV/manifest：`synapse_test_selection/20261022T044607Z_1ba7c5b9`。
- 配置：`configs/synapse/vit_query_mul_scale_fusion_asee_plan.yaml`，完整220轮，estimated_steps=30360；没有max_steps/limit_train_batches冒烟覆盖，没有诊断回调。
- CLI只改本次输出及W&B离线日志位置；run内保存`launch.json`、`status.json`、`train.log`、`run_training.py`和`organ_label_map.json`。
- 正式解析配置与最终baseline manifest逐项比较：仅网络类/ASEE新增选项、日志名称和产物路径不同；数据、预算、seed、优化器/调度/损失、增强、精度/编译、测试选择窗口均无额外变化。26份受保护源文件/配置/原材料hash无漂移，见formal_manifest_verification.json。
- 继续按最后50轮171–220评估完整12病例；保存best_test/last；原空掩码Dice，HD95单位体素，按核验label1..8关联器官名称。早期没有test CSV属于预期。没有新建自动定时检查任务。

本轮没有修改模型/训练/评估生产源码、三份创新配置或五份baseline配置；新增真实先验、官方文本编码器、运行证据和本记录，并更新AGENTS及历史准备/审查文档的最新状态入口。没有删除文件。
