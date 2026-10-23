# EoSeg Synapse 工作约定

本文件适用于当前 EoSeg Synapse 项目。依据当前代码、用户已确认的决定和 `.codex` 文档编写，更新于 2026-10-21。它不是其他数据集或其他项目的通用训练配置。

## 工作原则与依据

- 目标是在忠于作者 baseline 的基础上做创新。不得为了跑通、节省显存或提高分数，静默更改 baseline 的模型、数据、优化器、训练预算或指标计算。
- 最新用户指令优先；具体参数以所选 YAML、实际调用代码及解析后的 CLI 配置核对。文档与代码不一致时先指出冲突，不凭文档直接改代码。
- `.codex/synapse_reproduction_baseline.md` 记录主基线；`.codex/synapse_config_guide.md` 解释五套配置；`.codex/synapse_baseline_run_audit.md` 记录审计及未完成项；`.codex/terminal_conventions.md` 记录终端约定。历史问题列表不能当作当前状态。
- 不猜测作者未公开的实现或实验策略，不以其他数据集的配置补齐 Synapse 的未知项。

## 终端权限与服务器

用户已于 2026-10-21 明确授权代理直接控制终端和 SSH。取消“终端命令必须由用户执行”的限制；任务范围内的检查、测试和修复由代理直接完成，不默认把命令交给用户代执行。是否启动正式长训依照当前用户任务，不因拥有终端权限而自行开跑。

服务器项目目录固定为：

```bash
cd /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse
```

不要使用旧路径 `/Disk/HDD/Sk/EoSeg-main/EoSe-main-innovation_point`。

每次服务器 Python、检查或训练命令都带以下前缀：

```bash
HF_HUB_OFFLINE=1 \
TORCH_HOME=/Disk/HDD/Sk/.cache/torch \
TIMM_USE_OLD_CACHE=1 \
python ...
```

- 训练均在服务器 RTX 5090D（32GB）执行。SSHFS 仅代表文件映射，本地 PowerShell/Python 不等于服务器进程；禁止用本地 GPU 替代正式训练。
- 2026-10-21 已通过 SSH 实测 hostname=`5090`、上述项目路径及 GPU=`NVIDIA GeForce RTX 5090 D`。当时可执行文件为 `/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python`，PyTorch=`2.13.0+cu132`、CUDA runtime=`13.2`，CUDA 可用。环境可能变化，后续运行仍核验实际解释器和设备，不假定非交互 SSH 已激活 Conda。
- 可显式使用已核验的环境 Python，或先激活对应 Conda 环境；不得假定 base。正式长训前核验实际 GPU 占用、可用显存、驱动及磁盘空间。
- 不擅自安装/升级依赖、改 CUDA 环境或清理用户数据/历史输出。连接失败时报告实际错误，不把失败描述为用户必须代执行的长期限制。

## 当前主 baseline 不变量

默认采用第三套 `configs/synapse/vit_query_mul_scale_fusion.yaml`。以下是该配置及当前代码的值，不能混用其他四套配置。

| 项目 | 固定设置 |
|---|---|
| 网络 / 训练模块 | `EoSegMultiQSegFusion` / `MaskClassificationSemantic` |
| 编码器 / 主干名称 | `models.vit.ViT` / `vit_large_patch14_reg4_dinov2` |
| 输入 / patch / 类别 | 384×384 / **16** / 9（背景 0，ignore 255） |
| Query 与融合 | 16 个 query，在最后三个 block 前注入；最后层 query 决定三层预测的融合权重；共享预测头，只监督融合结果 |
| Attention 设置 | `masked_attn_enabled=False`，`attn_mask_annealing_enabled=False` |
| Batch / workers | 16 / 8；训练 shuffle=True、drop_last=True |
| Epoch / 基础 LR | **220 / 2e-4** |
| 优化器 | AdamW，weight_decay=0.05；不另加 bias/Norm 零衰减例外 |
| LLRD | 0.8，llrd_l2_enabled=True，lr_mult=1.0 |
| 调度 | 非主干预热 500 step；主干前 500 step LR=0，之后预热 1000 step；poly_power=0.9，按 step 更新 |
| 精度 / 梯度裁剪 / seed | 16-mixed / norm 0.01 / 0 |
| 设备与编译 | 作者入口默认 devices=1、accelerator=auto；fit 默认 torch.compile，matmul precision=medium |
| 损失采样 | num_points=12544，oversample_ratio=3，importance_sample_ratio=0.75 |
| 损失系数 | mask/dice/class=5/5/2，no-object=0.1；匈牙利匹配 |

- **patch_size=16 是用户明确确认的 baseline，不是待修复项。** 384 输入对应 24×24 网格；不能因主干或权重名称中的 patch14 改成 14、392 输入或额外补边。
- 五套配置保留各自预算：第三套 220 epochs，其余四套 200；LR、损失及优化器差异见配置说明。第一套反卷积基线没有公共 LLRD/调度器，不能统一套用第三套规则。
- 不额外写入已撤回的 YAML accelerator/devices 覆盖；实际训练必须确认 GPU，不能将 accelerator=auto 的 CPU 回落当作正式训练成功。
- 不因 OOM 默认减 batch、改尺寸、改精度或增加梯度累积；先诊断并报告需要改变的变量。

## 数据与预处理

- 使用 `project_TransUNet/data/Synapse/train_npz` 的二维 `.npz` 切片和 `project_TransUNet/data/Synapse/test_vol_h5` 的三维 `.npy.h5` 病例，字段为 `image`、`label`。
- 划分来自 `project_TransUNet/TransUNet/lists/lists_Synapse/train.txt` 与 `test_vol.txt`：固定 18 个训练病例、2211 张切片，12 个测试病例，无病例交叉；不随机重划，不改原始列表。
- 无独立验证集，val_dataloader 返回空列表，主配置关闭验证。各阶段都校验两份列表及病例交叉，按 fit/test 阶段检查相应文件；长训前提前检查测试体积可读性。
- 保留现有在线增强：颜色增强、水平翻转、scale_range=(0.5,2.0)、补零和 384 裁剪。灰度复制为三通道，ImageNet 标准化在模型内部完成；不得重复归一化或增加 /255。
- 当前没有原始 NIfTI/DICOM 的 HU 窗、spacing 重采样及完整在线预处理链路，不能声称支持原始 CT 直接训练。
- 不凭文件扩展名认定离线处理已全部核验，不编造器官名称映射。指标保持现有实现，HD95 当前没有 spacing 参数，不能直接标为毫米；空 mask 行为不得静默修改。

## 预训练、创新与兼容性修改

- 用户目标文件是 `dinov2_vitl14_reg4_pretrain.pth`，本地权重接口和主干改进按用户计划后续处理。当前默认 timm 自动预训练流程成功运行，不等于已加载该指定文件。
- 加载适配时核验实际来源、文件哈希、register/LayerScale、patch kernel、位置编码转换及 missing/unexpected keys；不能直接假定原始 pth 可 strict load。
- 创新另建模型/模块、配置、run 名称和输出目录，不覆盖 baseline。对照保持固定数据划分及未研究变量一致，明确列出改变的因素。
- 保留已验证的基础设施修复：RVSA 惰性导入、安全日志/代码归档、无 logger 的兼容、checkpoint 子类元数据及 run_manifest；不要为了恢复作者文件外观而重新引入启动错误。

## 测试、产物与完成标准

- 冒烟使用独立 CLI 覆盖或临时配置及输出目录；可以缩短轮数/步数、关 logger/compile 或减少 workers 以检查链路，但不能把这些覆盖写回 baseline。报告中明确哪些路径尚未验证。
- 当前保留用户已指定的 `SynapseTestSelection`：220 轮完整训练，最后 50 轮评估完整 12 个测试病例，仅维护 best_test.ckpt 与滚动 last.ckpt，不按测试结果提前停止。无需另建独立评估策略长文，也不据模糊配置推断作者未说明的流程。
- best/last 包含整个模型及主干权重，缺少 optimizer/scheduler 等完整续训状态，不能声称支持精确断点续训。
- 每次 run 使用独立目录，保留 CSV、selection_summary.json 和 run_manifest.json；核对实际命令、解析配置、代码/列表哈希及环境。不要覆盖或自动删除旧结果。
- 2026-10-21 已检查 `outputs/smoke/synapse_test_selection/20261021T024815Z_d63b7ad3`：服务器单卡 FP16、batch16、384/patch16，2 个训练 step，完整 12 病例测试，生成两份 checkpoint、逐轮/逐病例 CSV 和 manifest；manifest 源码哈希与当前文件一致。此次 workers=0、compile_disabled=True、logger=False、max_epochs=1，仅证明该冒烟链路。随后在 outputs/audit/20261021T123535Z 补齐2223文件全量扫描、服务器28项回归及默认compile/FP16/batch16/workers8/离线W&B短测；实际捕获3图，允许原有断图回退，GPU peak allocated约16.94GiB。独立加载388键/shape/值均正确；旧eager指标精确一致，compile对独立eager在case0003/class7有微差，相同compile上下文已精确重现在线CSV。详见运行审计第15、19条；不得声称跨模式逐位一致，或为消除差异静默改编译/精度/指标。不代表指定本地预训练文件已接入或正式220轮完成。
- 2026-10-21新增默认关闭的`--compile_test`，只为独立test临时编译forward。复现默认compile训练的checkpoint时显式开启；eager训练的checkpoint保留默认eager测试，不与`--compile_disabled`合用。当前服务器12例逐类指标已严格匹配在线CSV，31项回归通过。该接口不改原fit、精度、损失或YAML；新代码哈希见`.codex/baseline_test_interface_snapshot.json`，旧快照作为历史证据保留。
- `scripts/audit_synapse_baseline.py` 是独立只读数据审计及可选短测监测工具，不由baseline配置自动导入；监测回调仅在隔离CLI覆盖中使用。
- 按改动范围运行必要检查。区分静态检查、CPU 小模型测试、服务器真实冒烟、完整训练及论文成绩，不相互替代；保留现有测试，新增测试只覆盖必要行为。

## 回复与交付

- 用户于2026-10-21明确要求：后续baseline及创新实验的训练/测试结果汇总，必须将器官名称与对应Dice匹配展示，不再只给class_1..8。统一读取`priors/synapse_label_map.verified.json`，以实际`label_id`关联`dice_class_{id}`（以及HD95），不能按论文表格、原始BTCV编号或其他项目的排列配名。
- 当前映射：1=主动脉、2=胆囊、3=左肾、4=右肾、5=肝脏、6=胰腺、7=脾脏、8=胃；不整理背景0。报告保留标签ID、器官名称、Dice原值/百分数的明确标识；有HD95时标为体素，同时说明所取epoch/checkpoint。8器官平均来自同一checkpoint，不跨epoch逐类挑最高值。
- 该映射为当前数据的抽样语义核验，证据及边界见`.codex/synapse_label_mapping.md`。后续若更换数据或预处理版本，重新核对；不得直接使用已发现错误的`visualize_synapse_results.py`旧名称表。添加名称只属于展示/导出，不改变训练标签、模型输出通道、损失、指标计算或当前baseline进程。

- 使用中文，先说明结果，再说明验证及未完成项；不得把计划写成已执行结果。
- 每次对话结束必须列出本轮新增、修改、删除的文件及用途；无修改则明确说明。模型/训练参数若有变化，必须逐项说明，不能只写“优化”或“修复”。

## 本次运行保护与后续工作

2026-10-21用户已授权启动正式baseline，运行记录与单次8小时后任务见`.codex/baseline_followup_task.md`。准备创新代码之前先核对该记录；baseline运行期间保持原模型/训练文件、配置、数据和输出不变，创新代码与配置独立。不因定时检查到点或创新代码准备完成而停止未完成的baseline；不自行启动创新训练，准备完成后等待用户指示。

## 用户授权的物理HD95扩展（2026-10-21）

- 用户已明确要求增加真实物理距离HD95。使用独立 `configs/synapse/vit_query_mul_scale_fusion_hd95_mm.yaml`，说明见 `docs/synapse_physical_hd95.md`；本次正在运行的原baseline和原CSV继续采用体素口径，不热切换。
- 真实间距表位置为 `priors/synapse_spacing.verified.json`，顺序Z,Y,X、单位mm，必须按病例核验原始来源、当前H5的shape和标签哈希；缺表或来源/几何未核验时必须报错，不推测spacing。目前真实表尚缺失。
- 新口径汇总按已核验名称表展示HD95_mm，同时保留HD95_voxel。上述原baseline条款中的“HD95标体素”仍用于旧run，新物理run必须明确标mm。仅更改距离单位，不能顺带修改Dice、原空掩码规则、预测后处理、模型或训练参数。

## 当前默认Dice口径与最终baseline（2026-10-21用户确认）

- 用户明确要求继续沿用原空掩码规则作为当前baseline和后续创新的默认对照：pred非空/GT空时Dice=1，双方空时Dice=0。已知问题不等于授权修正；后续用户明确要求时才切换，原始结果始终保留。
- baseline已完成220轮、退出码0，最终最佳177轮：默认原口径平均Dice=84.718466%；同一权重仅修正错误满分分支的复核平均Dice=83.676799%。逐器官和两种口径分别记录于 `.codex/synapse_baseline_results.md`，后者不是新训练，也未覆盖正式成绩。
- 用户要求暂缓物理HD95，待原始文件完成下载后再处理。ASEE代码审查/修复不授权启动创新正式训练或替换baseline的模型、预算、增强、划分和评分规则。

## 最新创新训练授权

- 用户最新指令已授权：ASEE审核有问题先修复，无论是否发现问题，随后启动新一轮训练。该授权取代此前“准备后等待、不启动创新训练”的阶段性限制。
- 本轮选用主Plan配置 `configs/synapse/vit_query_mul_scale_fusion_asee_plan.yaml`：Plan外层LN公式、文本投影可训练、patch/query和text/probability双分支启用；其他两个变体保留，不同时启动。
- 先核验服务器环境，补齐官方来源的本地BioClinicalBERT和仅训练集生成的真实先验，真实GPU FP16/default compile冒烟通过后正式220轮；不改变baseline变量、默认原空掩码Dice规则或既有结果。实际启动信息随后记录在 `.codex/innovation1_training_run.md`。
