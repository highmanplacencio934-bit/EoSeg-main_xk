# EoSeg Synapse 工作约定

本文件适用于当前 EoSeg Synapse 项目。依据当前代码、用户已确认的决定和 `.codex` 文档编写，更新于 2026-10-28。它不是其他数据集或其他项目的通用训练配置。

## 工作原则与依据

- 目标是在忠于作者 baseline 的基础上做创新。不得为了跑通、节省显存或提高分数，静默更改 baseline 的模型、数据、优化器、训练预算或指标计算。
- 最新用户指令优先；具体参数以所选 YAML、实际调用代码及解析后的 CLI 配置核对。文档与代码不一致时先指出冲突，不凭文档直接改代码。
- `.codex/synapse_reproduction_baseline.md` 记录主基线；`.codex/synapse_config_guide.md` 解释五套配置；`.codex/synapse_baseline_run_audit.md` 记录审计及未完成项；`.codex/terminal_conventions.md` 记录终端约定。历史问题列表不能当作当前状态。
- 不猜测作者未公开的实现或实验策略，不以其他数据集的配置补齐 Synapse 的未知项。

## 后续CCPS训练入口约定（2026-10-28）

- 后续新启动且已获用户授权的CCPS主模型训练（EoSeg + ASEE + CCPS），必须使用 scripts/run_synapse_deterministic_v2.py fit，搭配 configs/synapse/deterministic_v2/ccps.yaml；保留 deterministic_protocol=ce2d_stableseg_execution_v2、deterministic_profile=ccps，不能只换入口却混用旧配置或baseline profile。
- 启动前核对 outputs/determinism/ce2d_stableseg_execution_v2/readiness.json 为passed、包含ccps且当前源码哈希匹配。仅CCPS采用已验证的局部BN编译边界及backward_pass_autocast=off，原模型公式、数据增强和其余科学训练变量保持。
- 原main.py、scripts/run_synapse_roi.py、v1确定性入口及旧CCPS启动器保留作历史复现，不作为当前已修复CCPS的新训练入口。不得将旧v1失败记录改成通过；用户明确要求复验旧路径时另立实验并说明协议。CCPS组件消融须保留其各自开关，准备对应独立v2配置并核验适用性，不能直接套用主配置冒充消融。
- 详细依据见 .codex/cuda_ccps_boundary_followup.md 的“后续CCPS训练入口”。本约定不授权立即启动CCPS长训；当前G6是纯baseline重复，不能替代CCPS自身完整220轮重复性验证。

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

## 服务器GPU并发与显存规则（2026-10-26 用户更新）

- 已授权的训练、测试和冒烟任务允许与服务器其他用户的GPU任务并发。只要现场核验的剩余显存足以覆盖本次任务的预计峰值及必要临时开销，即可直接执行；不要求整卡空闲，也无需另行申请并发例外。
- 启动前查询实际GPU进程与剩余显存，按相同配置的训练/推理峰值评估需求，包含梯度、优化器状态、compile和工作区开销，不能只看模型权重大小。其他进程存在或GPU利用率较高本身不是禁止执行的理由。
- 剩余显存不足时不启动，报告需求与余量或等待资源释放。不得通过kill、暂停、重启、GPU reset或“清理显存”等方式中止或干扰其他用户任务；发生OOM也只处理本次任务，不能停止他人任务腾显存。
- 不为塞入剩余显存静默更改batch、输入尺寸、精度或其他baseline变量。任务授权范围、哈希验收、数据/结果保护、本项目既定队列顺序及防重复锁继续适用；允许资源并发不代表自动授权新实验。
- 统一细则见`.codex/server_gpu_execution_rules.md`。本节替代旧文档中的“有其他GPU任务就不能执行”“必须GPU空闲”“每次并发需额外确认”等资源限制；历史运行记录和未获授权的实验不因此改写或自动开跑。

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

## 2026-10-23教师测试接口接入

- 用户已要求接入HD95_test.py，并明确主HD95使用真实物理距离。已在AgileFormer-main/AgileFormer-main/utils.py找到并核验实际test_single_volume/calculate_metric_percase；原utils不修改，SHA256固定。原函数不传spacing，其体素结果仅作对照。
- HD95_test.py现为独立EoSeg教师评估入口，新配置vit_query_mul_scale_fusion_asee_teacher_hd95.yaml；主报告mm，来自已核验逐病例ZYX间距。同一预测上保留老师Dice/voxel，原空掩码规则不改，不重训、不覆盖旧run。老师原入口已归档。
- main.py仅在独立物理/教师test保留显式包装类，防止checkpoint旧类覆盖；普通fit/test仍沿用原加载路径。75项CPU回归通过。实现、适配细节与本轮状态见.codex/innovation1_teacher_hd95.md；真实spacing现已生成，旧条款“缺少原始文件/spacing”为历史状态。

- 教师接口的真实GPU复核发现额外softmax会将极小EoSeg像素分数舍入成并列；现使用FP32 log-score适配，保持原argmax。76项CPU回归和12例1568切片GPU诊断通过。不得恢复直接把像素分数当logits传softmax的旧适配；失败run只供审计。最新补跑路径以outputs/teacher_hd95/innovation1_best/latest_path.txt为准。


## 2026-10-23 Innovation2 CCPS 用户授权

- 用户授权修复CCPS五项问题、补充两项实验解释，随后按单次任务两小时后启动主A3，五小时后检查并在首次未执行时补启动。具体时间、任务ID和验证见.codex/innovation2_ccps_runbook.md。
- 只训练configs/synapse/vit_query_mul_scale_fusion_asee_ccps.yaml：visual_prior、patch门控开、query门控关、presence=0、init=.95、both。新增context_only/residual_only仅作独立消融配置，不自动启动。
- 必须先有outputs/innovation2/ccps_readiness.json passed及哈希匹配；通过scripts/launch_innovation2_ccps.py幂等启动，保留baseline/innovation1代码与旧输出，不自动变更参数或重训失败run。
- CCPS主HD95为真实mm且保留voxel，按Dice选checkpoint；原空掩码规则不变。初始化RNG隔离不代表GPU训练必然逐位复现；.95初始门控不是off等价替代。

## 2026-10-25 论文消融与统一8-view约定

- 论文实验清单入口为 `.codex/synapse_journal_ablation_plan.md`。目标期刊未指定，其中组件矩阵是研究建议，不冒称期刊硬性规定。
- 当前主模型补测清单仅两项：M0=baseline+当前最优8-view，M1=baseline+innovation1（ASEE）+同一8-view，均使用各自已有best checkpoint。完整ASEE+CCPS结果仅作历史参考，联合增强任务独立管理，不另列M2/M3补测任务。无需为此重训，不安排论文“有/无TTA”消融表，但Methods仍需公开TTA定义。
- 固定8-view为四尺度1/.875/1.125/.75分别配原方向与水平翻转，保持现有覆盖融合；不引入垂直翻转或左右肾通道交换。不同结构须核验模型/权重兼容，不直接套用CCPS配置。
- 此条“等待启动、不停止当前训练”已由下方2026-10-25最新定时执行授权取代；执行前仍检查真实GPU资源与任务归属。所有补测使用独立输出目录，不覆盖旧权重和结果。
- 后续ASEE、CCPS组件消融也统一8-view，最新已授权范围及顺序见下节；CCPS依赖ASEE，不虚构无ASEE的CCPS独立对照。
- 历史best统一推理不等于训练预算/选择窗口已匹配；如实标记提前停止、测试集参与选择及legacy空掩码口径。保留历史结果，不擅自改参数、增强、划分或指标。


## 2026-10-25 定时早停、评估加速与消融执行授权

- 用户明确授权Denver 10月25日14:56温和早停唯一联合增强run outputs/augmentation_joint/rotation_scale_seed0/20261025T042216Z，按实际轮次记录，不保证205轮；任务归属重新核验，保留原产物及真实退出码，不杀其他进程、不自动重启。
- 15:30/16:30/17:30/18:30/19:30/20:30/21:30共7个单次后续任务。先按GPU_CPU_HD95_Optimization_Guide.md完成未来独立入口的指标等价优化与测试，再按.codex/synapse_journal_ablation_plan.md依次执行M0→M1→S1→S2→S3→S4→A1→A2→A10→A11；A0只做兼容检查，A3复用历史参考。
- 用户最新明确：所有新增消融使用原数据增强（含ScaleJitter 0.5–2.0，保留原颜色/水平翻转/补零/裁剪）及seed=0，不使用本次新增旋转或0.75–1.50尺度；M0/M1不重训，其余组件按原220轮独立训练并以同一best固定8-view报告。
- 详细流程与停止保护见.codex/synapse_ablation_scheduled_runbook.md。输出独立放outputs/ablations/journal_20261025/，队列锁/幂等回执/实际PID核验保证本项目既定消融队列顺序和防重复；与其他用户并发按本文件显存规则判断；只有完整完成并核验才更新消融MD第11节为已完成并启动下一项。
- 科学训练变量、默认legacy Dice空mask规则、真实HD95_mm+voxel、预测及固定8-view保持；CPU加速先同预测全12病例×8类逐值对照，未通过不启正式消融，不放宽容差。旧源码默认入口及历史结果保留。
- 此授权取代“仅文档、暂不启动”和旧任务禁止早停的阶段限制；不扩展到未列变体、多seed、额外增强实验、失败自动重训或无限后续监控。

## 2026-10-25 15:30独立评估加速验收接续

- 新增scripts/run_synapse_roi.py及training/synapse_roi_metrics.py、synapse_roi_evaluation.py，原默认入口保持。状态以outputs/ablations/journal_20261025/optimization/readiness.json为准，validating/false时不能启动正式消融。
- 已完成212项CPU回归与真实完整12例FP16/compile 8-view等价复核、三步fit加12例选择冒烟；CPU第二轮重复对照在后台执行。后续任务先按同目录continuation.md核验实际所有者和退出码，再运行scripts/finalize_synapse_roi_readiness.py，完整通过才启M0。不得重复或杀掉正常验证，不放宽数值门槛。

## 2026-10-25 独立ROI评估入口已验证

- 后续消融使用scripts/run_synapse_roi.py，原main.py及原指标默认路径保留。fit仍1-view选择best，test固定8-view；要求明确物理包装类与verified spacing，teacher接口不支持此ROI范围。
- 优化门槛为outputs/ablations/journal_20261025/optimization/readiness.json及hash，详情summary.md。使用单CPU联合前景ROI+halo1调用原MedPy，未改变原legacy空mask、Dice、mm/voxel或科学训练变量。不要把CPU阶段加速倍数声称为整场训练倍数；未启用CPU并行池/流水。

## 2026-10-25 16:30队列推进已执行

- M0（baseline best177）与M1（ASEE best196）固定8-view完整12例补测已完成，汇总在outputs/ablations/journal_20261025/reports/main_model_eight_view_comparison.md。两组未重训；新启动器后处理问题保留recovery并CPU修复，未改CSV或重新推理。
- S1文本单分支（use_probability=false）已于17:09:59 Denver启动，输出outputs/ablations/journal_20261025/S1/20261025T230957Z。原增强/seed0/220轮与所有原训练设置保持，401个原优化器组逐项一致。当前运行期间不得修改已冻结的scripts/launch_synapse_journal_component.py及其共享依赖；只读检查PID、日志、last及manifest。
- M0/M1入口为scripts/launch_synapse_journal_ablation.py；组件入口为scripts/launch_synapse_journal_component.py，--next严格跟随registry，不跳过/重训失败。完整训练及同best8-view后才标完成并下一项；deferred仅恢复测试，不从weights-only续训。


## 2026-10-26 TTA 旧协议删除

- 用户明确要求删除旧旋转/镜像 TTA。已删除传统 1/2/4/8-view 实现、独立 CLI、四组对比运行器、专用 YAML 和旧变换测试；不恢复这些入口。
- 当前仅保留 `training/synapse_multiscale_tta.py` 的 `scale_hflip` 家族：2/4/6/8-view 分别使用 1/2/3/4 个既定尺度，各配原方向和水平翻转。论文和当前消融固定 8-view：1/.875/1.125/.75 × 原方向/水平翻转。
- 公共像素分数提取、肾脏诊断和产物记录迁移至 `training/synapse_tta_common.py`；该文件没有旋转、垂直翻转或传统镜像视角分支。原 1-view baseline 与 fit 选择逻辑仍由原入口实现。
- 旧输出、权重、CSV、运行清单和历史文档作为证据保留；其中的旧命令已退役，不代表当前可执行接口。现行入口为 `scripts/evaluate_synapse_multiscale_tta.py` 和 `scripts/run_synapse_roi.py`。
- 本次为代码整理；多尺度变换、覆盖融合、类别通道、Dice/HD95、模型和训练变量保持。验收证据见 `outputs/audit/tta_legacy_removal/20261026T084000Z`；未执行新训练或 GPU 补测。

- 2026-10-26 S1 文本单分支 best187 已完成同一 scale_hflip 8-view 的12例测试（推理退出0）：Dice 87.054982%，HD95_mm 8.874617，HD95_voxel 6.599391；结果见 `outputs/ablations/journal_20261025/S1/20261025T230957Z/eight_view_tests/20261026T091846Z/summary.md`。S1 已登记完成；本轮未启动 S2。


## 2026-10-26 baseline-2 同seed重复训练

- 用户已授权纯EoSeg baseline重复训练，seed=0，名称baseline-2；不启用ASEE/CCPS。直接沿用原主YAML与main.py fit，原增强、优化器、220轮和171–220原1-view选择路径全部保持。
- 输出`outputs/baseline-2/20261026T101442Z`，指针`outputs/baseline-2/latest_path.txt`；实际启动/核验见`.codex/baseline_2_training_run.md`和run内startup_verification.json。模型/数据/解析科学参数及373组优化器与首轮一致，变化只有输出路径与日志名称。
- 本轮重现首轮原voxel HD95评估入口；不得为加速或统一展示而热切换ROI、物理包装类、TTA、增强或随机性选项。比较1-view同口径结果，独立TTA成绩不得直接混比。
- 运行中冻结被run_manifest记录的训练依赖、主配置和数据；不得自动早停/重启或中止它为其他实验腾资源。其他已授权任务在剩余显存足够时可按本文件服务器规则并发；PID必须现场重核验，详见run记录。


## 2026-10-26 删除独立旋转/尺度训练增强

- 用户明确授权删除此前新增的训练增强：旋转±10度/p=.3及ScaleJitter .75–1.50。8个独立源码/入口/回调/YAML/测试文件，以及outputs/augmentation_joint与outputs/audit/joint_augmentation_20261025已删除；不依据历史Plan自动恢复。
- 后续训练继续使用原作者datasets/synapse_transforms.py与SynapseSemantic：颜色、水平翻转、ScaleJitter .5–2、补零及随机384裁剪。原增强源码、数据、模型和当前多尺度水平翻转TTA保留。
- baseline-2继续运行，未停止/重启或改变参数。其历史manifest对全项目的快照可能包含上述已删除而未使用的分支；只对用户授权的这8项允许记录删除，实际baseline依赖仍严格哈希核验，不改写原manifest。
- 当前可执行代码、配置和测试不得残留该分支的import、class_path或启动接口；历史文档/清单中的旧路径仅作记录。未来ROI验收哈希在CPU回归通过后移除这些已退役条目，不改预测、指标或训练算法。


## 2026-10-26 baseline-2已停止并删除输出

- 用户最新明确要求停止baseline-2并删除outputs/baseline-2，现已完成。训练进程、监督器及所属子进程均退出；权重、日志、CSV、manifest、启动回执和latest_path指针随该目录一起删除。此前“baseline-2正在运行/冻结保护”条款为历史状态。
- 本次是用户主动停止，不记作自然训练故障或完整220轮完成；没有可用的本轮最终复现差值报告。不得继续读取已删除指针、自动重启、重新训练或自动启用ROI。
- 首轮baseline、innovation1/2、M0/M1/S1等消融结果、TTA和原数据保留。模型、训练源码和YAML未改；最小停训删除凭证在outputs/audit/baseline_2_stop_delete/20261026T125002Z/。


## 2026-10-26 新baseline-2已授权从头启用ROI训练

- 用户在停止并删除旧非ROI baseline-2后，明确要求使用ROI加速重新训练baseline-2；该授权取代上一阶段“不自动重训/启用ROI”的限制。当前新run为outputs/baseline-2/20261026T130228Z，指针outputs/baseline-2/latest_path.txt；文档.codex/baseline_2_roi_training_run.md。
- 使用独立configs/synapse/baseline_2_roi.yaml和scripts/run_synapse_roi.py fit，seed0纯EoSeg，ASEE/CCPS关闭，原增强/模型/373组优化器/220轮/1-view最后50轮Dice选择保持。原作者YAML和训练源码不改，不加载旧分割权重续训。
- 启动已验证ROI绑定；388个初始权重及Python/NumPy/PyTorch RNG与原baseline相同，实际解析科学变量和所有优化器组一致。ROI仅加速CPU指标计算，记录真实mm与voxel；与首轮比较HD95需用同单位。
- 此run正在训练；按当前显存并发规则执行其他已授权任务，但不得中止本run腾资源或修改其冻结依赖/配置。用户未授权自动启动后续消融或TTA；不因为训练结束就擅自开始其他实验。


## 2026-10-26 S2/S3四次定时执行授权

- 用户明确要求在新ROI baseline-2完成后依序执行S2、S3，均启用ROI；这取代上一节对这两组“未授权后续自动启动”的限制，不扩展到其他实验。共同基准Denver2026-10-26 07:10:19，四次单次检查为12:10:19/12:40:19/17:10:19/17:40:19；任务ID为s2-5、s2-5-5、s3-10、s3-10-5。
- 执行前读.codex/s2_s3_scheduled_runbook.md；使用隔离scripts/launch_synapse_s2_s3_scheduled.py显式指定S2或S3。S2为use_text=false/use_probability=true；S3为enhance_patch=true/enhance_query=false，其余ASEE开关按原各自YAML。
- 原增强/seed0/220轮/科学训练设置保持；最后50轮1-view选同best，随后固定8-view完整12例，真实mm+voxel和legacy口径保持。只有完整完成才登记并启动后继。前序仍运行或显存不足时等待，不停止它、不重复训练、不跳过。其他用户任务允许按可用显存并发，不能中止他人任务。
- 本次只设置四次单次任务；当前baseline-2继续，未立即启动S2/S3。失败保留证据不自动重训；不启动S4或其他变体；四次结束不新增长期监控，仍在运行的训练继续。

## 2026-10-26 S2用户主动早停

- 用户明确要求停止当前S2；已对核验身份的训练进程SIGINT温和停止，训练/监督器及所属子进程已退出，其他用户任务保留。输出与真实退出码1保留，状态stopped_by_user，不记作完整实验或自然故障。
- S2运行outputs/ablations/journal_20261025/S2/20261026T231139Z；最后保存训练211轮/29118step，完整测试210轮，best195平均Dice85.214482%，权重哈希见early_stop_report.json。未自动启动S3、重训或最终8-view；四条一次检查均已结束。

## 2026-10-27 同seed复现差异已完成GPU最小定位

- 同历史best177旧/ROI入口12×8 Dice逐值一致，均84.718466%；ROI不改变该权重的Dice。真实双入口初始化388状态及CPU/CUDA RNG一致。
- 受控同输入/最终采样下，CUDA损失/反向可分叉；旧入口自身重复也如此。原生前2Batch一致，第1步LR0但AdamW moments不同，第2步非主干LR4e-7后23个参数张量不同。grid_sample反向及原3D CE归约非确定性已实际复现，不再仅为警告推测。不能唯一量化历史1.1953pp各来源。
- 生产算法与参数未改；确定性CE/point_sample仅为独立后续方案，不能直接替换原baseline或声称开一个deterministic开关即可。证据与适用边界见.codex/baseline_repeat_four_step_audit.md及baseline_repeat_solution_plan.md，审计根outputs/audit/baseline_repeat_diagnosis/20261027T023508Z/。

## 2026-10-27 默认关闭额外体素HD95（用户最新授权）

- 当前physical/ROI/多尺度TTA、CCPS及教师适配评估默认compute_hd95_voxel=False，仅计算真实HD95_mm；需要恢复时显式--model.init_args.compute_hd95_voxel=true。代码保留，没有删除。此规则取代旧文档对未来物理run强制同时计算mm+voxel的要求，历史结果保持原样。
- 关闭时CSV体素列空白、JSON null、报告“未计算”，hd95_voxel_enabled=False；不造0/NaN。Dice、毫米距离、legacy空mask和所有训练/预测变量保持。启动器支持新模式且历史缺flag仍按旧双单位验收。
- 205项CPU回归、best173完整12例1-view（192个Dice/mm值）及best177单病例8-view（16值）严格等价，实际体素调用均0；ROI/TTA readiness已依新证据刷新，旧快照归档。见.codex/hd95_voxel_disabled.md与outputs/audit/hd95_voxel_disabled/20261027T073138Z/validation.json。
- 原作者旧main体素专用入口及历史双单位审计仅保留作对照，不代表当前physical/ROI默认仍计算体素；不因本次关闭启动训练或恢复S2/S3。


## 2026-10-27 独立CUDA确定性修复（用户授权实施与测试）

- 新协议ce2d_stableseg_v1使用独立training/deterministic_*、scripts/run_synapse_deterministic.py与configs/synapse/deterministic三配置；原main/ROI/baseline损失/模型/YAML及历史结果不替换。新协议改变CE归约布局、采样反向累加顺序与strict设置，数学目标保持，不宣称与旧CUDA数值逐位相同。
- 30项CPU回归、baseline/ASEE/CCPS实际入口CPU初始化/RNG/全部优化器组一致；GPU G1/G2各8条件×30次、真实criterion G3两路径×30次严格重复及预先数学门槛通过。状态详见.codex/cuda_deterministic_implementation.md和cuda_solution_plan.md第12节。
- G4/G5尚未执行：完整模型原batch16/FP16/default compile短轨迹需22000MiB安全余量，现场仅18192MiB；不降低变量、不停止他人任务。G6完整220轮重复尚未授权。不得把G1–G3等同完整训练可重复性。
- 独立outputs/determinism/ce2d_stableseg_v1/readiness.json当前passed=false，正式fit被阻止；后续先逐项完成完整模型/冷暖缓存/低侵入验收，再讨论长训授权。compute_hd95_voxel=False和原legacy空mask继续保持；不自动恢复S2/S3或新消融。


## 2026-10-27 G4/G5已接续（用户最新授权）

- 用户明确要求继续未完成的测试；GPU释放后在outputs/determinism/ce2d_stableseg_v1/full_model_20261027T121833Z/串行执行baseline/ASEE/CCPS的固定Batch、20原生Batch冷暖缓存和低观察器短测，保持原batch16/workers8/FP16/default compile/220轮调度预算；仅由Callback结束短测。
- baseline固定双进程和原生20Batch冷启动已通过；其余以matrix_status、退出码及完整comparison为准。独立readiness在全部验收前仍false。源代码冻结，不修改正在短测的模型/配置/损失依赖，不停止其他用户任务。
- 详见.codex/cuda_g4_g5_validation.md。G6完整220轮重复仍未授权，不自动恢复S2/S3/消融或安排长期监控。


## 2026-10-27 G4/G5部分通过与G6条件授权

- 本轮GPU有界诊断全部结束。ce2d_stableseg_v1的baseline/ASEE通过G4及20 Batch冷/暖缓存、低观察器G5；CCPS输入/前向/loss相同但有限梯度分叉，G4失败、G5未执行。readiness仍false，validated_profiles为空，不改正式fit全门槛保护。
- 末端BN局部编译边界候选固定双进程短测通过，但尚无G5/低观察器证据，未接入生产。详见.codex/cuda_g4_g5_validation.md、cuda_solution_plan.md第14节。
- 40项CPU回归通过；诊断比较器支持原生AMP首批溢出跳步，但逐值相等标准不变。原101份文件及4份独立数值运行文件hash保持，GPU producer AST不变，原baseline模型/参数/增强/损失/指标及历史结果未改。
- 用户已条件授权：G4/G5通过后两次纯baseline完整220轮可作为G6，配合3/3.5/8/8.5小时启动/检查；未通过则完善Plan。当前执行未通过分支，四任务未创建、G6未启动。此条取代此前“G6未获任何授权”的历史表述，不等于无条件长训。


## 2026-10-27 CCPS独立execution_v2通过与G6安排

- 用户要求继续定位后，组合候选（feature_upscale.10局部BN eager边界+AOT backward_pass_autocast=off）通过三独立固定进程、原生20 Batch完整梯度冷/暖对、低观察器冷/暖对及交叉，局部GPU数学参照/30次重复通过。单独任一控制均出现分叉；不唯一归因某个CUDA内核或BN公式。outer compile/FP16及模型/损失目标/训练变量保持。
- 新增training/deterministic_execution_v2.py、scripts/run_synapse_deterministic_v2.py、configs/synapse/deterministic_v2三配置。仅CCPS启用两控制，baseline/ASEE数值路径保持v1。真实入口G0–G5已passed，v2 readiness路径outputs/determinism/ce2d_stableseg_execution_v2/readiness.json；46项CPU回归通过。旧v1入口/数值实现及原101份文件不改，旧v1失败状态保留。新路径须显式选择，不热切换旧run。
- G6启动器scripts/launch_synapse_g6.py经8项CPU回归与默认离线W&B、FP16/compile/batch16/workers8两BatchGPU冒烟验证；初始权重与已验证baseline一致。Lightning回调接收解包模型，用_compiler_ctx识别compile；不可因没有_orig_mod误判未编译。
- 用户条件授权的四条单次任务已落实，基准Denver2026-10-27 22:26，目标10月28日01:26首次G6、01:56兜底、06:26第二次G6、06:56兜底。ID：g6-baseline-3h、g6-3-5h、g6-baseline-8h、g6-8-5h。按.codex/synapse_g6_runbook.md执行，输出outputs/determinism/g6_baseline_execution_v2/，只做纯baseline seed0完整220轮重复；当前尚未启动正式G6。
- 原增强/384/patch16/batch16/workers8/220轮/AdamW/LR/LLRD/损失系数/FP16/默认compile/划分/legacy Dice和HD95_mm-only保持；G6 baseline使用已验证的确定性CE与采样协议，不冒称旧未修复CUDA路径逐位相同。第一轮完整220轮及50轮12例测试后才启动第二轮；已有失败不重训，显存不足不中止他人任务，不恢复S2/S3/额外TTA。完整训练可重复性尚待G6，不由短测通过保证。


## 2026-10-28 G6首次baseline-repair已立即启动

- 用户最新明确要求开始G6，输出名称baseline-repair，并再次确认不加入创新模块。本次只运行纯EoSeg baseline（ASEE/CCPS均关闭），使用已验收scripts/run_synapse_deterministic_v2.py与configs/synapse/deterministic_v2/baseline.yaml，seed0、原增强、batch16/workers8、384/patch16、220轮/30360步及其余原科学变量保持。
- 当前run：outputs/baseline-repair/G6_1/20261028T072539Z；回执outputs/baseline-repair/repeat_1.json；latest指针outputs/baseline-repair/latest_path.txt。启动器输出根现为outputs/baseline-repair，仅名称/路径改动已通过8项CPU回归与逐字源代码恢复对照；新preparation保留原GPU冒烟依据和变更证明。原outputs/determinism/g6_baseline_execution_v2仅保留历史准备/计划元数据，不当作当前run根。
- 当前实际进程/进度以status/launch/PID-starttime-cmdline-cwd/GPU/last为准。不要因为旧01:26定时安排再启动repeat1；第二轮依然等待第一轮完整结束，不擅自停止、重训或切换参数。详见.codex/synapse_g6_runbook.md顶部当前状态。

启动后实际推进核验：截至2026-10-28T07:36:57.659160+00:00，滚动last.ckpt已完整保存7轮/966step，训练PID=3107921仍匹配starttime/cmdline/cwd；初始388项权重与已验收baseline一致，query/class_head/layer_gate均已发生有限数值更新。ASEE=False、CCPS=False。证据为当前run/startup_training_verification.json；此轮数为本次核验快照，不作为后续实时状态。
