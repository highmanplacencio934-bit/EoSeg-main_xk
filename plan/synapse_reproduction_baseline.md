# Synapse 第三套主配置复现基线

## 正式 baseline 已启动（2026-10-21）

用户已明确授权220轮正式训练，UTC 13:26:35（America/Denver 07:26:35）启动在服务器RTX 5090 D。实际解析配置为220 epochs、batch16、workers8、FP16、默认compile、LR2e-4，estimated_steps=30360；训练进程PID3477678，输出为outputs/baseline/20261021T132635Z。仅使用离线日志和独立输出路径，模型/训练源码、YAML、数据及原有权重未改。

已设置本对话单次跟进：2026-10-21 15:24 America/Denver（UTC21:24）。先检查训练/test结果并汇总，再阅读两份创新Plan及两个参考实现准备创新点1代码；不启动创新训练，准备完成后停止工作等待指示。8小时未结束时不自动终止baseline。详见baseline_followup_task.md。

以下“未正式训练/未完成220轮”等内容为启动前检查阶段的历史状态；本次是否完成以status.json、退出码与CSV为准，不把启动成功当作最终成绩。

## 2026-10-21 服务器审计状态

- 用户已授权代理直接操作终端/SSH，环境与正确路径已实测，取消用户必须代执行命令的限制。
- 新证据目录：outputs/audit/20261021T123535Z。2211训练切片和12测试体积已全量读取并保存逐文件SHA256；有限值、shape、整数标签0..8均通过，两份划分哈希未变。
- 服务器28项测试全部通过、原样CLI配置解析通过；原workers8增强batch的bool mask/标签检查通过。
- 默认compile、FP16、batch16、workers8及离线W&B的2-step短测已完整评估12例并保存best/last，退出0。loss/373组梯度有限，AdamW状态已初始化，GPU peak allocated约16.94GiB、reserved约17.78GiB；Dynamo捕获3图，损失/日志等仍按原代码断图回退，未改编译路径。
- 两份真实checkpoint独立CLI加载的388键/shape/值均一致；旧eager冒烟12病例指标精确重现。compile产物对独立eager仅case0003/class7有微差（Dice约1.26e-7、HD95约6.78e-4体素），单病例相同compile路径精确重现在线CSV；跨模式严格比较断言未通过，未改路径或放宽阈值。详见运行审计第15、19条和completion.json。
- 原审计后按用户要求新增默认关闭的--compile_test：独立测试可使用相同compile前向，真实12例逐类Dice/HD95严格差为0，31项回归通过；默认test仍eager，原fit、模型/损失/优化器/YAML不变。主入口和对应测试的预期哈希变化单列于baseline_test_interface_snapshot.json，历史快照保留。未进行正式220轮训练，也未接入用户指定的本地预训练文件。

初次审计日期：2026-10-19；数据接口更新：2026-10-20（用户时区）。初次记录仅为代码/本地轻量检查；当前已补充上节服务器真实短测和全量数据审计，未启动正式220轮训练。不以论文、其他仓库、其他数据集设置补齐未知项。
项目实际读取路径：\\sshfs.r\sk@anhuiuniversity.icu!21103\Disk\HDD\Sk\EoSeg-main_Synapse_acdc\EoSeg-main_synapse（当前工作区 SSHFS 路径）。PowerShell 默认目录异常回落到 C 盘，本审计全部使用项目绝对路径，没有采用本机同名代码。

## Baseline 忠实性与创新隔离约定（用户明确要求）

此项目的目标是在可靠复现 baseline 的基础上做创新，不能为了可运行、节省显存、提高分数或方便测试而静默更改 baseline 的模型或训练参数。

- 当前第三套 baseline 固定：EoSegMultiQSegFusion、16 query、最后三层注入及原融合方式；输入384×384、patch_size=16、9类、batch16、220 epochs、基础LR=2e-4、LLRD=0.8、weight_decay=0.05、warmup=[500,1000]、poly_power=0.9、16-mixed、seed0、norm裁剪0.01，以及现有损失/增强/固定18+12划分。
- 最新核对确认以上参数未改变；models/eoseg.py、datasets/synapse_transforms.py、training/mask_classification_loss.py、training/two_stage_warmup_poly_schedule.py 的 SHA256 与最初审计快照一致。
- 已授权的基础设施修复仅涉及无关分支的惰性导入、日志/归档、checkpoint子类元数据和配置/环境记录。设备设置已恢复作者入口默认值；正式运行仍使用既定服务器单卡，不更改训练算法。
- 未来创新使用独立模型类/模块、独立配置、独立run名称和输出目录；不直接覆盖 baseline 配置或原模型实现。对照实验保持数据划分、预算、损失、增强、优化器和评估选择规则一致；确需改变的变量单独列出。
- 主干和预训练权重仍按用户计划后续改进；若它们属于创新，必须与 baseline 分开；若为 baseline 的加载适配，必须核验权重来源、转换规则及实际结构，不能把结构替换混作加载修复。patch_size=16不因接口工作而改变。
- 显存不足、依赖冲突或版本问题先报告并诊断，不默认减batch/改尺寸、改学习率、切换精度、增加梯度累积、改损失或改指标口径。CPU小模型仅用于回归测试，不作为正式baseline的参数替代。

## 当前选定实验：第三套配置（2026-10-20 用户确认）

本项目后续默认复现第三套 `configs/synapse/vit_query_mul_scale_fusion.yaml`。下文的模型、优化器、损失、训练与保存设置均以这份配置及其调用代码为准，不混用另外四套对照配置的值。五套配置的用途与差异单独见 [Synapse 五套配置说明](synapse_config_guide.md)。

| 项目 | 选定基线 |
|---|---|
| 任务和数据 | Synapse；固定 18 个训练病例/2211 张切片、12 个测试病例；无独立验证集 |
| 网络 / 训练模块 | EoSegMultiQSegFusion / MaskClassificationSemantic |
| 主干名称 | vit_large_patch14_reg4_dinov2（包装类默认名称） |
| 目标本地预训练文件 | dinov2_vitl14_reg4_pretrain.pth；尚未完成本地权重接入及核验 |
| 输入 / patch / 类别 / batch | 384×384 / **16（已确认）** / 9（含背景）/ 16 |
| query / 注入层 | 16 个；在最后三个 transformer block 之前注入 |
| 融合 | 用最后一层 query 特征产生逐 query 的三层权重，融合最后三层 mask 和 class logits |
| epochs / 基础 LR | **220 / 2e-4** |
| 优化器 | AdamW；weight_decay=0.05 |
| LLRD | 0.8；llrd_l2_enabled=True；lr_mult=1.0 |
| 调度 | 非主干预热 500 step；主干先等待 500 step，再预热 1000 step；poly_power=0.9 |
| 精度 / 设备 | 16-mixed / devices=1；训练在服务器 RTX 5090D（32GB）上执行 |
| 梯度裁剪 / 默认 seed | norm=0.01 / 0 |
| 评估 / 保存 | 关闭独立验证；第 171–220 轮末评估完整 12 个测试病例；按测试 mean Dice 保存 best_test.ckpt，另滚动保存 last.ckpt |

保留第三套的 384 输入、patch_size=16、220 epochs、2e-4 LR 等设置。用户已明确确认 patch_size 就是 16，它不是待修复错误，不改成 14，也不因此更改输入尺寸或补边。主干网络和预训练权重接口由用户计划在后续改进，本轮只记录现状。详见 [运行前详细审计](synapse_baseline_run_audit.md)。

## 训练执行位置与硬件约定（2026-10-20 用户确认）

- 本项目的训练均在服务器上执行，指定 GPU 为 **NVIDIA RTX 5090D，32GB 显存**。该型号和环境已于2026-10-21直接通过SSH实测，最新证据见运行审计。
- 本地 Windows/Codex 侧用于代码编辑、SSHFS 文件访问及可执行的轻量检查；不得将本地运行环境或本地 GPU 当作正式训练环境。
- SSHFS/UNC 映射只提供服务器文件访问，不表示当前 PowerShell/Python 进程运行在服务器上。正式训练及依赖相关验证应通过服务器终端或远程执行完成。
- 启动训练前，在服务器实际核验并记录 hostname、项目与数据绝对路径、Python/Conda 环境、PyTorch/CUDA 及驱动版本、nvidia-smi 中的 GPU 型号和可用显存；不假定已安装依赖或已经核验过服务器环境。
- 本文历史记录中的“本机缺少 PyTorch/Lightning”“集成测试跳过”仅描述当时本地检查的限制，不代表服务器缺少这些依赖。相关集成测试应在服务器训练环境补跑。
- 此约定不改变各配置原有的 epochs、lr、llrd、batch_size、precision、devices 等参数。32GB 为用户指定的显存容量，训练时可用显存以实际查询为准。

## 基准配置与取值规则

README 指定的主配置为 configs/synapse/vit_query_mul_scale_fusion.yaml。其他 Synapse YAML 是消融配置，不混入本基线。取值依据为主 YAML 显式设置、对应构造函数默认值及 main.py 的 trainer_defaults；命令行覆盖将改变最终值。以下是复现当前代码应保留并核验的基线，不代表所有设置均合理或已成功运行。

## Backbone 与分割结构

| 项目 | 当前代码值/行为 |
|---|---|
| 训练模块 | training.mask_classification_semantic.MaskClassificationSemantic |
| 网络 | models.eoseg.EoSegMultiQSegFusion |
| 编码器 | models.vit.ViT |
| backbone_name | vit_large_patch14_reg4_dinov2（ViT 构造默认） |
| patch_size | **16（用户明确确认的 baseline 不变量）**；384 输入对应 24×24 网格，不因模型名中的 patch14 改成 14 |
| 输入尺寸 | 384×384，由 main.py 从 data 配置链接至模型及编码器 |
| timm 构造 | num_classes=0；img_size=配置尺寸；patch_size=16；pretrained=(ckpt_path is None) |
| ckpt_path | 主 YAML 未指定，构造默认 None，因此走 timm 自动预训练分支 |
| 用户目标权重 | dinov2_vitl14_reg4_pretrain.pth，仅用户指定，未配置或核验加载 |
| 通道 | 训练 make_rgb=True，将灰度复制为 3 通道；测试也复制三通道 |
| 标准化 | 网络内部 (x-mean)/std；mean=[0.485,0.456,0.406]；std=[0.229,0.224,0.225] |
| 主干冻结 | 本路径没有显式冻结主干，优化器遍历全部参数 |
| query | num_q=16，torch.randn 初始化 |
| 注入位置 | num_blocks=3，在最后 3 个 transformer block 之前加入 query |
| masked_attn_enabled | False，网络内固定 |
| attn_mask_annealing_enabled | False，主 YAML 显式指定 |
| 主干执行 | 手动 patch_embed → 存在时调用 _pos_embed → 逐 block 执行；非完整 backbone.forward |
| 多层输出 | 收集最后三层 tokens，每层经共享 backbone.norm 与预测头 |
| mask head | Linear(C,C) → GELU → Linear(C,C) |
| class head | Linear(C,10)：9 个语义类别与 1 个 no-object |
| 上采样头 | 两次转置卷积 kernel=2/stride=2：C→C/2→C/4；每级接 BN、GELU、depthwise 3×3 Conv、BN、GELU；最后 1×1 Conv 回到 C |
| patch tokens | 取序列末尾 grid_h×grid_w 个 token；网格由 patch_embed.grid_size 决定 |
| mask 生成 | mask embedding 与空间特征 einsum，双线性插值到输入尺寸，align_corners=False |
| 融合门 | 最后层 query → Linear(C,C/2) → GELU → Linear(C/2,3)，加零初始化 bias，再 softmax |
| 融合输出 | 逐 query 权重融合三层 mask logits 与 class logits；只返回一组融合结果 |

主干结构依赖 timm 版本。已在本机 timm 1.0.28 用 meta device、不加载权重核验：维度 1024、24 层、16 heads、MLP 隐层 4096、4 个 register tokens，patch/grid=16/24×24；结构检查不代表服务器结构或预训练加载已核验。完整数值、层学习率和资源估算见运行前审计。

## 优化器、LLRD 与调度

来源：主 YAML、training/mask_classification_semantic.py、training/lightning_module.py、training/two_stage_warmup_poly_schedule.py。

| 参数 | 当前值 |
|---|---|
| 基础 lr | **2e-4**（覆盖模块默认 1e-4） |
| llrd | **0.8** |
| llrd_l2_enabled | True |
| lr_mult | 1.0 |
| optimizer | torch.optim.AdamW |
| weight_decay | 0.05 |
| 参数组 | 每个参数单独分组，先主干组，再其他参数组 |
| bias/Norm 不衰减例外 | 没有，统一应用 weight_decay=0.05 |
| AdamW betas/eps 等 | 代码未显式指定，依赖安装的 PyTorch 默认；运行时记录 |
| scheduler | TwoStageWarmupPolySchedule；interval=step，frequency=1 |
| warmup_steps | [500,1000]，顺序为非主干预热、主干预热 |
| poly_power | 0.9 |

设实际主干有 L 个 block，编号 i 从 0 开始，block 内参数基准 lr=2e-4×0.8^(L-1-i)。非主干参数 lr=2e-4；名称含 backbone.norm 的主干参数恢复为 2e-4。非 block 主干参数还依赖逆序参数遍历时保留的 block_i 状态，须导出实际参数组核验，不能一概归为同一层。
当前 llrd_l2_enabled=True、lr_mult=1.0，不触发最后三层强制恢复基础 lr 的分支。应用 lr_mult 的 elif 条件已被前一个 if 覆盖，不能按变量名假定倍率生效。

非主干前 500 step 从 0 线性预热，之后多项式衰减；主干前 500 step 的 lr 为 0，再用 1000 step 线性预热到各组 LLRD 后的基准值，之后多项式衰减。lr=0 不等于显式冻结。总步数取 trainer.estimated_stepping_batches；当前固定2211切片、batch16、单卡且accumulate=1时预计138×220=30,360，最终以实际运行记录为准。

## 训练运行与保存

- max_epochs=220；precision=16-mixed；devices=1（main.py 默认）。
- gradient_clip_val=0.01；gradient_clip_algorithm=norm。
- seed_everything_default=0（入口 LightningCLI）；它不是数据划分 seed。
- float32 matmul precision=medium；torch._dynamo.config.capture_scalar_outputs=True、suppress_errors=True。
- fit 默认 torch.compile(model)，--compile_disabled 可关闭。
- 设备配置恢复作者入口默认：devices=1、accelerator=auto；正式运行仍按用户约定在服务器 RTX 5090D 上执行，启动时核验实际 GPU，不修改模型/训练参数。
- limit_val_batches=0、check_val_every_n_epoch=1000000：主实验关闭验证。
- 当前第三套使用 SynapseTestSelection：禁用标准自动 ModelCheckpoint；总训练 220 epochs，第 171–220 轮每轮末完整评估测试集，以 mean Dice 严格增大更新 best_test.ckpt（同分保留先前）；每个完成轮次滚动覆盖 last.ckpt。每次 fit 创建独立目录，只长期保留这两份模型权重。
- 原始本地主配置是 monitor=epoch、mode=max、save_top_k=50、save_last=True；该历史设置已被上述策略替换，不作为当前运行设置。
- 测试集不参与梯度更新，但用于 checkpoint 选择；这是本项目明确采用的策略，不归因于作者未公开说明的 Synapse 评估流程。
- 两份 checkpoint 通过 Trainer.save_checkpoint(weights_only=True) 保存，包含整个分割模型及主干，不含优化器/调度器状态，不是完整训练续训快照。后续独立测试使用明确文件路径。
- WandbLogger：project=eomt，name=synapse_384_learnable_fusion_query_2e_test，resume=allow。
- 入口默认 ModelSummary(max_depth=3)、LearningRateMonitor(logging_interval=epoch)。
- save_config_callback=None；自定义 CLI 现在向 trainer 传递解析配置，由 SynapseTestSelection 在 run_manifest.json 中自动归档配置、实际命令、环境、代码/划分哈希等。
- sitecustomize.py 是空覆盖文件，用于避免集群全局 Tensor.to 补丁；需核验实际导入位置。

## Synapse 数据划分与数据契约

来源：datasets/synapse_semantic.py、synapse_npz_dataset.py、synapse_transforms.py、lightning_data_module.py。

| 项目 | 当前代码设置 |
|---|---|
| train_base_dir | project_TransUNet/data/Synapse/train_npz |
| test_base_dir | project_TransUNet/data/Synapse/test_vol_h5 |
| list_dir | project_TransUNet/TransUNet/lists/lists_Synapse |
| 训练列表 | train.txt，每行 name 对应 name.npz，读取二维切片 |
| 测试列表 | test_vol.txt，每行 case 对应 case.npy.h5，读取完整三维病例 |
| 文件字段 | image、label |
| 类型 | 图像 float32，标签 int64 |
| num_classes | 9，背景为 0；ignore_idx=255；drop_background=False |
| 训练目标 | 切片每种有效类别生成一个二值 mask，不按连通域拆分实例 |
| 训练 batch | 16，shuffle=True，drop_last=True |
| 测试 batch | 固定 1 个病例，shuffle=False |
| num_workers | 8 |
| pin_memory | True |
| persistent_workers | 训练/验证由基类默认为 True（workers>0）；测试未显式指定 |

代码没有随机划分或交叉验证，固定读取 project_TransUNet 中的两个原有 txt。已核验：训练 18 个病例、2211 张切片，测试 12 个病例；列表无重复，所引用文件全部存在，训练/测试病例 ID 无交集。数据接口在每次 setup 时校验两份列表，按运行阶段检查相应文件，保存 split_summary 并记录日志；不会改写原始列表。
当前固定划分没有独立验证集。val_dataloader 已改为返回空列表，不再复用训练数据；主配置仍关闭验证。

## 预处理与在线增强

- 现有读取器只支持已转换的 .npz/.npy.h5，没有原始 NIfTI/DICOM 在线读取链路。
- 未实现 HU 窗截断、强度归一化、spacing 重采样、原始标签映射；不能仅凭扩展名认定这些离线步骤已经完成。
- 训练默认 color_jitter_enabled=True；scale_range=(0.5,2.0)。
- 增强顺序：过滤 crowd → 颜色增强 → 水平翻转 → ScaleJitter → 右/下补零 → 384×384 随机裁剪 → 移除空 mask；全部 mask 为空时递归重试。
- brightness 因子幅度 32/255，contrast 幅度 0.5，saturation 幅度 0.5，hue 幅度 18/360；各子操作以 0.5 概率执行，contrast 与 saturation/hue 顺序随机。
- 水平翻转及其他未显式参数采用 torchvision 实现默认，依赖版本；不是代码中明确写定的数值。
- 三通道图像在模型内做 ImageNet 标准化，不应在 Dataset 重复执行。
- 首个 global_step 打印图像 dtype/min/max 并断言 max<=1.5；未检查最小值，不能据此认定严格在 [0,1]。
- forward 中 x/255 仅为注释，不执行。

## 损失

来源：training/mask_classification_semantic.py、mask_classification_loss.py。

| 参数 | 默认值（主 YAML 未覆盖） |
|---|---|
| num_points | 12544 |
| oversample_ratio | 3.0 |
| importance_sample_ratio | 0.75 |
| mask_coefficient | 5.0 |
| dice_coefficient | 5.0 |
| class_coefficient | 2.0 |
| no_object_coefficient | 0.1 |
| mask_thresh / overlap_thresh | 0.8 / 0.8 |

损失继承 transformers Mask2FormerLoss，匹配采用 Mask2FormerHungarianMatcher，匹配代价使用相同 mask/dice/class 系数；mask 损失按匹配 mask 数归一化，分布式归约后除 world_size，分母最小为 1。当前主网络仅返回融合输出，因此没有最后三层分别输出的三份辅助损失。mask_thresh/overlap_thresh 不作为下述 Synapse 逐像素 argmax 测试路径的判定阈值。

## 测试和评估口径

来源：training/lightning_module.py 的 test_step、to_per_pixel_logits_semantic、calculate_metric_percase。

- 完整病例逐切片推理，没有在该 test_step 调用滑窗或随机测试增强。
- 图像用 scipy.ndimage.zoom(order=3) 缩放到模型输入尺寸，再复制三通道。
- mask sigmoid 与 class softmax（排除 no-object）按 query 合成逐像素类别分数；必要时双线性上采样分数，之后 argmax。
- 预测用 zoom(order=0) 恢复原切片尺寸，组装三维预测。
- 对完整病例计算类别 1..8 的 Dice 与 HD95，背景不参与；病例内部对 8 类算术平均，经 self.log 记录并由框架聚合。
- HD95 未向 medpy 传 voxelspacing，不能声明使用真实毫米间距。
- 当前空 mask 特殊逻辑：预测/真值均非空时正常算；预测非空且真值空时 Dice=1、HD95=0；其余返回 Dice=0、HD95=0。这是当前实现，后续修正必须记录为基线变更。

## 当前接入风险与复现前待核验项

1. patch_size=16 已由用户确认，保持不变。模型名称/权重文件名中的 14 不覆盖显式参数；本机 timm 1.0.28 存在将 patch kernel 重采样到目标 16 并调整位置编码的加载流程。主干和预训练接口后续改进时需保持这条 baseline 约定，不将 patch_size 当作错误修复。
2. ViT 普通 timm 分支没有本地权重加载操作；ckpt_path 非空仅关闭自动预训练。上层又把同一路径当完整分割模型 state_dict 加载并检查 missing/unexpected keys，因此不能把原始 DINOv2 权重直接当成已正确接入。
3. 自动预训练来源取决于安装的 timm、配置与缓存。正式运行应核验权重来源、SHA256、键名转换、位置编码与张量形状。
4. 已移除 main.py 中指向不存在构造参数的 network.init_args.img_size 与 data.stuff_classes/model.stuff_classes 链接；保留 data→model、data→encoder 的有效尺寸链接。当前本机缺少 PyTorch/Lightning，尚未执行完整 CLI 解析及模型实例化。
5. requirements.txt 未固定版本，入口使用 Lightning/jsonargparse 私有 API；必须保存实际 torch、timm、torchvision、transformers、lightning 等版本。
6. 已核验真实病例列表、数量、文件对应及病例交叉情况。待补：类别/器官映射、准确预处理来源、完整数组校验、spacing、最终解析配置、实际主干结构和参数组、运行环境及训练/测试结果。
7. 对照实验应保持已确认基线项；必要修复、数据替换和设计变更单独记录。本文只固定当前代码，不代表作者论文设置或成绩已经复现。

## 证据文件

README.md；configs/synapse/vit_query_mul_scale_fusion.yaml；main.py；models/vit.py；models/eoseg.py（EoSegMultiQSegFusion）；datasets/synapse_semantic.py；datasets/synapse_npz_dataset.py；datasets/synapse_transforms.py；datasets/lightning_data_module.py；training/mask_classification_semantic.py；training/lightning_module.py；training/two_stage_warmup_poly_schedule.py；training/mask_classification_loss.py；requirements.txt；sitecustomize.py。


## 2026-10-20 配置差异与接口变更记录

各配置保留原有 epoch，绝不跨实验统一设置：

| 配置 | 网络 | max_epochs | YAML lr |
|---|---|---|---|
| vit_deconv_384.yaml | EoSegDeconvSeg | 200 | 未显式指定 |
| vit_query_mask.yaml | EoSegQueryMaskSeg | 200 | 未显式指定 |
| vit_query_mul_scale_fusion.yaml | EoSegMultiQSegFusion | **220** | 2e-4 |
| vit_query_mul_scale_fusion_mul_layer.yaml | EoSegMultiQSegFusionMultiLayer | 200 | 2e-4 |
| vit_query_mul_scale_fusion_vit.yaml | EoSegMultiQSegFusion | 200 | 1e-4 |

差异来自当前代码中的不同模型/消融配置；代码没有说明作者选择各 epoch 的理由，不能推测。本文主基线仍指融合主配置，所以训练控制中的 220 保留。不是所有 Synapse 配置均为 220，也不是均为 200。

数据接口：
- 五份 Synapse 配置均指向已有 project_TransUNet 数据和病例列表。
- 相对路径统一相对项目根目录解析，不依赖启动目录；仍接受服务器绝对路径。
- train_split=train、test_split=test_vol 暴露为接口参数，默认值即原始列表。
- 新增 datasets/synapse_splits.py：校验重复、空列表、非法条目、训练切片/测试体积类型、缺失文件、病例交叉；每个目录只扫描一次，减少 SSHFS 元数据请求。
- 两份列表仍必须存在以检查病例交叉；fit 只要求训练切片存在，test 只要求测试体积存在，setup(None) 检查两侧文件。
- 验证加载器返回 []；测试 collate 改为模块级函数，避免 lambda 在 spawn worker 下无法序列化。
- 本次不修改 LR、LLRD、模型结构、预训练权重或原始数据/划分文件。

验证：7 项纯 Python 单元测试通过；真实数据校验为 18 训练病例/2211 切片/12 测试病例/无病例交集；五份 YAML 路径、原始 epoch 值与改动 Python 文件语法检查通过。本机未安装 torch、torchvision、lightning、jsonargparse，未执行真实 DataLoader、完整训练 CLI 或 GPU 训练，不将静态检查称为训练通过。


## 2026-10-20 Review P2 修复：按阶段检查数据文件

问题：原 setup 在运行阶段分支之前同时扫描训练和测试数据目录，独立测试会因训练数据未挂载而失败，训练也被强制依赖测试体积。

修复：validate_synapse_split 新增 keyword-only 的 check_train_files/check_test_files 参数，默认仍检查两侧；SynapseSemantic.setup 根据 stage 传入标志。fit 仅扫描训练目录，test 仅扫描测试目录，None 检查两侧，validate/predict 只校验列表（当前没有独立验证集或预测加载器）。任何阶段仍读取两份列表，检查条目、重复和病例交叉；不要求非活动阶段的数据目录存在。每次 setup 都重新校验，避免同一实例先 fit 后 test 时因缓存跳过测试文件检查。

验证：15 项测试中 12 项通过、3 项 DataModule 集成测试因本机未安装 PyTorch/torchvision/Lightning 而跳过。新增覆盖缺少非活动目录仍能通过、活动文件缺失仍报错、病例交叉仍报错；集成测试覆盖 test-only、fit-only 和 fit→test 重检，待有训练依赖的环境执行。真实数据列表在训练侧检查、测试侧检查和双侧检查下均通过，结果为 18 个训练病例/2211 张切片/12 个测试病例；Python 语法与各配置原有 epoch 检查通过。未启动训练，未修改配置、模型或数据。


## 2026-10-20 运行前审计与 patch 约定

用户明确确认 patch_size=16；主干和预训练权重另行后续改进。本轮不修改训练源代码或 YAML。新增 [详细运行审计](synapse_baseline_run_audit.md) 和 baseline_audit_snapshot.json，记录本机实际 probe、源文件/配置/划分 SHA256、已确认与待服务器核验项。

新定位的启动/评估风险：当前主入口无条件导入 RVSA→MMSeg→mmcv，本机缺 mmcv 导致原样 --print_config 失败；默认 W&B 代码归档会打开当前不存在的 .gitignore；隔离小骨干 probe 中，Lightning 2.6.5 的 test --ckpt_path 超参数重解析报 ignore_idx is not accepted（模型保存时忽略了 _class_path）。这些不涉及将 patch_size 改成 14，且不能被此前 22 项数据/回调测试通过替代。服务器尚未完成对应复现，未启动正式训练。


## 2026-10-20 审计修复完成记录

按顺序完成：惰性导入 RVSA；安全代码归档/.gitignore；禁用 logger 的入口及 LRMonitor 兼容；新 checkpoint 保留 _class_path；第三套明确单 GPU；自动 run_manifest。28 项测试和真实原样 --print_config 通过。详见运行审计顶部“修复执行状态”；该状态覆盖本文件之前的历史问题列表。模型结构、patch16、220 epochs、lr/llrd、数据和测试选模口径保持不变，其他四套 YAML 没改。

服务器 GPU/FP16/compile/显存与完整 checkpoint 推理重现仍待实际认证环境；主干和预训练接口按用户计划延期；两份纯权重不支持精确续训。旧 checkpoint 元数据迁移无真实样本，未擅自覆盖。


## 2026-10-20 最终参数核对与回退

模型结构、数据读取/增强、损失和 LR 调度源文件与审计前 SHA256 一致；220 epochs、patch16、384输入、batch16、query16、lr=2e-4、llrd=0.8等参数不变。额外写入第三套 YAML 的 accelerator=gpu、devices=1 两行已撤回，恢复作者入口默认设备配置。保留导入、日志、checkpoint元数据和归档兼容性修复。

不再维护独立评估方式长文，仅保留运行所需的文件/指标说明。在线评估和best/last保存代码按用户已指定的行为保留；未据作者配置的模糊性推定任何未明确的作者做法。今后创新必须另建配置，不覆盖本 baseline。

## 最终baseline与用户确认的Dice口径

2026-10-21：正式220轮已完成、退出码0，最佳177轮。用户确认继续使用原空掩码逻辑，默认平均Dice84.718466%；同一预测仅修正错误满分分支的对照复核83.676799%，独立记录不启用。逐器官和来源见 `.codex/synapse_baseline_results.md`。原评分和训练参数未改；物理HD95暂缓。
