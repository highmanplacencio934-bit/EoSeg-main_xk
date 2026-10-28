# Synapse baseline 运行前详细审计

## 正式 baseline 已启动（2026-10-21）

用户已明确授权220轮正式训练，UTC 13:26:35（America/Denver 07:26:35）启动在服务器RTX 5090 D。实际解析配置为220 epochs、batch16、workers8、FP16、默认compile、LR2e-4，estimated_steps=30360；训练进程PID3477678，输出为outputs/baseline/20261021T132635Z。仅使用离线日志和独立输出路径，模型/训练源码、YAML、数据及原有权重未改。

已设置本对话单次跟进：2026-10-21 15:24 America/Denver（UTC21:24）。先检查训练/test结果并汇总，再阅读两份创新Plan及两个参考实现准备创新点1代码；不启动创新训练，准备完成后停止工作等待指示。8小时未结束时不自动终止baseline。详见baseline_followup_task.md。

以下“未正式训练/未完成220轮”等内容为启动前检查阶段的历史状态；本次是否完成以status.json、退出码与CSV为准，不把启动成功当作最终成绩。

更新：2026-10-21。当前文档以服务器实测及代码为准，已将旧故障与当前状态分开；不再依靠顶部更正覆盖正文中的过时结论。项目目标仍是忠于 baseline，再单独开展创新。

原审计证据目录：`outputs/audit/20261021T123535Z/`；可选测试编译接口验证目录：`outputs/audit/20261021T130147Z_compile_test/`。命令、退出码、环境、完整数据扫描和短测日志均保存在此；用户此前完成的冒烟仍保留在 `outputs/smoke/synapse_test_selection/20261021T024815Z_d63b7ad3/`。

原审计的baseline_integrity.json记录当时23项文件无变化。随后按用户要求优化独立测试：main.py新增默认关闭的--compile_test，tests/test_baseline_startup.py新增回归，其余21项快照文件保持原哈希；移除新增接口可精确还原旧main.py哈希，原fit逻辑、默认eager测试、模型、损失、优化器、YAML、数据与权重均未改变。当前代码哈希见baseline_test_interface_snapshot.json；历史两份快照不覆盖。

## 1. 当前 baseline 与不变量

| 项目 | 当前值 |
|---|---|
| 主配置 | configs/synapse/vit_query_mul_scale_fusion.yaml（第三套） |
| 模型 / 编码器 | EoSegMultiQSegFusion / models.vit.ViT |
| 主干名称 / patch | vit_large_patch14_reg4_dinov2 / **16** |
| 输入 / patch 网格 | 384×384 / 24×24 |
| 类别 / query | 9（含背景）/ 16 |
| Epoch / batch | **220 / 16** |
| 基础 LR / LLRD | **2e-4 / 0.8** |
| 优化器 / weight decay | AdamW / 0.05 |
| 预热 / poly power | [500,1000] / 0.9 |
| 精度 / 裁剪 / seed | 16-mixed / norm 0.01 / 0 |
| 默认设备 / compile | devices=1、accelerator=auto；fit 默认启用 torch.compile |

其余四份配置各自保留 200 epochs 及原有模型、LR、损失/优化器差异，不统一成第三套。完整对照见 `synapse_config_guide.md`。

## 2. 已修复启动问题与回归结果

| 项目 | 当前状态 |
|---|---|
| 无关 RVSA→mmcv 顶层导入 | 已惰性导入；DINO 不要求 mmcv，本轮服务器原样 print_config 通过 |
| .gitignore 缺失及代码归档 | 已有排除规则、固定项目根目录及异常回退；服务器回归通过 |
| logger=False / LearningRateMonitor | 已兼容；无 logger 时移除不兼容回调，正常 logger 保留 |
| checkpoint 子类 schema | 已保留 _class_path；服务器真实 CLI 小模型回归通过；真实大模型重载结果见第 15 条 |
| 额外 accelerator/devices YAML 覆盖 | 已撤回；恢复作者入口默认，不改变既定单卡服务器要求 |
| 运行配置与来源记录 | 已自动写 run_manifest，含 resolved config、命令、代码/列表哈希及实际参数组 |

原审计服务器回归28项通过；新增可选接口后 `python -m unittest discover -s tests -v`：**31 项全部通过，无跳过**；原样 `main.py fit --config ... --print_config` 退出码 0。日志见 `unit_tests.log`、`resolved_config.log`、`commands.json`。

旧版 mmcv、缺 .gitignore、logger=False、ignore_idx schema 的失败均是修复前证据，不能继续列成当前故障。测试中的 tiny backbone 只用于接口回归，不替代真实大模型测试。

## 3. patch 与预训练边界：保持不变

patch_size=16 已由用户确认，不改为 14，不改 384 输入，不额外补边。名称中的 patch14 说明预训练架构来源，并不推翻包装类的显式 patch16。

用户指定 `dinov2_vitl14_reg4_pretrain.pth`，但当前代码没有将此文件接入 timm 分支。默认 ckpt_path=None，仍走 timm 自动预训练流程；离线缓存可用不等于指定文件已加载。主干与本地权重接口按用户计划后续改进，本轮不动。

timm 的权重过滤可能执行 patch kernel/位置编码重采样；不能据名称直接判错，也不能声称原始 pth 可以直接 strict load。未来接入须核验文件 SHA256、来源、转换、register/LayerScale 及 missing/unexpected keys，并区分加载适配与结构创新。

## 4. 模型结构细节

本机 timm 1.0.28 的 meta 结构结果：embedding=1024、24 blocks、16 heads、MLP 隐层4096、1 CLS+4 register；patch projection=(1024,3,16,16)，pos_embed=(1,576,1024)。服务器 timm=1.0.29，运行参数总量为309,291,280，与此前结构审计一致。服务器真实checkpoint进一步确认patch projection=(1024,3,16,16)、pos_embed=(1,576,1024)、4个register、query=(16,1024)、class head=(10,1024)，均为FP32；见checkpoint_structure.json。

16 个 query 在 0-based block21 之前加入，token 数从581变为597；末尾576个空间 token 恢复网格。主网络手动执行 patch_embed/_pos_embed/blocks，换主干不能只检查 embedding 维度。

最后三层共享 backbone.norm、mask/class head、上采样头；24→48→96，再插值到384。含 BatchNorm 的共享上采样头每个训练 batch 调用三次，running statistics 按调用顺序更新。

最后层 query 为每个 query 产生跨三层 softmax 权重，分别融合 mask/class logits；只监督融合结果，没有三份辅助损失。num_blocks=3、masked_attn_enabled=False、attention mask annealing=False。

## 5. 三种 checkpoint 入口

| 入口 | 当前含义 |
|---|---|
| 用户目标 dinov2...pth | 主干预训练文件，接口后续处理 |
| model.init_args.ckpt_path | 构造阶段加载完整分割模块 state_dict，同时链接编码器的 pretrained 条件；不等于精确续训 |
| fit/test 顶层 --ckpt_path | Lightning 框架加载路径，包含 CLI 模型超参数重解析 |

独立 test 在加载完整 checkpoint 前仍先构造模型，可能先读取 timm 预训练缓存；本轮保持该行为并使用 HF_HUB_OFFLINE=1。不能将模型构造用的预训练文件与完整分割 checkpoint 混填。旧缺 _class_path 的 checkpoint 无真实样本，未做迁移或覆盖。

## 6. 数据路径、固定划分与全量检查

- 训练：project_TransUNet/data/Synapse/train_npz；train.txt 对应2211张二维切片，来自18病例。
- 测试：project_TransUNet/data/Synapse/test_vol_h5；test_vol.txt 对应12个三维病例，共1568切片。
- 列表目录：project_TransUNet/TransUNet/lists/lists_Synapse；无重复、无病例交叉，不重新随机划分。
- 相对数据路径按项目根目录解析，接受服务器绝对路径。fit/test 分别检查活动文件，两份列表始终检查病例交叉；无独立验证集，val_dataloader=[]。

列表 SHA256：

| 列表 | SHA256 |
|---|---|
| train.txt | 2a3afc3b126f560d023c09e9a943e31c9ab95cea544afa69bf6b39f80ade59f5 |
| test_vol.txt | 7a1cdb883ad2f76f79cb922cee782eb8618ee7a72e75b640bbcee1367d66c2a6 |

本轮已完整读取 **2223 个文件的 image/label 数组**：全部有限、形状匹配，label 为整数0..8，图像全局范围[0,1]；失败0。训练全为512×512、image/label=float32；测试为[D,512,512]、image=float32、label=uint8。逐文件 SHA256、shape、dtype、极值、类别计数见 `preflight/data_files.json`，汇总见 `preflight/data_summary.json`。本次是全量内容扫描，不再只是此前54张/4例抽查。

本地 TransUNet/datasets/README.md 写明窗[-125,275]、逐3D归一化及切片/h5转换步骤，但缺本批数据的原始转换记录，不能据此证明实际来源与处理链完全一致。当前代码仍没有原始 NIfTI/DICOM 在线处理链。项目中未找到可靠的标签→器官名称映射，继续使用class_1..8，不能凭外部常见顺序猜填。

## 7. 采样、batch 与总步数

训练 batch16、shuffle=True、drop_last=True，2211//16=138 batch/epoch，每轮丢弃打乱后的3张切片；不按病例/器官平衡。默认单卡、accumulate_grad_batches=1、无 batch 限制时预计30,360优化步，最终以实际 estimated_stepping_batches/global_step 为准。

服务器原数据加载器实测：workers8、persistent_workers=True、pin_memory=True、138 batches/epoch。测试 batch 固定1病例、shuffle=False、workers8，使用模块级 collate；测试与训练的 batch 单位不同。证据见 `loader_check.json`。

## 8. 在线增强与输入契约：保持不变

灰度复制三通道，读取器不/255、不重复 ImageNet normalize。顺序为过滤crowd→颜色增强→水平翻转→ScaleJitter→右/下补零→384随机裁剪→过滤空mask；scale_range=(0.5,2.0)，水平翻转p=0.5。brightness幅度32/255、contrast/saturation各0.5、hue=0.05，各子操作概率0.5。

服务器workers8实测增强batch=(16,3,384,384)、float32；mask为bool、标签int64且0..8，无插值新标签。最大值1.0000002384属于浮点误差，不额外clip。模型内部ImageNet标准化不改。

补边的各类别mask均补零，不自动补背景mask。背景参与训练；全mask为空时递归重试的原行为保留。改变颜色增强、补边监督、重试采样、背景处理或归一化均可能改变baseline，不能以数据优化名义调整。

## 9. 优化器与分层 LR：保持不变

AdamW，基础LR=2e-4、weight_decay=0.05、llrd=0.8、llrd_l2_enabled=True、lr_mult=1.0。全部参数参与梯度计算；单参数分组，bias/Norm没有no_decay例外。

服务器实际373个参数组，共309,291,280个参数。调度前基准LR：

| 参数 | initial_lr |
|---|---|
| block0、patch projection等 | 1.1805916207174128e-6 |
| block21 / 22 / 23 | 1.28e-4 / 1.6e-4 / 2e-4 |
| backbone最终norm、query、分割头 | 2e-4 |

block i 通常为2e-4×0.8^(23-i)。运行manifest在调度初始化后采集，起始lr=0并非参数组设置丢失。AdamW.defaults中的lr=0.001是PyTorch构造默认值，所有实际组已显式设置LR，不能误认训练基础LR变为1e-3。服务器betas=(0.9,0.999)、eps=1e-8。

源码中lr_mult的elif被前一个if覆盖；修正它会改变非默认消融语义。本轮不改，只确认当前lr_mult=1及实际分组。

## 10. 预热与数值检查

非主干前500 step线性预热，之后poly衰减；主干前500 step LR=0，再预热1000 step。每轮138 step时，500/1500分别约3.62/10.87轮。LR=0不等于冻结，仍计算梯度并可能更新Adam矩估计。

`schedule_boundaries.json` 已用原调度器检查0、1、499、500、501、1499、1500、1501、30359、30360的LR；这是CPU边界探针，不冒充长训实际轨迹。正式第500/1500步数值及最终AMP状态只能随真实训练核验，不能为检查而改warmup或把两步冒烟当作完成预热。

裁剪保持norm=0.01，FP16缩放/反缩放由Lightning处理。短测反缩放后的梯度、loss/scaler和Adam状态结果见运行报告；不修改loss或优化器以绕过异常。

## 11. 损失与 query 语义：保持不变

使用MaskClassificationLoss及匈牙利匹配；num_points=12544、oversample=3、importance=0.75，mask/Dice/class=5/5/2、no-object=0.1。训练类别0..8，class head输出10类（含no-object）。

16 query不与8器官固定一一对应；目标按切片中的有效语义类别生成mask，不按连通域拆实例。背景mask参与训练损失，测试均值排除背景。当前只监督融合结果，不添加辅助损失，也不套用第一套的0.5 CE+0.5 Dice。

## 12. 服务器、FP16 与 compile

已直接SSH核验：hostname=5090；服务器项目=/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse；Python=/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python。GPU=NVIDIA GeForce RTX 5090 D，驱动580.178.04，nvidia-smi总显存32607MiB。

当前PyTorch=2.13.0+cu132、CUDA runtime=13.2，编译架构含sm_120；Lightning=2.6.6、timm=1.0.29、transformers=5.17.0。完整包版本见installed_versions.json。没有升级/安装依赖，不把本机版本当服务器锁定要求。

默认compile、capture_scalar_outputs=True、suppress_errors=True、matmul precision=medium保留；suppress_errors可能允许回退，必须结合日志和Dynamo计数判断实际捕获，不能只看配置开关。本轮保持原FP16/compile/batch16/workers8，临时max_epochs=1、limit_train_batches=2、last_n_epochs=1、离线W&B及独立绝对输出目录，已完成完整12病例评估并退出0。两步loss=10.98704/11.12897、373个参数梯度均有限，scaler=65536未回退，AdamW状态373项已初始化。GPU peak allocated=18,187,970,048字节（16.94GiB），peak reserved=19,092,471,808字节（17.78GiB）；证据为compiled_retry/runtime_fit.json。Dynamo捕获1358个调用、3个图，损失显式disable、日志/循环等有断图及eager回退，并非全图编译；本轮不改变原路径。

显存不足时不默认减batch/输入、切bf16、增梯度累积或启用梯度检查点；这些涉及运行或训练行为改变，需要作为独立变量处理。

## 13. CPU RAM、I/O 与共享资源

启动前实测主机内存约78.4GiB、可用约71GiB；/dev/shm约39.2GiB；项目磁盘可用约1.09TiB。余量会变化，长训启动前应复查。GPU有其他用户进程占用约3.8GiB，本轮未停止或改动该进程，短测在可用显存超过27,000MiB时启动。

12例共1568个512×512切片，图像float32+标签int64原始载荷约4.59GiB。训练persistent workers与测试预取可能同时驻留，GT转换/预测/锁页内存另有开销；不能只查GPU。

独立审计回调只观测资源、loss/梯度和编译计数，不改模型或RNG；GPU峰值由allocator统计，RAM为若干时点进程树RSS之和，可能重复计算共享页，不冒充连续RAM峰值或磁盘吞吐。共享GPU、编译和监测开销下的短测耗时不外推为220轮性能。

## 14. 测试推理与指标：保持不变

完整3D病例、逐2D切片推理；zoom(order=3)到384，复制三通道，mask sigmoid与去no-object的class softmax合成逐像素分数；插值分数后argmax，再zoom(order=0)回原尺寸。无TTA、滑窗、连通域后处理或阈值搜索；mask_thresh/overlap_thresh不参与这条路径。

类别1..8按病例/类别等权平均；Dice为0..1，展示百分比才乘100。HD95未传voxelspacing，单位为体素坐标距离，不改成毫米。resize后不额外clip。

原空mask规则：双方空→(0,0)，仅预测非空→(1,0)，仅真值非空→(0,0)，非空重合→(1,0)。它不符合部分常见约定，但改动会改变分数及选模，本轮不改。

## 15. 独立重载与一致性核验

两份真实大模型checkpoint均通过原`main.py test --ckpt_path`的超参数解析与加载；各388个state_dict键无missing/unexpected/shape mismatch，加载后逐张量值完全一致，均不含_orig_mod键。checkpoint为epoch=0/global_step=2，对应CSV第1轮；各次best/last的SHA256及张量值也完全相同，评估没有改动权重。证据见reload_*/checkpoint_schema.json、checkpoint_pairs.json、checkpoint_structure.json。

| 检查 | 实际结果 |
|---|---|
| 旧eager冒烟→独立CLI eager test | 12病例×8类Dice/HD95全部精确一致，退出0 |
| 新compile冒烟→独立CLI eager test | 12例均评估完成、权重恢复正确；严格指标相等断言未通过，退出1 |
| 差异范围 | 仅case0003/class7：Dice绝对差1.2630000937e-7，HD95绝对差0.00067838089体素，其余病例/类别相同 |
| 单病例上下文定位 | eager的inference_mode与no_grad相同；使用Lightning同样from_compiled包装的compile/no_grad后精确匹配在线CSV |

默认行为：main.py在fit中torch.compile，独立test仍为eager；训练内回调因此可能与独立eager推理产生浮点数值差异。本次探针已确认该病例差异来自compile/eager，不能误判为checkpoint未加载或权重改变。**加载闭环通过，但跨执行模式不能声称指标完全一致。** 历史断言失败保留，不放宽检查，不改默认test/fit路径、FP16或指标。

详细数值及退出码见reload_commands.json、reload_eager/runtime_test.json、reload_compiled/runtime_test.json；定位脚本与结果为case_compile_probe.py、case_compile_comparison.json，汇总见completion.json。单病例同路径复现不等于已证明未来所有训练checkpoint跨模式相同。

### 可选优化：独立测试使用相同compile前向

用户于2026-10-21要求优化该差异，现新增**默认关闭**的`--compile_test`。显式开启时，仅临时对模型forward执行torch.compile，checkpoint加载、test_step、病例循环、精度及指标仍交给原Trainer.test；加载权重后才首次触发实际编译，结束或异常时恢复原forward。不会将整个病例/HD95循环一起编译。`--compile_test`与`--compile_disabled`同时出现会报错。

服务器真实验证：对同一compile短测checkpoint，完整12病例、8类Dice/HD95与训练内CSV的最大绝对差为**0**，严格比较通过、退出码0；388键/shape/逐张量权重也一致。31项回归全部通过，覆盖默认eager不变、仅编译forward、异常恢复、冲突参数及真实CLI解析。证据为20261021T130147Z_compile_test目录下commands.json、unit_tests.log、compiled_reload/runtime_test.json与checkpoint_schema.json。

使用原则：默认compile训练产生的checkpoint，独立复现时显式加`--compile_test`；`--compile_disabled`训练产生的checkpoint保留eager测试。不加新开关的历史行为不变，不能笼统声称不同执行模式永远没有微差。本次仅验证当前checkpoint和服务器环境，不外推为跨版本/硬件的逐位保证。

```bash
cd /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse
HF_HUB_OFFLINE=1 \
TORCH_HOME=/Disk/HDD/Sk/.cache/torch \
TIMM_USE_OLD_CACHE=1 \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python main.py test \
  --config configs/synapse/vit_query_mul_scale_fusion.yaml \
  --ckpt_path /absolute/path/best_test.ckpt --compile_test
```

## 16. 保存、目录与续训限制：保持不变

当前用户已指定的代码行为保留：220轮训练，最后50轮完整测试，best_test.ckpt与滚动last.ckpt；不提前停止。无需另建独立评估策略文档，也不推测作者未说明的Synapse流程。

每次fit创建独立UTC时间+ID目录；last在本轮测试前保存。CSV为1-based epoch，checkpoint通常0-based；必须核对CSV/JSON/同份权重，不能跨轮拼Dice与HD95。两份权重包含整个模型及主干；旧运行不自动删除。

weights_only不含optimizer/scheduler/scaler/完整RNG及跨进程best历史，**不支持精确续训**。本轮不增加第三份完整状态、不改恢复策略、不覆盖历史权重。输出父目录使用独立服务器绝对路径，实际run_dir见manifest/日志。

## 17. 日志、归档、随机性与环境

原W&B配置project=eomt、name=synapse_384_learnable_fusion_query_2e_test、resume=allow保留；同名不等于同run，也不等于断点续训。短测使用WANDB_MODE=offline，不联网发布；未检查/打印API key或测试在线凭据。LearningRateMonitor在正常logger下保留。

自动run_manifest已存在，包含解析配置、实际callbacks、来源/列表哈希、绝对数据路径、版本/GPU及逐组LR。save_config_callback=None不再等于“没有配置归档”；正式220轮是否完整结束仍需退出码与指标/资源报告证明。

seed=0不等于deterministic=True或精确恢复worker随机流；不擅自开启确定性模式。所有服务器Python命令保留HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1。

本轮sitecustomize未被启动加载；但torch.Tensor.to实测为原生method_descriptor，没有发现全局Python包装补丁，见tensor_to_check.json。不能声称空覆盖文件已生效，也不为此更改PYTHONPATH。

## 18. 检查记录与尚未处理的范围

- 全量数组/文件哈希、列表与病例交叉：通过，2223文件失败0。
- 服务器回归测试：28/28通过；原配置CLI解析：通过。
- 原workers8真实增强batch、布尔mask和合法标签：通过。
- 原模型/训练参数/五份YAML保持不变；主入口仅增可选test接口，相关测试增加，旧快照其余21项一致，新快照记录接口版本。
- 默认compile/FP16/batch16/workers8及离线W&B真实短测：2步、12例、保存完成，退出0；梯度有限。
- 真实大模型独立加载和逐权重核验：通过；旧eager指标精确一致，新compile对独立eager有第15条所列微小差异，未掩盖断言失败。
- 新增审计回调首次因服务器未装psutil中断，已改为标准库/proc，不安装依赖；失败目录及日志保留，重试结果单列，不能算baseline故障。
- 主干/指定本地预训练接口按用户计划留后；标签器官语义、原始转换来源仍缺证据；旧缺元数据checkpoint无样本。
- 正式220轮、第500/1500步实际轨迹、长期资源变化及论文成绩未执行，不能由短测代替。

## 19. 正式服务器运行前的门槛与处置

| 检查 | 状态 / 处置 |
|---|---|
| hostname/项目/Python/5090D | 已完成；environment.json |
| 原样print_config及DINO依赖 | 已完成；退出码0 |
| logger=False、归档异常回退 | 已完成回归；离线W&B真实短测通过，在线凭据不测试 |
| 主干与指定预训练文件 | 按用户计划延期；本轮不改加载接口/初始化来源 |
| 列表SHA、全部文件和内容 | 已完成；2223文件全读、原划分不变；器官名称仍缺来源 |
| 正式模型/损失/AdamW初始化后峰值 | 已完成；两步loss/梯度有限，373项AdamW状态已初始化，GPU峰值见第12条；RAM为采样RSS，不冒充连续总峰值 |
| 原FP16/compile/batch16/workers8 | 已完成；默认compile真实捕获3图，允许原有断图回退，完整测试/保存通过 |
| 评估/保存/独立重载闭环 | 已完成；--compile_test完整12例精确复现在线CSV；默认eager保留 |
| 编译键、逐张量值与同份权重指标 | 388键/shape/值正确；旧eager及新可选compile各自同路径精确一致；跨模式微差记录保留 |
| 精确续训限制/输出目录/磁盘 | 已明确不支持精确续训；独立绝对输出路径及磁盘核验完成 |
| resolved config/run_manifest | 已实现并检查；正式220轮的最终记录随实际训练生成 |

本轮可在不改baseline前提下进行的检查已执行并记录；没有启动正式220轮。尚缺的权重接口、器官映射/原始处理来源、长期资源与数值轨迹不伪造完成状态。

改变baseline的项不动：模型/预训练初始化来源、lr_mult算法、训练增强/归一化/背景与损失、batch/尺寸/精度/累积/梯度检查点、空mask/spacing/后处理，以及额外完整续训策略。纯检查通过不等于授权更改这些项。
