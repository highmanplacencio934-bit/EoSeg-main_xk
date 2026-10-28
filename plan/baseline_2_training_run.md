# baseline-2：seed=0 纯 EoSeg 重复训练

> 记录范围更新（2026-10-26）：本页仅记录已停止并删除的旧非ROI run。用户随后明确授权了新的ROI baseline-2，从头训练的新run为outputs/baseline-2/20261026T130228Z；当前运行信息见[ROI baseline-2记录](baseline_2_roi_training_run.md)。下文旧的“当前状态/禁止自动重训”仅适用于旧run，不能当作新run状态。

> 当前状态更新（2026-10-26T06:51:44.855231-06:00）：用户明确要求停止baseline-2并删除其全部输出，现已完成。训练、监督器及所属子进程均退出；outputs/baseline-2及其权重、日志、CSV、清单和指针已删除。以下内容仅保留历史启动记录，不代表仍在训练或可续训；不自动重训或开启新实验。停训删除凭证：outputs/audit/baseline_2_stop_delete/20261026T125002Z/summary.json。

用户于2026-10-26明确授权：以首轮baseline一模一样的参数重新训练，seed=0，不开启创新模块，命名baseline-2。本次用于观察同seed重复运行的结果差异，不新增其他研究变量。

## 运行身份

- 服务器项目：`/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse`，hostname=5090。
- Python：`/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python`。
- 启动时间（America/Denver）：2026-10-26T04:14:43.007913-06:00。
- 本轮输出：`outputs/baseline-2/20261026T101442Z`；指针：`outputs/baseline-2/latest_path.txt`。
- 监督器PID：2671989；训练PID：2671992。后续操作必须重新核验PID/start_ticks/cmdline/cwd，不能只凭旧PID操作。
- 参照run：`outputs/baseline/20261021T132635Z`；首轮完整220轮，best177，原1-view平均Dice=84.71846610153186%。首轮best的8-view=86.8495283529797%属于另一推理协议，不与本轮1-view混比。

## 实际入口与固定值

直接使用`main.py fit --config configs/synapse/vit_query_mul_scale_fusion.yaml --seed_everything=0`；完整可执行argv在本轮`launch.json`。只覆盖default_root_dir、W&B save_dir/name（baseline-2）和原SynapseTestSelection的output_dir；没有复制/改写作者YAML。

- 网络：models.eoseg.EoSegMultiQSegFusion；训练类：training.mask_classification_semantic.MaskClassificationSemantic。ASEE/CCPS未实例化。
- 384×384、patch16、9类、16query、batch16、workers8、220epochs。
- LR=2e-4，AdamW、weight_decay=.05、LLRD=.8；原预热/多项式调度、梯度裁剪norm=.01、损失及采样均保持。
- 原颜色增强、水平翻转、ScaleJitter(.5,2)、原补零/384裁剪；18训练病例/2211切片、12测试病例原列表。
- 16-mixed、默认torch.compile、matmul precision=medium；不额外设置deterministic、benchmark或其他随机性选项。
- 原离线timm预训练初始化流程，不加载baseline/innovation已训练分割checkpoint；未替换为用户指定本地pth接口。
- 每次Python命令保留HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1；W&B offline。

## 相同性核验

启动前除main.py外，首轮launch中22份源码/配置/审计文件的SHA256完全一致。main.py保留了随后已有的独立test兼容功能；本轮沿用原fit函数和纯baseline分支，不能声称入口文件整体字节哈希与首轮相同。

启动后的`startup_verification.json`确认：

- 解析后的模型、数据、seed、训练、编译设置与首轮完全一致，归一化比较仅移除四项输出/日志命名差异。
- 373个优化器组的顺序、LR、initial_lr、参数数目及优化器defaults逐项一致。
- estimated_steps=30360，依赖版本全部一致；train/test列表SHA一致。
- 实际device=cuda:0，RTX5090D；启动前GPU空闲，显存可用约31GiB，磁盘可用约1.1TiB。
- 所有训练模型、数据、增强、损失、优化器、YAML与原权重文件未修改。

## 评估、最终比较与状态

本轮刻意使用首轮相同的SynapseTestSelection和1-view路径：171–220共50轮，每轮完整12例，按原legacy Dice选择同一best；保留best_test.ckpt与滚动last.ckpt。训练内HD95仍是首轮voxel口径，不标mm；本次不把独立物理包装类、ROI入口或TTA加入原训练对照。

监督器仅在完整220轮退出0、50个epoch记录与600个病例记录完整后生成本轮summary.json/summary.md，按已核验1–8器官名称输出同一best的Dice/HD95_voxel与首轮Dice差值；不能按器官跨epoch挑最高值。失败保留真实退出码，不自动重训；weights-only不可冒充精确续训。未创建额外定时监控或启动其他消融。

当前状态以run下status.json、train.log、run_manifest及真实PID为准。baseline-2运行期间保持其模型/训练依赖、YAML、数据和seed冻结。此记录表示已启动并核验，不代表220轮或最终比较已经完成。


## 2026-10-26 删除未使用的独立增强分支

用户随后授权删除8个独立旋转/尺度增强文件及其专属输出。baseline-2实际数据类仍为SynapseSemantic，原Transforms、模型、数据、YAML与优化器均未改，训练未停止。原run_manifest曾扫描记录这些未使用的文件，历史快照不改；这些经授权删除的全项目快照条目不是本轮模型的活跃依赖。后续核验按删除清单排除该独立分支，实际baseline训练依赖继续严格比对。
