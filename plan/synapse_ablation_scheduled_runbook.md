# Synapse 早停、评估加速与消融定时执行说明

更新：2026-10-25。以下为用户已授权的未来执行任务；本次设置阶段不提前停止现有训练，不提前启动优化或消融。

## 时间与阶段

时区均为America/Denver（本日UTC-06:00）。

| 日期/时间 | 内容 |
|---|---|
| 10月25日14:56 | 温和早停指定联合增强run，记录实际轮次及完整结果 |
| 10月25日15:30 | 后续任务1：依据优化指南更新未来独立评估入口并验证 |
| 10月25日16:30 | 后续任务2：通过优化门槛后，从M0开始串行队列 |
| 10月25日17:30 | 后续任务3：核验当前实验；完成则记录并启动下一项 |
| 10月25日18:30 | 后续任务4：同上 |
| 10月25日19:30 | 后续任务5：同上 |
| 10月25日20:30 | 后续任务6：同上 |
| 10月25日21:30 | 后续任务7：同上；不自动另建后续检查 |

只有14:56任务操作指定旧run；七个后续任务不代表七组实验或七小时内全部做完。每次先读取账本与实际进程，不按时刻机械启动下一实验。若优化未通过，后续时刻继续未完成的验证阶段；不绕过门槛。最后一次检查后已正常运行的实验继续运行，未开始项目保留待执行，不自动追加无限监控或从头重训。

本机需保持开机、Codex应用运行、SSH连接配置及项目可访问；任务实际执行时间受唤醒与资源影响。不得把尚未执行的动作写成已完成。

## 服务器与环境

SSH别名anhuiuniversity.icu；项目 /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse。
Python /Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python。
每次Python带HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1。CPU验证可单独CUDA_VISIBLE_DEVICES为空及OMP/MKL线程限额，正式GPU任务不得继承隐藏GPU的环境。启动前核验真实解释器、GPU进程/显存、磁盘、清单与源码/配置/权重/先验hash。

## 14:56指定早停

唯一目标：
outputs/augmentation_joint/rotation_scale_seed0/20261025T042216Z。
历史training PID645665、supervisor PID645616仅用于定位；发送信号前必须重新核验/proc命令、cwd、启动时间及run归属，避免PID复用误杀。
14:56按时间停止，不等待“完成205轮”；205只是按此前速度的估计。

1. 记录status、launch、checkpoint/CSV/manifest位置与hash，最后训练epoch/global_step、最后完整12例评估epoch及最佳epoch。
2. 向核验后的训练子进程发送SIGINT，温和退出；分段等待确认，禁止模糊pkill、SIGKILL或停止其他用户/其他run。若未退出，记录仍在运行，不能假报成功。
3. 等监督器写完状态后保留原status副本，再将用户停止标记为stopped_by_user，同时保留真实退出码。不要篡改原日志或把非零退出码抹成0。
4. 此启动器训练非零退出不会自动TTA；仍须核验。如果到点同run已自然完成并启动TTA，仅温和停止核验属同run的test子进程，保留完成训练与中断测试两个真实状态；已经自然结束则不再发送信号。不得给其他任务发信号。
5. 保留best/last/CSV及原有历史输出，不重启该run，不自动补最终8-view或从weights-only声称精确续训。
6. 写outputs/ablations/journal_20261025/stop_report.json与reports/early_stop.md，按实际状态汇总器官结果，区分训练轮数和测试完成轮数。早停后正在计算的部分病例不能计成完整成绩。

## 优化阶段：仅未来独立入口

依据根目录GPU_CPU_HD95_Optimization_Guide.md。已有候选在outputs/audit/hd95_acceleration_candidate_20261025/，仅完成局部验证，不能直接视为全量通过。

先保留原指标实现与默认入口，通过独立模块/配置实现pred|gt联合包围盒+halo，仍调用原MedPy Dice/HD95_mm/HD95_voxel，完整原病例几何校验在裁剪前进行。保留远处误检、真实ZYX spacing、原空mask逻辑与返回状态，不热注入旧run，不降低精度或改公式。
先合成边界/空mask/各向异性回归，再在同一保存预测上覆盖12病例×8前景类别逐值对照。若现有保存预测不齐，可在旧训练已确认停止且GPU空闲后，用一个固定历史权重收集所缺预测；这属于评估验证，不重训。不能比较两次不一致的推理结果后把差异归于指标容差。
先单进程ROI，验证通过后才评估2个CPU worker/有界流水；若收益和资源不足则保留已通过的单进程方案，不能为追求利用率改模型。需重复交替计时、记录端到端耗时/RAM/误差及病例顺序、checkpoint归属/RNG保护。任何非逐值一致先定位，不自行放宽容差。并行不是必选项。
建立optimization/readiness.json，仅在实际完成必要回归、真实12例全类对照、端到端计时及必要GPU默认FP16/compile接口冒烟后置passed，保存源码/环境/配置hash。若比较不通过则不启动正式消融。后续所有run记录加速实现与worker数，不覆盖历史成绩。

## 已授权队列与固定变量

严格顺序：
M0 → M1 → S1 → S2 → S3 → S4 → A1 → A2 → A10 → A11。
A0仅兼容性检查，A3仅核验已有完整模型参考，不作为新增重复训练项。

- M0：纯baseline已有best，固定当前最优scale_hflip_8，完整12例测试。
- M1：baseline+ASEE已有best，相同8-view，完整12例测试。
- S1/S2：ASEE文本单分支/共现单分支；S3/S4：ASEE patch单分支/query单分支。分别独立训练，不能只在完整权重上关模块冒充训练消融。
- A1/A2/A10/A11：CCPS global/visual/context_only/residual_only；主Plan其余变量保持，以文档为准，独立训练。
- 所有新增训练统一seed=0、原数据增强（颜色增强、水平翻转、ScaleJitter 0.5–2.0、原补零/384裁剪）；不使用本次联合增强±10度/p0.3/尺度0.75–1.50。
- 384/patch16、batch16/workers8、220轮、AdamW/LLRD/LR/调度/损失/FP16/default compile、固定18/12划分及训练集先验均保持各自原对照设置。训练从相应原预训练初始化，不从历史分割best微调。
- 新训练仍按原最后50轮完整12例1-view Dice选择best，再用同一best固定8-view作为最终报告；不能按8-view重新挑epoch。不加入有/无TTA消融。
- 默认legacy空mask逻辑保留；HD95主mm且保留voxel。原空mask问题、测试参与模型选择与不同历史训练预算如实记录。
- 不启动A4–A9、额外超参数搜索、多seed、增强2×2或未列出的变体。当前完整模型历史8-view结果不算新训练完成证明。

## 输出、状态与防重复

专用根目录：outputs/ablations/journal_20261025/。
已预置orchestration_state.json、experiment_registry.json；以后每项使用独立ID/UTC时间戳目录，logs、launch/status、CSV、manifest、最终器官报告分别保留，不覆盖旧outputs/baseline、innovation1/2或augmentation_joint。

先建立共享队列锁与幂等启动回执：锁和带进程启动时间的所有权记录也覆盖优化代码编写、验证、状态更新，防止15:30工作跨到16:30时第二个任务同时修改；后续任务若检测有效优化所有者则只读检查。GPU任务同一时刻最多一项；即使上次代理仍工作或小时任务重叠，也不得重复启动。启动锁需覆盖检查和登记/Popen的竞态窗口；持久化PID、启动时间、run目录、阶段和launch intent，已有进程即只检查。不能仅凭stale状态删除锁或当作可以重跑；核实进程/启动凭据。缺少账本不等于没有运行。

每次任务：
1. 核验旧训练已停及优化passed/hash；若未停先处理/报告该阻碍，不抢占资源。不得仅因早停任务失败就并发启动新任务。
2. 当前任务仍运行：核验日志、epoch/step或完整病例进展；保持运行。训练结束但最终8-view未完，仍算当前任务。
3. 当前任务退出：核对真实退出码、12病例×8类结果、8-view配置、权重hash及来源；新训练还需220轮与171–220的50次完整选择评估。任一缺失不能标已完成。
4. 验证通过，写summary.md/json、逐器官Dice%/HD95_mm/voxel及平均（核验名称映射、不整理背景），更新registry和消融MD第11节为“已完成”，注明路径/hash/epoch/预算，再启动下一未做项目。
5. 失败记failed/blocked与证据；仅修复本任务适配/加速实现的问题并重新验证，保留失败目录。不得静默更改科学变量、自动重训或把失败跳过后假称顺序完成。涉及模型设计歧义或需改变对照变量时报告，不擅自扩大。
6. 完成一次检查后暂停/删除自身一次任务，不删除后续任务，不创建额外监控。状态不变且无异常时保持安静；完成/新启动/失败/需要用户决定时报告。

MD“已完成”只能在真实输出核验通过后更新；预置账本状态全部planned，不代表当前已开跑。

## 已创建的任务标识

- 2026-10-25T14:56:00-06:00：synapse-14-56（已执行早停；实际SIGINT 14:58:01，训练14:58:08退出；本次自动任务已删除，后7项保留）。
- 2026-10-25T15:30:00-06:00：synapse-15-30（实现与功能回归已完成，第二轮CPU对照继续；本次自动任务已删除，后6项保留）。
- 2026-10-25T16:30:00-06:00：synapse-16-30（已完成优化验收、M0/M1及S1启动；此前删除受额度审批失败影响，20:30复核后已删除）。
- 2026-10-25T17:30:00-06:00：synapse-17-30（当时未实际检查，受此前额度审批阻碍；20:30清理已过期一次任务）。
- 2026-10-25T18:30:00-06:00：synapse-18-30（当时未实际检查，受此前额度审批阻碍；20:30清理已过期一次任务）。
- 2026-10-25T19:30:00-06:00：synapse-19-30（当时未实际检查，受此前额度审批阻碍；20:30清理已过期一次任务）。
- 2026-10-25T20:30:00-06:00：synapse-20-30（已只读检查S1正常训练；本次一次任务已删除，21:30保留）。
- 2026-10-25T21:30:00-06:00：synapse-21-30（已执行：S1完整训练退出0，最终8-view因其他用户GPU任务延后；本次任务已删除，无新增监控）。

## 14:56任务实际执行结果

2026-10-25已温和早停指定run，SIGINT于14:58:01 Denver发出，训练14:58:08退出；PID645665和645616均消失、GPU计算进程为空。last保存205轮/28290step，完整12例测试仅至204轮（171–204共34轮），第205轮测试未完成。最佳192轮1-view Dice=85.762305%，HD95_mm=22.807769，voxel=19.879132。原退出码1与监督器failed内容已归档，当前状态正确记录stopped_by_user。详见outputs/ablations/journal_20261025/reports/early_stop.md。未启动优化/消融或最终8-view。

## 15:30实现完成、重复计时接续（当时验证中）

新增独立ROI模块/入口已完成；212项CPU回归、完整12例GPU8-view（288值精确复现）及3步fit/12例选择链路通过。CPU全12例首轮96类dict对照已通过，第二轮交替重复仍执行；readiness为validating/false，尚不启M0。接续步骤见outputs/ablations/journal_20261025/optimization/continuation.md。CPU完成且退出0后运行scripts/finalize_synapse_roi_readiness.py，只有其最终passed才推进。16:30任务可在同一回合分段等候临近结束的原验证，不能另起重复作业。

## 15:30优化验证完成

2026-10-25T22:31:52.467112+00:00：独立入口scripts/run_synapse_roi.py验证通过；CPU完整缓存评估21.30倍，12例8类精确一致，212项CPU回归、完整GPU8-view和三步fit链路通过。单CPU ROI启用，未实现并行池/流水，不改变原入口。详见outputs/ablations/journal_20261025/optimization/summary.md。本阶段未启动M0/M1。

## 16:30任务实际完成与后续入口

优化双轮已完成并严格passed，完整缓存CPU评估中位数1394.998秒→65.500秒（21.30倍）；这不是整场训练倍数。M0和M1已有best的完整12例固定8-view已完成，汇总见outputs/ablations/journal_20261025/reports/main_model_eight_view_comparison.md；M0 Dice86.849528%、HD95_mm12.968490，M1 Dice86.086021%、HD95_mm11.928217。保留了新启动器后处理导入/文档表格定位错误的recovery证据，仅恢复CPU结算，没有重推理或改分。

S1已于2026-10-25 17:09:59 Denver正式开始原增强/seed0的220轮训练；run outputs/ablations/journal_20261025/S1/20261025T230957Z，supervisor1897260、training1897309仅供定位，检查时重验/proc启动时间和命令。startup_verified.json核验时为8轮/1104step；原scale0.5–2、无旋转、384/patch16、batch16/workers8、FP16/default compile、基础LR2e-4/LLRD及401个优化器参数组均与原ASEE一致；唯一研究改动use_probability=false。240项CPU回归及S1三步GPU/2次真实optimizer更新通过。

17:30起任务：先看active/registry/实际PID与日志，S1正常训练保持安静，不重复启动或修改已冻结runtime脚本。只有S1完整220轮+50次12例评估+同best8-view全核验后，才运行scripts/launch_synapse_journal_component.py --next启动S2；之后依次S3/S4/A1/A2/A10/A11。M0/M1使用scripts/launch_synapse_journal_ablation.py，该入口不启动组件。训练完成但测试GPU忙是deferred，只恢复同run测试，不能重训。前171轮无测试CSV是预期；rolling last.ckpt每轮保存，可只读CPU mmap检查epoch/global_step。

## 20:30检查与一次任务清理

本次于2026-10-26T02:52:14.316910+00:00复核：S1进程与启动时间、命令、cwd均一致，训练正常；rolling last已保存第204轮/28152step，完整12例评估171–203共33轮，当前最佳第187轮1-view平均Dice=86.696942%。这些是中间状态，尚无S1最终8-view结果。优化readiness的122份来源哈希、8份验证报告及当前运行脚本/配置哈希全部保持。未修改运行代码、科学训练变量、registry或orchestration_state；未停止训练，也未重复启动下一项。

发现其他用户xr的GPU进程2085630，未操作该进程；后续启动或最终测试仍须重新核验资源。17:30、18:30、19:30未实际完成检查，不能把本次状态倒填为当时结果；此前最终清理受自动审批额度限制。现已通过应用工具删除16:30至20:30五个已过期/本次一次任务，21:30任务保留，不创建额外监控。详见outputs/ablations/journal_20261025/reports/20_30_followup_result.json与20_30_file_changes.json。

## 21:30最终检查：S1训练完成，8-view待GPU

实际训练结束21:37:27 Denver（2026-10-26T03:37:27 UTC），220轮/30360step，退出0。171–220全部50轮及12病例×8器官记录齐全，训练完成证明training_completion.json已经生成；best187同一权重的1-view平均Dice86.696942%、HD95_mm17.270780、voxel14.998108。该成绩不是最终8-view。逐器官与哈希详见outputs/ablations/journal_20261025/reports/s1_training_complete_pending_8view.md。

当前GPU仍有其他用户xr进程2085630，启动器已进入training_completed_tta_deferred；本次不抢占GPU，不标S1完成、不启S2。账本已由监督器更新，MD同步为“训练完成，8-view待GPU”。原训练子进程与监督器均已退出，未发送停止信号。后续仅允许在资源与原哈希核验后恢复同S1的best8-view，完整核验后才按队列推进。

21:30为最后一个授权单次检查，现已通过应用工具删除；七次窗口结束，不创建额外监控。S2–S4、A1/A2/A10/A11仍待执行。所有原代码/模型/增强/seed/训练参数/指标与历史产物保持。检查记录见outputs/ablations/journal_20261025/reports/21_30_followup_result.json，文件变更见outputs/ablations/journal_20261025/reports/21_30_file_changes.json。
