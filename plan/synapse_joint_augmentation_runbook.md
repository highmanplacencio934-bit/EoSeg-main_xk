# Synapse联合增强独立实验运行记录

更新：2026-10-24T22:24:02.790779-06:00。用户本次已明确授权独立实现并直接训练完整EoSeg+ASEE+CCPS。只启动这一组联合增强对照。

## 实际启动

- 服务器：5090；GPU：RTX5090D 32GB；Python：`/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python`。
- 根目录：`/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/augmentation_joint/rotation_scale_seed0/20261025T042216Z`。
- 定位指针：`outputs/augmentation_joint/rotation_scale_seed0/latest_path.txt`。
- supervisor PID：645616；training PID：645665，已读取/proc及nvidia-smi匹配。
- 实际状态：running/training；这不是已完成实验成绩。
- 日志：run下`train.log`、`supervisor.log`、`launch.json`、`status.json`；原始manifest位于`synapse_test_selection/20261025T042229Z_db8ab4fc/run_manifest.json`。
- 运行配置与优化器对照：`runtime_control.json` passed；完整最终文件列表见`file_changes.json`。

## 唯一科学改动

1. 新增小角度旋转：[-10,10]度均匀采样，p=0.3，expand=False。
2. ScaleJitter从0.5–2.0改为0.75–1.50。

图像双线性、mask最近邻；旋转共用角度、标签不换，填充沿用全mask零。保留原颜色增强、水平翻转p0.5、右下补边/随机384裁剪和空目标重试。没有垂直翻转或旋转TTA。

DataModule/transform/CLI/YAML均为新入口，原`datasets/synapse_transforms.py`与`datasets/synapse_semantic.py`字节未改；新路径旋转禁用且尺度复原时与原输出及RNG状态一致。初始化阶段不消耗额外随机数。原模型、损失及优化器实现没有修改。

seed=0，220轮、batch16、workers8、384/patch16、LR2e-4、AdamW/LLRD/调度/梯度裁剪与原完整模型一致；FP16 mixed/default compile；ckpt_path=None，从原离线timm预训练流程初始化。不能说是从原best继续训练，也不能声称指定的另一个本地预训练文件已新接入。

固定18训练病例/2211切片、12测试病例，不更换列表或创建15/3划分。ASEE/CCPS参数、训练先验与默认legacy空mask规则不改。HD95采用核验后的逐病例mm并保留voxel。

## 对照与结束后评估

参考：`outputs/tta/innovation2_best_scale_hflip8/20261024T123754Z/results_summary.json`；旧完整模型seed0 best187，8-view Dice86.9787030901%、HD95_mm11.1421893937。目标+0.5pp即87.4787030901%，尚无新成绩或提升保证。

训练最后50轮171–220，每轮原1-view完整12病例Dice选best，保留best_test.ckpt/last.ckpt。额外归档截至213轮的匹配窗口best为`best_test_through_epoch213.ckpt`与`matched_window_summary.json`。这额外占用约1.2GB，仅在此独立新实验，未改原回调。

训练退出0且核验220轮/50轮测试完整后，启动同一best的固定scale_hflip_8（尺度1.0/0.875/1.125/0.75×原方向/水平翻转、384画布、原覆盖融合）。只做一次最终8-view，不逐轮8-view选优；输出`final_eight_view/`、`tta.log`、`eight_view_results.md/json`和8器官命名表。名称按verified label_id关联，背景不汇总。GPU若被其他任务占用则记录TTA deferred，不抢占。

旧实验完整评估到213轮，新预算220轮；报告必须同时提供171–213匹配窗口1-view结果。如best晚于213，额外候选轮次影响必须说明，不能全归因增强。只有一组组合实验，不能拆分旋转/尺度贡献或宣称多seed稳定增益。测试集参与checkpoint选择，不能描述为独立盲测。

## 验证证据

`outputs/audit/joint_augmentation_20261025/`：153项CPU回归（15项新增），50张仅训练集样本检查/增强示例，配置逐项对照，101份原文件哈希一致。独立GPU冒烟：batch16/384/workers8/FP16/default compile；真实loss11.1333/11.2956/11.5370；原生GradScaler65536降32768后两次有限非零梯度更新；峰值17.316GiB；12例geometry preflight；固定8-view输出[1,9,384,384]有限，新旧2-view max diff=0。

首个冒烟观测器在默认GradScaler处理前错误断言scaled梯度有限，已将检查移到unscaled梯度并允许框架原生跳步；失败日志保留。仅修正隔离观测器，未改训练scaler或精度。测试也按已安装torchvision的make_params接口核验共享角度。

门禁：`outputs/augmentation_joint/joint_readiness.json` passed，包含原文件、新源码/配置和证据哈希。启动器加锁且已有receipt则返回，不会重复训练；失败不会自动改参数/重训。

## 本轮文件

新增：datasets/synapse_joint_transforms.py、datasets/synapse_joint_semantic.py；training/synapse_joint_selection.py；scripts/run_synapse_joint_augmentation.py、scripts/launch_synapse_joint_augmentation.py；configs/synapse/vit_query_mul_scale_fusion_asee_ccps_joint_aug.yaml；tests/test_synapse_joint_augmentation.py、tests/test_synapse_joint_launcher.py；本文档；独立audit/readiness/run产物。

修改：根目录增强Plan，补充最新授权与真实实现/启动状态。

删除：无。原增强、baseline/innovation1/innovation2模型/配置/数据/权重/结果均未覆盖。

## 状态查询与操作

通过已配置SSH anhuiuniversity.icu执行；每次Python保留HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1。SSHFS不是服务器进程。不要复用旧innovation2启动器开新训练。

用户随后授权一次30分钟启动检查，任务信息见下节。需要状态时读指针、status/log、真实PID/GPU与manifest；既有best/last是weights-only，不能声称支持精确续训。未获新指令不切换增强、不启动其他消融或种子训练。

最新启动后核验：UTC 2026-10-25T04:24:53.158310+00:00，已完整训练第2轮、global_step=276，last.ckpt大小1270204959字节；训练仍运行。原默认compile触发动态mask数量的recompile limit警告，保留框架原有回退行为，没有关闭compile或改变参数。

## 用户追加的一次30分钟检查

已创建当前对话heartbeat，任务ID=`30`。以2026-10-24 22:25:20 America/Denver为基准，于22:55:20检查（UTC2026-10-25 04:55:20；北京时间12:55:20）。只核对本run进程、GPU、日志、manifest及epoch/step，向用户汇报一次；异常只读排查，不自动重启或改参数。完成后删除或暂停自身任务，不停止训练或最终8-view。请求副本：run下`30min_startup_check_request.json`。

## 30分钟检查完成（任务30已删除）

检查时间：2026-10-24T22:56:29.845349-06:00（America/Denver）。训练正常推进，滚动last已完整第39轮、global_step=5382，相较启动核验2轮/276step继续增长。supervisor/training PID均存活且命令匹配；RTX5090D训练PID645665占用19484MiB，检查时GPU利用率76%。115项门禁哈希与manifest源码哈希均未变化，日志无Traceback/OOM。测试从171轮开始，当前没有Dice/HD95测试记录符合计划。

已删除一次任务30；未停止、重启训练或改变任何训练参数。新增run内`30min_startup_check_result.json`保存证据；修改本文档记录检查完成；无删除项目文件。
