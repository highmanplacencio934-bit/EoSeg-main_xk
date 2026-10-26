# 完整EoSeg+ASEE+CCPS seed42 单变量对照

用户2026-10-24授权：暂不研究新增数据增强，训练完整baseline+innovation1+innovation2（EoSeg+ASEE+CCPS）的seed42，与seed0比较；做好消融，不轻易改变现有代码。

唯一科学变量为seed_everything=0→42。直接复用configs/synapse/vit_query_mul_scale_fusion_asee_ccps.yaml及原训练/模型/增强/数据划分/损失/优化器/调度/指标代码，CLI覆盖seed42；不接入新增强，不加载seed0训练权重、不从ASEE/CCPS checkpoint续训，保持原离线timm预训练初始化。只改变日志名称、日志目录、输出目录以隔离结果。220轮、batch16、workers8、384/patch16、2e-4、AdamW/LLRD/warmup/poly/clip及默认FP16/compile保持原值。

服务器/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse，经SSH anhuiuniversity.icu；Python /Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python。每次Python保留HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1；日志WANDB_MODE=offline与seed0一致。

输出定位：outputs/innovation2/ccps_seed42/latest_path.txt；准备路径prepared_path.txt。正式run的requested_launch.json、config_comparison.json记录唯一科学差异和输出标识差异；parse print_config省略linked targets的问题仅在审计层按原main.py链接展开，未改原参数。启动后必须用真实run_manifest.resolved_cli_config再逐项比较。

seed0参考outputs/innovation2/ccps_plan/20261023T121754Z，训练用户主动停止于214轮、完整测试到213轮；43个完整候选epoch171–213、最佳187轮，无TTA Dice=86.242979%、HD95=29.744758mm。seed42保持220轮原预算，除了171–220完整结果，还必须按同样171–213窗口选最佳Dice并比较，以避免把候选数量差异混入seed差异。主对比均无TTA；不能拿seed0八视角86.978703%与seed42无TTA比较。HD95主mm并保留voxel，原legacy空掩码规则及核验器官映射不变。只报告两个seed差异，不把两次运行当作统计显著性证据。

用户同时要求两个单次检查：30分钟检查于2026-10-24 America/Denver 07:42:45（13:42:45 UTC），任务seed42-30；2小时兜底于09:14:42（15:14:42 UTC），任务seed42-2。两者都保留，不能互相替换。2小时检查若正常启动和持续训练则保持安静。异常时先只读排查进程、日志、资源；不自动重训、改参数或停止正常训练。如处理需修改现有代码或影响复现性，停止并报告。每条检查结束后暂停/删除自身，不建立长期监控，不影响训练或另一条任务。

当前为准备阶段，后续启动核验追加到本文档。原代码、YAML、先验、数据和历史输出保持，新增内容均为独立run监督/比较记录。


## 已正式启动并核验

输出outputs/innovation2/ccps_seed42/20261024T131734Z，2026-10-24 07:22:06 America/Denver（13:22:06 UTC）启动。训练PID=3814933、监督器3814884，CUDA:0 RTX5090D，GPU约19486MiB。原readiness及70份seed0训练源码哈希一致；真实run_manifest逐项比较只差seed与独立日志/输出标识，estimated_steps=30360，220轮、seed42、默认compile、16-mixed；未加载训练checkpoint。

本轮核验时已保存训练第4轮last.ckpt，global_step=552；完整测试在171轮开始前没有正式test结果属于预期。

run目录新增run_train.py作为独立监督器，复用原preflight与main.py；compare_seed_runs.py只在训练正常结束后CPU汇总，不参与模型/训练/预测。对比脚本已验证seed0的43轮/4128行及best187、逐器官/逐病例均值一致、匹配窗口排除214以及并列取更早轮。输出seed_comparison_summary.md/json、seed_comparison_organ_metrics.csv、seed_comparison_case_metrics.csv和seed42_full220_best_organs.csv。

2个定时任务均已创建ACTIVE：seed42-30（07:42:45）和seed42-2（09:14:42），2026-10-24 America/Denver，均只执行一次。第二次正常时静默；两次异常均不得改动影响复现性的代码/参数或自动重训。


## 用户明确早停及诊断（2026-10-24 America/Denver晚间）

已SIGINT正常退出，训练/监督器退出、GPU释放；状态stopped_by_user，原退出码1另存。last198、完整测试197、27轮324病例、best186 Dice84.090398%/HD9549.705577mm。未完成220轮，原171–213比较窗口不能使用；改按共同171–197的27候选比较，seed0最佳187为86.242979%，seed42下降2.152581pp。早停报告及证据见innovation2_seed42_early_stop_analysis.md和run内seed_gap_analysis.json。

3个异常病例全体积只读重放的48项Dice与原CSV一致，证实seed42在GT无器官切片上产生大量左肾误检；代码/配置/版本/优化器/划分/spacing/映射核验一致。没有重训或更改生产代码。30分钟和2小时单次任务均已完成并删除，不存在本run长期监控。
