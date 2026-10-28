# baseline差异：20分钟后一次分析任务

用户要求基于当前时间20分钟后生成解决方案，任务已创建ACTIVE heartbeat，ID=20-baseline。
基准：2026-10-26 21:53:48 America/Denver（2026-10-27 03:53:48 UTC）。
执行：2026-10-26 22:13:48 America/Denver（2026-10-27 04:13:48 UTC）。
本次仅一次，结束删除/暂停自身，不新增长期监控。

先读AGENTS.md和baseline_repeat_four_step_audit.md。当前诊断根outputs/audit/baseline_repeat_diagnosis/20261027T023508Z/。优先读取实际双入口12×8评分结果、初始化快照、固定Batch首步及随机点/梯度/优化器更新记录。核验正在运行的诊断身份和进度，不能重复启动或因时间到而停止。

生成.codex/baseline_repeat_solution_plan.md：区分已证实原因、已排除项、待验证项；每项修复注明是否改变baseline、最小回归、验收标准与顺序。未完成的GPU验证明确记录，不用CPU等价或grid_sample警告代替历史差值唯一归因。不改生产训练参数、不重训、不启动S3，保留S2早停和所有旧结果。通过SSH操作服务器，固定环境Python及三个离线前缀，资源按剩余显存规则。

本轮双入口评分由evaluate_same_checkpoint_entries.py old/roi运行，输出four_step/evaluation_old、evaluation_roi及evaluation_old.log/evaluation_roi.log。固定checkpoint=首轮best177，SHA321a085a076773c889af671081b7f9586d66b6ddf8b9ce446eaac8c2c9bbe7cf。以实际progress.json/dice96.json/退出记录为准，20分钟不是实验超时。

最新接续：ROI12例已退出0，96Dice与原baseline177在线CSV逐值零差。旧入口新评估还在运行，真实PID3805828/start77898979，SSH执行会话18717；其完整结果在four_step/evaluation_old/dice96.json，退出后运行compare_entry_evaluations.py严格比较两边及历史。ROI会话23851已确认0并保存exit.json。固定Batch诊断脚本first_step_entry_probe.py与batch已准备，CPU自测通过；运行GPU需要22000MiB（最低不能<21606），当前其他用户12392MiB仍在，不够则写未完成，不改batch/精度。首步脚本由/root/first_step_diagnostic准备，交付说明以first_step_*文件为准，不把CPU接口试验当GPU首步完成。

## 单次任务执行完成

已完成同checkpoint双入口96Dice严格比较并核验两侧退出0；已生成baseline_repeat_solution_plan.md及solution_plan_evidence.json。GPU首步资源仍不足，明确待验证。本任务完成后删除20-baseline，不另建监控。
