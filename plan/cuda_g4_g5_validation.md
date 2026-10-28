# 原v1 G4/G5整模型验收结果（历史证据）

**范围说明：本文保留原v1的失败矩阵；当前独立v2已通过G0–G5，最新结果见cuda_ccps_boundary_followup.md。不要把原v1失败改写为通过，也不要将本历史段落当作当前v2状态。**

所有本轮GPU诊断已结束；结论为**部分通过，整体未通过**。不再处于显存等待或后台执行阶段。

证据根：outputs/determinism/ce2d_stableseg_v1/full_model_20261027T121833Z/。最终汇总为final_validation_summary.json。matrix_status的stopped_on_unpassed_gate表示验收失败停止推进，不是仍在训练。原始失败及比较器初版日志保留。

| 模型 | G4固定Batch/实际AdamW | G5原生20 Batch | 结论 |
|---|---|---|---|
| 纯EoSeg baseline | 两独立冷启动，2 Batch/2次AdamW调用；首次LR0，第二次正LR，已记录边界一致 | 冷缓存对、暖缓存对、低观察器对，以及跨冷暖/跨观察器公共边界一致 | 通过本轮有界验收 |
| EoSeg+ASEE | 两独立进程各3 Batch，首批原生AMP跳步，随后2次AdamW调用；70个阶段一致 | 同上，全部通过 | 通过本轮有界验收 |
| EoSeg+ASEE+CCPS | 两GPU进程退出0，但首次反向scaled_gradients已出现真实有限差异，后续有效更新的梯度/moments/权重亦不同 | 按G4失败门槛未执行 | 未通过 |

原生20 Batch：baseline第3批跳步，19次AdamW调用含第一次LR0，18次正LR更新；ASEE第1批跳步，19次调用均正LR。不能把global_step/调用次数当作参数更新次数。没有取消跳步、手调scale或改变调度预算。

短轨迹通过不等于完整220轮可重复、Dice提升或跨环境确定性。重/低观察器最终RNG边界不同，交叉比较限于公共loss/LR/scaler/模型/optimizer；最终RNG由低观察器双进程另验。

## CCPS的已证实问题与候选

原始两进程初始化、RNG、固定输入、前向、随机点、Matcher及首批loss一致；首批反向有限梯度不同，后续实际更新继续分叉。部分上采样层梯度不同，而末端BN参数/末端卷积/预测头的首批梯度及BN运行统计一致。只能缩小到反向/编译执行边界，不能证明BN公式或CCPS数学设计有错。

诊断全部限定在ccps_isolation：
- 原真实criterion及65536缩放重放各路径30次一致，不能替代整网证明。
- 编译器确定性控制能稳定孤立CCPS模块，但整网固定Batch仍失败。
- 单独对齐backward autocast=off仍失败。
- autograd单线程的一对短测通过，但组合控制的一对仍失败，不能推广。
- 给loss输入加观察hook能使两次网络梯度一致，说明观察器会改变执行，不能把hook当修复。
- 最后一对将feature_upscale.10.forward作为局部eager边界，同时使用该实验的编译确定性及backward-autocast配置；外层compile仍开。两进程各3 Batch/2次AdamW、70阶段一致。**仅候选固定短测通过，尚无G5/低观察器验证，未接入生产。** 多条件共同变化，不能唯一归因BN。
- 一次辅助缩放criterion重放遗漏独立缓存变量，已在scaled_criterion_replay/cache_scope_note.json记录，不作协议放行证据；主验收使用独立缓存，未清理共享缓存。

## 比较器与CPU回归

旧诊断器错误假定首次成功AdamW调用必然LR0。现在支持原生首批AMP溢出跳过LR0，但必须核验GradScaler降scale及后续真实正LR参数变化。逐值相等标准没有放宽。ASEE初版误判日志保留。

新增scripts/validate_synapse_deterministic_trace.py；原probe仅CPU compare分支转用该验证器。GPU采集/前后向/优化器代码AST未改，证明见compare_branch_ast_proof_installed.json。全部40项CPU回归通过，见final_cpu_regression.log。原101份源码/配置及4份独立数值运行文件hash保持，见final_source_protection.json。

低观察器暖态batch墙时中位：baseline约0.289–0.291秒、ASEE约0.295–0.296秒；allocated峰值约17713/18045 MiB，reserved约20124/20448 MiB。仅20 Batch，不能外推完整训练评估总时长。重度存档的数百秒主要含CPU拷贝/hash，不代表训练吞吐。

## G6条件及文件变更

readiness继续false，validated_profiles为空，不改正式入口的全门槛保护。用户条件授权通过后两次完整纯baseline重复可作为G6，并安排3/3.5/8/8.5小时四次启动/检查。当前矩阵未全部通过，按“未通过则完善Plan”执行：**未创建四任务、未启动G6。** 若只放行baseline，须明确缩小验收范围并另留scope，不能静默抹去CCPS失败。

下一步见cuda_solution_plan.md第14节。原baseline模型/科学参数/增强/损失、生产CCPS、历史结果、mm-only HD95与legacy规则不改。未停止其他用户进程。

本次接续新增低观察器、CPU轨迹比较器、两份测试和本说明；修改独立probe的CPU比较分支、LR0合成测试夹具、Plan、实现记录和AGENTS。outputs保存全部诊断产物。删除：无。
