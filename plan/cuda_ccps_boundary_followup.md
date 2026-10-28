# CCPS编译边界接续验证

## 后续CCPS训练入口（当前规则，2026-10-28）

后续已获用户授权的CCPS主模型训练，固定使用以下组合：

|项目|必须使用|
|---|---|
|训练入口|scripts/run_synapse_deterministic_v2.py fit|
|主模型配置|configs/synapse/deterministic_v2/ccps.yaml|
|协议|deterministic_protocol=ce2d_stableseg_execution_v2|
|模型profile|deterministic_profile=ccps|
|启动门槛|outputs/determinism/ce2d_stableseg_execution_v2/readiness.json passed、ccps已验证及当前源码hash匹配|

旧main/ROI/v1入口与旧CCPS启动器不包含这套完整组合修复，不用于后续新CCPS训练；它们保留作历史对照。组件消融需要保持对应科学开关并准备、核验独立v2配置，不能直接把主配置当作全部消融配置。此处只是明确后续执行入口，不启动任何训练。

v2已通过有界G0–G5；CCPS自身220轮重复性尚未完成，纯baseline G6不替代它。下文含早期pending/未安装等阶段日志，它们属于历史状态，以本节及“最终结论与真实入口验收”为准。

用户已授权继续定位与测试，当前仅隔离P1–P5，未启动正式训练。证据根：outputs/determinism/ccps_boundary_followup/20261028T030813Z/。

状态读取status.json、p1_results.json、各launch/exit、comparison及最终summary，不能把进程退出0当作数值通过。原失败轨迹重放仍为红；没有覆盖旧矩阵和readiness。

P1优先只加末端BN局部边界，保留原编译/反向假设；每候选三独立冷进程。必要时逐项增加backward-autocast对齐与编译器确定性控制，单变量递进。首个三进程一致的候选仍须P2数学/状态/RNG核验、G5冷暖缓存及低观察器验证，不能直接放行。

P2 CPU数学对照通过：FP32/double各3次训练态forward/backward，输出、输入/参数梯度、BN统计与num_batches_tracked精确一致。只证明边界包装不改变这些CPU语义，不能替代真实GPU验证。包装不改参数/Buffer ID、state键或RNG；GPU实际入口也逐次记录。

本轮仅新增独立outputs诊断包装器/控制器/证据与本文档，原模型、训练参数、生产入口和历史结果不变。测试结束再据实汇总，readiness维持false。


阶段进展：仅末端BN边界的三次冷进程中，a/b一致但c首批反向分叉，不能视为修复。首次梯度hash差异集中在selector的6个参数；存档的解码器原始梯度在该边界一致，后续全局裁剪后出现真实有限差异。BN局部边界+backward autocast=off（未加编译确定性flag）三独立冷进程均一致，已进入原生20 Batch低观察器冷/暖缓存复核。低观察器冷进程对已通过。后续以最终状态为准。

候选相对旧v1首批Loss有约1.4e-4变化，说明局部边界改变了浮点执行路径；数学目标、参数和训练变量保持，不冒称旧新逐位相同。若最终采用，须独立协议版本与匹配对照，不热替换历史baseline。


反向设置的依据：安装Lightning实际回调已记录backward不在autocast中（旧ccps_backward_off_a.log）；安装PyTorch默认backward_pass_autocast=same_as_forward。官方建议实际backward位于autocast外时对compiled region设off，见https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_backward.html。该来源只支持执行上下文对齐，不证明它单独消除了整网非确定性。

原生20 Batch低观察器冷对、暖对和冷暖交叉均通过，包含完整初末状态/RNG/scheduler/optimizer及loss序列。逐Batch完整梯度冷/暖对仍在运行，不能提前登记候选完整通过。可恢复执行控制已在独立outputs内准备并通过6项CPU行为检查，尚未安装到生产。


候选最终验收：固定3冷进程、原生20 Batch完整梯度冷对/暖对、低观察器冷对/暖对、冷暖交叉和重/低公共边界全部通过。局部GPU BN对CPU double参照及30次重复通过预先阈值。单独off的3进程仍失败，单独BN边界也失败。组合候选没有启用额外编译确定性flag，outer compile/FP16及原科学变量保持。

已新增独立v2：training/deterministic_execution_v2.py、scripts/run_synapse_deterministic_v2.py、configs/synapse/deterministic_v2三份YAML和tests/test_deterministic_execution_v2.py。旧文件未替换。46项CPU回归通过。v2仅CCPS安装局部BN边界与off上下文；baseline/ASEE保持v1执行，无新参数/Buffer或RNG消耗。v2正式readiness仍false，真实入口初始化/GPU短轨迹正按entry_verification_v2/run_entry_checks.py验证。

本节是阶段状态，不能据候选通过就宣称真实v2已全通过或已启动G6。


## 最终结论与真实入口验收

本轮接续已完成。独立v2的G0–G5已passed，baseline/ASEE/CCPS均列入validated_profiles。真实v2 CPU初始化双进程的全部状态、RNG及优化器组与各自v1相同；baseline/ASEE的原生20 Batch数值轨迹逐值匹配既有v1，CCPS实际入口固定双进程及20 Batch冷/暖低观察器逐值匹配本轮已完整验证的候选。见entry_verification_v2/actual_entry_summary.json。G1–G3复用的是未修改、同环境的原确定性CE/采样实现及原数学验收，不将元数据改名当作算子重测。

这是有界确定性验收，不证明完整220轮或跨环境逐位一致。单独BN边界失败、单独off失败、两者组合在本轮全矩阵通过，只证明该组合是当前已验证的有效方案；尚不能唯一锁定某个CUDA内核的所有因果机制。未改CCPS数学公式、共享结构、梯度目标或训练变量。局部边界改变浮点执行路径，旧新首批Loss不逐位相同，故采用独立版本，不覆盖旧结果。

## G6准备与定时

G6启动器8项CPU保护通过；v2全部46项CPU回归通过。启用默认离线W&B、FP16/compile/batch16/workers8及220/30360调度的2 Batch真实GPU冒烟退出0，初始388项权重与验证baseline一致，373优化器组、CUDA及compile均确认，last.ckpt为epoch0/global_step2。首版启动回调误用_orig_mod识别编译而失败，已依据Lightning.from_compiled的_compiler_ctx修正并加入回归；失败日志保留，未改训练编译或模型。

用户已条件授权的四个单次任务现已创建，基准Denver2026-10-27 22:26：10月28日01:26首次G6、01:56检查、06:26第二次G6、06:56检查。详见synapse_g6_runbook.md。正式G6尚未开始；两次完整纯baseline重复将检验全220轮差异，第一轮未完整结束不能开始第二轮。只使用新独立协议，不恢复S2/S3或其他消融/TTA。

## 本轮文件变更

新增10份：
- training/deterministic_execution_v2.py：按模型profile隔离执行控制、状态记录及readiness保护。
- scripts/run_synapse_deterministic_v2.py：独立入口；baseline/ASEE保留v1数值路径，仅CCPS应用组合控制。
- configs/synapse/deterministic_v2/baseline.yaml：纯baseline独立配置。
- configs/synapse/deterministic_v2/asee.yaml：ASEE独立配置。
- configs/synapse/deterministic_v2/ccps.yaml：CCPS独立配置。
- tests/test_deterministic_execution_v2.py：6项CPU行为与门槛回归。
- scripts/launch_synapse_g6.py：两次G6队列/幂等/监督/完整覆盖/分器官汇总与差值。
- tests/test_synapse_g6_launcher.py：8项启动器保护与编译识别回归。
- .codex/cuda_ccps_boundary_followup.md：本轮实证、方案、适用范围与变更清单。
- .codex/synapse_g6_runbook.md：四个单次任务、运行参数和异常边界。

修改4份：
- .codex/cuda_solution_plan.md：同步v2实际通过与G6安排。
- .codex/cuda_g4_g5_validation.md：标明原v1历史范围，链接当前v2结果。
- .codex/cuda_deterministic_implementation.md：增加v2实现与验收状态。
- AGENTS.md：记录当前入口、验证范围、冻结规则和G6授权落实。

新增产物：本轮独立outputs诊断包装器/对照/日志/梯度/缓存/数学参照、v2 readiness、G6 preparation与schedule账本。8项回归验证启动器防重复/失败不重训/前后顺序/实际PID/轮次覆盖，未模拟正式成绩。四个任务通过应用工具创建，未手改任务配置。原101份源码/配置及v1数值实现保持。删除：无。
