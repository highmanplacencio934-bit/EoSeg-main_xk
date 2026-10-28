# CUDA确定性独立实现与验收记录

当前新增独立v2已通过G0–G5，详见cuda_ccps_boundary_followup.md；以下保留原v1实施历史。

更新：2026-10-27。协议：`ce2d_stableseg_v1`。

本轮已按用户授权实现独立CE/采样/CLI/配置，并完成G0的CPU初始化边界、G1–G3服务器GPU验收。**最新：baseline/ASEE有界G4/G5通过，CCPS G4失败、G5未执行，整体未通过。本轮GPU诊断均结束，最新见文末及cuda_g4_g5_validation.md。** 未启动220轮训练、S3、消融或新定时任务。不能据此保证最终Dice或跨环境逐位一致。

主证据：`outputs/determinism/ce2d_stableseg_v1/20261027T082449Z/`；独立门槛：`outputs/determinism/ce2d_stableseg_v1/readiness.json`，当前passed=false。

## 实际修改与不变量

- CE将原B×Q×10分类logits直接reshape为(BQ)×10，保留weighted mean、原target构建/匹配/no-object9权重.1及外围class系数2。背景仍0。目标函数不变，浮点归约路径改变；不宣称与旧CUDA loss逐位相同。
- 点采样保留原native bilinear/zeros/align_corners=False前向、原坐标映射与AMP策略。反向对四邻域贡献按目标像素stable-sort，再做固定分段树形scan，每个像素唯一写入；不用重复索引原子累加，也不用全局cumsum相减。
- Matcher、uncertainty、GT及最终预测采样均使用独立局部依赖；保留随机draw顺序、top-k、最终坐标和原mask/Dice归一化。显式拒绝坐标requires_grad，当前训练本来不需要坐标梯度。不改site-packages或全局HF函数。
- criterion和采样各保留显式eager边界；原criterion本就compiler.disable。独立入口仍使用默认外层torch.compile、FP16和matmul medium，不以关闭compile/改FP32通过测试。
- 模型、数据划分/标签、原随机增强、seed0、batch16/workers8、384/patch16、220轮预算/30360原预估步、损失参数、AdamW/LLRD/原预热/裁剪均保持。三个独立YAML仅换协议开关、输出/日志路径及显式保持HD95_voxel=false。
- strict=True/warn_only=False，cuDNN deterministic=True/benchmark=False；CUBLAS_WORKSPACE_CONFIG=:4096:8和PYTHONHASHSEED=0在Python启动前设置。这些是新计算协议的显式数值差异，不偷偷写入旧入口。
- 原baseline/ROI源码与原YAML保持；对开工快照中100份既有文件逐项核验，0修改/0缺失。旧权重/结果/旧readiness不覆盖。关闭独立协议后，真实入口初始化、RNG及优化器也与原ROI一致。

## 真实验收结果

| 阶段 | 状态 | 实证 |
|---|---|---|
| G0 | CPU边界通过；GPU首步边界待G4 | 三模型各两个独立进程走真实入口，初始化/RNG/全部优化器组一致；所有进程退出0 |
| G1 | 通过 | 8种CE组合（FP32/FP16、eager/默认compile、含匹配/全no-object），每种30次loss与梯度hash完全一致，strict通过 |
| G2 | 通过 | 8种12544点采样组合（FP32/FP16、eager/默认compile、NP2/NHW2、非连续/重复/边界/越界），每种30次前向与梯度完全一致；native前向逐位相同，CPU double参考及gradcheck通过 |
| G3 | 通过 | 重放真实batch16保存的criterion输入，eager和compile包装各30次；匹配、最终坐标、全部局部采样、RNG、分项loss及输入梯度严格一致；旧新数学差异在事先落盘门槛内 |
| G4 | baseline/ASEE通过；CCPS失败 | CCPS初始化/输入/loss一致，有限梯度分叉，后续真实更新也不同 |
| G5 | baseline/ASEE通过；CCPS未执行 | 两组20原生Batch冷/暖缓存、低观察器及交叉公共边界一致；CCPS由G4失败阻止 |
| G6 | 条件授权未触发 | G4/G5全部通过后两次完整baseline重复；当前未启动或设置定时任务 |

G0：baseline为388项状态/373组；ASEE为419项状态/401组；ASEE+CCPS为433项状态/415组。创新模型含非张量extra_state，按内容核验，不能把所有项都称为Tensor。CPU入口没有前反向，Lightning提示CPU precision回落不能当作FP16训练通过；GPU FP16证明来自G1–G3。

CPU回归共30项通过（含验证器本身拒绝不同轨迹/不完整或无真实更新的门槛）。CPU数学报告另存`cpu/cpu_validation.json`；三组对照在`g0/comparison.json`。首次ASEE诊断遇到extra_state序列化错误，仅修正独立诊断器，保留失败日志；未改ASEE模型。

GPU环境：5090 / RTX5090D / 驱动580.178.04 / torch2.13.0+cu132 / runtime13.2，timm1.0.29、Transformers5.17.0。默认timm旧缓存文件`/Disk/HDD/Sk/.cache/torch/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth`，SHA256=`36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51`；本轮不改变原patch16适配。

阈值在各次实验前写入validation_thresholds.json，没有事后放宽。CUDA G1/G2峰值allocated约96MiB、reserved128MiB；G3约415MiB/644MiB。这些仅为loss/算子占用，不能当作完整ViT训练峰值。

## 初始性能与资源记录（历史阶段，最新见文末）

同一真实输入，去掉逐张量hash观察器后，3次warmup+30次CUDA Event计时：原criterion中位28.4465ms，候选45.0091ms，约1.5822倍（增加16.5626ms）。这是损失前反向单项，**不是整场训练慢58%**；整模型编译/显存/吞吐尚待G4/G5测量。详见criterion_performance.json。

G4/G5现场剩余18192MiB，原配置安全门槛22000MiB（最低保护21606MiB），差3808MiB。其他任务仍运行；没有停止任何其他进程、减batch、关compile或改精度。完整模型验证未开启。现场值会变化，下次必须重新查询，不把此值当作长期可用量。

## 初始执行模板与保护（接续状态见文末）

独立入口：`scripts/run_synapse_deterministic.py`。正式fit（probe_batches=0）当前会主动拒绝，因为readiness未passed；该保护已真实测试，退出1且未开始训练。`--deterministic_enabled=false`恢复原criterion/flags路径，仅供受控对照，不自动代表新的正式训练授权。

资源足够后，先用`probe_synapse_deterministic_entry.py`准备固定Batch，再运行两次独立candidate固定Batch/两次原生Batch短轨迹；G5用`--batch-target=20`。使用全新输出tag，原220轮/30360估步保持，仅Callback在目标后结束；不要改max_steps/limit_train_batches来重写调度。采集Loss、scaled/unscaled/clipped梯度、原生GradScaler跳步、AdamW真实调用/LR/moments/参数、输入/点采样/RNG；比较器遇到差异返回非零。冷缓存两次分别用独立目录，暖缓存两次使用同组已热缓存；不清理共享缓存。最后再补低侵入全模型复核，才能考虑readiness passed。

服务器命令始终使用：

```bash
cd /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse
HF_HUB_OFFLINE=1 TORCH_HOME=/Disk/HDD/Sk/.cache/torch TIMM_USE_OLD_CACHE=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 WANDB_MODE=offline \
TORCHINDUCTOR_CACHE_DIR=<本次独立缓存绝对路径>/inductor \
TRITON_CACHE_DIR=<本次独立缓存绝对路径>/triton \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python \
scripts/probe_synapse_deterministic_entry.py deterministic \
--profile baseline --output-dir <本次独立证据目录> --tag <全新tag> \
--target-updates 2 --max-attempts 16
```

以上是待资源满足后的模板，不是本轮已执行GPU整模型命令。固定Batch先用CPU的prepare模式，并在采集时加--fixed-batch；native阶段不加。GPU命令不能继承CPU验证的CUDA_VISIBLE_DEVICES空字符串。ASEE/CCPS也须分别验收，不能仅凭CPU初始化通过就标为完整validated_profiles。

## 本轮文件变更清单

新增：
- `training/deterministic_mask_loss.py`
- `training/deterministic_point_sampling.py`
- `training/deterministic_protocol.py`
- `scripts/run_synapse_deterministic.py`
- `scripts/validate_synapse_deterministic.py`
- `scripts/probe_synapse_deterministic_entry.py`
- `configs/synapse/deterministic/baseline.yaml`
- `configs/synapse/deterministic/asee.yaml`
- `configs/synapse/deterministic/ccps.yaml`
- `tests/test_deterministic_mask_loss.py`
- `tests/test_deterministic_point_sampling.py`
- `tests/test_deterministic_protocol.py`
- `.codex/cuda_deterministic_implementation.md`

修改：
- `.codex/cuda_solution_plan.md`：同步实际实施、测试与尚未完成阶段。
- `AGENTS.md`：追加独立确定性协议和当前未验收限制。

新增产物：上述outputs/determinism证据根、独立readiness、数学阈值/报告、初始化对照/日志/命令/退出码、性能、资源与源码保护记录。失败/初版诊断证据保留。删除文件：无。


## 接续G4/G5最终状态及本轮变更

所有有界GPU诊断结束，baseline/ASEE通过；CCPS失败。局部末端BN编译边界候选仅固定双进程短测通过，尚无G5，未采用。readiness仍false，40项CPU回归通过。原101份既有源码和4份独立数值运行文件hash保持；probe仅CPU compare分支改动，GPU producer AST相同。详见cuda_g4_g5_validation.md、Plan第14节与outputs下final_validation_summary.json。

新增：scripts/probe_synapse_deterministic_low_observer.py、scripts/validate_synapse_deterministic_trace.py、tests/test_deterministic_low_observer.py（4项）、tests/test_deterministic_amp_trace_gate.py（6项）、.codex/cuda_g4_g5_validation.md。
修改：独立probe的CPU比较分支、tests/test_deterministic_protocol.py的合成LR0夹具、Plan、本记录和AGENTS；保存全部独立outputs证据。删除：无。无科学训练参数变化。

低观察器暖态batch中位墙时baseline约0.290秒、ASEE约0.295秒；allocated峰值约17713/18045 MiB、reserved约20124/20448 MiB。不能将重度逐梯度hash耗时或20 Batch外推为完整训练耗时。

G6已获有条件授权，但CCPS未通过整体门槛，因此按用户失败分支完善Plan，未创建3/3.5/8/8.5h四任务，未启动完整训练，不恢复S2/S3或其他消融。


## 独立execution_v2已完成接续

新增独立执行模块、CLI和三配置，仅CCPS采用末端BN局部eager边界与AOT backward autocast=off；baseline/ASEE与v1数值轨迹逐值相同。原v1数学实现/入口不改。实际v2 G0–G5、46项CPU、启动器8项CPU保护和真实两Batch离线日志冒烟均通过。原101份文件保留；完整220轮重复性仍待G6。

四个单次G6任务按Denver10月28日01:26/01:56/06:26/06:56建立，未立即启动长训，详见synapse_g6_runbook.md。源代码/配置及数据在正式run期间保持冻结；不热切换CCPS控制到原main/ROI或其他创新入口。
