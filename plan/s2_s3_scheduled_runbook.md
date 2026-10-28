# S2/S3：四次单次定时执行

用户于2026-10-26明确授权顺序执行S2、S3，均启用ROI。共同基准为2026-10-26 07:10:19 America/Denver（13:10:19 UTC）。四个任务均已创建ACTIVE heartbeat，COUNT=1，附着本对话；执行结束暂停/删除自身，不新增长期监控。

| 任务 | 相对基准 | Denver时间（2026-10-26，MDT） | UTC | automation ID |
|---|---|---|---|---|
| baseline-2完成后启动S2 | +5h | 12:10:19 | 18:10:19 | s2-5 |
| S2启动兜底 | +5.5h | 12:40:19 | 18:40:19 | s2-5-5 |
| S2完成后启动S3 | +10h | 17:10:19 | 23:10:19 | s3-10 |
| S3启动兜底 | +10.5h | 17:40:19 | 23:40:19 | s3-10-5 |

## 服务器与当前前序

通过SSH anhuiuniversity.icu访问`/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse`。Python=`/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python`；每次Python使用HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1。SSHFS/本地PowerShell不代表服务器进程。正式任务不能继承CPU验证的空CUDA_VISIBLE_DEVICES。

前序固定为新ROI baseline-2：`outputs/baseline-2/20261026T130228Z`，不是已删除的旧run。完整220轮、退出0、最后50轮171–220全12例、4800病例器官行及best/last来源核验后，才允许S2。训练仍运行则保留，禁止为定时任务提前停止或从weights-only假续训。

## 模型和不变量

- S2：`configs/synapse/ablations_journal/s2_asee_probability_only.yaml`，use_text=false/use_probability=true，enhance_patch=true/enhance_query=true。
- S3：`configs/synapse/ablations_journal/s3_asee_patch_only.yaml`，enhance_patch=true/enhance_query=false，use_text=true/use_probability=true。
- 两组均为纯EoSeg+ASEE组件实验，不启CCPS。原增强、seed0、384/patch16、batch16/workers8、220轮、AdamW/LR/LLRD/原调度/损失/FP16/default compile与各原对照保持；不恢复已删除旋转/尺度增强。
- 从原离线timm预训练初始化，不从baseline/S1/完整模型best微调。训练入口scripts/run_synapse_roi.py，fit固定1-view；171–220完整12例按原legacy Dice选同best，随后同best固定scale_hflip_8（1/.875/1.125/.75×原方向/Hflip）最终测试。HD95真实mm+voxel，verified名称，不整理背景。

## 独立启动器和实际命令

使用`scripts/launch_synapse_s2_s3_scheduled.py`；原scripts/launch_synapse_journal_component.py及当前baseline-2冻结依赖未修改。新入口只允许S2/S3，不会意外启动S4。

启动S2：
```bash
HF_HUB_OFFLINE=1 TORCH_HOME=/Disk/HDD/Sk/.cache/torch TIMM_USE_OLD_CACHE=1 /Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python scripts/launch_synapse_s2_s3_scheduled.py --next --expected-experiment S2
```
启动S3：同命令改为`--expected-experiment S3`；若S2尚未完整完成，入口拒绝跳过。

若S2或S3训练已完整完成、状态为training_completed_tta_deferred，只补同run的best测试：
```bash
HF_HUB_OFFLINE=1 TORCH_HOME=/Disk/HDD/Sk/.cache/torch TIMM_USE_OLD_CACHE=1 /Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python scripts/launch_synapse_s2_s3_scheduled.py --resume-test 该run的完整路径 --expected-experiment S2
```
S3对应改为S3。不从weights-only继续fit，不把训练完成但最终8-view未完成记成实验完成。

共享队列锁、幂等receipt、PID启动时间和launch intent沿用原机制。先读`outputs/ablations/journal_20261025/experiment_registry.json`和orchestration_state.json。已有run运行只核验；已有失败run不自动覆盖/重训。结果仍放同根目录的S2/S3独立时间戳子目录，完整完成后更新消融Plan第11节。

## 显存与顺序

按最新server_gpu_execution_rules.md允许与其他用户并发；检查可用显存，而不是要求无其他进程。训练预算24576MiB（同类训练约20GiB进程峰值加余量），最终8-view预算6144MiB（同类实测约3.2GiB加余量）。不足等待，不中止其他用户、不降batch或改科学设置。既定前序和本项目队列仍按顺序，不同时启动S2和S3。

第1/2次：baseline-2完整结束后，S2没有正式run才启动；S2正常则不重复。第3/4次：S2训练及最终8-view完整后启动S3；S2只差TTA先完成测试。若较早任务因额度等未执行、S2仍缺失但baseline-2已完整结束，可先补S2，不能跳到S3。前序未结束时保持运行；最后检查时尚在正常训练的任务继续，不因检查窗口结束停止。

仅在完成、新启动、失败或需用户决定时报告。四次以外不自行新建长期任务，不自动启动S4/A1/A2/A10/A11。

## 准备验收

11项CPU测试、CLI参数解析通过；缺失/重复病例、未完整220轮、失败退出、S3跳过S2、显存不足及越权S4均有拒绝测试。两份配置的原科学值CPU核对通过（准备阶段仅模拟前序完成，真实执行门槛没有绕过）。8个训练命令/验证/结果汇总函数AST与原启动器完全一致；GPU资源准入和前序编排为明确差异。未启动S2/S3或新GPU冒烟。

证据：`outputs/audit/s2_s3_scheduled_preparation/20261026T132246Z/validation.json`。每次执行仍核验readiness=passed及配置/代码/先验/数据清单哈希；准备不是正式训练完成证明。

## 2026-10-26 17:10检查执行结果

前两次检查时baseline-2仍训练，本次核验其已于19:01:35 UTC完整结束220轮、退出0，171–220共50轮/600病例/4800病例器官行齐全，best173的1-view平均Dice=85.913756%。已通过独立启动器完整前置/哈希校验，补启动S2：outputs/ablations/journal_20261025/S2/20261026T231139Z，训练PID3559687，监督器3559541。实际解析为共现单分支、patch/query开启、原增强/seed0/220轮/FP16/default compile；ROI已绑定。S2正在训练，未启动S3。检查凭证outputs/audit/s2_s3_scheduled_checks/s3-10/check.json；实际PID以后续现场核验为准。

## 2026-10-26 17:40最后一次检查

S2实际训练进程3559687及启动时间/命令/cwd核验一致，已保存完成36轮、4968 step；GPU占用19304MiB。运行源码与ROI readiness哈希均一致、passed。S2尚未完成220轮及最终8-view，故S3未启动，保持S2正常运行。本次为最后一条定时检查，执行后删除s3-10-5，不新增长期监控；S2监督器仍负责本run训练后同best8-view，不自动启动S3。S3继续待执行，需后续检查/指令推进。凭证outputs/audit/s2_s3_scheduled_checks/s3-10-5/check.json。未修改模型/训练源码、配置、数据或参数。

## S2用户主动早停

用户随后明确要求早停S2。已核验实际PID并SIGINT温和停止，训练、监督器及所属子进程均退出；最后保存训练211轮/29118step，完整12病例测试到210轮，共40轮/480病例/3840病例器官行。最佳195轮1-view平均Dice=85.214482%。保留全部权重/CSV/manifest/log及真实退出码1，原监督器failed状态归档，当前状态stopped_by_user。未完成220轮及最终8-view，不自动启动S3。详见该run/early_stop_report.json。
