# G6：同seed纯baseline完整重复与四次单次任务

## 当前执行状态（2026-10-28用户最新要求）

用户明确要求立即开始第一次G6，并将输出命名为baseline-repair；本次只验证纯EoSeg baseline，ASEE/CCPS均不开启。实际已于UTC 2026-10-28T07:25:39.982155+00:00 启动，当前run为outputs/baseline-repair/G6_1/20261028T072539Z，唯一回执outputs/baseline-repair/repeat_1.json，latest指针outputs/baseline-repair/latest_path.txt。监督器PID 3107856，训练PID以该run/launch.json现场核验，不能只凭本段旧PID操作。

当前输出根目录必须使用outputs/baseline-repair；scripts/launch_synapse_g6.py仅改输出根与离线日志名称，8项CPU启动保护通过，其余源代码逐字不变。v2 baseline YAML、模型、数据、seed0、原增强、优化器/调度、220轮预算及评估规则未改。启动前现场GPU剩余32092MiB，实际5090D/torch2.13.0+cu132/CUDA13.2与原预训练SHA已核验。

原计划01:26的首次启动已由本次立即启动取代；不得为旧计划再开一次repeat1。已存在运行时后续只检查。第二轮仍须第一轮完整220轮及50轮测试完成后才能启动。旧定时任务g6-baseline-3h的更新接口返回任务不存在；本轮没有重建定时任务，也没有更改其余定时安排。下方旧路径/未启动/ACTIVE条款为此前记录，以本段、当前回执和应用实际任务状态为准。

本轮v2真实入口G0–G5及启动器准备已通过，见outputs/determinism/ce2d_stableseg_execution_v2/readiness.json和outputs/determinism/g6_baseline_execution_v2/preparation.json。原v1的CCPS失败证据不覆盖；v2 baseline/ASEE执行与v1数值路径相同，只有CCPS应用局部BN eager边界及backward_pass_autocast=off。G6仅跑纯baseline，不启用ASEE/CCPS。

## 时间与授权

用户已授权通过后：3h第一次baseline G6、3.5h启动检查、8h第二次G6、8.5h检查。此前失败分支未建任务，本轮全部验收通过后按当前时间落实。基准2026-10-27 22:26:00 America/Denver（2026-10-28 04:26:00 UTC）。

|任务|Denver|UTC|
|---|---|---|
|G6-1启动|10月28日01:26|10月28日07:26|
|G6-1检查/未启动兜底|10月28日01:56|10月28日07:56|
|G6-2启动|10月28日06:26|10月28日12:26|
|G6-2检查/未启动兜底|10月28日06:56|10月28日12:56|

四个任务只各执行一次，不额外轮询。第一轮未完整结束时保持运行，第二轮不并发；8.5h仍未满足条件则如实报告，不自动增加下一次任务。

## 服务器与不变量

通过SSH anhuiuniversity.icu操作/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse。Python固定/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python。每次Python保留HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1；独立协议还需CUBLAS_WORKSPACE_CONFIG=:4096:8、PYTHONHASHSEED=0、OMP_NUM_THREADS=1。不用本地PowerShell/Python执行训练。

配置configs/synapse/deterministic_v2/baseline.yaml；入口scripts/run_synapse_deterministic_v2.py。纯EoSeg、seed0、原增强、固定18/12病例、384/patch16、batch16/workers8、220轮/30360步、原AdamW/LR/LLRD/预热/损失系数/裁剪/FP16/默认outer compile。计算协议含已验证的2D CE和稳定采样反向；不是未修复的历史入口。HD95仅真实mm，legacy空mask不改。fit为1-view，171–220轮完整12例选同一best。不自动加TTA或创新模块。

## 幂等执行

只通过scripts/launch_synapse_g6.py --repeat 1或--repeat 2启动。检查用同脚本--check。脚本自己设置完整训练环境、独立缓存和离线W&B日志；formal fit的probe_batches=0，禁止把冒烟2/100 Batch覆盖带入正式训练。

输出outputs/determinism/g6_baseline_execution_v2/；repeat_1.json/repeat_2.json为唯一回执，G6_1/G6_2子目录保存独立run。队列锁由监督器持有到训练及汇总完成。已有回执时只能核查，不能重复启动；失败、用户停止或监督器异常不自动覆盖/重训，不从weights-only冒充精确续训。

启动前检查v2 readiness及当前源码hash、G6 preparation和启动器SHA、解释器/torch2.13.0+cu132/CUDA13.2/5090D、原预训练SHA36e4deff…、磁盘>=30GiB和现场GPU余量>=22000MiB。允许其他用户任务并发，显存足够即可；不足等待/报告，绝不中止他人进程，不减batch或更改参数。

检查回执、status.json、launch.json、真实training PID/start_ticks/cmdline/cwd、GPU、train.log、startup_verification.json、run_manifest、滚动last。阶段没有171轮测试属于预期。默认compile断图/原AMP溢出跳步不能单凭警告判失败。

## 完整标准与对照

退出0只是必要条件。还需last.ckpt epoch219/global_step30360、171–220全50轮、600病例及4800病例器官行，逐轮12病例×8类完整、best来源及epoch正确。不以某次完整测试或last文件存在冒充220轮完成。

监督器保存初始状态/RNG、逐器官同best Dice及HD95_mm和平均、best/last状态内容hash及文件来源SHA。两次均完成后comparison.json记录初始状态/RNG一致性、best epoch、best/last内容一致性、全部逐轮/逐病例CSV，以及各器官Dice百分点和HD95_mm差值。器官名来自priors/synapse_label_map.verified.json，背景不列，不跨epoch拼最优。

任务结束暂停/删除自身，保留其他三条任务与正常训练。正常检查无变化保持安静；完成、启动、失败或需决定时通知。定时检查时间不是训练超时，不在检查点停止正常训练。

## 实际任务登记

四个线程单次任务均已创建并核对为ACTIVE，分别为g6-baseline-3h、g6-3-5h、g6-baseline-8h、g6-8-5h。当前未启动正式G6。具体日期见上表；每条仅一次，完成后暂停/删除自身。

启动后实际推进核验：截至2026-10-28T07:36:57.659160+00:00，滚动last.ckpt已完整保存7轮/966step，训练PID=3107921仍匹配starttime/cmdline/cwd；初始388项权重与已验收baseline一致，query/class_head/layer_gate均已发生有限数值更新。ASEE=False、CCPS=False。证据为当前run/startup_training_verification.json；此轮数为本次核验快照，不作为后续实时状态。
