# Innovation2 CCPS 执行记录

用户授权：修复五项审查问题并解释两项实验因素；两小时后启动主训练，五小时后检查并在未启动时补启动。2026-10-23 America/Denver：06:17启动，09:18:50保障检查（UTC 12:17、15:18:50）。任务ID innovation2-ccps、innovation2。当前创建的是Codex线程定时任务，触发时仍需应用/连接/模型额度可用；第二个任务可在首次未执行时补启动，但不能承诺绕过额度限制。

## 主配置与不变量

configs/synapse/vit_query_mul_scale_fusion_asee_ccps.yaml：visual_prior，patch_gate=true、query_gate=false、presence=0、init=.95、hidden=256、patch_gate_effect=both。ASEE text/probability双先验及patch/query增强保留。原离线timm预训练初始化；无innovation1 checkpoint微调。

保留220epochs、batch16、workers8、384/patch16、16query、9类、seed0、16-mixed、默认torch.compile、AdamW 2e-4/WD.05/LLRD.8、500/1000warmup、poly.9、normclip.01和原损失/增强/18+12划分。171–220轮完整12例测试，mean Dice选择best，last滚动。HD95主mm并保留voxel，原空掩码规则及器官映射保持。

## 修复与实验解释

五项修复和两处机制解释详见根目录EoSeg_CCPS_Implementation_Plan.md第9节。参考代码包保持原始材料，修复后的生产文件在根目录models/training/scripts/tests/configs下，不要重新复制原包覆盖。

CPU初步98项全量回归通过（其中22项CCPS）。冒烟第一次仅因隔离审计回调误读不存在的data.batch_size属性停止，未开始训练step；已修复审计回调，保留gpu_smoke_attempt1.log。当前GPU冒烟还在验证，最终状态以outputs/innovation2/ccps_readiness.json为准。禁止将pending当passed。

审计目录 outputs/audit/ccps_integration_20261023T101700Z：保护文件哈希、CPU日志、隔离GPU冒烟、完整解析配置及manifest。冒烟只运行1epoch/2batch、logger=false与审计callback，未将覆盖写回主配置，不作正式成绩。

## 定时启动

服务器/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse，SSH anhuiuniversity.icu，Python /Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python，必须保留HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1。

执行 scripts/launch_innovation2_ccps.py（无参数）只启动一次；--check-only仅检查。启动前验证readiness=passed及源码/配置/先验/清单哈希，hostname=5090、解释器、CUDA、空闲GPU至少28000MiB与磁盘15GiB。若不满足直接报错，不改训练参数。不使用新定时器或shell延迟代替现有两个Codex任务。

正式输出 outputs/innovation2/ccps_plan/<UTC时间>，latest_path.txt定位。fcntl文件锁+launch_receipt.json保证两个定时任务不重复启动；已有失败run也不会自动重训。监督器记录status.json、launch.json、train.log和退出码。checkpoint/CSV/manifest位于其synapse_test_selection子目录，named物理报告必须按label_id关联器官名。脚本不会安排长期监控。


## 最终验证通过

最终全套CPU回归101/101通过（25项CCPS，76项既有回归）；源码语法与13份YAML解析通过。真实GPU 2训练step：batch16、384、workers8、默认compile、16-mixed；选择器梯度有限非零、参数实际更新，全模型权重有限。峰值allocated=14.9115GiB。真实GPU Linear hook确认选择器MLP输出全部FP32；真实checkpoint经LightningCLI CPU完整恢复成功（0测试batch，仅加载验证，不是精度评估）。presence消融CLI解析成功。12病例几何预检通过，未用测试病例计算冒烟成绩。

受保护baseline/innovation1共8个核心文件SHA256保持不变。已有训练与评估结果未覆盖。生产配置仍220轮，正式训练尚未启动，等待06:17定时任务。readiness将记录passed及启动所需哈希；其后修改受保护源码应重新验证。


## 正式训练已启动（定时任务实测）

2026-10-23 06:17:54 America/Denver（12:17:54 UTC）经SSH检查readiness=passed、全部69项哈希无变化、GPU空闲、磁盘约1.1TiB可用后，通过幂等启动器启动主A3。

- 输出：outputs/innovation2/ccps_plan/20261023T121754Z
- 监督器PID=2089726，训练PID=2089767。实际GPU进程PID匹配，开始执行首批训练。
- manifest：outputs/innovation2/ccps_plan/20261023T121754Z/synapse_test_selection/20261023T121806Z_c99f1c48/run_manifest.json；解析配置确认220轮、16-mixed、默认compile，独立输出和物理评估回调。
- 启动环境hostname=5090、指定Conda解释器、PyTorch2.13.0+cu132/CUDA13.2；保留三项离线环境前缀。
- 12病例、50轮测试窗口尚未开始，当前没有正式test成绩。
- 本次启动任务innovation2-ccps已执行完并删除，防止重复触发；五小时保障检查任务innovation2仍保留，09:18:50 America/Denver执行。

本轮仅更新本运行记录并生成正式run的launch/status/train.log、run_manifest等产物；未更改模型、配置、训练参数或既有结果。


## 五小时保障检查（2026-10-23 09:19 America/Denver）

已检查同一正式run：status=running，实际训练PID2089767及GPU PID匹配，显存占用19748MiB。last.ckpt已保存第172轮训练权重、global_step=23736；完整12例测试已完成至第171轮。第171轮临时mean Dice=85.582458%，HD95_mm=31.072210mm；尚非最终220轮结果。当前继续后续轮次/测试，未重启或启动第二份训练。日志的compile重编译上限提示来自目标mask数量变化，为已允许的原路径回退，未修改编译设置。

单次五小时检查任务innovation2执行结束后删除，不再定时检查。仅更新本记录，未改源码、配置、参数或已有结果。


## 用户主动停止（2026-10-24T08:22:13.908577+00:00）

已按最新指令向核验后的训练PID2089767发送SIGINT并正常退出，监督器亦退出，GPU已释放。保存训练第214轮，完整测试到第213轮，最佳第187轮。共43轮完整测试/516病例，保留原权重和CSV。平均Dice=86.242979%，HD95=29.744758mm。status标stopped_by_user，不冒充220轮完成。TTA尚未执行，已删除等待任务tta；未经后续指令不自动重启。结果见.codex/innovation2_ccps_results.md。
