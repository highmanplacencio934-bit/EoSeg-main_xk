# 本次 baseline 启动与单次后续任务

> 服务器并发规则更新（2026-10-26，用户明确授权）：其他用户正在执行GPU任务时，剩余显存足够本次任务即可并发，无需再申请并发例外；不足时不得启动，也不得中止其他用户任务。详见[统一显存规则](server_gpu_execution_rules.md)。本文旧的“GPU必须空闲/并发需新授权”等条款已被本规则取代，历史执行事实保留。

用户于2026-10-21明确授权：现在启动baseline；约8小时后先检查训练及测试结果并汇总；然后阅读两个Plan和两个参考实现，准备创新点1代码；完成后停止继续工作，等待用户指示。

## 已启动的正式 baseline

- 服务器：hostname=5090，NVIDIA GeForce RTX 5090 D；Python=/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python。
- 项目：/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse。
- 配置：configs/synapse/vit_query_mul_scale_fusion.yaml；220 epochs、384输入、patch16、batch16、workers8、16-mixed、seed0、LR2e-4、LLRD0.8、AdamW/WD0.05、warmup[500,1000]、默认compile，未改科学参数。
- 实际estimated_stepping_batches=30360。训练/测试固定18+12病例；最后50轮171–220完整测试，保持best_test.ckpt与last.ckpt。
- 本次仍使用已验证的默认timm预训练/离线缓存路径，不宣称用户指定dinov2_vitl14_reg4_pretrain.pth的本地接口已接通。
- 启动时间：2026-10-21 07:26:35 America/Denver（UTC 13:26:35）。
- 后台supervisor PID=3477677，training PID=3477678；以后以status.json与/proc实际进程为准，不只凭PID认定存活。
- 输出根目录：outputs/baseline/20261021T132635Z。
- 权重/CSV目录：outputs/baseline/20261021T132635Z/synapse_test_selection/20261021T132645Z_81f3214a。
- train.log记录stdout/stderr；launch.json记录命令/环境/来源；status.json记录PID、启动/退出时间及退出码；run_manifest.json记录解析配置与来源。
- supervisor是独立服务器会话，SSH断开不终止训练。启动后已核验CUDA实际运行和原配置；不因8小时到了而中断220轮。
- 日志使用WANDB_MODE=offline；输出路径及日志存储路径是仅有的CLI配置覆盖，不改变模型和训练算法。原配置及源文件未改。

服务器Python命令必须保留以下环境变量，非交互SSH应使用明确的上述解释器，不假定base：

```bash
HF_HUB_OFFLINE=1 \
TORCH_HOME=/Disk/HDD/Sk/.cache/torch \
TIMM_USE_OLD_CACHE=1 \
python ...
```

## 单次定时跟进

- automation id：8-synapse-baseline-1，kind=heartbeat，附着于本对话；已创建ACTIVE状态的单次规则。
- 触发时间：2026-10-21 **15:24 America/Denver（MDT/UTC-6）**，即UTC 21:24；按用户“从现在起约8小时”设置。规则包含单次执行限制，勿改成无结束的循环。
- 本机系统时区已核实与America/Denver一致。本地项目/技能的定时任务需要电脑开机、Codex客户端运行，SSH及项目挂载可访问；baseline本身运行在服务器，不依赖客户端持续连接。
- 任务完成后暂停自身自动化，避免任何重复代码准备；不要启动后续循环或另建对话。

## 到时依次执行

1. 先读AGENTS.md和本文件，读取status.json、train.log、run_manifest.json，验证真正的训练进程和退出码。
2. 核对220轮是否结束，CSV是否有171–220共50轮、每轮12例（共600逐病例行），best/last/selection_summary是否一致。报告最佳epoch、逐类Dice/HD95及均值；Dice按0..1和百分数区分，HD95是体素单位，器官编号未核验时不猜名字。
3. 未完成则报告真实进度；未到171轮没有test结果属于预期。未结束时不停止、不重启、不更改baseline，绝不把部分结果冒充最终成绩。失败则报告错误与退出码，不擅自改变参数或把纯权重当作精确续训。需要独立复现compile checkpoint时使用已验证的--compile_test。
4. 将baseline结果/状态汇总到本次输出目录的独立Markdown报告，并向用户汇报。
5. 随后按顺序完整阅读根目录MGTN.md、ASEE算法改进方案_v2.pptx（两份Plan）、asee_module.py、build_prior.py（参考实现）。这四个文件已确认存在，本轮尚未提前解析其设计。按适用PPT技能读取幻灯片、图示和备注，不仅依赖抽取文字。
6. 核验两份Plan的对应关系及参考代码，准备创新点1的独立模块/模型、配置、必要的训练集先验构建接口和使用说明。不得覆盖baseline或参考源文件。新增变量明确列出，不能凭空补齐Plan缺失设计。
7. 构建训练先验只能使用训练数据，除非用户后续明确授权其他用途。阅读参考代码后先检查顶层副作用，避免导入时自动执行大规模数据处理。
8. 仅做必要CPU/小规模接口和语法检查；不争抢仍在训练的baseline GPU，不启动创新训练或正式大规模先验生成。
9. 代码准备完成后汇总文件变更、验证和待定项，停止工作等待用户指示。这里的“停止”不代表结束仍未完成的baseline进程。

## 本次单次任务执行完成（2026-10-21 UTC21:44）

- 已执行八小时只读检查并保存独立报告；baseline尚未结束，训练继续，未停止/重启/修改。
- 初检最近保存192/220轮、测试171–191；准备结束复查完整测试至192，共264病例，进程仍运行。当前最佳177轮，Dice84.718466%，HD95=47.389567体素；均为部分结果。
- 已按顺序读取两个Plan及两个参考代码，准备独立创新模块/模型、3份显式变体配置、训练先验接口、13项新增CPU测试和说明；原31项加新增13项共44项通过，配置解析3/3通过，27个受保护文件哈希无漂移。
- 未生成正式先验、未运行创新GPU冒烟或训练；未停止baseline。待映射/本地文本编码器来源及Plan分歧决定，详见`.codex/innovation1_preparation.md`和`docs/innovation1_asee.md`。
- 本次单次自动化已通过app工具删除（8-synapse-baseline-1，deleteStatus=deleted），不再重复唤醒；不是终止服务器训练。现在停止继续工作，等待用户指示。
