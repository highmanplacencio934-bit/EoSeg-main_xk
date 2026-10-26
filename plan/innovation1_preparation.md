> 最新训练授权与状态见 `.codex/innovation1_training_run.md`：用户已要求审核后启动主Plan训练；真实官方文本编码器和仅训练集先验已补齐。下文“等待/缺少编码器/未生成先验”为该次审查或首次准备时的历史状态。

> 最新状态：baseline已完成220轮、正常退出；原口径最终最佳177轮Dice84.718466%。用户确认继续保留原Dice规则，修正分支复核83.676799%单独记录。ASEE已完成后续审查/修复和66项CPU回归，见[审查报告](asee_code_review.md)。以下为首次8小时准备的历史记录。

# 创新点1准备完成记录

执行任务：8-synapse-baseline-1；2026-10-21 UTC21:24触发的单次8小时后续任务。已完成baseline检查、材料阅读、独立代码与必要CPU验证，现在停止工作等待用户指示。

## Baseline结果边界

八小时检查UTC21:25:27，训练PID3477678实际存活，最近保存192/220轮；完整测试171–191共21轮252病例。代码准备结束UTC21:43:34复查：进程仍运行，完整测试到192轮，共22轮264病例；selection_summary同步至192。当前最佳177轮，mean Dice=0.8471846610153185（84.718466%），对应mean HD95=47.38956681663292体素。尚无220轮最终结果或最终退出码。没有中止、重启或修改baseline。

逐类表与初次证据：`outputs/baseline/20261021T132635Z/followup_8h_summary.md`、`followup_8h_status.json`。结束复查：同目录`code_preparation_end_status.json`。原baseline CSV/权重由训练进程自行继续更新，本次未改写。

## 本轮新增文件与用途

- `models/asee.py`：先验加载及ASEE，两种显式公式与分支开关。
- `models/eoseg_asee.py`：独立集成模型，只在最后三层前增强空间patch/query，保留CLS/register和原融合/头。
- `scripts/build_synapse_asee_prior.py`：固定训练清单的真实NPZ流式共现统计、CPU离线文本特征提取接口；未实际构建正式先验。
- `configs/synapse/asee_label_map.example.json`：未核验的映射示例，默认拒绝投入构建。
- `configs/synapse/vit_query_mul_scale_fusion_asee_plan.yaml`：Plan外层LN公式、文本映射可训练。
- `configs/synapse/vit_query_mul_scale_fusion_asee_reference.yaml`：参考代码门控残差公式。
- `configs/synapse/vit_query_mul_scale_fusion_asee_plan_frozen_text.yaml`：Plan公式、文本映射整程冻结；未编造解冻计划。
- `tests/test_asee_innovation.py`：13项必要CPU接口/数值/隔离/原损失反传回归。
- `docs/innovation1_asee.md`：四份材料的冲突、公式、对照变量、先验格式、使用模板及待定项。
- `.codex/innovation1_preparation.md`：本次完成记录（本文件）。
- `outputs/innovation1/preparation_20261021/*`：受保护文件前后哈希、源文件快照、CPU日志/验证状态、三份解析配置、PPT只读抽取/关键方法图渲染/备注清单及渲染脚本。
- `outputs/baseline/20261021T132635Z/followup_8h_status.json`、`followup_8h_summary.md`、`code_preparation_end_status.json`：只读检查的独立报告产物。

本轮修改：`.codex/baseline_followup_task.md`追加执行结果；准备过程中修订了本轮新文件，最终内容以上述文件为准。本轮没有删除项目文件；已通过app工具移除单次自动化`8-synapse-baseline-1`（deleteStatus=deleted），避免重复执行。

## 验证与待定项

服务器CPU、CUDA_VISIBLE_DEVICES为空、线程2，原31项+新增13项共44项通过；3份配置均经main.py fit --print_config解析成功。已验证禁用ASEE时与原模型train/eval输出逐值一致，参考门控实现的数值对照、原匈牙利损失反传、优化器归组与来源加载。目标1024维ASEE参数量7,360,768（Plan）/7,360,770（参考门控）。仅读取1张真实训练NPZ检查格式，其余先验测试用隔离临时fixture；没有正式大规模先验生成。

23份baseline受保护源码/配置及4份材料共27个哈希均未改变。原5套baseline YAML、模型、数据、损失、训练预算、精度、增强、评估和预训练路径未改。创新配置新增模块变量及独立输出路径，其余与第三套baseline一致。

待核验/决定：标签ID与器官名的可信映射来源；本地BioClinicalBERT文件来源/路径；两种残差公式；映射是否冻结及是否需要明确的解冻计划。PPT没有备注文件，热图是示意，不充当真实映射或统计。GPU/FP16/真实compile冒烟、正式先验与创新训练均未执行，等待用户指示。指定dinov2_vitl14_reg4_pretrain.pth本地加载仍未接通，未借本次创新修改该baseline流程。
