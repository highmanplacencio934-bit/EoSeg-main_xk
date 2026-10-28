# 当前物理评估默认关闭体素HD95

用户于2026-10-27明确要求不再计算额外体素HD95，保留代码以便恢复。后续物理/ROI/多尺度TTA、CCPS以及教师适配入口默认compute_hd95_voxel=false，只计算真实spacing的HD95_mm。

## 使用与恢复
模型构造参数：compute_hd95_voxel: false（默认）。
LightningCLI可显式覆盖：
    --model.init_args.compute_hd95_voxel=false
需要恢复双单位时：
    --model.init_args.compute_hd95_voxel=true

低层binary_metrics_mm/binary_metrics_mm_roi也支持同名keyword-only开关。关闭时非空器官只执行一次带voxelspacing的MedPy HD95；不执行不带spacing的第二次调用。空mask原Dice/mm/status分支保持，体素字段为None。

## 产物契约
hd95_voxel_enabled=false；CSV体素列为空、JSON为null、Markdown显示“未计算”，不能把未计算写成0或NaN。Dice和HD95_mm仍正常输出。启用True时恢复原双单位数值。报告/启动器按实际metadata模式校验，拒绝混模式和假零；历史缺flag文件仍按旧双单位严格校验。原有CSV/checkpoint/manifest未覆盖。

## 验证
outputs/audit/hd95_voxel_disabled/20261027T073138Z/validation.json：
- 205项CPU回归通过，含真实调用次数、物理/ROI/TTA/教师/CCPS、CSV导出、恢复True与启动器验收。
- 原baseline-2 best173完整12病例GPU FP16/compile：96Dice+96HD95_mm与历史CSV逐值相同；实际HD95毫米调用95次，体素调用0次。
- 原baseline best177固定8-view真实case0008冒烟：8Dice+8HD95_mm严格匹配历史M0；毫米调用8次、体素0次。此项为单病例，不是新跑完整12例TTA。
- ROI/TTA readiness在旧快照归档及新证据通过后刷新源码hash，记录了新验收范围；历史ccps_readiness未伪更新。

模型、原训练增强、seed、损失、AdamW/LR/LLRD、预算和预测计算不改；本次没有启动正式训练。历史原作者main.py/LightningModule的体素专用指标入口保留作对照，它不属于物理评估的额外第二次计算；后续当前实验使用physical/ROI入口。历史validate_synapse_roi_acceleration.py明确True仅用于双单位回归，不在默认训练中调用。
