# baseline同seed差异：四步服务器验收

更新：2026-10-27 America/Denver。最新真实GPU证据根：outputs/audit/baseline_repeat_diagnosis/20261027T023508Z/。生产源码、配置、数据、模型与默认运行协议均未改；原S2保持用户早停，未启动S3或正式重训。

## 当前结论

已在受控同seed运行中实证训练轨迹从CUDA损失/反向阶段分叉。评分入口差异已排除；本次受控初始化、输入、增强、CPU/CUDA RNG、最终随机采样点与匹配均一致。旧入口自身重复也发生同类差异，因此不依赖ROI入口。

原3D分类交叉熵的CUDA前向归约与grid_sample CUDA反向均实测非确定性。前者造成微小loss显示/归约差异，并不能仅由此推断全部梯度差；后者固定输入重复时直接产生不同梯度。其他CUDA算子尚未全部独立排除，不能把历史最终1.1953pp精确归给单一算子。

## 四步验收状态

| 用户步骤 | 最新状态 | 证据 |
|---|---|---|
| 同checkpoint双入口12×8评分 | 完成，96项逐值相同，两侧退出0，均84.7184661015% | four_step/same_checkpoint_comparison.json及两侧dice96.json/exit.json |
| Seed0真实入口初始化 | 完成，388项及CPU RNG一致；首步GPU native CUDA RNG也一致 | first_step_cpu_interface_comparison.json；first_step_*/first_step_trace.json |
| 同固定Batch首步loss/梯度/一次更新 | 完成，首个记录差异为CE2.38418579e-7；scaled/unscaled/clipped梯度不同 | first_step_comparison_old_roi.json；first_step_detailed_difference.json |
| 原生Batch/最终采样、非零LR更新及CUDA隔离 | 完成最小分叉定位：原生前2Batch相同，第1步moments不同、第2步23个参数张量不同；两个CUDA非确定路径实证 | trajectory_detailed_comparison.json；cuda_loss_ops_isolation.json |

这是最小定位完成，不代表新做了220轮训练、所有seed已稳定，或精确复演两次历史正式训练。

## 第3步的关键解释

固定Batch16/384，FP16/default compile、原220/30360预算不变；old/roi与old_repeat三个独立GPU进程均退出0。初始参数、native CPU/CUDA RNG、网络输出、原始随机draw与匹配相同；CE为2.2414004802703857与2.241400718688965，差2.384185791015625e-7。mask/dice分项及加权总loss相同；367个unscaled梯度键、373个裁剪后梯度键不同。

首个真实AdamW调用时373组LR均为0，因此可训练参数暂未移动。相对初始化变化的12项均为BatchNorm运行缓冲，两入口也一致，不能把这些缓冲写成AdamW权重更新。旧入口再次运行仍出现loss/梯度差异，说明ROI不是必要条件。

## 第4步原生批次对照

使用原生DataLoader，未注入固定Batch，保留原增强/shuffle/workers8。old与roi均真实执行2次AdamW后主动退出0；不是修改max_steps预算。

- 两次尝试均input/RNG/network输出/原始随机draw/最终sample_point坐标与输出/matcher严格相同。
- 第1步：LR全0，369个unscaled梯度键不同，373个参数对应AdamW内部状态不同，权重仍相同。
- 第2步：原预热使非主干LR=4e-7、主干LR=0，366个unscaled梯度键不同；23个可训练参数张量出现差异，包括query_embed、mask/class头、上采样和layer_gate；BatchNorm缓冲无差异。
- Query最大绝对差约5.96e-8，所记录参数最大差约1.19e-7。这是早期数值微差，不等于最终Dice只差这么多。

## CUDA隔离

固定同输入的原分类CE30次出现两个loss值，梯度在该孤立CE试验中相同；严格确定性报nll_loss2d_forward_out_cuda_template不支持。固定同输入grid_sample30次前向loss标量相同，但产生30组梯度（隔离试验未保存完整前向输出hash），最大差1.144409e-5；严格确定性报grid_sampler_2d_backward_cuda不支持。真实MaskClassificationLoss固定输入/RNG20次也产生20组梯度。

诊断用flatten为2D的CE30次稳定且strict通过；仅作为候选，不应用生产，因为数值归约顺序改变。grid_sample仍需独立确定性实现与梯度验证，不能仅开deterministic=True或只修CE就承诺整模型稳定。

## 保护与边界

gpu_validation_summary.json汇总全部退出码、脚本hash和证据。对照S2 run_manifest的生产源码hash零变化；全部诊断GPU进程已结束，其他用户xr进程未操作。历史baseline/baseline-2和全部权重、CSV保留。历史未保存初始CUDA状态/逐batch梯度，故不能精确量化1.195289936pp的全部成因。不自动重训、改FP32、关compile、禁增强或切换默认deterministic。
