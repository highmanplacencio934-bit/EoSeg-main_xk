# 更高视角TTA的改进候选与预检（2026-10-24）

## 结论

当前直接可实施的方向是固定384画布的轻度尺度变化×水平翻转，替代已失效的垂直/双轴翻转。该方案已完成独立正式CLI/回调接入、138项CPU回归及真实GPU一致性冒烟；用户授权的新4视角完整12例测试正在运行。此前切片预检没有证实稳定提升平均Dice，正式结果尚未产生。原mirror/D4 TTA及其历史结果保留。

现有2视角正式12例平均Dice86.442904%、HD95 26.298295mm；无TTA86.242979%、29.744758mm，Dice提升0.199925个百分点。不能因希望提分而把视角数视为收益保证。

## 代码约束与候选

ASEE forward依赖固定num_patches/grid，ViT构造输入384、patch16。候选保持网络输入384×384和24×24网格；先对原384输入缩放，再居中补零或裁剪回384。它是固定画布zoom TTA，不是修改模型输入分辨率的标准动态分辨率多尺度测试。

| 名称 | 缩放比例（相对384画布） | 每个尺度的镜像 | 总视角 |
|---|---|---|---:|
| scale_hflip_2 | 1.0 | 原方向、水平翻转 | 2 |
| scale_hflip_4 | 1.0、0.875 | 原方向、水平翻转 | 4 |
| scale_hflip_6 | 1.0、0.875、1.125 | 原方向、水平翻转 | 6 |
| scale_hflip_8 | 1.0、0.875、1.125、0.75 | 原方向、水平翻转 | 8 |

顺序和比例在本次GPU预检之前固定，不是根据结果逐个器官选择。4/6/8的新名称必须附带scale_hflip，不能与旧mirror4/D4_8混淆。

核心处理：先反水平翻转，再恢复缩放坐标；缩小时去掉补边对应的预测；放大裁剪时将不可见区域的覆盖权重设为0。FP32累加各视角分数，除以该像素实际覆盖权重，最后argmax。保留1.0尺度确保全图覆盖。不得把裁掉区域作为背景票，也不交换左右肾通道，不引入额外softmax。

## 已执行验证

- 6项CPU测试通过：identity与旧2视角分数逐值一致、常数分数在覆盖归一化后保持、放大视角不可见边缘排除、缩小补边预测丢弃、水平翻转类别通道保持、无identity拒绝。
- 服务器RTX5090D空闲时，使用同一innovation2 best187、FP16/default compile对3例代表切片执行2/4/6/8候选。新2视角与原predict_tta_scores(...,2)真实GPU逐值一致。
- checkpoint SHA256：b82664367030fa68ce211cb0d0388fe6aab0609c5e69fe33f5c0f8ca6af5627d。
- 三例切片按最大化min(左肾GT面积,右肾GT面积)选择，和前一轮诊断一致；仅用于定位，不作为正式测试均值。

| 病例 / z（0起） | 原2视角平均Dice% | 多尺度4视角 | 多尺度6视角 | 多尺度8视角 |
|---|---:|---:|---:|---:|
| case0022 / 61 | 81.9176 | 81.5657 | 81.9033 | 81.6493 |
| case0036 / 127 | 72.3342 | 68.2268 | 67.5158 | 66.0396 |
| case0001 / 110 | 87.3250 | 89.8269 | 89.6694 | 89.7232 |

表中为单个2D切片上GT存在类别的平均Dice，不是3D病例指标，不可当作Synapse平均Dice。所有候选在这些切片上的两种跨肾误分计数都是0；这不保证全12病例也为0。case0001的右肾明显改善，case0036其他器官下降，说明尺度也存在收益/损失。此次未计算切片HD95，不能由此预报完整体积HD95。

## 后续优先级

1. 若继续正式比较，先将候选接入独立测试CLI/回调、记录尺度/裁剪/覆盖合同和新源码readiness；旧TTA默认及旧readiness不改。
2. 固定同一best187，完整12病例比较scale_hflip_4和scale_hflip_6，再决定是否运行8；逐器官Dice/HD95_mm、左右肾跨类误分和耗时均保留。以统一候选配置比较，不能拼凑各病例/器官最优结果。
3. 若尺度分支不稳定，可另行测试以原2视角为主、附加尺度较低权重的融合；当前未实现/验证此项，不能声称优于等权。
4. 若确需垂直翻转或D4方向，需要独立训练实验纳入相应训练增强并与当前训练对照；这是改变训练变量，不能写回原baseline或声称无需重训就能获得方向鲁棒性。

多尺度+翻转的通用方法参考MMCV官方文档：https://mmcv.readthedocs.io/en/latest/api/generated/mmcv.transforms.MultiScaleFlipAug.html 。本候选为适配本项目固定输入的独立实现，效果结论仅来自上述实际预检。

## 文件与状态

新增outputs/audit/tta_scale_probe_20261024/candidate.py、test_candidate.py、probe.py、probe.log、probe_results.json、verification.json、file_changes.json，以及本文档。生产代码、正式配置、权重、训练增强、指标公式均未变；84份原生产源码/配置哈希一致。没有启动新训练或完整12例候选测试。


## 本轮正式接入及启动（2026-10-24）

用户要求完成对应代码并启动新4视角。新增training/synapse_multiscale_tta.py、scripts/evaluate_synapse_multiscale_tta.py及独立multiscale_tta配置；支持scale_hflip_2/4/6/8，但本次仅启动4。旧mirror/D4实现、旧readiness和训练配置均未改。

验证：138项CPU回归（新增13项）通过；新入口FP16/compile真实GPU冒烟中新2视角与旧2视角max diff=0，新4视角与隔离候选max diff=0。完整12例测试尚未结束。证据outputs/audit/tta_multiscale_interface_20261024；新哈希门禁outputs/tta/multiscale_readiness.json。

输出：outputs/tta/innovation2_best_scale_hflip4/20261024T111222Z；监督器PID=3673200，测试PID=3673239。已确认CUDA设备、16-mixed、compile_test、best187及其SHA256、固定12例、真实spacing和命名表、run_manifest。新4视角为1.0/0.875×原方向/水平翻转，网络画布384、patch16，原legacy空掩码规则。

监督器在结束时自动校验12例96条病例器官记录与8器官均值，输出results_summary.md/json、organ_comparison.csv、kidney_case_comparison.csv及comparison_1_2_scale4.csv。比较基准为同一best187的无TTA和已完成旧2视角；不跨权重或拼接逐器官最优，不把旧mirror4与新scale_hflip4混称。未启动6/8视角或新训练。


## 新6视角启动（2026-10-24，最新状态）

新4视角已完整完成12例、exit0：Dice=86.452230%，HD95=19.041325mm，耗时约24分13秒。

用户随后要求启动新6视角，输出outputs/tta/innovation2_best_scale_hflip6/20261024T114632Z。使用同一innovation2 best187，尺度1.0、0.875、1.125各配原图/水平翻转；固定384画布、patch16、FP16/compile_test、真实mm+voxel和原legacy空掩码规则。监督器PID=3709912，测试PID=3709974，已核验GPU权重加载和6视角manifest，状态running。

复用已通过138项CPU回归和GPU冒烟的多尺度接口，readiness及源码/配置/先验/列表哈希通过，本轮不改核心代码或训练参数。结束时监督器自动校验12例96条指标，生成分器官及平均Dice/HD95报告、左右肾病例诊断和同权重无TTA/2视角/新4视角/新6视角对照。只启动6视角，不启动8视角或新训练。


## 新8视角启动（2026-10-24，最新状态）

新6视角已完整完成12例、exit0：Dice=86.835536%，HD95=19.065135mm，耗时约24分06秒。新4视角为86.452230%/19.041325mm；以上是同一best187。

用户要求再做新8视角，输出outputs/tta/innovation2_best_scale_hflip8/20261024T123754Z。同一innovation2 best187；尺度1.0、0.875、1.125、0.75各配原图/水平翻转；固定384画布、patch16、FP16/compile_test、真实mm+voxel及legacy空掩码规则。监督器PID=3765892，测试PID=3765925，已核验GPU权重加载、8视角manifest及全部源码哈希，状态running。

本次使用既有scale_hflip_8，不是D4旋转8视角。复用138项CPU回归和既有GPU冒烟证据，readiness哈希一致，无核心代码或训练参数改动。监督器结束时校验12例96条指标，输出命名器官及平均Dice/HD95、左右肾病例诊断和同权重无TTA/2/新4/新6/新8对照。只启动新8视角测试，不启动新训练。
