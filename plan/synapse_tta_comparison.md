# Innovation2 固定最佳权重的 TTA 对比

用户授权：比较不同TTA视角，追加8视角，并要求重点考虑左右肾可靠性。独立评估，不改变原训练、checkpoint选择或现有结果。

## 视角定义

- 1：原图，调用原物理evaluate_synapse_case，作为重放对照。
- 2：原图、水平翻转。
- 4：原图、水平、垂直、双轴翻转。
- 8：二维D4全部8种不同变换，等价于0/90/180/270度旋转各配原样和水平翻转。实现顺序保留4镜像为前4项，再加其余4种旋转/镜像，无重复。不是三维8种轴翻转。

参考轴组合思路：https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py 的 _internal_maybe_mirror_and_predict；本项目使用EoSeg像素分数而非nnUNet网络logits。当前训练只有水平翻转，没有垂直翻转或90度旋转，因此4/8效果必须实测，不能预设优于2。

## 推理契约

保留384输入、原cubic输入缩放、模型内部ImageNet归一化、FP16模型及原per-pixel转换、bilinear到384。每个视角独立重算ASEE/CCPS，输出9类像素分数；先逆空间变换，再FP32均值，最后argmax与nearest回原512网格。禁止平均query顺序或已argmax硬标签，禁止额外softmax。完整3D恢复后以已核验ZYX spacing计算毫米HD95并保留voxel；原空掩码规则不改。所有组同一最终best，不重选checkpoint，不用test GT控制门控或推理。

## 左右肾

固定label3=左肾、label4=右肾，不合并，不自动交换通道。当前训练翻转图像和mask后仍保留语义标签ID，这与逆变换后保持原通道的定义一致。不能因语义ID定义正确就推定翻转模型预测可靠。

每个视角/每病例报告两侧独立Dice和HD95_mm，并记录pred_left_on_gt_right_voxels、pred_right_on_gt_left_voxels及两侧GT体素数。所有统计只在预测完成后计算。对1视角仅在原metric调用处观察mask，不改变计算或预测。比较各组相对单视角的左右肾退化；误分体素不是额外HD95定义，也不凭此自动交换通道或修改预测。

## 执行与产物

scripts/evaluate_synapse_tta.py 是独立test入口，--tta_views=1/2/4/8，--compile_test匹配正式训练。main.py及原训练/物理评估文件保持不变。配置 configs/synapse/vit_query_mul_scale_fusion_asee_ccps_tta.yaml。

等innovation2 status=completed/exit0且220轮、50次12例评估完整后，启动 scripts/run_synapse_tta_comparison.py。顺序1→2→4→8，单GPU，无并发，不重训；固定最终best的epoch和SHA256。1视角96条病例器官指标必须严格重放原记录（1e-10阈值），不一致即停，不放宽。每组保存合同、命名器官指标、逐病例时间及肾脏误分。

输出 outputs/tta/innovation2/<UTC时间>；comparison_summary.md/json、comparison_organs.csv、kidney_comparison.csv，分别标出平均Dice最高和HD95最低组。四组对比使用同一权重，不将TTA增益记作模型结构增益；各组总时间含启动/编译，亦保留test_seconds和逐病例时间。

目前只进行CPU与CLI加载验证。正在运行的innovation2训练不占用额外GPU；真实TTA GPU评估将在训练完成后执行。定时检查只用于兑现本次对比请求，完成或失败后删除，不另启长期监控。


## 最终验证与等待任务

118项全套CPU回归通过，其中17项TTA。D4八种变换互异且反变换严格恢复；单视角原指标观察钩子不改变结果；query顺序变化不影响像素通道；左右肾交叉误分统计、非有限分数拒绝及汇总生成通过。真实CCPS checkpoint经8视角独立CLI在CPU完整恢复（0测试batch，不是TTA精度验证）。训练核心6文件哈希未变。

outputs/tta/tta_readiness.json已标passed，含源码/配置/先验/清单哈希，表示可在训练完成后进入正式独立评估。GPU TTA尚未运行。首个1视角重放将作为真实GPU接口核验，不匹配会停止后续组。

Codex线程任务id=tta，每20分钟检查前置训练与对比状态；正常训练/等待时不通知。训练完成后只启动一次四组顺序对比，完成或失败汇报后删除任务。该任务仍需Codex运行、SSH与账户额度可用。


## 最新状态：用户主动停训后取消等待任务

2026-10-24用户要求停止当前innovation2训练并汇总结果。训练保存至214轮，完整测试至213轮，最佳187轮。TTA尚未执行，等待任务tta已删除，原“完整220轮成功后评估”的触发前提不再成立。保留实现、测试与readiness证据，不自动开跑；后续若继续TTA，需明确以此次用户停止的同一best为输入并更新执行前提，不能伪称训练完成220轮。


## 2026-10-24 两项代码审查修复

1. 每组独立test由SynapseTTASelection写入run_manifest.json和readiness_snapshot.json。记录实际解析CLI配置、Python/软件版本、CUDA/设备、matmul精度、源码（含scripts入口和运行器）及数据清单哈希、checkpoint来源、TTA合同和指定离线环境变量。保存当次readiness原始字节及哈希，后续全局readiness更新不影响旧记录；不采集任意环境变量。不访问estimated_stepping_batches、训练loader或optimizer。汇总表保存每组manifest路径与哈希，缺清单时拒绝出正式汇总。

2. 左右肾均值增加baseline_dice_percent、baseline_hd95_mm、dice_delta_pp、hd95_delta_mm，以及退化/不可比较病例数。新增kidney_case_comparison.csv（4×12×2=96行）和kidney_degraded_cases.csv，按(case,label_id)对齐单视角，不按行号配对。包含每病例Dice/HD95原值与变化、双向肾脏误分变化、两侧GT体素数与mask状态。Markdown展示均值变化和逐病例退化子集。

Dice退化定义为严格数值下降，HD95退化定义为两侧记录均nonempty时严格数值上升，不代表统计显著性。legacy空mask的HD95=0保留，但标记hd95_comparable=false、hd95_degraded=null，不把占位零值当作真实边界改善。聚合指标公式、模型和推理路径不变。

本轮仅修复两项，不放开已停训run的启动前提，不恢复已删除的定时任务，不运行GPU评估。验证结果见outputs/audit/tta_review_fixes_20261024。

最终验证：125项全套CPU回归通过（其中24项TTA），3份修改Python文件语法检查通过，9份训练/指标核心文件SHA256保持不变。readiness已刷新为本轮源码哈希，保留先前CLI恢复证据；未执行GPU评估，也未修改已停训run的启动限制。


## 本轮明确授权：仅运行innovation2 best的2视角

2026-10-24用户要求先对已停训innovation2的best_checkpoint执行2视角。通过独立evaluate_synapse_tta.py直接指定已核验第187轮best_test.ckpt；不使用要求220轮完成的四组运行器，不篡改原训练stopped_by_user状态。

- 输出：outputs/tta/innovation2_best_views2/20261024T084750Z
- 监督器PID=3512379，测试PID=3512421；已实测GPU进程和真实权重恢复，16-mixed、compile_test、原图+水平翻转、完整12例。
- checkpoint SHA256=b82664367030fa68ce211cb0d0388fe6aab0609c5e69fe33f5c0f8ca6af5627d。
- 已生成完整run_manifest、readiness快照、checkpoint/spacing/name合同。结束后保存命名器官Dice、HD95_mm/voxel及相对原无TTA第187轮CSV的变化，左右肾按病例单独报告。
- 原无TTA训练未采集对侧误分体素计数，不能伪造其基线/增减；2视角本组计数正常保存。
- 本次只启动2视角评估，不恢复训练或旧定时任务，未自动启动1/4/8视角。


## 2视角已完成，按用户要求独立启动4视角

2视角完整12例已结束exit0：同一best187，平均Dice=86.442904%，HD95=26.298295mm；无TTA参考86.242979%/29.744758mm。

用户随后要求再次测试4视角；已启动原图、水平、垂直、双轴翻转的独立12例评估。输出outputs/tta/innovation2_best_views4/20261024T091548Z，监督器3543154，测试PID=3543197。实际恢复best187，SHA256=b82664367030fa68ce211cb0d0388fe6aab0609c5e69fe33f5c0f8ca6af5627d；FP16/compile_test，原spacing/legacy规则不变，不交换肾脏通道。完成后额外保存相同权重1/2/4平均指标对照CSV。仅启动4视角，无训练或8视角任务。


## 4视角完成与已运行三组汇总

输出outputs/tta/innovation2_best_views4/20261024T091548Z，完整12例，exit0，耗时约24分04秒。4视角平均Dice=60.152770%，HD95=71.539775mm。2/4组checkpoint哈希、模型配置、执行源码哈希、精度和spacing一致；8视角尚未执行。结果见本run comparison_1_2_4.md及organ_comparison_1_2_4.csv。

4视角两侧肾脏12例Dice均退化，出现对侧肾GT区域误分。垂直/双轴翻转是新增实验变量，未做单视角预测隔离，不能把原因说成已证实的左右肾互换。当前结果不支持采用4视角，不自动继续8视角。


## 2026-10-24 垂直翻转退化诊断

已完成真实GPU单视角隔离：三个病例的代表切片中，垂直和双轴翻转预测在正确复原后两侧肾脏Dice均为0；四视角等权融合受到这些错误预测污染。手工融合与生产函数逐值一致，eager/FP32和固定原图CCPS门控未消除退化。训练增强只有水平翻转，新增视角存在方向适应问题；尚未证实某一网络层的具体机制。没有修改训练/生产推理/指标或交换器官通道，未启动8视角。完整证据和边界见tta_vertical_flip_diagnosis.md及outputs/audit/tta_drop_diagnosis_20261024。


## 新多尺度4视角已启动（2026-10-24）

按用户要求完成独立scale_hflip入口并启动outputs/tta/innovation2_best_scale_hflip4/20261024T111222Z。同一innovation2 best187，尺度1.0/0.875，各配原方向/水平翻转；384/patch16、FP16/compile_test、真实mm+voxel和legacy空掩码规则。测试PID=3673239，状态running，已确认权重加载、GPU及manifest。138项CPU回归和真实GPU分数一致性冒烟通过。新测试未结束，不预报增益；6/8视角未启动。实现/验证见synapse_multiscale_tta_plan.md。


## 新6视角启动（2026-10-24，最新状态）

新4视角已完整完成12例、exit0：Dice=86.452230%，HD95=19.041325mm，耗时约24分13秒。

用户随后要求启动新6视角，输出outputs/tta/innovation2_best_scale_hflip6/20261024T114632Z。使用同一innovation2 best187，尺度1.0、0.875、1.125各配原图/水平翻转；固定384画布、patch16、FP16/compile_test、真实mm+voxel和原legacy空掩码规则。监督器PID=3709912，测试PID=3709974，已核验GPU权重加载和6视角manifest，状态running。

复用已通过138项CPU回归和GPU冒烟的多尺度接口，readiness及源码/配置/先验/列表哈希通过，本轮不改核心代码或训练参数。结束时监督器自动校验12例96条指标，生成分器官及平均Dice/HD95报告、左右肾病例诊断和同权重无TTA/2视角/新4视角/新6视角对照。只启动6视角，不启动8视角或新训练。


## 新8视角启动（2026-10-24，最新状态）

新6视角已完整完成12例、exit0：Dice=86.835536%，HD95=19.065135mm，耗时约24分06秒。新4视角为86.452230%/19.041325mm；以上是同一best187。

用户要求再做新8视角，输出outputs/tta/innovation2_best_scale_hflip8/20261024T123754Z。同一innovation2 best187；尺度1.0、0.875、1.125、0.75各配原图/水平翻转；固定384画布、patch16、FP16/compile_test、真实mm+voxel及legacy空掩码规则。监督器PID=3765892，测试PID=3765925，已核验GPU权重加载、8视角manifest及全部源码哈希，状态running。

本次使用既有scale_hflip_8，不是D4旋转8视角。复用138项CPU回归和既有GPU冒烟证据，readiness哈希一致，无核心代码或训练参数改动。监督器结束时校验12例96条指标，输出命名器官及平均Dice/HD95、左右肾病例诊断和同权重无TTA/2/新4/新6/新8对照。只启动新8视角测试，不启动新训练。
