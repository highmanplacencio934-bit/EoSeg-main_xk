# EoSeg Synapse CUDA确定性方案与分阶段验收Plan

更新：2026-10-27，America/Denver。

**状态：原v1的CCPS G4失败记录保留。独立v2已完成G0–G5：baseline/ASEE保留v1数值路径，CCPS采用局部末端BN编译边界与backward_pass_autocast=off；实际入口复核、46项CPU回归与G6启动器8项回归/2 Batch GPU冒烟通过。用户条件授权的两次纯baseline G6及四个单次任务已落实，正式220轮尚未开始，完整可重复性待G6验证。** 本Plan依据当前代码和服务器实测，目标是建立独立、可验证的确定性计算协议；不预先承诺一定可修复、精度提高或跨环境逐位复现。

## 1. 当前证据与问题边界

证据根：outputs/audit/baseline_repeat_diagnosis/20261027T023508Z/。相关说明为baseline_repeat_four_step_audit.md、baseline_repeat_solution_plan.md。

| 检查 | 已观察到的事实 | 可以得出的结论 |
|---|---|---|
| 同best177的两个真实入口 | 完整12病例×8器官，96个Dice逐值相同，均84.7184661015%，两侧退出0 | ROI没有把同一权重的Dice算高 |
| 实际入口初始化 | baseline的388个状态张量及Python/NumPy/Torch状态一致；GPU短轨迹初始CUDA RNG也一致 | 本次受控初始化一致 |
| 原生前2个Batch | 顺序、增强后输入、标签、最终采样坐标/输出、Matcher和网络输出相同 | 本次受控输入/采样不同不是分叉起点 |
| 第1次真实AdamW调用 | 学习率全0，梯度与AdamW moments已经不同，参数暂未移动 | optimizer调用次数不能代替非零LR参数更新；不能把BN缓冲当权重 |
| 第2次真实更新 | 非主干LR按原预热为4e-7，23个训练参数张量分叉 | 数值微差已进入真实参数轨迹 |
| 旧入口自身重复 | 不经过ROI也出现损失/梯度差异 | 分叉不依赖ROI入口 |

已定位的两条CUDA路径必须分开解释：

- 原3D weighted CE：固定输入30次出现两个loss值；该孤立试验的梯度hash相同。严格模式报nll_loss2d_forward_out_cuda_template不支持。
- grid_sample反向：固定输入30次的前向loss标量相同，梯度却有30组，最大绝对差约1.1444e-5；严格模式报grid_sampler_2d_backward_cuda不支持。该试验未保存完整前向输出hash，不能把标量相同扩大为整个输出逐位相同。
- 真实MaskClassificationLoss的固定输入/RNG重复也出现不同梯度。不能把CE标量微差直接说成全部梯度差的原因，也未逐一排除其他CUDA算子。

因此当前代码存在已实证的CUDA数值非确定性。但历史两个完整run没有首批CUDA状态、逐batch梯度等全部快照，不能把最终1.1953个百分点精确分摊给某个算子。短轨迹定位不等于已修复或已重跑220轮。

## 2. 保持的对照变量与最新指标设置

候选方案只研究计算确定性。以下设置按当前baseline固定，创新模型按各自原配置逐项核对：

| 范围 | 保持设置 |
|---|---|
| baseline网络 | EoSegMultiQSegFusion，原ViT主干，384×384、patch16、16 queries、9个语义类别 |
| 数据 | 固定18训练病例/2211切片、12测试病例，原列表及病例ID，不随机重划 |
| 在线增强 | 原颜色、水平翻转、ScaleJitter0.5–2.0、原补零/随机384裁剪；保留随机性 |
| 训练 | seed0、batch16、workers8、220轮、30360个原预估step |
| 优化器与调度 | AdamW、LR2e-4、weight_decay0.05、LLRD0.8、原分组/预热/poly、norm裁剪0.01 |
| 数值路径 | 首选保留16-mixed、默认torch.compile、matmul precision=medium |
| 损失 | 原采样12544、oversample3、importance0.75、mask/dice/class系数5/5/2、no-object0.1 |
| 训练选择 | 最后171–220轮完整12例、1-view Dice选择同一个best，原legacy空mask规则 |
| 最终推理 | 需要时固定scale_hflip8：1/.875/1.125/.75各配原方向/Hflip，不恢复旧旋转/垂直翻转 |
| 当前物理HD95 | compute_hd95_voxel=False，仅真实mm；体素列空白/JSON null，不重新打开体素计算 |

第1步LR0来自现有预热调度，不是新增冻结主干或为了复现修改学习率。不能取消增强、固定所有裁剪、改变采样点数、Loss权重或模型结构来换取“稳定”。

沿用已核验的实际离线timm预训练来源及hash；本方案不顺带接入其他pth、修改patch_size或转换预训练权重。关闭体素HD95是已单独完成的评估开关，不是CUDA训练确定性修复。旧诊断在该指标开关修改前完成，后续新实验应重新记录当前源码hash。

## 3. 独立实现范围与可恢复性

以下独立文件现已创建，原入口保留；实际结果见第12节：

- training/deterministic_mask_loss.py：独立损失类，复用原目标构建、匹配、系数和归一化。
- training/deterministic_point_sampling.py：保持点采样语义的候选确定性实现。
- scripts/run_synapse_deterministic.py：独立入口，在模型创建前设置确定性，保留原入口。
- configs/synapse/deterministic/：独立配置，明确CE/采样/确定性开关及协议ID。
- tests/中的对应数学、梯度、重复性测试。
- outputs/determinism/<protocol_id>/<run_id>/：配置、环境、hash、RNG、轨迹、性能与结果。

不直接改site-packages，不在生产流程用全局monkey patch替换Transformers函数。优先采用独立子类/局部依赖，使同进程的其他模型不会被悄悄改变。

如替换criterion，必须确认模块state_dict、参数及buffer键/shape/值、RNG消耗和原优化器分组均正确；不能因重新构造criterion或模型多消耗随机数。baseline与ASEE/CCPS分别核验，不能把baseline的388状态/373组硬套给创新模型。

旧源码、权重、CSV、readiness快照和历史协议保留。确定性配置关闭时的恢复行为必须测试；新实现应有独立readiness，不能仅刷新旧hash就宣称新协议验收完成。

## 4. 环境与严格检查设置

最近已核验环境为RTX5090D、PyTorch2.13.0+cu132、CUDA runtime13.2。实施前仍检查实际解释器、驱动、依赖、预训练hash及可用显存，不升级当前环境。

下列为独立入口的环境要求；本轮GPU算子/criterion验证采用这些前缀，完整模型有界验证已执行（最新见第14节）：

~~~bash
HF_HUB_OFFLINE=1 \
TORCH_HOME=/Disk/HDD/Sk/.cache/torch \
TIMM_USE_OLD_CACHE=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONHASHSEED=0 \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python <独立确定性入口> <已授权任务参数>
~~~

CUBLAS_WORKSPACE_CONFIG和PYTHONHASHSEED应在启动Python前设置并记录。PYTHONHASHSEED属于新计算协议的显式环境差异，不静默写入原baseline。候选入口在模型创建和首次CUDA操作前执行：

~~~python
import torch

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=False)
torch.set_float32_matmul_precision("medium")  # 保持原数值设置
~~~

这些设置是检测/约束，不是已经实现的算子修复（[官方开关说明](https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html)）。当前环境已经实测上述两个算子在strict模式下报错，不必再把“开开关后是否仍有问题”写成未知。

不要捕获报错后自动退回warn_only、关闭确定性、改FP32或关compile再继续训练。若有新算子报错，记录调用边界后停止该候选验证；CUBLAS配置和cuDNN选项不能保证覆盖所有算子。

记录TORCHINDUCTOR_CACHE_DIR、TRITON_CACHE_DIR、compile后端/选项、图断点/回退及CPU线程设置。候选缓存使用本方案独立目录，不清理或改写全局缓存及其他用户缓存。冷缓存重复与暖缓存重复分组验证；不把一个冷启动与一个暖启动的偶然一致当作完整结论。跨PyTorch/驱动/GPU或不同并发负载，不预先承诺逐位一致，通用边界见[官方复现说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。

## 5. CE首选候选：保持目标函数，改变归约路径

当前调用链是MaskClassificationLoss.loss_labels → Transformers Mask2FormerLoss.loss_labels → weighted CrossEntropyLoss。

实际分类logits为B×Q×10：9个语义类别（背景0及器官1–8）加no-object索引9。no-object不是背景；未匹配query仍以索引9监督，权重0.1，其余类别权重1。保持原匈牙利匹配、target_classes填充/索引与类别映射。

首选候选是保留同一个CrossEntropy公式，改输入布局：

~~~python
# 示意；候选实现仍需继承当前完整target构建逻辑。
loss_ce = torch.nn.functional.cross_entropy(
    pred_logits.reshape(-1, pred_logits.shape[-1]),
    target_classes.reshape(-1),
    weight=self.empty_weight,
    reduction="mean",
)
~~~

不能先transpose再错误reshape，也不能遗漏不连续tensor布局、ignore_index（若未来输入出现）、dtype及AMP行为。

对当前整数类别标签，weighted mean的分母是有效target的类别权重之和：

~~~text
L = sum_i(w[y_i] * CE_i) / sum_i(w[y_i])
~~~

以上按[CrossEntropyLoss的weighted mean定义](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)保留原整数标签目标。不能改成简单除以B×Q、匹配数量或mask数量；no-object占比变化时这些表达式不等价。外围class系数2保持原位置/次数，不重复乘权重。

现有孤立试验中，flatten2D候选30次稳定且strict通过；新旧loss仍有ULP级差异。因此它是数学目标保持、数值路径改变的候选，不能宣称与旧CUDA实现逐位相同。

候选验收覆盖：真实保存的criterion输入、不同batch/query/匹配数量、含背景与不含背景、所有未匹配query、no-object占比、非连续logits、原AMP/dtype及Loss/梯度对照。旧协议与新候选的数学正确性用CPU双精度参考和预先记录的数值误差门槛判断；新候选自身重复应严格一致，不能把两种验收混为一谈。

## 6. 点采样候选：先限定语义，再选择实现

当前损失输入为匹配后的pred_masks，实际sample_point封装调用F.grid_sample；坐标原为[0,1]，传入grid前映射为2*coordinates-1。保持二维bilinear、padding_mode=zeros、align_corners=False、相同shape/dtype/device/AMP边界。

必须保持以下流程和原顺序：
1. 原随机坐标draw；
2. 过采样位置的logits、uncertainty与top-k；
3. 选出的最终坐标及追加随机坐标；
4. GT point labels与pred point logits；
5. 原mask/Dice损失、匹配与归一化。

候选必须覆盖Matcher中、uncertainty中和最终监督中全部相关sample_point调用，不只替换最后一次函数。仅记录torch.rand相同不够，还需记录top-k结果、最终坐标及采样输出。出现uncertainty并列时要检查tie行为，不能擅自新增排序规则改变选点。

当前loss_masks对选点与GT采样使用no_grad；需要反向的是最终预测采样的input_features梯度。先按真实需要设计实现；若候选宣称通用grid_sample替代，还必须支持坐标梯度或对不支持输入显式报错，不能悄悄detach本来需要梯度的grid。

固定所有坐标、换nearest、减少点数、关闭uncertainty、改padding/align_corners均不是本Plan的等价修复。原点采样反向的重复累加必须以固定顺序处理；仅把它改写为含重复index的gather/scatter/index_add，不代表CUDA反向已经确定。

已授权后选用原生前向+stable-sort/分段固定树反向；其他方案不自动实施：
- 首选可证明重复性且保持上述语义的确定性GPU采样/反向；
- CPU双精度作为数学与梯度参照，保持自动求导连接；CPU回退不是默认生产方案；
- 自定义算子/内核仅在必要时引入，并单独记录构建、版本、compile兼容和性能；
- 新PyTorch版本只在独立环境另作候选，不能原地升级或把切版本当作已修复。

## 7. 分阶段验收与推进门槛

每个阶段完成后记录真实状态和证据；没有通过则停止推进，不自动进入完整训练。

| 阶段 | 验收内容 | 通过标准 |
|---|---|---|
| G0：入口/环境 | 科学配置、初始参数/buffer、RNG、373组优化器（baseline）、源码/预训练/数据hash | 与各自预定协议一致；新旧协议差异明确 |
| G1：CE单项 | 固定输入/targets/权重，Loss及梯度，strict检查 | 至少30次；数学对照达预定门槛，新候选重复逐值一致 |
| G2：采样单项 | 固定特征和坐标，前向tensor及input梯度 | 至少30次；真实点数、边界/越界/重复坐标、CPU参考/gradcheck、FP16/compile均验证 |
| G3：完整criterion重放 | 同保存输入、CPU/CUDA RNG、匹配、最终坐标、分项Loss/梯度 | 两修复组合后strict通过；完整重复记录一致 |
| G4：真实首步/第二步 | 固定Batch，再用原生DataLoader；AMP、裁剪、AdamW moments及参数 | 至少2个独立进程；捕获第1步LR0及第2步非零LR，不凭global_step判断真实更新 |
| G5：连续短轨迹 | 原生20–100个Batch/step并明确实际更新次数 | 所有已记录边界一致；冷/暖缓存分组，减少观察器干扰后再复核 |
| G6：完整重复 | 220轮、171–220完整选择及需要的同best8-view | 用户另行授权后至少2次从头独立运行；不自动开跑 |

数学等价误差门槛需在候选实验之前定义并落盘，不能看到结果后放宽。重复性门槛是新候选在固定条件下逐值一致；旧新实现之间不要求字节/浮点完全相同。不能仅用最终Dice接近、标量loss相同或strict不报错代替梯度/轨迹验证。

每次都核验输入、初始权重、RNG、随机点和匹配后，再解释数值差异。原GradScaler的overflow跳步必须保留；同时记录未缩放loss、缩放后loss、scaled/unscaled/裁剪后梯度、scale/growth tracker、真实optimizer.step次数、LR、exp_avg/exp_avg_sq与权重。不能为凑一次成功更新而手动调scale或将跳步计作权重更新。

至少保留一次真实GPU算子/短轨迹检查，不用CPU通过代替CUDA通过。观察器不额外消耗RNG；同步hash会改变时序，所以记录其开销，并用低侵入重复检查避免“只在重度仪器下稳定”的误判。

## 8. 结果一致性、性能与恢复状态

比较checkpoint时，按排序后的state_dict键、dtype/shape及tensor内容hash判断；若评估训练状态，还比较optimizer/scheduler/GradScaler/RNG等相关内容。不要要求整个.ckpt文件SHA相同：保存路径、归档成员和元数据可能不同。权重内容与epoch/step、best选择及逐病例原始指标应分别核验。

日志时间、run路径、UUID、W&B ID不参与科学数值相等判定；记录它们用于追溯。训练指标比较同一记录粒度与聚合方式，不能拿显示四舍五入值代替原始数值。器官名按verified label_id映射，8器官来自同一checkpoint，不逐类跨epoch挑最优。

现有best/last是weights-only，不能精确续训。若未来需要完整恢复，另行保存模型/优化器/调度/GradScaler及RNG、数据顺序等状态；persistent workers与预取状态也需设计。这个Plan的完整重复指从头重复，不承诺中断恢复逐位一致。

计时区分编译冷启动、GPU前后向、采样/损失、CPU指标与端到端；GPU计时采用适当同步。记录峰值allocated/reserved与nvidia-smi进程峰值，不能用权重大小估算。性能下降或显存增加超出资源预算时报告取舍，不自动减batch、改precision、关闭compile。

服务器运行仍遵守AGENTS：其他用户任务存在时，剩余显存足够峰值加开销即可运行；不足则等待，不停止他人任务。本项目受控对照串行执行，实施前重新核验现场资源，不把过去的余量当作当前可用值。

## 9. 论文对照与历史结果

旧baseline的84.7185%和baseline-2的85.9138%是已保存的两次1-view历史结果，不因此作废或覆盖。原ASEE/CCPS各组成绩、选择窗口、提前停止与TTA需查各自原始记录，不在本Plan预填未经核验的汇总数。

如果最终决定把新确定性协议用于论文主对照，应按同一新协议和匹配预算/seed重做实际用于比较的baseline、创新模型及组件；这是后续待授权的实验安排，不是当前已批准的三套220轮训练。若仅做诊断，原协议成绩可以保留，但不能把新旧协议差值全部归因于创新模块。已有CCPS提前停止等预算差异也应说明。

确定性不等于更高Dice，也不等于跨seed训练方差消失。新入口不能为追求论文分数而放宽标准或筛掉失败/负面结果。需要多seed统计时另行授权。

## 10. 交付清单与当前待定项

每阶段应交付并据实记录：独立代码/配置、依赖和来源hash、真实命令/解析参数、输入/RNG/采样/梯度/优化器轨迹、数值误差与重复性报告、性能/显存、失败算子与回退记录、下一阶段状态。

实施状态与后续决定（以第12节实际结果为准）：
1. 确定性点采样的具体实现及compile支持；
2. 新旧数学对照的误差门槛与性能预算；
3. 若候选确需额外依赖/自定义kernel/独立环境，先提供具体变更再决定；
4. 短轨迹全部通过后，是否采用新协议及授权完整重复。

**当前G4/G5部分通过，CCPS尚未通过；最新状态及下一步见第14节。未启动完整训练或定时任务。**

## 11. 依据与引用

项目的直接依据：
- baseline_repeat_four_step_audit.md：四步服务器定位与范围。
- baseline_repeat_solution_plan.md：实证结果与候选修复边界。
- hd95_voxel_disabled.md：当前HD95_mm-only默认规则。
- 审计根下cuda_loss_ops_isolation.json、trajectory_detailed_comparison.json、gpu_validation_summary.json及真实输入重放文件。
- 当前training/mask_classification_loss.py、安装Transformers5.17.0的Mask2FormerLoss.loss_labels/loss_masks/sample_point。以后实施仍检查实际版本与源码hash。

正式文档仅用于通用语义，不能替代本服务器实测：
- [PyTorch：确定性算法开关](https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html)。
- [PyTorch：CrossEntropyLoss加权mean语义](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)。
- [PyTorch：复现条件与随机性](https://docs.pytorch.org/docs/2.14/notes/randomness.html)。

检索时官方stable文档指向2.14；本项目最近运行版本为2.13.0+cu132，因此不再写“2.13文档保证本例grid_sample可确定”。本例两个strict报错以已保存的实际GPU记录为准。

## 12. 初始实施与测试记录（历史阶段，最新见第14节）

用户已明确授权“根据plan进行修复，修复完之后进行测试”。实现/逐项变更和完整证据见`.codex/cuda_deterministic_implementation.md`。

- 独立协议`ce2d_stableseg_v1`：2D weighted CE、原生前向+固定分段反向采样、局部loss子类、独立CLI、baseline/ASEE/CCPS三份配置。原baseline和历史结果保留；100份开工时既有文件hash无改动。
- CPU回归30项通过；三模型真实入口初始化/RNG/优化器分别388/373、419/401、433/415一致；disabled恢复对照通过。这只是CPU边界，不是GPU整模型训练通过。
- G1/G2：各8种GPU条件、每种30次，严格重复通过。G3：真实保存的batch16 criterion输入，两种包装各30次重复通过，原匹配/最终坐标/RNG一致，数学误差均在预先阈值内。继续保持FP16/default outer compile和HD95_voxel=false。
- G4/G5：剩余18192MiB低于22000MiB安全门槛，未执行，未停止其他用户任务。损失单项中位28.4465ms→45.0091ms，整模型速度/显存待验证。
- 独立readiness目前passed=false、validated_profiles为空；正式长训保护已验证。G6仍需另行授权，不能把当前结果表述为完整同seed训练已经可逐位复现。

本轮没有更改baseline科学参数，也没有220轮重训或新增定时任务。初始metadata/CPU测试夹具问题均在独立代码内修正并重验，不把其失败当作训练故障。


## 13. G4/G5接续启动记录（历史阶段，最新见第14节）

用户已授权继续未完成测试；服务器GPU释放后，独立完整模型短测已启动。baseline固定Batch双进程与原生20Batch冷启动双进程已通过，其他阶段仍执行中。动态证据见`.codex/cuda_g4_g5_validation.md`及outputs/determinism/ce2d_stableseg_v1/full_model_20261027T121833Z/；readiness仍false，不再将当前阻碍写成旧18192MiB显存不足。没有启动G6或正式长训。


## 14. G4/G5最终部分通过与CCPS下一步Plan

本轮GPU诊断均已退出。baseline和ASEE固定Batch、原生20 Batch冷/暖缓存、低观察器及公共边界交叉比较通过。CCPS首次有限scaled梯度已不同，后续有效更新分叉，G4失败，G5未执行。CPU回归40项通过。证据见cuda_g4_g5_validation.md与full_model_20261027T121833Z/final_validation_summary.json，不再将当前阻碍写为显存不足。

末端BN局部编译边界候选双进程各3 Batch/2次实际AdamW、70阶段一致，但未做G5，未接入生产。strict无报错/某一对通过都不是充分证明；观察hook能改变现象，不能凭此唯一归因BN或认定CCPS数学设计有错。

### 下一轮顺序（用户已要求继续，正在独立接续）

| 顺序 | 工作 | 验收与停止条件 |
|---|---|---|
| P1 | 保留失败/通过输入、源码、缓存及编译/反向上下文；候选逐一撤除额外开关，每次只改一个条件 | 至少3独立冷启动进程；不以某一对偶然通过放行 |
| P2 | 核对局部BN边界前后state、参数ID、优化器组、RNG、BN统计更新次数和前向/输入梯度数学参照 | 不改BN公式、eps/momentum、FP16、共享权重或原outer compile；数学阈值预先落盘，重复性仍严格相等 |
| P3 | 同一最小候选重做CCPS固定Batch G4，覆盖原AMP跳步、正LR更新、scaled/unscaled/clipped梯度、moments及参数 | 记录全部边界；有差异则继续缩小编译分段/共享解码器反向累加边界，不先改结构 |
| P4 | G4通过才重做CCPS G5：20–100原生Batch，冷缓存对、暖缓存对、低观察器重复和交叉公共边界 | 不依赖观察hook；输入/采样/RNG/loss/optimizer/最终状态严格一致，任何差异停止推进 |
| P5 | 候选G4/G5全通过后考虑独立新协议版本，再重验baseline/ASEE | 不覆盖v1失败记录，不用刷新hash代替重验，不热切入原生产入口 |
| P6 | 最终门槛满足后两次从头完整220轮纯baseline，同seed0/原增强/预算/预训练/列表，作为G6 | 完整171–220选择窗口，比较参数内容hash、逐轮/逐病例指标、同checkpoint；不以整个ckpt归档字节相同为必要条件 |

待验证假设：①编译分段/内核选择改变反向归约，局部稳定边界应在不同冷暖缓存中持续消除差异；②共享解码器多路径梯度累加受执行次序影响，应在相同梯度入口及低观察器中复核；③观察器/调度仅暂时掩盖差异，新增进程或低观察器会使其重现。以上不能写成已确认根因。禁止为稳定而取消多层融合、改FP32、改损失或关闭全部compile。

### G6条件授权

用户要求G4/G5通过后，以当时基准安排3h第一次baseline G6、3.5h检查、8h第二次、8.5h检查；未通过则完善Plan。纯baseline按同一新协议从头完整重复两次可以作为G6。当前整体未通过，故执行Plan分支：四个任务未创建、G6未启动。不自动将“baseline有界通过”改成“整个协议通过”。

未来进入G6时两run独立目录与共享幂等锁，第二run不能抢先于未结束的第一run；检查时间不是超时，不停止正常训练，失败不覆盖重训，显存不足不改batch或停止他人任务。不设置无限轮询。


### CCPS接续验证已启动

用户最新“继续”授权按上表接续P1–P5。新证据根outputs/determinism/ccps_boundary_followup/20261028T030813Z/，说明见cuda_ccps_boundary_followup.md。优先只增加末端BN边界，在三个独立冷启动中对照；如果未通过，逐项增加backward-autocast对齐与编译控制，不同时盲改生产代码。CPU边界语义已核验，真实GPU状态以status/exit/comparison为准。原模型、训练设置及旧readiness不改，整体尚未通过前不触发G6。


## 15. CCPS接续验证完成与v2/G6落实

完整结果、失败控制、局部数学参照、真实入口与文件变更见cuda_ccps_boundary_followup.md。新readiness为outputs/determinism/ce2d_stableseg_execution_v2/readiness.json，当前G0–G5通过；旧v1 readyness仍失败，不覆盖历史。

已证实：仅BN边界的三进程包含分叉；仅off的三进程也分叉；组合候选三固定进程、原生20 Batch完整梯度冷/暖对及低观察器冷/暖对、跨缓存/跨观察器全部严格一致；局部GPU数学对CPU double参照通过事前门槛，30次重复一致。没有使用额外编译确定性flag。真实v2入口数值结果与验证候选一致，baseline/ASEE与v1轨迹一致。具体所有CUDA内核因果尚未唯一锁定，不宣称模型公式出错或通用硬件问题。

G6已按验收完成时当前时间落实四条单次任务。基准Denver2026-10-27 22:26；2026-10-28 01:26/01:56/06:26/06:56。严格两次从头纯baseline seed0/220轮，启动器scripts/launch_synapse_g6.py，配置configs/synapse/deterministic_v2/baseline.yaml，输出outputs/determinism/g6_baseline_execution_v2/。第一轮完整完成后第二轮才可开始；异常不自动重训，检查不等于超时。当前没有正式G6在运行。具体验收和任务ID见synapse_g6_runbook.md。
