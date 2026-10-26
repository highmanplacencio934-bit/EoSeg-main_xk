# HD95再次复核：同GPU/FP16/compile的真实预测证据

日期：2026-10-21 America/Denver。本轮针对用户再次质疑“Dice高而HD95很高，公式是否算错”进行进一步只读诊断，不修改baseline。已读取AGENTS.md和前次诊断结果；采用实际调用链检查、原函数重放与掩码干预对照，不把重复调用同一个库当作全部证据。

## 结论与范围

已在第177轮best_test checkpoint的case0035上，使用服务器RTX5090D、CUDA FP16 autocast、torch.compile复核。全部8个前景类别的Dice和HD95与原CSV精确一致，最大绝对差均为0。该病例的肝脏高HD95确实由图像边缘的远处误检碎片造成；非空HD95公式、传入类别掩码和均值汇总未查出实现错误。

这次只执行了用户明确授权的1例GPU复核。其余case0029、case0003、case0038仍只有前次CSV分解，未逐例重推，不能把case0035的因果证据冒称为四例全部都已定位。

## 原始路径重放与资源边界

- checkpoint自带epoch=176（0-based），对应177轮；完整388个state_dict键strict加载成功，并在转GPU前逐值核对全部权重/缓冲区一致。
- 使用原models.vit.ViT、EoSegMultiQSegFusion、MaskClassificationSemantic及原evaluate_synapse_case；没有另写预测解码。审计初始化跳过预训练下载，随后加载完整训练checkpoint，主干也来自该checkpoint，不是随机初始化或另换预训练。
- 单病例94张切片，原384输入/patch16；matmul precision=medium，CUDA FP16 autocast、no_grad/eval、compiled forward，与原训练中测试的上下文一致。指标入口只增加掩码拷贝，不改其输入和返回值。
- 用户先前“不争抢GPU”限制导致第一次并发尝试被自动审批拒绝；该次命令未启动。随后用户明确选择“允许现在并发执行单病例GPU复核”，并补充显存不足时的处理意见。本轮例外仅用于此单病例，不是长期并发GPU授权，也不授权创新训练。
- GPU复核设置PyTorch显存fraction上限0.17；实测peak allocated=1,334,810,624 bytes（约1.24GiB），peak reserved=1,358,954,496 bytes（约1.27GiB），无OOM。进程已正常退出并释放GPU；只读检查确认原baseline PID3477678继续运行。未停止/重启/改写原训练，未声称并发计算对速度完全没有影响。

## case0035：实际数值与因果对照

|项目|实际值|
|---|---:|
|原GPU CSV肝脏Dice|95.259680%|
|本次GPU重放肝脏Dice|95.259680%|
|原GPU CSV肝脏HD95|181.231896体素|
|本次GPU重放肝脏HD95|181.231896体素|
|GT体素数|391,391|
|预测体素数|402,817|
|交集体素数|378,280|
|假阳性/假阴性体素数|24,537 / 13,111|
|GT连通域数（6连通）|1|
|预测连通域数|82|
|不与GT重叠的预测碎片|81个，12,420体素|
|移除这些碎片后丢失的GT交集|0体素|
|双向合并表面点数|129,826|
|表面距离超过100体素的点|9,515（7.3290%）|
|距离90分位数|2.236068体素|
|距离95分位数|181.231896体素|
|GT→预测表面距离95分位数|1.732051体素|
|预测→GT表面距离95分位数|218.091838体素|

独立使用二值腐蚀提取表面、SciPy EDT求最近距离并合并取95分位，得到181.23189564753773，与原metric入口完全相同；Dice直接用2×378280/(402817+391391)也与原函数一致。对两侧表面坐标作相同轴排列后，抽样独立KDTree距离最大差为0，排除了将相同坐标顺序误认成异常距离的解释。

诊断副本的对照：

|掩码|Dice%|HD95（体素）|
|---|---:|---:|
|原预测|95.259680|181.231896|
|仅保留最大连通域（诊断对照）|96.773038|1.414214|

该对照保持GT和所有指标公式不变，只从诊断副本移除81个无GT交集碎片。表明高HD95来自这些错误表面，不能归因于95分位公式的算术错误。最大连通域没有加入baseline，也没有覆盖原预测、CSV或checkpoint；这些对照值不是新baseline成绩。

较小的误检体积不等于较少的误检表面：碎片仅占预测体积约3.08%，但贡献10,044个额外表面点；其中9,515个离GT超过100体素，占双向表面采样7.33%，超过了HD95最高5%的边界。因此90%距离仍小于约2.24体素，而95分位可以跳到181.23。

核验图real_prediction_liver_diagnosis.png：绿色为肝脏GT，红色为原GPU预测。肝脏主体两条轮廓基本重合，但图像最右侧、最下侧的背景区域可见成排红色误检碎片；这不是肝脏本身整体偏移181个体素。

## 对代码调用链的复核

1. training/lightning_module.py:204：image和label均取同一个病例的3D体积；仅去掉可能存在的batch维。
2. :230：每张图像用order=3缩放至384输入，灰度复制三通道；模型内部做原归一化。
3. :766：sigmoid(mask)乘softmax(class)，排除最后no-object通道，输出每个像素各类的分数。
4. :258：按通道argmax得到类别；此处没有“所有分数很低就归背景”的额外规则。
5. :262：预测类别用order=0恢复至原512×512，再写回同一个z切片；没有用order=3插值类别编号。
6. :271：以pred_vol==c和gt_vol_np==c形成同shape的布尔掩码，逐类计算。本次肝脏确认为c=5，未误传多类标签或其他器官。
7. :1010：非空分支调用MedPy0.5.2，内部距离为float64，没有将距离存成uint8或产生255截断。
8. training/synapse_selection.py：先按12病例等权汇总各类，再取8前景类平均；这里没有交换列、混epoch或把HD当HD95。

当前非空公式：HD95=percentile95(concat(预测表面→GT表面最近距离,GT表面→预测表面最近距离))。无voxelspacing时按单位网格计算，单位为体素，不能标毫米。前一轮20组KDTree与体素计数交叉检查均通过；本轮又在真正的GPU预测输入上独立重算通过。[MedPy作者源码](https://loli.github.io/medpy/_modules/medpy/metric/binary.html#hd95)。安装版本源码已在前次outputs/audit/metric_diagnosis_20261021归档；不以旧版本网页代替环境核验。

解码策略会保留微弱的错误前景响应：只要肝脏分数比背景更高，argmax就会选肝脏，不要求它达到某个额外置信阈值。单切片CPU FP16探针发现，远处误检在384空间、恢复512之前就存在（582体素），因此不是恢复尺寸后才凭空产生的类别。其sample(0,373)的肝脏分数约8.33e-5、背景约2.15e-5，仍选为肝脏。分数是query加权的语义分数，不能简单当成已校准概率，也不能未经实验就硬加某个阈值。

本轮定位到的是“低分前景响应被当前解码保留，并形成边缘碎片”的直接机制；没有通过训练干预确定最初产生响应的权重、位置编码、上采样或增强中的具体训练原因。未把棋盘状外观等同于已证明某个ConvTranspose层有代码bug。

## 补充：缺少真实spacing是否能解释高值

用户提供的网页分析对单位的判断正确：当前未传voxelspacing，报告的是单位网格距离，原12病例肝脏平均HD95=75.167970不能标为mm。缺少真实spacing会影响物理距离的数值和不同病例之间的比较，但不能单独证明当前单位网格的异常高值由此产生。

本次直接读取case0035的原H5：只有image、label两个数据集，文件和两者的attributes均为空，没有可用的真实spacing。本轮没有从shape猜测spacing，也没有把某个假定Z间距传入baseline。

在已保存、与原CSV精确匹配的GPU预测上，补做CPU几何分解：

|诊断条件|HD95|
|---|---:|
|原3D单位网格|181.231896体素|
|忽略Z坐标，仅测XY距离，保留原3D表面样本权重|181.069048像素网格单位|
|原3D单位网格，诊断副本仅保留最大连通域|1.414214体素|

第二行使用原3D表面点的XY投影集合寻找最近距离，仍按原3D各点采样、双向合并后取95分位。它是去掉Z约束后的几何下界诊断，不是真实spacing指标，也不是可替换baseline的2D评估。病例Z尺寸94，Z坐标最大差仅93；去掉Z后仍有7.3290%的双向表面点XY距离超过100，证明本例的高值主要来自平面内边缘误检。仅修正Z间距不能消除这些XY距离。真实XY间距仍会改变最终毫米数值，目前不能推算其大小。

原始预测与仅移除无GT交集碎片的诊断副本都使用相同的单位网格公式，HD95从181.23降到1.41。这个对照与是否拥有真实spacing无关，进一步支持误检表面是本例高值的直接原因。没有将此因果结论扩展为其余三个异常病例均已逐例证实。

要补充物理距离，须找回原NIfTI affine/spacing、DICOM信息或预处理记录中的逐病例spacing，并核对其与当前H5的轴顺序、裁剪和重采样的关系。不能对所有病例的平均75.168直接乘同一个猜测系数。以后如补算mm结果，应另列结果并保留原baseline口径。本轮未更改任何指标公式或正式结果。

证据：outputs/audit/hd95_replay_20261021/gpu_case0035_20261022T024443Z/spacing_direction_review.json。

## CPU/精度差异：必须保留的另一项发现

此前没有原始预测，所以先做了CPU FP32重放；这次CPU结果为肝脏Dice89.618640%、HD95269.432552，明显不同于线上，不能用作原模式的精确复现。其保存图和mask明确标CPU，未与最终GPU证据混用。

同一真实切片的CPU FP32/FP16比较：FP32有3,395个远处误检体素（384空间），FP16有582个；FP16有6,754个像素的所有类别分数都变为0，FP32为0个。在一些背景位置，FP32肝脏响应仅1e-10左右，FP16下溢为0，argmax的胜出类别会改变。这解释了为何低置信区域可对精度/内核很敏感，但不表示降低精度是正确的后处理，也不授权把baseline改为FP32。

本次最终结论以GPU FP16/compile的单病例全16项指标精确匹配及其保存掩码为依据。CPU探针仅用于解释敏感性和检查缩放环节。

## 已知其他问题与处理边界

前次发现的“pred非空、GT为空返回Dice=1”问题仍保留，未修改；本例肝脏GT为391391体素，未走该分支，所以它不是本次肝脏HD95的原因。本例胆囊GT为空的相关结果仍按原baseline保存，不将修正后的反事实值混入原成绩。

若以后增加最大连通域、背景置信规则或体外区域过滤，都会改变推理/评估口径，应作为单独明确的实验比较，不能静默覆盖baseline。还不能推断作者论文使用了哪种后处理。

## 本轮文件变更与复现

新增outputs/audit/hd95_replay_20261021下的replay_liver_cpu.py、probe_decoding_cpu.py、replay_liver_gpu.py、analyze_captured_masks.py，及各自manifest/result/progress、原掩码压缩文件、表面/连通域分析JSON和证据PNG。gpu_case0035_20261022T024443Z为最终同模式核验目录，CPU同名产物单独保留。新增.codex/hd95_replay_review.md（本说明）；更新.codex/synapse_metric_diagnosis.md，指向这次已完成的实证，保留前轮历史。

服务器命令均保留HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1前缀，使用DeepLearning_xiakun环境。GPU并发需新的明确授权；脚本默认不执行，活动baseline时还需--allow-concurrent-gpu保护标志。此次已授权的一例已完成，不自动重复。

分析重放保存的GPU掩码（CPU执行，不会再推理）：

```bash
cd /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse
HF_HUB_OFFLINE=1 TORCH_HOME=/Disk/HDD/Sk/.cache/torch TIMM_USE_OLD_CACHE=1 \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python \
  outputs/audit/hd95_replay_20261021/analyze_captured_masks.py \
  --replay-dir outputs/audit/hd95_replay_20261021/gpu_case0035_20261022T024443Z
```
