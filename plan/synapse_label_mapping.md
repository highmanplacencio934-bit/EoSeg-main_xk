# 当前Synapse数据的标签与器官映射核验

核验日期：2026-10-21 America/Denver。范围是本工作区project_TransUNet/data/Synapse的实际预处理数据，不假定所有同名Synapse数据都使用相同重编号。

## 结论和证据等级

本批数据的语义映射通过实际CT/label叠加图核对：0背景；1主动脉、2胆囊、3左肾、4右肾、5肝脏、6胰腺、7脾脏、8胃。证据是两个测试体积case0001/case0008的逐类最大面积层与多层叠加图，以及两个训练病例case0005/case0009共10张切片交叉检查。训练抽样中case0005覆盖全部8类；没有将仅仅np.unique出现1..8当作器官名证据。

这是实际数据抽样的语义核验，不是原始转换脚本或原始NIfTI逐体素比对。H5只有image/label，文件属性为空，没有完整方向/spacing/标签名元数据。左右肾通过肝脏与脾脏的侧别关系交叉区分；绘图使用array[z].T，不能把未经转置的数组行列直接称作解剖左右。此处不假称已有人工临床复核或预处理来源的完整追溯。

|当前标签ID|器官|英文|原始BTCV标签ID（用于对照）|实物核对要点|
|---|---|---|---|---|
|1|主动脉|Aorta|8|椎体前方的增强血管横断面，跨多个头尾层面连续出现。|
|2|胆囊|Gallbladder|4|肝脏下方的胆囊轮廓；case0001较扁长，case0008较清晰。|
|3|左肾|Left Kidney|3|图像右侧、肝脏对侧的肾轮廓，与左侧脾脏的同侧关系一致。|
|4|右肾|Right Kidney|2|图像左侧、与肝脏同侧的肾轮廓，区别于label3。|
|5|肝脏|Liver|6|右上腹最大实质器官，轮廓跨左右叶。|
|6|胰腺|Pancreas|11|位于胃后方、横行跨中线的胰腺实质轮廓。|
|7|脾脏|Spleen|1|左上腹、胃外侧的脾实质；case0009训练图也清楚支持此对应。|
|8|胃|Stomach|7|上腹部胃腔及胃壁；case0001内有明显高密度口服对比剂。|

## 为什么不能直接复制其他映射

- 本地project_TransUNet/README.txt声明数据来自TransUNet/Synapse，但没有提供重编号脚本或标签名表；本次补充实物证据，不将README当成完整来源链。
- 原始BTCV 13器官ID可对照MONAI官方示例：[MONAILabel Radiology](https://github.com/Project-MONAI/MONAILabel/blob/main/sample-apps/radiology/README.md)。那套原始ID不能直接用于当前8类版。
- H-SAM作者仓库给出的另一套8类重编号为1脾、2右肾、3左肾、4胆囊、5肝、6胃、7主动脉、8胰腺：[H-SAM preprocessing](https://github.com/Cccccczh404/H-SAM/blob/main/preprocess/README.md)。它是该项目的真实定义，但与这里实际标签图明显不同，不适用于本批数据。
- 论文表格按器官排列，不等于数组标签编号；没有用论文表格的排列或每类Dice高低推断映射。
- 原参考build_prior.py的8行文本次序恰好与本次核对一致；这只是在实物核对后确认可复用其文本，不把参考代码自身当来源证明。四份Plan/参考文件保持原样。

## 现有显示名称问题

scripts/visualize_synapse_results.py的SYNAPSE_CLASS_NAMES现为background、Spleen、Right Kidney、left_kidney、right_kidney、Gallbladder、Esophagus、Liver、Stomach。对应当前数据时，ID1、2、5、6、7名称错误，且出现重复右肾。该表在此脚本中用于图例显示，baseline训练/CSV指标按数字ID计算，不因本次补名改变。按本轮核验范围未自动改写该脚本；后续修正显示名称应使用上表。

## 文件与使用范围

- 核验图、mask_geometry.json、training_sample_check.json在outputs/audit/organ_mapping_20261022T014306Z；这些只读检查没有推理或更改标签。
- priors/synapse_label_map.verified.json记录有序8类、英文文本、核验范围和来源，可作为后续ASEE构建器的label-map-json输入。verified仅指本次明确记录的抽样语义核验；original_preprocessing_provenance_verified仍为false。
- 构建器未来仍只统计训练集，本次未执行正式先验生成或创新训练。未从测试体积构造共现矩阵、概率或任何数值训练先验。
- 上一份第177轮指标表新增metrics_with_organs.csv/md，只添加本次核实的名字；指标数值和原始CSV不变，不把中途结果改称220轮最终结果。
- 历史审计中的“映射未核验”是当时状态；以本记录更新其当前语义状态，原始数据处理来源仍未完成验证。

## 后续训练结果汇总约定（用户于2026-10-21确认）

后续baseline与创新训练的逐类结果，按`priors/synapse_label_map.verified.json`的label_id匹配`dice_class_{id}`/`hd95_class_{id}`，同时展示器官名称和标签ID，不只输出数字类别。Dice须区分0..1原值与百分数，HD95标体素，不列背景；注明所选epoch/checkpoint。各器官来自同一选定checkpoint，不跨epoch逐类拼出最佳分数。此约定已写入根目录AGENTS.md，属于汇总展示要求，不修改baseline算法或当前运行。
