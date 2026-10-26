# Synapse_RawData原始数据与HD95_test.py核验

结论：原始数据可用于本实验真实物理HD95，已经核验；老师脚本作为参考保留，不直接替换当前EoSeg评估。

## 数据对应与几何

Synapse_RawData/RawData包含Training/img的30份imgXXXX.nii.gz、Training/label的30份labelXXXX.nii.gz，以及Testing/img的20份图像。30组训练图像/标签的size、spacing、origin和direction均一致。当前18+12划分对应这30份有标签病例，当前12例的原始标签均在Training/label中；不需要用网站另外20例无标签Testing替换测试集。

已执行scripts/build_synapse_spacing.py：逐例检查NIfTI明确mm单位、正交网格，原BTCV13类映射到当前8类，再核对轴排列/翻转；12例完整标签与H5逐体素完全相等，无需猜测裁剪/重采样。随后--validate-only再次通过。新建priors/synapse_spacing.verified.json，包含逐病例ZYX间距、shape、标签哈希、原始标签文件哈希与匹配变换。

例如case0001的spacing ZYX=(3.0,0.66796875,0.66796875)mm；case0022=(5.0,0.7675780058,0.7675780058)mm。Z间距分别有3mm和5mm，XY也逐例不同，不能统一写成1或直接乘旧HD95标量。

## 老师脚本的适用性

HD95_test.py是AgileFormer2D测试入口，依赖config.get_config、datasets.dataset_synapse、datasets.dataset_ACDC、networks.agileFormer_2d及utils。这些并非当前EoSeg入口，不能直接拿它加载EoSeg checkpoint。脚本在顶层parse_args，要求--cfg；本轮只读，没有导入或执行。

关键test_single_volume由from utils import *引入，上传文件没有给出该函数和底层calculate_metric_percase定义，根目录也没有utils.py。因而不能从此入口核验实际HD95公式、connectivity、双向距离聚合、95分位或空掩码规则。

入口中Synapse的z_spacing固定为1，未看到读取逐病例XY/Z间距或原始NIfTI的代码；但不能据此断言底层函数一定没传spacing，因为底层函数尚缺。z_spacing也可能仅用于保存预测文件的header，传了这个参数并不能证明指标计算使用它。

建议沿用本项目既有独立物理评估接口，在恢复到H5原网格的同一预测上调用MedPy hd95(pred,gt,voxelspacing=(sz,sy,sx))；同时保留voxel结果、器官名称和原空掩码Dice规则，不修改训练、划分、精度/compile或添加后处理。该接口无需老师入口才能运行；如要对齐老师的指标定义，还需要对应utils.py及其调用的指标函数。

## 本轮范围

只完成原始数据核验和间距表构建，尚未重新推理baseline/ASEE checkpoint，也未产出毫米HD95分数。没有修改HD95_test.py、原始数据、既有模型/训练/评估源码或历史CSV。后续评估可直接使用已训练checkpoint，无需重新训练。
