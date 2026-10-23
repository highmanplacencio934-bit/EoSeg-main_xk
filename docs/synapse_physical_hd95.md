> 2026-10-22更新：averaged的238份NIfTI有mm间距且MD5均匹配，但83份标签均为二值，与12例H5无任何轴排列下的shape候选，不能构建verified spacing。ASEE已完成220轮。详见 `.codex/averaged_hd95_geometry_audit.md`；下文仍在训练的描述为首次准备时历史状态。

# Synapse 真实物理距离 HD95

2026-10-21，用户明确授权增加毫米口径。当前 baseline 正在运行，原代码、主配置、CSV、checkpoint 不变；新增独立入口供有真实几何元数据的后续训练或独立测试使用。当前缺少原始 spacing，因此接口已准备，真实病例的毫米成绩尚未生成。

## 几何数据的位置和来源

- 原始标签 NIfTI 预留位置：`project_TransUNet/data/Synapse/raw_nifti/labels/`；例如 `label0001.nii.gz`。当前只有上一级 README，没有原始 CT/标签。
- 核验后的间距表：`priors/synapse_spacing.verified.json`。当前没有生成；不能用示例文件冒充。`configs/synapse/spacing_manifest.example.json` 故意使用 null 和 verified=false，不能运行。
- 每个病例分别记录 `spacing_zyx_mm=[sz,sy,sx]`、H5 `shape_zyx`、规范化标签 SHA256、原始来源和几何映射证据。毫米值对应最终恢复到 H5 网格的预测，不是384输入网格，不再乘384/512。
- 12例 H5 目前只包含 image/label，文件与数据集 attrs 全为空；在当前项目数据目录及旧 EoSeg 目录中未找到原始 Synapse NIfTI/DICOM/spacing 文件。这不是已穷尽服务器所有位置的声明。

## 构建和校验 spacing

`build_synapse_spacing.py` 仅在用户提供真实原始标签后执行；本轮没有运行正式构建。它使用已有 SimpleITK，要求 NIfTI-1 头明确声明 mm、3D正交网格和正的间距。SimpleITK 数组是Z,Y,X，GetSpacing是X,Y,Z，必须反序；再根据与H5完整标签匹配得到的轴排列调整间距，翻转不改变正间距。

默认从原BTCV的13前景类中按已核验映射提取8类，其余类置0；若原标签已是Synapse8类，显式使用 `--source-labels synapse8`。工具逐病例要求重排/翻转后的标签与H5逐体素完全一致，保留全部匹配变换、文件哈希和来源。若经历无法解释的裁剪、重采样、类别映射不同、单位不明或存在导致不同spacing的轴序歧义，会报错，不猜数值；不修改原始文件或H5。只有全部病例成功才写出间距表，并拒绝覆盖已有表。

原始文件到位后，在服务器执行：

```bash
cd /Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse
HF_HUB_OFFLINE=1 TORCH_HOME=/Disk/HDD/Sk/.cache/torch TIMM_USE_OLD_CACHE=1 \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python \
  scripts/build_synapse_spacing.py \
  --raw-label-dir project_TransUNet/data/Synapse/raw_nifti/labels
```

已有表的只读校验：

```bash
HF_HUB_OFFLINE=1 TORCH_HOME=/Disk/HDD/Sk/.cache/torch TIMM_USE_OLD_CACHE=1 \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python \
  scripts/build_synapse_spacing.py --validate-only
```

## 毫米评估入口

新配置：`configs/synapse/vit_query_mul_scale_fusion_hd95_mm.yaml`。它完整保留主配置内容，只替换评估模型/回调类，添加间距表和独立输出位置。模型继承原类，forward、推理缩放/argmax、网络权重结构、损失、训练增强、数据划分、220轮、batch16、LR2e-4、LLRD0.8、AdamW、FP16和compile设置均不变。不是在当前运行进程里热切换口径。

`SynapsePhysicalHD95Semantic` 在fit/test的setup阶段先校验整个测试列表的spacing、shape和标签哈希，每例评估前再次核验真实标签。它继承原 `evaluate_synapse_case`，在同一预测上计算：

```python
metric.binary.hd95(pred, gt, voxelspacing=(sz, sy, sx))
```

保持MedPy原先的表面定义、connectivity=1、双向表面距离合并后95分位；同时保留不传spacing的体素值。Dice及按mean Dice选最佳checkpoint的逻辑不变。原始病例大小恢复后再计算距离，不改变预测，不添加最大连通域或任何后处理。非空物理距离不能由旧CSV的HD95标量直接换算：各向异性会改变最近表面点及95分位，必须对预测掩码重新计算。

独立测试命令（真实表已核验、GPU允许使用后才执行；本轮没有执行）：

```bash
HF_HUB_OFFLINE=1 TORCH_HOME=/Disk/HDD/Sk/.cache/torch TIMM_USE_OLD_CACHE=1 \
/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python main.py test \
  --config configs/synapse/vit_query_mul_scale_fusion_hd95_mm.yaml \
  --ckpt_path outputs/baseline/20261021T132635Z/synapse_test_selection/20261021T132645Z_81f3214a/best_test.ckpt \
  --compile_test --trainer.accelerator=gpu --trainer.devices=1 --trainer.logger=false
```

这里best_test仍可能在原训练完成前变化；正式报告必须核对实际checkpoint的epoch/文件哈希，不能默认仍是177轮。接口与原state_dict键兼容，无需为了换距离单位重新训练。当前只有CPU微型网络验证，没有执行真实12病例GPU毫米重评。

## 输出、器官名称与空掩码

输出在 `outputs/physical_hd95/` 的独立run目录，不复用原baseline目录。fit仍保留best_test/last和最后50轮的完整12例评估；新增回调仍按Dice选择权重，不按HD95改变选择。

- `metric_units.json`：声明主HD95为mm、轴序、spacing/名称表哈希、全病例几何校验和空掩码规则。
- `spacing_manifest.json`、`label_map.json`：本次使用的元数据副本。
- `physical_case_organ_metrics.csv`：epoch、case、label_id、英文/中文器官名、Dice原值与百分数、`hd95_mm`、`hd95_voxel`、本例spacing、状态。
- `physical_epoch_organ_metrics.csv`：同一epoch下每个器官对完整病例等权汇总，含空掩码病例数。不整理背景，不跨epoch逐类挑分。
- 独立test另写 `checkpoint_source.json`（实际加载权重的epoch/global_step、路径和文件哈希；未知epoch留空，不用测试循环轮数冒充）与 `physical_test_summary.json`；fit继承的 `test_*metrics.csv` 中通用 `hd95_*` 列在这个新run内代表mm，由metric_units明确标识；不能与旧run拼接。

空掩码分支是原baseline中另外一个已知问题，本次只更改距离单位，不顺带修改Dice或边界规则。pred非空/GT空仍继承Dice=1、HD95=0，单侧或双侧空的HD95=0都标成 `*_legacy_zero`；这些0是继承的占位规则，不是实际测得的物理表面距离。汇总仍沿用原规则并明确记录数量。后续若修正空掩码规则，需作为另一项显式口径变更，不能和spacing换算混为一谈。

真实物理距离可能更大或更小，不保证降低肝脏HD95。此前case0035的远处误检仍在，只改单位不会去除它。原作者单位网格结果与新mm结果应分别标注，不能直接作为同单位数值比较。

## 验证与当前缺口

CPU检查和CLI解析记录见 `outputs/audit/physical_hd95_20261021/`。测试涵盖三轴不同spacing的可手算位移、独立KDTree毫米坐标距离、spacing=1时与旧指标完全一致、空掩码规则不变、完整轴重排/翻转匹配、NIfTI单位验证、原始NIfTI到表的端到端流程、微型原推理路径的Dice/state_dict一致，以及微型fit/test的命名CSV。合成fixture的spacing仅用于测试，不是本数据集的真实spacing。

最终核验：58/58项CPU回归通过（原44项+新增14项），CLI配置解析通过；47个既有代码/配置/参考文件哈希无变化。新增检查以176（0-based）的合成checkpoint核对导出为177轮，验证实际checkpoint来源记录。baseline PID3477678仍在运行，本轮没有新增GPU进程。上述验证不代表真实病例的毫米评估已经完成。

待补：12例真实原始标签或有来源且与当前H5网格对应的逐病例spacing。未获得这些数据前，不能生成真实毫米成绩，也不会自动启动新的正式训练、GPU重评或覆盖旧结果。

来源：[MedPy作者源码](https://loli.github.io/medpy/_modules/medpy/metric/binary.html#hd95)、[SimpleITK空间与图像概念](https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html)、[NiBabel坐标与affine说明](https://nipy.org/nibabel/coordinate_systems.html)。运行实现以服务器实际安装版本和CPU交叉验证为准。
