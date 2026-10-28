# Synapse HD95毫米接口状态（2026-10-21）

用户本轮明确授权把HD95增加为真实物理距离计算。独立实现和使用方法见 [docs/synapse_physical_hd95.md](../docs/synapse_physical_hd95.md)。原baseline仍按体素网格口径完成，不热切换、不混写原CSV；新增配置为 configs/synapse/vit_query_mul_scale_fusion_hd95_mm.yaml，科学训练参数保持第三套原值，只有评估单位接口和输出目录不同。

真实间距表约定 priors/synapse_spacing.verified.json；原始标签预留 project_TransUNet/data/Synapse/raw_nifti/labels/。目前12例H5均无spacing，在已检查的当前/旧EoSeg数据目录未找到原始NIfTI/DICOM/间距表。尚未生成verified表，尚未执行真实毫米评估；缺失元数据时直接报错，不回退为1、不假装mm。

新增核心/模型/回调、NIfTI匹配构建工具、CPU回归和说明；保留原Dice及其已知空掩码规则，逐器官输出名称、Dice、HD95_mm、HD95_voxel和空掩码状态。CPU/CLI和受保护源文件哈希核验结果见 outputs/audit/physical_hd95_20261021/verification.json。

最终核验：58/58项CPU检查通过（新增14项），配置解析成功，47个既有源文件/配置哈希无漂移；真实spacing仍缺失，未执行真实GPU毫米重评。独立测试导出使用checkpoint真实epoch并记录路径/哈希，已用合成177轮权重验证。

## 用户要求执行后的实际结果（2026-10-21）

已执行 spacing --validate-only 前置校验，因真实间距表仍缺失退出1。当前/旧项目、个人SSD和三个现有压缩包未发现原始NIfTI；其他个人挂载不可访问。官方Synapse API的本轮三个请求均HTTP403/GEO_RESTRICTION。未规避限制、未假设spacing、未启动GPU重评或改动原baseline。执行证据：outputs/physical_hd95/preflight_20261022T034254Z/execution_status.json、spacing_preflight.log、execution_report.md。待提供已取得的原始NIfTI标签或可靠逐病例几何数据；此前CPU接口验证不能代替真实毫米结果。
