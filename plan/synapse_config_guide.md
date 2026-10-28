# Synapse 五套配置说明

更新日期：2026-10-20。依据为当前项目的五份 Synapse YAML、models/eoseg.py、models/vit.py、training/pixel_classification_semantic.py、training/mask_classification_semantic.py、training/lightning_module.py 及 README。本文解释代码中的模型对照关系，不代表已完成训练或已复现论文成绩。

## 当前采用的配置与共同设置

用户已选定**第三套 vit_query_mul_scale_fusion.yaml** 为当前主实验；README 也将其列为主配置。完整复现参数、数据接口与服务器硬件约定见 [第三套主配置复现基线](synapse_reproduction_baseline.md)。其余四套保留作者原有设置，作为模型对照，不与第三套混用。

五套共用当前 Synapse 预处理数据和固定列表：
- 训练：project_TransUNet/data/Synapse/train_npz；18 个病例、2211 张切片。
- 测试：project_TransUNet/data/Synapse/test_vol_h5；12 个病例。
- 列表：project_TransUNet/TransUNet/lists/lists_Synapse 下的 train.txt 与 test_vol.txt；病例无交集，不重新随机划分。
- 输入 384×384，灰度复制三通道；9 类（含背景），训练 batch size=16，测试每批 1 个病例。
- 当前没有独立验证集；各 YAML 关闭验证。文件完整性按 fit/test 阶段分别检查，两份列表始终参与病例交叉检查。
- 训练均在服务器 RTX 5090D（32GB 显存）执行；硬件来源为用户约定，运行环境仍须在服务器实测。

## 对照总表

| 顺序 | 配置文件（configs/synapse/ 下） | 模型与主要作用 | Epoch | 基础 LR | 主干名称 |
|---|---|---|---|---|---|
| 1 | vit_deconv_384.yaml | EoSegDeconvSeg；反卷积像素分类基线 | 200 | 1e-4（模块默认） | 默认 DINOv2 名称 |
| 2 | vit_query_mask.yaml | EoSegQueryMaskSeg；只在输出头使用 query | 200 | 1e-4（模块默认） | 默认 DINOv2 名称 |
| **3** | **vit_query_mul_scale_fusion.yaml** | **EoSegMultiQSegFusion；最后层 query 决定三层预测融合权重；当前主实验** | **220** | **2e-4** | 默认 DINOv2 名称 |
| 4 | vit_query_mul_scale_fusion_mul_layer.yaml | EoSegMultiQSegFusionMultiLayer；三层 query 拼接决定融合权重 | 200 | 2e-4 | 默认 DINOv2 名称 |
| 5 | vit_query_mul_scale_fusion_vit.yaml | EoSegMultiQSegFusion；同融合结构更换普通 ViT 主干 | 200 | 1e-4 | vit_large_patch16_384 |

默认 DINOv2 名称为 models.vit.ViT 构造函数中的 vit_large_patch14_reg4_dinov2，前四套未显式覆盖。这里的“默认”是静态代码取值，不等于已经成功加载用户指定的权重文件。

## 1. 反卷积像素分类基线

配置：vit_deconv_384.yaml；网络：EoSegDeconvSeg；训练模块：PixelClassificationSemantic。

ViT 先提取图像 token，代码截取 patch 特征并恢复为空间网格，经过转置卷积、ReLU 和 1×1 卷积，直接输出每个像素的类别。转置卷积的 kernel_size、stride 在该网络中都写为 14；必要时插值回输入大小。没有 query，也没有三层预测融合。

损失为 0.5×CrossEntropy + 0.5×DiceLoss；CE 使用 ignore_idx=255，Dice 对类别 1..8 计算，不含背景。

该模块重写 configure_optimizers：直接对网络参数建立 AdamW，lr=1e-4、weight_decay=0.05。虽然构造签名也包含 llrd、warmup_steps 等参数，当前优化器方法没有应用它们，也没有返回学习率调度器。因此不能把第三套的 LLRD 和分阶段预热写成此配置的实际行为。

## 2. 只在输出头使用 query 的掩码分类对照

配置：vit_query_mask.yaml；网络：EoSegQueryMaskSeg；训练模块：MaskClassificationSemantic。

主干独立提取图像特征。num_queries 未显式给出时取 num_classes，当前为 9。可学习 query 经输出头生成 mask embedding 与类别 logits，mask embedding 与上采样图像特征组合生成 mask。query 不注入主干 transformer block，不与图像 token 一同经过这些 block。

损失使用带匈牙利匹配的掩码分类损失。基础 lr 取模块默认 1e-4；优化器继承公共实现，采用 LLRD 与分阶段预热、多项式衰减。该设置观察的是只在输出头使用 query 的行为；与第一套相比还改变了损失和优化方式，不能把性能差异全部归因于 query。

## 3. 当前主实验：query 注入主干与三层预测融合

配置：vit_query_mul_scale_fusion.yaml；网络：EoSegMultiQSegFusion；训练模块：MaskClassificationSemantic。

1. 使用 16 个可学习 query，在最后三个 transformer block 之前拼入 token 序列。
2. query 与原有 token 共同经过最后三层，分别收集三层输出。
3. 每层经过共享的归一化和预测头，生成 mask logits、class logits 和 query 特征。
4. 仅使用最后一层 query 特征，经门控网络生成每个 query 对应的三层 softmax 权重。
5. 同一组三层权重分别融合 mask logits 和 class logits，最终只返回一组融合预测。

保留 220 epochs、lr=2e-4、llrd=0.8、AdamW、weight_decay=0.05、warmup_steps=[500,1000]、poly_power=0.9。分阶段预热的具体执行顺序见基线文档，不将两个数解释为同时开始的独立预热。

三层预测参与融合，但当前只对返回的融合结果计算损失，没有三份独立辅助监督。num_blocks=3、masked_attn_enabled=False 在网络中固定；主 YAML 另设 attn_mask_annealing_enabled=False。

## 4. 融合权重来源对照：三层 query 特征拼接

配置：vit_query_mul_scale_fusion_mul_layer.yaml；网络：EoSegMultiQSegFusionMultiLayer。

总体仍为 16 个 query、最后三层注入与预测融合。关键变化在权重来源：主模型使用最后一层 query 特征，而本模型将最后三层 query 特征沿特征维拼接为 [B,Q,3C]，再交给融合门生成 [B,Q,3] 的权重。

它比较决定融合比例时是否利用三层信息。仍只返回融合后的 mask/class logits，没有新增三份辅助监督。该配置为 200 epochs、lr=2e-4；与第三套 epoch 不同，因此未经控制训练预算，不能声称两者仅有门控输入这一处实验差异。

## 5. 主干对照：普通 ViT

配置：vit_query_mul_scale_fusion_vit.yaml；网络仍为 EoSegMultiQSegFusion，显式设置 encoder.init_args.backbone_name=vit_large_patch16_384。

保留 16 个 query、最后三层注入，以及由最后一层 query 决定的三层融合结构。主干名称与前四套不同，且配置为 200 epochs、lr=1e-4。因此它是作者保留的另一套主干对照配置，并非只替换主干且训练预算、LR 都完全相同的严格单变量实验。

这份普通 ViT 配置不对应用户指定的 dinov2_vitl14_reg4_pretrain.pth。不要为了使用该文件，直接将其加载到 vit_large_patch16_384 主干。

## 阅读配置时必须保留的区别

- 文件名中的 mul_scale 在当前代码主要体现为不同深度层的预测融合；这些层的 patch 网格尺寸相同，不是 U-Net 式不同空间分辨率的特征金字塔。
- 不把 220 统一成 200，也不把其他配置改成 220；选择哪份配置，就采用它对应的 epoch、LR、主干、损失及优化器实现。
- 第一套使用普通 AdamW、无公共调度器；其余四套才使用公共的 LLRD 和分阶段预热实现。
- **patch_size=16 是用户明确确认的 baseline 设置**，保持 384 输入与 24×24 网格，不因名称中的 patch14 改成 14。各 YAML 的 ckpt_path 尚未显式指向用户本地文件；主干和预训练加载接口按用户计划后续改进。本机 timm 1.0.28 的权重过滤流程支持 patch kernel 重采样及位置编码调整，不能将 14→16 直接判为错误。
- 本文记录现有代码，不推测作者选择不同 epoch/LR 的原因，也不把对照配置解释为已经完成的实验结果。

运行前的完整细节、接口风险与证据层级见 [运行审计](synapse_baseline_run_audit.md)。

2026-10-20：运行审计修复状态已更新；第三套保持作者入口默认设备配置，修复启动/日志/checkpoint schema 并自动归档 run_manifest。模型及各自 epoch/LR 不变。
