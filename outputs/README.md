由于原始的输出文件包含checkpoints（文件内容太大，就不上传了）
现在的里面包含baseline和innovation的指标

以下是outputs木下面的一些文件夹的内容及作用
| ID | 唯一研究因素 | 对照/解释 |
|---|---|---|
| S1 | use_text=true,use_probability=false | 文本分支单独效果 |
| S2 | use_text=false,use_probability=true | 共现先验分支单独效果 |
| M1 | 两分支均启用 | 与S1/S2比较联合先验贡献 |
| S3 | enhance_patch=true,enhance_query=false | patch增强单独效果 |
| S4 | enhance_patch=false,enhance_query=true | query增强单独效果 |
| M1 | patch/query均启用 | 与S3/S4比较注入位置贡献 |

| ID | 模型与推理 | 训练增强 | 工作内容 |
|---|---|---|---|
| M0 | baseline + 当前最优8-view TTA | 原增强 | 使用原EoSeg baseline的best checkpoint，补做完整12病例测试 |
| M1 | baseline + innovation1（ASEE）+ 当前最优8-view TTA | 原增强 | 使用ASEE的best checkpoint，补做完整12病例测试 |


