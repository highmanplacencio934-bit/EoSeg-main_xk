# Synapse 论文消融实验清单与执行约定

更新：2026-10-25。状态：实验设计与执行约束；本文件不代表所有实验已完成。

## 1. 适用范围与期刊依据

目标期刊尚未指定，因此不能宣称某个期刊强制要求以下实验数量、三种seed或具体模块组合。本清单以支撑论文创新主张、控制变量和可复现报告为原则。目标期刊确定后再核对其作者指南。
参考：医学影像AI的[CLAIM 2024](https://doi.org/10.1148/ryai.240300)强调数据划分、模型和评估的透明报告；[Nature Portfolio报告规范](https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards)提供代码与数据可用性的要求。这些来源不规定本项目具体消融矩阵，下述矩阵是针对ASEE/CCPS设计的研究建议。

本轮用户确认：论文不安排有/无TTA消融。所有论文模型与组件对照统一固定8-view；baseline和baseline+ASEE直接补测已有best checkpoint，无需重新训练。已有1-view结果作为选权重和内部诊断记录保留，不包装成另一项论文创新。

## 2. 需要补做的两项主模型实验

本节仅安排以下两个补测实验，均使用各自已有best checkpoint，不重新训练。

| ID | 模型与推理 | 训练增强 | 工作内容 |
|---|---|---|---|
| M0 | baseline + 当前最优8-view TTA | 原增强 | 使用原EoSeg baseline的best checkpoint，补做完整12病例测试 |
| M1 | baseline + innovation1（ASEE）+ 当前最优8-view TTA | 原增强 | 使用ASEE的best checkpoint，补做完整12病例测试 |

两组使用同一个已固定的scale_hflip_8方案。“当前最优”指此前完整模型已测试方案中表现最好的8-view，不代表已证明它在baseline或ASEE上也最优；本次不为各组重新搜索TTA。

M1−M0用于比较统一8-view下ASEE的整体贡献。已有完整模型（baseline+ASEE+CCPS）的8-view结果仅作为历史参考，不另设M2补测任务；独立联合增强训练也不列为本节M3，不因本清单追加或重跑。涉及CCPS或联合增强贡献的结论仍须单独核验公平对照，不能仅由M0/M1推导。

最新指令已授权按定时队列执行M0/M1，先14:56早停旧任务，再完成评估加速验证；本次设置阶段不提前启动。后续核验权重、入口兼容性及GPU资源，使用独立输出目录；不覆盖旧结果、不重新挑checkpoint，不影响已有训练任务。

## 3. 固定8-view的严格定义

使用现有scale_hflip_8：尺度1.0、0.875、1.125、0.75，每个尺度包含原方向及水平翻转，共8视角；保持384画布、现有插值/居中裁剪补边/逆变换/有效覆盖融合和FP32分数融合，再argmax。
不加入垂直翻转、90/180度旋转或左右肾通道交换，不换成旧mirror4/D4。所有模型同一实现、精度、compile测试方式和后处理。

不能直接把CCPS专用YAML套在baseline/ASEE权重上。先核验模型包装类、state_dict键/shape、缺失及额外键、原1-view一致性，再接通同一TTA接口；任何适配独立实现，不覆盖原入口。固定8-view属于推理方法，论文Methods仍需明确说明；不做有无TTA表不等于隐去TTA。

## 4. ASEE内部机制：已纳入后续定时队列

用户已明确授权S1→S2→S3→S4在M0/M1后依次独立训练。统一原增强、seed=0、相同原训练预算与固定8-view最终测试，锚点为M1。

| ID | 唯一研究因素 | 对照/解释 |
|---|---|---|
| S1 | use_text=true,use_probability=false | 文本分支单独效果 |
| S2 | use_text=false,use_probability=true | 共现先验分支单独效果 |
| M1 | 两分支均启用 | 与S1/S2比较联合先验贡献 |
| S3 | enhance_patch=true,enhance_query=false | patch增强单独效果 |
| S4 | enhance_patch=false,enhance_query=true | query增强单独效果 |
| M1 | patch/query均启用 | 与S3/S4比较注入位置贡献 |

保留patch_update=plan、文本投影可训练等其他设置。关闭分支需要独立训练，不能只在已训练完整权重上关闭然后称作训练消融。若只做测试时关闭，应标明推理干预。开关是否真正断开对应路径、参数是否进入优化器，均须实现核验。

冻结文本投影、plan/reference_gated比较作为可选扩展；只有论文强调这些设计时才提高优先级。参考实现差异不能混充单一开关实验。

## 5. CCPS内部机制：A1/A2/A10/A11已纳入后续定时队列

用户已明确授权在S4后依次训练A1→A2→A10→A11。均以ASEE完整结构、原增强、seed=0、8-view为基础，presence=0，query gate关闭，其他训练设置一致；A0为兼容检查、A3为已有参考，均不重复安排正式训练。

| ID（沿用原Plan） | 设置 | 目的 |
|---|---|---|
| A0 | selector off | 验证关闭CCPS与原ASEE兼容；同权重forward等价是接口检查，不证明完整训练逐位一致 |
| A1 | global，patch gate开 | 全局类别权重对照 |
| A2 | visual，patch gate开 | 内容相关选择对照 |
| A3（已有完整模型） | visual_prior，patch gate开 | 视觉与先验联合选择 |
| A10 | 主配置但context_only | 仅类别上下文重加权 |
| A11 | 主配置但residual_only | 仅残差强度调节 |
| A3（已有完整模型） | both | 两种作用联合效果 |

优先补A1、A2、A10、A11；A0先做兼容检查，已有M1能否作为训练对照需核对初始化、优化器参数组、RNG和配置，而非自动假定一致。A1→A2也可能改变选择器结构/参数量，应报告，不能说绝对只改变输入信息。

原Plan的A4/query-only、A5/patch+query、A6/A7 presence监督、隐藏维度和初始化概率搜索属于可选扩展。主模型没有使用的presence损失无需为了表格数量强行加入。不要把“不如主配置”的组删除。

## 6. 增强实验的表述边界

独立联合增强run按最新指令于2026-10-25 14:56 America/Denver提前停止，实际状态以后续停止报告为准；不属于新增消融队列。后续所有消融恢复使用原增强，seed统一0。被计划早停的历史联合增强方案为：旋转±10度、p=0.3，尺度0.75–1.50；保留原水平翻转、颜色增强和监督规则。现有联合增强入口独立，不改原增强。

本轮不追加旋转单独/尺度单独训练。该联合增强对照只能用于评估组合效果，不能分别声称旋转或尺度贡献，也不能证明协同。如果论文要作这些机制主张，才另行建议2×2设计：原增强、仅旋转、仅尺度、二者联合，全部同一8-view；须用户另行授权。
提高0.5个百分点是研究目标，不是成功标准或保证，不按结果筛掉负面实验。

## 7. 公平比较与现有证据限制

固定18训练病例/2211切片及12测试病例，384输入、patch16、batch16、AdamW/LLRD/LR/预热/损失/精度/compile及先验版本等按原配置核验；先验只能来自训练数据。新增消融单独模型开关/YAML/输出目录，运行清单记录全部差异。

历史baseline与ASEE、CCPS的训练预算和评估窗口必须从各run核验。已知CCPS seed0仅完成171–213的逐轮评估，baseline完成220；不能标为所有组均完整220轮。当前M0/M1按用户要求各自历史best补测8-view，首先作为“历史best统一推理对照”；这并不能消除选权重窗口不等的影响。
严格匹配窗口时需171–213同一规则挑选并有对应权重。仅有best/last时可能无法恢复该窗口最佳权重；应标记不可获得，不能拿全程best冒充，也不自动重训。

目前测试集参与checkpoint选择、TTA方案和增强开发，不能把这12例描述为完全未参与开发的独立最终测试。保持现有代码与历史成绩，但论文如实说明；独立验证/外部测试或新划分属于未来设计，须另行授权，不悄悄改划分。

## 8. 指标与不确定性

每组固定一个checkpoint，完整12病例，按核验label_id展示8器官Dice(%)、HD95_mm及8器官平均；保留HD95_voxel为追溯列，不混合单位、不报告背景。器官映射读取priors/synapse_label_map.verified.json；间距来自已核验spacing表。

默认沿用用户指定legacy空掩码规则。此规则存在pred非空/GT空给Dice=1的问题，论文须明确口径，记录触发数；不能仅因作者代码如此而称其数学上正确。已修正口径若补充评估，所有对照同一预测统一计算并单列，不能只修某组；本文件不授权切换默认逻辑。

训练重复建议在后续确认稳定性时使用至少3个预先确定且各组匹配的seed，报告全部结果、均值与样本标准差；这是建议，不是统一期刊硬规定。用户当前暂缓seed研究，不自动运行。已提前停止seed42不能当作与完整训练同预算的重复，更不能隐去不利结果。

病例配对差值/置信区间以病例为抽样单位，不能把切片或8器官当成独立患者。可对固定checkpoint的12病例做配对bootstrap，并明确它不包含训练seed及模型选择不确定性；大量组件比较如作显著性检验需预先指定比较和多重比较处理。12例证据有限，不强行写显著改善。

## 9. 论文表格与最小交付

1. 本次补测表仅M0和M1：统一当前最优8-view，逐器官Dice/HD95_mm与均值，注明seed、预算、checkpoint选择窗口。已有完整模型结果可在核验后单列历史参考，不增加本次补测项。
2. ASEE机制表：S1/S2/M1与S3/S4/M1，两组逻辑分开。
3. CCPS机制表：A1/A2/A3和A10/A11/A3，不重复训练相同主配置。
4. 效率表：参数量、同硬件8-view推理时间、峰值显存；GPU推理与CPU HD95计时分开，不把指标计算瓶颈归因于网络。
5. 补充材料：逐病例结果、失败病例、固定可视化样例和完整配置/版本/哈希。无需安排有无TTA消融表。

每个实验登记：状态(计划/已配置/冒烟/运行/完成/提前停止)、run目录、源码与权重SHA、seed、epoch/global_step、增强、TTA、实际12病例完整性、均值及逐类CSV。已有配置不是已完成实验，历史结果只能核验后复用。

## 10. 执行次序与最新授权

用户已于2026-10-25明确授权：14:56 Denver提前停止当前联合增强run，下一阶段按GPU_CPU_HD95_Optimization_Guide.md准备并验证保持指标一致的加速；之后依序执行M0→M1→S1→S2→S3→S4→A1→A2→A10→A11。
所有新增训练采用原增强与seed=0，不采用此次旋转/尺度联合增强。M0/M1仅测试已有best；S/A组件独立训练，科学变量保持本文件各节要求。
15:30起共7个单次任务，每隔1小时一次，第一项负责优化验证，之后检查并推进；时间表、进程保护、验证门槛与幂等约定见[synapse_ablation_scheduled_runbook.md](synapse_ablation_scheduled_runbook.md)。
输出专用目录outputs/ablations/journal_20261025/。只有当前实验完整完成且结果核验通过，才能标完成并推进；失败不静默重训，不自行增加七次之外的监控。
本授权取代此前“仅写文档、等待启动”的限制，不授权额外搜索、多seed、A4–A9、增强拆分或改变评分协议。

## 11. 实验状态登记

设置时以下项目均为“待执行”，不代表已开始。每次完成后更新状态、真实run路径、权重SHA、epoch与报告链接，并保留提前停止/失败的证据。

| 顺序 | ID | 内容 | 状态 | 结果与证据 |
|---|---|---|---|---|
| 前置 | OPT | 指标等价加速及完整回归 | 已完成 | outputs/ablations/journal_20261025/optimization/readiness.json；完整结果见同目录summary.md |
| 1 | M0 | baseline已有best + 8-view | 已完成 | outputs/ablations/journal_20261025/M0/20261025T224447Z/summary.md |
| 2 | M1 | baseline+ASEE已有best + 8-view | 已完成 | outputs/ablations/journal_20261025/M1/20261025T225349Z/summary.md |
| 3 | S1 | ASEE文本单分支训练+best 8-view | 已完成 | outputs/ablations/journal_20261025/S1/20261025T230957Z/eight_view_tests/20261026T091846Z/summary.md |
| 4 | S2 | ASEE共现单分支训练+best 8-view | 用户主动早停（未完成） | outputs/ablations/journal_20261025/S2/20261026T231139Z/early_stop_report.json |
| 5 | S3 | ASEE patch单分支训练+best 8-view | 待执行（前序未完成，四次定时检查已结束） | .codex/s2_s3_scheduled_runbook.md；17:40检查S2已完成36轮，未启动S3 |
| 6 | S4 | ASEE query单分支训练+best 8-view | 待执行 | — |
| 7 | A1 | CCPS global训练+best 8-view | 待执行 | — |
| 8 | A2 | CCPS visual训练+best 8-view | 待执行 | — |
| 9 | A10 | CCPS context_only训练+best 8-view | 待执行 | — |
| 10 | A11 | CCPS residual_only训练+best 8-view | 待执行 | — |

机器可读状态为experiment_registry.json与orchestration_state.json，位于专用输出目录。MD与账本按同一核验结果同步，不能只根据PID消失或只有训练结束就标为已完成。


## 2026-10-26 S2/S3定时安排

用户已明确授权新ROI baseline-2完成后执行S2，再在S2训练和同best8-view完整完成后执行S3。四次检查为Denver10月26日12:10:19、12:40:19、17:10:19、17:40:19；细则、命令和任务ID见s2_s3_scheduled_runbook.md。本轮仅完成准备和定时设置，尚未启动S2/S3，不将已定时写成已完成。
