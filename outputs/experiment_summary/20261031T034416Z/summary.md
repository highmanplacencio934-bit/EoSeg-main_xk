# outputs主实验、重复训练与消融全量汇总

核验UTC：2026-10-31T03:48:01.734912+00:00。本报告只读原始产物，不停止/重启训练，不修改模型、训练配置、原CSV或checkpoint。

## CUDA标记与统计口径

“CUDA修复=开”表示实际启用ce2d_stableseg_execution_v2：独立稳定CE/点采样、严格deterministic guard；CCPS另有局部BN边界/AOT backward off。不是“开启CUDA非确定性”。“关”表示原CUDA入口，未接入这套修复，不能据此断言每个算子都非确定。开也不能保证两次完整训练最终逐位相同。ROI只属指标基础设施，不等于CUDA修复。

所有训练主表为各自最佳checkpoint的1-view legacy Dice；8-view不混入主表。单位Dice原值0..1、表内百分数；HD95_mm/HD95_voxel分列，未计算为“—”，不是0。背景0不整理；器官按verified label_id映射，每组8器官来自同一best，绝不逐器官跨epoch挑最大。

发现16组已形成评分的正式训练，状态计数{'completed': 12, 'stopped_by_user': 3, 'running': 1}；另有12份完整12例独立推理结果（含明确标注的重复复测）。冒烟/审计/预备目录不作为论文主实验，未执行计划不伪造结果。

## 全部正式训练（主实验、组件消融、seed与收敛性检查）

|ID|实验|CUDA修复|ROI|seed|训练轮数/预算|状态|best epoch|1-view Dice%|HD95_mm|HD95_voxel|
|---|---|---|---|---:|---|---|---:|---:|---:|---:|
|E01|S2仅共现（V2）|开(V2)|开|0|220/220|completed|177|86.161264|—|—|
|E02|S1仅文本（V2）|开(V2)|开|0|220/220|completed|197|86.505362|—|—|
|E03|S1仅文本（原CUDA）|关|开|0|220/220|completed|187|86.696942|17.270780|14.998108|
|E04|S2仅共现（原CUDA，早停）|关|开|0|211/220|stopped_by_user|195|85.214482|33.453487|31.933530|
|E05|ASEE双先验主实验（V2修复）|开(V2)|开|0|220/220|completed|190|86.381785|—|—|
|E06|原始baseline|关|关|0|220/220|completed|177|84.718466|—|47.389567|
|E07|baseline-2（ROI版）|关|开|0|220/220|completed|173|85.913756|32.509347|27.659047|
|E08|baseline-lock 第1次|关|关|0|220/220|completed|175|86.464784|—|—|
|E09|baseline-lock 第2次|关|关|0|220/220|completed|181|86.731628|—|—|
|E10|baseline-repair 第1次|开(V2)|开|0|220/220|completed|184|86.283648|17.997721|—|
|E11|baseline-repair_2 第2次|开(V2)|开|0|220/220|completed|181|86.441231|20.644387|—|
|E12|ASEE+CCPS完整主模型（V2）|开(V2)|开|0|211/220|running|198|86.327317|—|—|
|E13|ASEE双先验主实验（原CUDA）|关|关|0|220/220|completed|196|85.447484|—|21.668917|
|E14|ASEE+CCPS 低LR补训100轮（V2）|开(V2)|开|0|100/100|completed|15|86.381773|25.987788|—|
|E15|ASEE+CCPS seed0（原CUDA）|关|关|0|214/220|stopped_by_user|187|86.242979|29.744758|29.113595|
|E16|ASEE+CCPS seed42（原CUDA）|关|关|42|198/220|stopped_by_user|186|84.090398|49.705577|48.269474|

低LR组是历史CCPS best187的weights-only补训，LR2e-5、额外100轮，优化器/调度器重建且使用V2；不冒充从原预训练的220轮对照。原CUDA CCPS seed0/42及旧S2均用户早停，候选窗口不等于完整50轮。运行中的完整主模型为本报告快照，后续可能改变。

## 正式训练逐器官Dice（%）

|ID|实验|主动脉|胆囊|左肾|右肾|肝脏|胰腺|脾脏|胃|平均|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|E01|S2仅共现（V2）|89.913135|77.722282|88.600370|85.114377|92.743988|73.958118|93.738782|87.499062|86.161264|
|E02|S1仅文本（V2）|90.708830|73.394044|88.849484|87.069460|95.896680|73.972658|93.619784|88.531956|86.505362|
|E03|S1仅文本（原CUDA）|90.410344|75.044345|90.092873|86.993407|96.268377|73.953752|94.167669|86.644768|86.696942|
|E04|S2仅共现（原CUDA，早停）|82.661272|75.501843|90.252582|86.887422|91.961103|73.710376|93.577375|87.163882|85.214482|
|E05|ASEE双先验主实验（V2修复）|90.666832|74.368865|90.616072|87.448442|95.841968|72.333267|93.175074|86.603760|86.381785|
|E06|原始baseline|84.612251|72.106025|86.538600|86.436181|95.344391|72.845100|92.165197|87.699983|84.718466|
|E07|baseline-2（ROI版）|89.234858|75.291771|89.027706|84.020253|95.529554|74.205599|93.786651|86.213656|85.913756|
|E08|baseline-lock 第1次|90.259389|73.649958|88.829459|86.299187|95.885783|75.154904|93.871470|87.768122|86.464784|
|E09|baseline-lock 第2次|90.513010|74.601248|90.156683|85.973517|95.366311|74.923933|94.322252|87.996069|86.731628|
|E10|baseline-repair 第1次|90.243415|73.669679|87.286497|86.609863|95.265816|74.966350|94.010170|88.217394|86.283648|
|E11|baseline-repair_2 第2次|90.562318|72.682647|88.247882|87.017682|96.115979|75.036556|93.632771|88.234014|86.441231|
|E12|ASEE+CCPS完整主模型（V2）|90.567153|74.708656|89.482432|86.706830|95.651358|72.738927|93.128338|87.634839|86.327317|
|E13|ASEE双先验主实验（原CUDA）|88.084674|73.072619|87.666605|85.723927|95.763757|72.583933|93.371459|87.312903|85.447484|
|E14|ASEE+CCPS 低LR补训100轮（V2）|90.452632|74.312219|88.774312|87.282680|94.707625|73.253199|93.670578|88.600938|86.381773|
|E15|ASEE+CCPS seed0（原CUDA）|90.314036|74.651480|87.828996|87.241810|94.511409|73.732226|93.539985|88.123890|86.242979|
|E16|ASEE+CCPS seed42（原CUDA）|87.062227|72.363241|77.918013|85.939919|95.948737|73.621851|93.337363|86.531832|84.090398|

## 独立推理：TTA、物理HD95及重复复测

表中CUDA标记为“源权重训练时”的修复开关；这些任务没有反向训练，不能用推理入口是否接入确定性loss来判断源训练协议。

|ID|源训练|源训练CUDA修复|视角/协议|Dice%|HD95_mm|HD95_voxel|重复说明|
|---|---|---|---|---:|---:|---:|---|
|T01|原始baseline|关|8 / scale_hflip|86.849528|12.968490|10.106415|独立测试|
|T02|ASEE双先验主实验（原CUDA）|关|8 / scale_hflip|86.086021|11.928217|9.587378|独立测试|
|T03|S1仅文本（原CUDA）|关|8 / scale_hflip|87.054982|8.874617|6.599391|独立测试|
|T04|baseline-repair 第1次|开(V2)|8 / scale_hflip|86.975607|9.701723|—|独立测试|
|T05|ASEE双先验主实验（原CUDA）|关|1 / 1-view|85.447484|23.235310|21.668917|独立测试|
|T06|原始baseline|关|8 / scale_hflip|86.849528|12.968490|10.106415|复测同T01|
|T07|ASEE双先验主实验（原CUDA）|关|1 / 1-view / 教师适配|85.447484|23.235310|21.668917|独立测试|
|T08|ASEE+CCPS seed0（原CUDA）|关|4 / scale_hflip|86.452230|19.041325|17.382331|独立测试|
|T09|ASEE+CCPS seed0（原CUDA）|关|6 / scale_hflip|86.835536|19.065135|17.705635|独立测试|
|T10|ASEE+CCPS seed0（原CUDA）|关|8 / scale_hflip|86.978703|11.142189|9.017792|独立测试|
|T11|ASEE+CCPS seed0（原CUDA）|关|2 / legacy_hflip_2|86.442904|26.298295|25.242640|独立测试|
|T12|ASEE+CCPS seed0（原CUDA）|关|4 / retired_flip_4_with_vertical|60.152770|71.539775|80.174001|独立测试|

当前8-view均为四尺度1/.875/1.125/.75×原方向/水平翻转；历史4-view含垂直翻转的结果已明确标为retired_flip_4_with_vertical，不当成当前多尺度4-view，也不恢复退役代码。M0与指定baseline best复测是同一权重/协议/结果，不能当两组独立训练。教师HD95与物理HD95同权重同指标但接口不同，保留各自来源。

## 独立推理逐器官Dice（%）

|ID|源训练/视角|主动脉|胆囊|左肾|右肾|肝脏|胰腺|脾脏|胃|平均|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|T01|原始baseline/8|90.617997|73.307718|88.888896|87.029893|96.364918|75.942083|93.863316|88.781406|86.849528|
|T02|ASEE双先验主实验（原CUDA）/8|90.512888|72.422171|87.761099|86.638421|96.292855|73.016605|93.677808|88.366325|86.086021|
|T03|S1仅文本（原CUDA）/8|91.247921|74.638180|89.502432|87.283013|96.460140|74.727555|94.337940|88.242680|87.054982|
|T04|baseline-repair 第1次/8|90.956851|74.631696|87.605655|86.983775|96.279375|75.940983|94.420610|88.985911|86.975607|
|T05|ASEE双先验主实验（原CUDA）/1|88.084674|73.072619|87.666605|85.723927|95.763757|72.583933|93.371459|87.312903|85.447484|
|T06|原始baseline/8|90.617997|73.307718|88.888896|87.029893|96.364918|75.942083|93.863316|88.781406|86.849528|
|T07|ASEE双先验主实验（原CUDA）/1|88.084674|73.072619|87.666605|85.723927|95.763757|72.583933|93.371459|87.312903|85.447484|
|T08|ASEE+CCPS seed0（原CUDA）/4|91.245835|74.391697|86.345549|86.963983|96.036577|74.086318|93.709673|88.838208|86.452230|
|T09|ASEE+CCPS seed0（原CUDA）/6|91.440313|75.782751|86.459757|87.214802|96.159502|74.631323|93.815943|89.179899|86.835536|
|T10|ASEE+CCPS seed0（原CUDA）/8|91.248864|75.345409|88.726999|87.040226|96.280853|74.227474|93.737375|89.222424|86.978703|
|T11|ASEE+CCPS seed0（原CUDA）/2|90.744857|74.533595|87.252385|87.166323|96.037341|73.635208|93.802767|88.370759|86.442904|
|T12|ASEE+CCPS seed0（原CUDA）/4|84.129127|55.753131|59.599992|29.671555|89.991506|50.122930|33.154989|78.798927|60.152770|

## 逐器官HD95与完整来源

training_organ_metrics.csv、independent_test_organs.csv逐行保存器官名/label_id/Dice原值与百分数/HD95_mm/HD95_voxel；experiments.json保存每组所有8器官和真实路径，evidence.json记录配置/入口/确定性guard的证据与哈希。

- E01 S2仅共现（V2）：[outputs/ablations/asee_probability_only_v2/20261030T185908Z](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/ablations/asee_probability_only_v2/20261030T185908Z>)；选择窗口171-220；来源[outputs/ablations/asee_probability_only_v2/20261030T185908Z/synapse_test_selection/20261030T185935Z_001efa2a/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/ablations/asee_probability_only_v2/20261030T185908Z/synapse_test_selection/20261030T185935Z_001efa2a/selection_summary.json>)。
- E02 S1仅文本（V2）：[outputs/ablations/asee_text_only_v2/20261030T125953Z](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/ablations/asee_text_only_v2/20261030T125953Z>)；选择窗口171-220；来源[outputs/ablations/asee_text_only_v2/20261030T125953Z/synapse_test_selection/20261030T130018Z_83a1d0b0/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/ablations/asee_text_only_v2/20261030T125953Z/synapse_test_selection/20261030T130018Z_83a1d0b0/selection_summary.json>)。
- E03 S1仅文本（原CUDA）：[outputs/ablations/journal_20261025/S1/20261025T230957Z/training](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/ablations/journal_20261025/S1/20261025T230957Z/training>)；选择窗口171-220；来源[outputs/ablations/journal_20261025/S1/20261025T230957Z/training/synapse_test_selection/20261025T231010Z_ca94e093/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/ablations/journal_20261025/S1/20261025T230957Z/training/synapse_test_selection/20261025T231010Z_ca94e093/selection_summary.json>)。
- E04 S2仅共现（原CUDA，早停）：[outputs/ablations/journal_20261025/S2/20261026T231139Z/training](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/ablations/journal_20261025/S2/20261026T231139Z/training>)；选择窗口171-210；来源[outputs/ablations/journal_20261025/S2/20261026T231139Z/training/synapse_test_selection/20261026T231154Z_63a12319/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/ablations/journal_20261025/S2/20261026T231139Z/training/synapse_test_selection/20261026T231154Z_63a12319/selection_summary.json>)。
- E05 ASEE双先验主实验（V2修复）：[outputs/asee-repair/repeat_1/20261029T075801Z](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/asee-repair/repeat_1/20261029T075801Z>)；选择窗口171-220；来源[outputs/asee-repair/repeat_1/20261029T075801Z/synapse_test_selection/20261029T075825Z_449bb9df/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/asee-repair/repeat_1/20261029T075801Z/synapse_test_selection/20261029T075825Z_449bb9df/selection_summary.json>)。
- E06 原始baseline：[outputs/baseline/20261021T132635Z](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/baseline/20261021T132635Z>)；选择窗口171-220；来源[outputs/baseline/20261021T132635Z/synapse_test_selection/20261021T132645Z_81f3214a/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/baseline/20261021T132635Z/synapse_test_selection/20261021T132645Z_81f3214a/selection_summary.json>)。
- E07 baseline-2（ROI版）：[outputs/baseline-2/20261026T130228Z](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/baseline-2/20261026T130228Z>)；选择窗口171-220；来源[outputs/baseline-2/20261026T130228Z/synapse_test_selection/20261026T130239Z_ffcc3345/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/baseline-2/20261026T130228Z/synapse_test_selection/20261026T130239Z_ffcc3345/selection_summary.json>)。
- E08 baseline-lock 第1次：[outputs/baseline-lock/20261030T031914Z](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/baseline-lock/20261030T031914Z>)；选择窗口171-220；来源[outputs/baseline-lock/20261030T031914Z/synapse_test_selection/20261030T031928Z_9fb167de/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/baseline-lock/20261030T031914Z/synapse_test_selection/20261030T031928Z_9fb167de/selection_summary.json>)。
- E09 baseline-lock 第2次：[outputs/baseline-lock/repeat_2/20261030T080537Z](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/baseline-lock/repeat_2/20261030T080537Z>)；选择窗口171-220；来源[outputs/baseline-lock/repeat_2/20261030T080537Z/synapse_test_selection/20261030T080551Z_e015b90f/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/baseline-lock/repeat_2/20261030T080537Z/synapse_test_selection/20261030T080551Z_e015b90f/selection_summary.json>)。
- E10 baseline-repair 第1次：[outputs/baseline-repair/G6_1/20261028T072539Z](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/baseline-repair/G6_1/20261028T072539Z>)；选择窗口171-220；来源[outputs/baseline-repair/G6_1/20261028T072539Z/synapse_test_selection/20261028T072656Z_c6ab2a6d/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/baseline-repair/G6_1/20261028T072539Z/synapse_test_selection/20261028T072656Z_c6ab2a6d/selection_summary.json>)。
- E11 baseline-repair_2 第2次：[outputs/baseline-repair_2/G6_2/20261029T022332Z](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/baseline-repair_2/G6_2/20261029T022332Z>)；选择窗口171-220；来源[outputs/baseline-repair_2/G6_2/20261029T022332Z/synapse_test_selection/20261029T022451Z_3453b054/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/baseline-repair_2/G6_2/20261029T022332Z/synapse_test_selection/20261029T022451Z_3453b054/selection_summary.json>)。
- E12 ASEE+CCPS完整主模型（V2）：[outputs/experiments/asee_ccps_full_v2/20261030T233233Z](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/experiments/asee_ccps_full_v2/20261030T233233Z>)；选择窗口171-211；来源[outputs/experiments/asee_ccps_full_v2/20261030T233233Z/synapse_test_selection/20261030T233258Z_37cace9d/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/experiments/asee_ccps_full_v2/20261030T233233Z/synapse_test_selection/20261030T233258Z_37cace9d/selection_summary.json>)。
- E13 ASEE双先验主实验（原CUDA）：[outputs/innovation1/asee_plan/20261022T044551Z](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/innovation1/asee_plan/20261022T044551Z>)；选择窗口171-220；来源[outputs/innovation1/asee_plan/20261022T044551Z/synapse_test_selection/20261022T044607Z_1ba7c5b9/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/innovation1/asee_plan/20261022T044551Z/synapse_test_selection/20261022T044607Z_1ba7c5b9/selection_summary.json>)。
- E14 ASEE+CCPS 低LR补训100轮（V2）：[outputs/innovation2/ccps_low_lr_2e-5_100ep/20261028T124208Z](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/innovation2/ccps_low_lr_2e-5_100ep/20261028T124208Z>)；选择窗口1-100；来源[outputs/innovation2/ccps_low_lr_2e-5_100ep/20261028T124208Z/synapse_test_selection/20261028T124805Z_f151bb6a/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/innovation2/ccps_low_lr_2e-5_100ep/20261028T124208Z/synapse_test_selection/20261028T124805Z_f151bb6a/selection_summary.json>)。
- E15 ASEE+CCPS seed0（原CUDA）：[outputs/innovation2/ccps_plan/20261023T121754Z](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/innovation2/ccps_plan/20261023T121754Z>)；选择窗口171-213；来源[outputs/innovation2/ccps_plan/20261023T121754Z/synapse_test_selection/20261023T121806Z_c99f1c48/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/innovation2/ccps_plan/20261023T121754Z/synapse_test_selection/20261023T121806Z_c99f1c48/selection_summary.json>)。
- E16 ASEE+CCPS seed42（原CUDA）：[outputs/innovation2/ccps_seed42/20261024T131734Z](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/innovation2/ccps_seed42/20261024T131734Z>)；选择窗口171-197；来源[outputs/innovation2/ccps_seed42/20261024T131734Z/synapse_test_selection/20261024T132218Z_f39d912f/selection_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/innovation2/ccps_seed42/20261024T131734Z/synapse_test_selection/20261024T132218Z_f39d912f/selection_summary.json>)。
- T01：[outputs/ablations/journal_20261025/M0/20261025T224447Z/evaluation/independent_test/20261025T224503Z_da030f9e/physical_test_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/ablations/journal_20261025/M0/20261025T224447Z/evaluation/independent_test/20261025T224503Z_da030f9e/physical_test_summary.json>)，源训练E06，best177。
- T02：[outputs/ablations/journal_20261025/M1/20261025T225349Z/evaluation/independent_test/20261025T225405Z_a5c6704b/physical_test_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/ablations/journal_20261025/M1/20261025T225349Z/evaluation/independent_test/20261025T225405Z_a5c6704b/physical_test_summary.json>)，源训练E13，best196。
- T03：[outputs/ablations/journal_20261025/S1/20261025T230957Z/eight_view_tests/20261026T091846Z/evaluation/independent_test/20261026T092041Z_6b93c117/physical_test_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/ablations/journal_20261025/S1/20261025T230957Z/eight_view_tests/20261026T091846Z/evaluation/independent_test/20261026T092041Z_6b93c117/physical_test_summary.json>)，源训练E03，best187。
- T04：[outputs/baseline-repair/G6_1/20261028T072539Z/tta_scale_hflip_8/20261028T121033Z/evaluation/independent_test/20261028T121047Z_b607ca34/physical_test_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/baseline-repair/G6_1/20261028T072539Z/tta_scale_hflip_8/20261028T121033Z/evaluation/independent_test/20261028T121047Z_b607ca34/physical_test_summary.json>)，源训练E10，best184。
- T05：[outputs/physical_hd95/innovation1_best/20261023T030526Z/independent_test/20261023T030542Z_ecb584c2/physical_test_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/physical_hd95/innovation1_best/20261023T030526Z/independent_test/20261023T030542Z_ecb584c2/physical_test_summary.json>)，源训练E13，best196。
- T06：[outputs/rechecks/baseline_best_8view/20261026T074728Z/evaluation/independent_test/20261026T074742Z_204712ee/physical_test_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/rechecks/baseline_best_8view/20261026T074728Z/evaluation/independent_test/20261026T074742Z_204712ee/physical_test_summary.json>)，源训练E06，best177。
- T07：[outputs/teacher_hd95/innovation1_best/20261023T091726Z/independent_test/20261023T091738Z_bb046923/physical_test_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/teacher_hd95/innovation1_best/20261023T091726Z/independent_test/20261023T091738Z_bb046923/physical_test_summary.json>)，源训练E13，best196。
- T08：[outputs/tta/innovation2_best_scale_hflip4/20261024T111222Z/independent_test/20261024T111315Z_f9bac2c5/physical_test_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/tta/innovation2_best_scale_hflip4/20261024T111222Z/independent_test/20261024T111315Z_f9bac2c5/physical_test_summary.json>)，源训练E15，best187。
- T09：[outputs/tta/innovation2_best_scale_hflip6/20261024T114632Z/independent_test/20261024T114646Z_7cdde881/physical_test_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/tta/innovation2_best_scale_hflip6/20261024T114632Z/independent_test/20261024T114646Z_7cdde881/physical_test_summary.json>)，源训练E15，best187。
- T10：[outputs/tta/innovation2_best_scale_hflip8/20261024T123754Z/independent_test/20261024T123808Z_bfa7fe37/physical_test_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/tta/innovation2_best_scale_hflip8/20261024T123754Z/independent_test/20261024T123808Z_bfa7fe37/physical_test_summary.json>)，源训练E15，best187。
- T11：[outputs/tta/innovation2_best_views2/20261024T084750Z/independent_test/20261024T084804Z_c9b10bc0/physical_test_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/tta/innovation2_best_views2/20261024T084750Z/independent_test/20261024T084804Z_c9b10bc0/physical_test_summary.json>)，源训练E15，best187。
- T12：[outputs/tta/innovation2_best_views4/20261024T091548Z/independent_test/20261024T091602Z_e2823c0a/physical_test_summary.json](<//sshfs.r/sk@anhuiuniversity.icu!21103/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/tta/innovation2_best_views4/20261024T091548Z/independent_test/20261024T091602Z_e2823c0a/physical_test_summary.json>)，源训练E15，best187。

## 未形成测试成绩的历史seed1/2条目

|条目|seed|状态|实际启动|已保存轮数|CUDA修复|平均Dice|
|---|---:|---|---|---|---|---|
|asee_seed1|1|cancelled_by_user|否|—|计划为关，未启动|未形成正式评估|
|asee_seed2|2|cancelled_by_user|否|—|计划为关，未启动|未形成正式评估|
|baseline_seed1|1|stopped_by_user|是|26|关（原入口）|未形成正式评估|
|baseline_seed2|2|cancelled_by_user|否|—|计划为关，未启动|未形成正式评估|

这4项不混入16组已有评分的总表，其中1项已启动后早停且未到评估阶段，3项在排队阶段取消。状态和路径另见unscored_seed_records.csv。

## 边界

S3/S4、A1/A2/A10/A11没有发现正式selection结果，不因为存在计划或配置就记为完成；ASEE修复第二次未发现正式run。已按用户删除的旧非ROI baseline-2和旋转增强输出不在当前outputs中，不从文档估造最终成绩。所有旧run文档标题只作提示，实际模型开关以归档配置/协议/状态核验为准。完整主模型相对单先验同时增加另一先验和CCPS，三组不能单独隔离CCPS贡献。不同训练协议、seed、预算、TTA和距离单位不可直接混成一个排行榜。
