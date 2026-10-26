# Innovation2 CCPS 用户停训结果

按用户要求停止；计划220轮，保存训练至第214轮，完整12例测试至第213轮。共43轮完整评估、516条病例记录。不是完整220轮结果。

同一第187轮best_test.ckpt，原空掩码规则，不使用TTA，12病例平均；HD95主单位mm。

|标签ID|器官|Dice (%)|HD95 (mm)|
|---|---|---:|---:|
|1|主动脉|90.3140|38.4320|
|2|胆囊|74.6515|17.3064|
|3|左肾|87.8290|58.2493|
|4|右肾|87.2418|11.4785|
|5|肝脏|94.5114|76.0930|
|6|胰腺|73.7322|7.5584|
|7|脾脏|93.5400|16.8046|
|8|胃|88.1239|12.0359|
|—|8器官平均|86.2430|29.7448|

胆囊1例空GT的HD95=0为原规则占位。预测、指标公式、原CSV和best/last权重保持；停止当前进程并取消待执行TTA，不自动重训或重启。

输出：/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/innovation2/ccps_plan/20261023T121754Z

结构化记录：/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse/outputs/innovation2/ccps_plan/20261023T121754Z/user_stop_summary.json
