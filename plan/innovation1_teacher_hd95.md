> 最新正式补跑目录：outputs/teacher_hd95/innovation1_best/20261023T091726Z。主指标HD95_mm；第一次直接像素分数适配已中止并作废，最新采用FP32 log-score适配。76项CPU回归通过，GPU全部1568切片额外softmax不再改变argmax。

# Innovation1老师HD95代码接入与补跑

用户要求接入HD95_test.py并明确主HD95采用真实物理距离。实际上传依赖位于AgileFormer-main/AgileFormer-main/utils.py，SHA256=6e7eceb5f6e1dd7571c38094efd85951c229989c89e6cdc8f90b8bed92309a2c。原始函数未修改，按哈希加载并实际调用；原HD95_test.py已归档至outputs/audit/teacher_hd95_integration_20261023T085802Z/HD95_test.original.py。

老师原calculate_metric_percase不传spacing，原z_spacing仅用于保存图像header。本次保留原函数的Dice和体素HD95，并在同一组预测/GT掩码上调用MedPy传入核验ZYX间距，主报告为mm。没有用固定z=1或从旧标量换算。原空掩码规则继承不变。

HD95_test.py已适配到EoSeg原CLI，默认新独立配置；移除不可用的AgileFormer模型构建、tiny/base文件名判断、顶层参数解析和通配导入依赖。新的EoSegPixelAdapter把输入灰度复制成3通道，将query输出按原方式变成像素分数并双线性上采样；教师函数执行切片缩放、softmax/argmax、最近邻恢复原尺寸。额外softmax使用FP32以避免额外FP16舍入；模型/解码仍为16-mixed和原compile路径，不重复归一化。保留原EoSeg cuDNN默认设置，不套用老师入口的deterministic开关。原始utils源文件和baseline训练源码/配置未改。

main.py的独立test类保留白名单加入SynapseTeacherHD95Semantic，防止checkpoint元数据把教师包装类覆盖回原训练类；普通fit/test继续原行为。新增7项教师函数/适配测试，全部既有和新增CPU回归75/75通过；真实checkpoint配置解析通过。CPU微型fixture仅将Tensor.cuda临时替换为identity，不影响GPU生产执行。

本次只补跑196轮best_test.ckpt，不启动训练；输出outputs/teacher_hd95/innovation1_best/20261023T090443Z。主入口、命令、原始/修改源码哈希、checkpoint哈希与单位见launch.json。每完成一例核对原Dice/体素HD95和前次毫米结果，小于1e-10，否则中止；case_replay_progress.json提供进度。正常完成后自动生成results_summary.md及本记录的完成节，status.json负责最终状态。

## 真实GPU发现的问题与修复

首次真实补跑发现首例Dice/HD95不一致，严格检查立即中止。诊断实际扫描全部12例1568切片，原分数直接输入softmax改变11194个384网格像素；一个像素的有效类别分数仅为5.960464477539063e-08，FP32 softmax后9类完全并列。修复是在模型/解码完成后取FP32 log-score，再交给老师softmax归一化，保留原argmax；全零像素保持class0。先有失败回归再修复，76项CPU回归及GPU全1568切片诊断均通过，后者变化像素为0。未改变主干、权重、精度、尺寸或距离公式，未放宽逐病例对照。诊断日志中的距离0是临时stub值，不是模型成绩；正式成绩来自独立完整补跑。

## 本次已正常完成

# Innovation1最佳权重：接入老师函数后的毫米HD95

第196轮best_test.ckpt，完整12病例、96条病例-器官记录。主HD95为真实物理距离mm；实际执行AgileFormer-main上传的test_single_volume和calculate_metric_percase，后者原始voxel结果作为对照。

平均Dice=85.447484%；平均HD95_mm=23.235310mm；平均HD95_voxel=21.668917体素。

|ID|器官|Dice (%)|HD95 (mm)|HD95 (体素)|
|---|---|---:|---:|---:|
|1|主动脉 (Aorta)|88.0847|48.5782|46.6431|
|2|胆囊 (Gallbladder)|73.0726|17.5065|16.8507|
|3|左肾 (Left Kidney)|87.6666|18.8443|17.6050|
|4|右肾 (Right Kidney)|85.7239|28.5751|28.3360|
|5|肝脏 (Liver)|95.7638|36.4183|39.5081|
|6|胰腺 (Pancreas)|72.5839|7.9095|5.0154|
|7|脾脏 (Spleen)|93.3715|17.0569|14.5000|
|8|胃 (Stomach)|87.3129|10.9937|4.8931|

全部12例的Dice/体素HD95与原训练196轮记录一致，毫米HD95与前一轮物理评估一致；最大差异<1e-10。保留原空掩码规则，胆囊1例空GT的HD95=0为继承占位值。未修改或重训checkpoint。teacher_metric_contract.json记录实际源文件与适配细节。

结果路径：outputs/teacher_hd95/innovation1_best/20261023T091726Z
