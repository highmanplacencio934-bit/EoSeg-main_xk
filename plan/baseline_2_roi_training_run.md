# baseline-2：seed0纯EoSeg＋ROI指标加速

当前用户已在停止并删除旧非ROI baseline-2后明确授权：使用ROI加速重新训练baseline-2。本次从原离线timm预训练流程重新初始化，不从已删除旧run或任何分割best续训。

## 运行身份

- 项目：`/Disk/HDD/Sk/EoSeg-main_Synapse_acdc/EoSeg-main_synapse`；hostname=5090；RTX5090D32GB。
- Python：`/Disk/HDD/Sk/miniconda3/envs/DeepLearning_xiakun/bin/python`；torch2.13.0+cu132，CUDA13.2。
- 启动时间（America/Denver）：2026-10-26T07:02:28.061881-06:00。
- 新run：`outputs/baseline-2/20261026T130228Z`；当前指针：`outputs/baseline-2/latest_path.txt`。
- 训练PID=2856404，supervisor PID=2856401。操作前重新核验启动时间/cmdline/cwd；不凭旧PID发信号。
- 配置：`configs/synapse/baseline_2_roi.yaml`；入口：`scripts/run_synapse_roi.py fit --seed_everything=0 --tta_views=1`。完整argv在新run/launch.json。

## 固定训练变量

纯models.eoseg.EoSegMultiQSegFusion，16query、384×384/patch16、9类、batch16/workers8、seed0、220epochs，预计30360steps。原颜色/水平翻转/ScaleJitter(.5,2)/补零/随机384裁剪，18训练病例2211切片与12测试病例原划分。

LR=2e-4、AdamW、WD=.05、LLRD=.8、原warmup/poly/损失和点采样、norm裁剪.01、16-mixed与默认compile全部保持。未启用ASEE、CCPS、已删除的联合增强或训练TTA；不额外设置deterministic/benchmark。

HF_HUB_OFFLINE=1、TORCH_HOME=/Disk/HDD/Sk/.cache/torch、TIMM_USE_OLD_CACHE=1，W&B offline；名称baseline-2。

## ROI和评估接口

复用已验证的SynapsePhysicalHD95Semantic物理评估包装类及SynapsePhysicalHD95Selection；原模型、损失与优化器继承保持。roi_metrics_scope在启动fit时绑定，仅在评估中以pred|GT联合前景ROI+halo1调用原MedPy；保留所有前景（含远处误检）、完整几何/标签哈希校验、真实ZYX spacing和原legacy空mask策略。单CPU worker，不新增并行池。

171–220最后50轮仍每轮完整12例、以原1-view mean Dice选择同一best，保留best_test.ckpt及滚动last.ckpt。物理包装类同时记录HD95_mm与HD95_voxel；本轮test_*metrics.csv的hd95字段为mm，命名器官CSV分别明确两种单位。与首轮比较Dice/选择策略一致，HD95需使用voxel列与原voxel口径对齐，不能混单位。

本轮没有引入8-view训练内选择或按TTA重新挑epoch。未来若补8-view，只对同一best独立评估。本次不自动启动其他消融或TTA。

## 已完成验证

- 既有ROI readiness=passed，111项当前源码/配置/先验哈希与8份原验证报告哈希严格核验；既有12病例×8器官逐值等价及真实GPU FP16/compile验证沿用，未改该实现。
- 新独立配置CPU解析与首轮比较，展开data链接的img_size/num_classes后，除明确评估包装、附加spacing和输出命名字段外，科学训练配置完全一致。最初print-config省略链接字段造成比较不通过，已按实际链接值复核，不更改科学参数。
- 原baseline与ROI包装分别seed0重新构建：388个初始state tensor逐值哈希相同，309291280个参数；Python/NumPy/PyTorch初始化后RNG哈希一致；训练hparams一致。见`outputs/audit/baseline_2_roi_preflight/20261026T125656Z/cpu_validation.json`。
- 新正式run的startup_verification.json：实际device=cuda:0，ROI启用日志已确认；全部373个优化器组及defaults、30360步预算、依赖版本和数据清单与首轮一致。
- 真实新run已保存第1轮/138step的last.ckpt；这是正式训练的实际进度，不是完整220轮结果。当前状态以PID、status.json和滚动checkpoint为准。

## 结束和文件保护

监督器等待实际fit退出；仅在完整220轮退出0、171–220的50个epoch、600病例、4800病例器官记录齐全时生成summary.json/summary.md，按核验label_id输出同一best的8器官Dice、HD95_mm/voxel及平均，并比较首轮1-view Dice。失败保留真实状态，不自动重训；weights-only不代表可精确续训。

运行期间冻结本run的源码/配置/数据和seed，不热切换。旧非ROI baseline-2已按用户指令停止并删除，其最小凭证在outputs/audit/baseline_2_stop_delete/；不要把旧run当作本run或声称本次是旧优化器状态的续训。
